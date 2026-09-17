"""Reusable temporal operators for feature engineering.

These operators transform time series (one value per generation step)
into fixed-length feature vectors. They're applied across all feature
families — attention entropy, key novelty, gate sparsity, velocity norms,
etc. all get the same temporal decomposition.

Two operators:
    windowed_stats  — Split into windows, compute mean/std/slope per window
    stft_features   — Spectral decomposition via STFT
    apply_operators — Convenience: both combined
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.signal import stft as scipy_stft

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

# Minimum time series lengths for each operator
_MIN_T_WINDOWED = 4
_MIN_T_STFT = 8


# ── Cached name builders (C5) ─────────────────────────────────────────────────
# Feature names are pure functions of (prefix, n_windows); the ~thousands of
# f-strings were rebuilt for every generation. Cached as tuples — callers must
# not mutate (list(...) at the return sites keeps the public contract).

@lru_cache(maxsize=None)
def _windowed_names(prefix: str, n_windows: int) -> tuple[str, ...]:
    return tuple(
        f"{prefix}_w{wi}_{stat}"
        for wi in range(n_windows)
        for stat in ("mean", "std", "slope")
    )


@lru_cache(maxsize=None)
def _stft_names(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}_spectral_centroid",
        f"{prefix}_bandwidth",
        f"{prefix}_low_band_energy",
        f"{prefix}_mid_band_energy",
        f"{prefix}_high_band_energy",
    )


def windowed_stats(
    time_series: F32,
    n_windows: int = 4,
    prefix: str = "",
) -> tuple[F32, list[str]]:
    """Split time series into windows, compute summary stats per window.

    Per window: mean, std, slope (linear regression coefficient).
    Total features: 3 * n_windows.

    Parameters
    ----------
    time_series : array [T]
        1D time series (one value per generation step).
    n_windows : int
        Number of non-overlapping windows.
    prefix : str
        Feature name prefix (e.g., "attn_entropy_L16").

    Returns
    -------
    (features, names) — same contract as extract_tier*().
    """
    T = len(time_series)
    features: list[float] = []
    names = list(_windowed_names(prefix, n_windows))

    if T < _MIN_T_WINDOWED:
        return np.zeros(3 * n_windows, dtype=np.float32), names

    ts = np.asarray(time_series, dtype=np.float64)
    window_size = T // n_windows

    for wi in range(n_windows):
        start = wi * window_size
        end = start + window_size if wi < n_windows - 1 else T
        window = ts[start:end]

        features.append(float(window.mean()))
        features.append(float(window.std()))

        # Slope via linear regression: y = a*x + b
        if len(window) >= 2:
            x = np.arange(len(window), dtype=np.float64)
            slope = float(np.polyfit(x, window, 1)[0])
        else:
            slope = 0.0
        features.append(slope)

    return np.array(features, dtype=np.float32), names


def stft_features(
    time_series: F32,
    nperseg: int = 64,
    prefix: str = "",
) -> tuple[F32, list[str]]:
    """Spectral decomposition of a time series via STFT.

    Returns 5 features:
        spectral_centroid — power-weighted mean frequency
        bandwidth         — power-weighted std of frequency
        low_band_energy   — fraction of power in [0, 0.1) normalized freq
        mid_band_energy   — fraction of power in [0.1, 0.3) normalized freq
        high_band_energy  — fraction of power in [0.3, 0.5] normalized freq

    Parameters
    ----------
    time_series : array [T]
        1D time series.
    nperseg : int
        STFT window length. Reduced if T < nperseg.
    prefix : str
        Feature name prefix.
    """
    T = len(time_series)
    feature_names = list(_stft_names(prefix))

    if T < _MIN_T_STFT:
        return np.zeros(5, dtype=np.float32), feature_names

    ts = np.asarray(time_series, dtype=np.float64)

    # Pin nperseg to a constant so the frequency grid is identical across
    # generations of different length; zero-pad series shorter than the window.
    # For our generations T >> nperseg, so this is a no-op in practice (and a
    # robustness fix for short sequences, e.g. kotodama turns).
    if nperseg < 2:
        return np.zeros(5, dtype=np.float32), feature_names
    if T < nperseg:
        ts = np.concatenate([ts, np.zeros(nperseg - T, dtype=np.float64)])
    actual_nperseg = nperseg

    try:
        freqs, _, Zxx = scipy_stft(
            ts,
            fs=1.0,
            nperseg=actual_nperseg,
            noverlap=actual_nperseg // 2,
        )
        # Power spectrum: average over time windows
        power = np.mean(np.abs(Zxx) ** 2, axis=1)  # [n_freqs]
    except Exception as e:
        logger.warning(f"STFT failed for {prefix}: {e}")
        return np.zeros(5, dtype=np.float32), feature_names

    total_power = power.sum()
    if total_power < 1e-12:
        return np.zeros(5, dtype=np.float32), feature_names

    # Normalized power distribution
    p_norm = power / total_power

    # Spectral centroid (power-weighted mean frequency)
    spectral_centroid = float(np.sum(freqs * p_norm))

    # Bandwidth (power-weighted std)
    bandwidth = float(np.sqrt(np.sum(p_norm * (freqs - spectral_centroid) ** 2)))

    # Band energies (normalized frequencies: 0 to 0.5)
    max_freq = freqs[-1] if len(freqs) > 0 else 0.5
    if max_freq > 0:
        normalized_freqs = freqs / max_freq * 0.5
    else:
        normalized_freqs = freqs

    low_mask = normalized_freqs < 0.1
    mid_mask = (normalized_freqs >= 0.1) & (normalized_freqs < 0.3)
    high_mask = normalized_freqs >= 0.3

    low_energy = float(p_norm[low_mask].sum()) if low_mask.any() else 0.0
    mid_energy = float(p_norm[mid_mask].sum()) if mid_mask.any() else 0.0
    high_energy = float(p_norm[high_mask].sum()) if high_mask.any() else 0.0

    features = np.array([
        spectral_centroid, bandwidth,
        low_energy, mid_energy, high_energy,
    ], dtype=np.float32)

    return features, feature_names


def apply_operators(
    time_series: F32,
    prefix: str,
    n_windows: int = 4,
    include_stft: bool = True,
    nperseg: int = 64,
) -> tuple[F32, list[str]]:
    """Apply both temporal operators to a time series.

    Convenience function: windowed_stats + optional stft_features.
    Returns ~17 features (12 windowed + 5 STFT) per time series.
    """
    all_features: list[F32] = []
    all_names: list[str] = []

    f, n = windowed_stats(time_series, n_windows=n_windows, prefix=prefix)
    all_features.append(f)
    all_names.extend(n)

    if include_stft:
        f, n = stft_features(time_series, nperseg=nperseg, prefix=prefix)
        all_features.append(f)
        all_names.extend(n)

    combined = np.concatenate(all_features) if all_features else np.array([], dtype=np.float32)
    return combined, all_names
