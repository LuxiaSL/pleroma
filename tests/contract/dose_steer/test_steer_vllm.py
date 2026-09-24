"""Contract: the vLLM steering pre-hook — ``pleroma.steer.vllm`` — and the dose
lanes that install it.

Every lane installs ONE module-level hook (span x time profile x dtype
guard); none carries its own injection closure. The hook is called DIRECTLY
with vLLM's decoder-layer pre-hook args ``(positions, hidden_states, ...)`` —
no vLLM, no GPU. vLLM flattens a batch to ``positions: [num_tokens]``,
``hidden: [num_tokens, H]``.

The span: the lanes default to UNIFORM — the span the loom serves — and keep
CONTINUATION (``positions >= prompt_len``) as the named
``--injection-span continuation`` option, pinned below bit-for-bit against a
reference continuation mask, so bands measured under continuation-only
injection (the installed 70B one included) can be reproduced. Every lane
receipt records the span.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from tests.contract.dose_steer import targets as T

H = 6
PLEN = 4
NTOK = 9
SITE = 5
CONT = T.InjectionSpan.CONTINUATION
UNIF = T.InjectionSpan.UNIFORM


def _hidden(n: int = NTOK, dtype: torch.dtype = torch.float32, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, H, generator=g).to(dtype)


def _vec(seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(H, generator=g) * 2.5


def _call(state: Any, positions: Any, h: torch.Tensor,
          extra: tuple[Any, ...] = ("RESIDUAL",), kw: dict[str, Any] | None = None) -> Any:
    return T.vllm_hook(state, SITE)(None, (positions, h) + extra, dict(kw or {"k": 1}))


def _legacy_continuation(h: torch.Tensor, positions: torch.Tensor, plen: int,
                         vec: torch.Tensor) -> torch.Tensor | None:
    """REFERENCE: the continuation-only write — ``+vec`` at every position
    ``>= prompt_len``, None when there is none — that the continuation-era
    vLLM bands were measured under."""
    mask = positions >= int(plen)
    if not bool(mask.any()):
        return None
    h2 = h.clone()
    h2[mask] = h2[mask] + vec.to(h.dtype)
    return h2


# ── CONTINUATION: the reproduction option, pinned against the reference mask ─


def test_continuation_prefill_writes_only_positions_at_or_after_prompt_len() -> None:
    """On the prefill forward, rows with position < prompt_len are untouched and
    rows >= prompt_len get +vec. The return value keeps a[0], replaces a[1],
    passes a[2:] and kw through."""
    vec = _vec()
    st = T.vllm_state(vecs={SITE: vec}, prompt_len=PLEN, span=CONT)
    h = _hidden()
    pos = torch.arange(NTOK)
    (a0, h2, *rest), kw = _call(st, pos, h)
    assert a0 is pos and rest == ["RESIDUAL"] and kw == {"k": 1}
    torch.testing.assert_close(h2[:PLEN], h[:PLEN], rtol=0, atol=0)
    torch.testing.assert_close(h2[PLEN:], h[PLEN:] + vec, rtol=0, atol=0)
    assert st.fired == 1


def test_continuation_is_bit_identical_to_the_legacy_lane_hook() -> None:
    """Over a battery of prompt lengths, position layouts and dtypes, the
    continuation option returns exactly the reference tensor (or both
    None) — `--injection-span continuation` reproduces continuation-era bands."""
    cases = [(PLEN, torch.arange(NTOK)), (0, torch.arange(NTOK)), (NTOK, torch.arange(NTOK)),
             (PLEN, torch.tensor([PLEN, PLEN + 1, 0, 1, PLEN + 3])), (2, torch.tensor([7])),
             (3, torch.tensor([3, 3, 3]))]
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for plen, pos in cases:
            h = _hidden(len(pos), dtype=dtype)
            vec = _vec()
            want = _legacy_continuation(h, pos, plen, vec)
            got = _call(T.vllm_state(vecs={SITE: vec}, prompt_len=plen, span=CONT), pos, h)
            if want is None:
                assert got is None, (plen, pos, dtype)
            else:
                assert got[0][1].dtype == dtype
                assert torch.equal(got[0][1], want), (plen, pos, dtype)


@pytest.mark.parametrize("span", [UNIF, CONT])
def test_decode_step_is_written(span: T.InjectionSpan) -> None:
    """A decode step (one token per sequence, position >= prompt_len) gets the
    full vector under either span; a batch of n samples of one prompt is n
    such rows."""
    vec = _vec()
    st = T.vllm_state(vecs={SITE: vec}, prompt_len=PLEN, span=span)
    h = _hidden(3)
    (_a0, h2, *_r), _kw = _call(st, torch.tensor([PLEN + 2] * 3), h)
    torch.testing.assert_close(h2, h + vec, rtol=0, atol=0)


@pytest.mark.parametrize("span", [UNIF, CONT])
def test_inert_paths_return_none_and_do_not_fire(span: T.InjectionSpan) -> None:
    """The hook returns None (vLLM keeps the original args) when: no vector for
    this site (the alpha-0 arm: vecs = {}); positions is not a tensor or is
    empty; fewer than 2 positional args; and, under continuation, every
    position is in the prompt. `fired` counts only real writes — the
    self-check reads it, so a hook that never fires (e.g. compiled away under
    CUDA graphs) is caught rather than looking steered."""
    vec = _vec()
    h = _hidden()
    for st in (T.vllm_state(vecs={}, prompt_len=PLEN, span=span),
               T.vllm_state(vecs={SITE + 1: vec}, prompt_len=PLEN, span=span)):
        assert _call(st, torch.arange(NTOK), h) is None
        assert st.fired == 0
    st = T.vllm_state(vecs={SITE: vec}, prompt_len=PLEN, span=span)
    assert _call(st, list(range(NTOK)), h) is None
    assert _call(st, torch.arange(0), h[:0]) is None
    assert T.vllm_hook(st, SITE)(None, (torch.arange(NTOK),), {}) is None
    assert st.fired == 0
    st = T.vllm_state(vecs={SITE: vec}, prompt_len=NTOK + 5, span=span)
    r = _call(st, torch.arange(NTOK), h)
    if span is CONT:
        assert r is None and st.fired == 0
    else:  # uniform ignores prompt_len
        assert r is not None and st.fired == 1


@pytest.mark.parametrize("span", [UNIF, CONT])
def test_input_hidden_is_not_mutated(span: T.InjectionSpan) -> None:
    """The caller's hidden tensor is left untouched (vLLM may reuse it)."""
    st = T.vllm_state(vecs={SITE: _vec()}, prompt_len=PLEN, span=span)
    h = _hidden()
    keep = h.clone()
    r = _call(st, torch.arange(NTOK), h)
    assert torch.equal(h, keep)
    assert r[0][1] is not h


@pytest.mark.parametrize("span", [UNIF, CONT])
def test_bf16_hidden_stays_bf16(span: T.InjectionSpan) -> None:
    """The model runs bfloat16 and the vector is float32: the write casts the
    vector to h.dtype, so the residual stays bf16 (a float32 residual crashes
    the next fused rms-norm)."""
    vec = _vec()
    st = T.vllm_state(vecs={SITE: vec}, prompt_len=0, span=span)
    h = _hidden(dtype=torch.bfloat16)
    (_a0, h2, *_r), _kw = _call(st, torch.arange(NTOK), h)
    assert h2.dtype == torch.bfloat16
    assert torch.equal(h2, h + vec.to(torch.bfloat16))


# ── time profiles (the delivery screen's step / ramp) ───────────────────────


def test_time_k_is_screen_deliverys() -> None:
    assert T.TIME_K == 64.0


def test_time_profiles_under_continuation() -> None:
    """step: s_t = [t >= TIME_K]; ramp: clamp(t/TIME_K, 0, 1) for t >= 0, with
    t = position - prompt_len under continuation (the temporal handle: WHEN in
    the continuation the push begins). Output dtype preserved (bf16)."""
    k = T.TIME_K
    vec = _vec()
    pos = torch.tensor([0, PLEN, PLEN + 32, PLEN + 64, PLEN + 100])
    t = (pos - PLEN).to(torch.float32)
    h = _hidden(len(pos))
    for prof, s in (("step", (t >= k).float()),
                    ("ramp", torch.clamp(t / k, 0, 1) * (t >= 0).float())):
        r = _call(T.vllm_state(vecs={SITE: vec}, prompt_len=PLEN, span=CONT,
                               time_profile=prof), pos, h)
        torch.testing.assert_close(r[0][1], h + s.unsqueeze(-1) * vec, rtol=0, atol=1e-6)
    hb = _hidden(len(pos), dtype=torch.bfloat16)
    rb = _call(T.vllm_state(vecs={SITE: vec}, prompt_len=PLEN, span=CONT,
                            time_profile="ramp"), pos, hb)
    assert rb[0][1].dtype == torch.bfloat16


def test_time_profiles_under_uniform_use_absolute_position() -> None:
    """Under uniform, t = absolute position (the delivery screen's uniform scope):
    prompt_len does not move the step/ramp."""
    k = T.TIME_K
    vec = _vec()
    pos = torch.tensor([0, 10, 32, 64, 100])
    t = pos.to(torch.float32)
    h = _hidden(len(pos))
    for plen in (0, PLEN, 50):
        for prof, s in (("step", (t >= k).float()),
                        ("ramp", torch.clamp(t / k, 0, 1))):
            r = _call(T.vllm_state(vecs={SITE: vec}, prompt_len=plen, span=UNIF,
                                   time_profile=prof), pos, h)
            torch.testing.assert_close(r[0][1], h + s.unsqueeze(-1) * vec, rtol=0, atol=1e-6)


def test_step_before_onset_is_inert() -> None:
    """A step whose onset no position has reached writes nothing and does not
    fire (the delivery screen's early decode steps)."""
    st = T.vllm_state(vecs={SITE: _vec()}, prompt_len=PLEN, span=CONT, time_profile="step")
    assert _call(st, torch.arange(NTOK), _hidden()) is None
    assert st.fired == 0


def test_constant_profile_equals_the_uniform_and_continuation_masks() -> None:
    """continuation/constant reproduces the reference continuation write;
    uniform/constant writes the prompt too — one hook, both spans."""
    pos = torch.arange(NTOK)
    vec = _vec()
    for dtype in (torch.float32, torch.bfloat16):
        h = _hidden(dtype=dtype)
        a = _call(T.vllm_state(vecs={SITE: vec}, prompt_len=PLEN, span=CONT), pos, h)
        assert torch.equal(a[0][1], _legacy_continuation(h, pos, PLEN, vec))
        b = _call(T.vllm_state(vecs={SITE: vec}, prompt_len=PLEN, span=UNIF), pos, h)
        assert torch.equal(b[0][1], h + vec.to(dtype))


def test_unknown_profile_or_span_is_refused_at_construction() -> None:
    """An unknown profile or span is refused at construction: a hook that
    silently returned None would be an unsteered run that looks steered."""
    with pytest.raises(T.SpanError, match="time profile"):
        T.vllm_state(vecs={}, span=UNIF, time_profile="saw")
    with pytest.raises(T.SpanError, match="injection span"):
        T.vllm_state(vecs={}, span="prompt-only")
    with pytest.raises(T.SteerError, match="time_k"):
        T.vllm_state(vecs={}, span=UNIF, time_k=0)


# ── guards: shape / dtype / sites ────────────────────────────────────────────


def test_shape_and_dtype_mismatches_raise_actionable_errors() -> None:
    st = T.vllm_state(vecs={SITE: torch.randn(H + 1)}, prompt_len=0, span=UNIF)
    with pytest.raises(T.VectorShapeError, match="wrong lever/map"):
        _call(st, torch.arange(NTOK), _hidden())
    st = T.vllm_state(vecs={SITE: _vec()}, prompt_len=0, span=UNIF)
    with pytest.raises(T.SteerDTypeError, match="floating"):
        _call(st, torch.arange(NTOK), torch.ones(NTOK, H, dtype=torch.int64))
    with pytest.raises(T.VectorShapeError, match="flattened"):
        _call(st, torch.arange(NTOK - 1), _hidden())
    with pytest.raises(T.VectorShapeError, match="1-D"):
        T.vllm_state(vecs={SITE: torch.randn(2, H)}, span=UNIF)
    with pytest.raises(T.SteerDTypeError):
        T.vllm_state(vecs={SITE: torch.ones(H, dtype=torch.int32)}, span=UNIF)
    with pytest.raises(T.SpanError, match="prompt_len"):
        T.vllm_state(vecs={}, span=CONT, prompt_len=-1)


class _FakeVllmLayer(nn.Module):
    """vLLM decoder-layer call shape: (positions, hidden, residual)."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[torch.Tensor] = []

    def forward(self, positions: torch.Tensor, hidden: torch.Tensor,
                residual: torch.Tensor | None) -> torch.Tensor:
        self.seen.append(hidden.clone())
        return hidden


def test_install_pre_hooks_registers_on_layers_site_and_is_all_or_none() -> None:
    """install_pre_hooks puts one SitePreHook (with_kwargs) on each
    layers[site]; a forward through the layer sees the write; an out-of-range
    site registers nothing; removing the handles restores the plain call."""
    layers = nn.ModuleList([_FakeVllmLayer() for _ in range(4)])
    st = T.steer_vllm.VllmInjectionState(span=UNIF)
    with pytest.raises(T.SiteError, match="out of range"):
        T.steer_vllm.install_pre_hooks(layers, [1, 4], st)
    assert all(len(layer._forward_pre_hooks) == 0 for layer in layers)
    handles = T.steer_vllm.install_pre_hooks(layers, [1, 3], st)
    vec = _vec()
    h = _hidden()
    pos = torch.arange(NTOK)
    st.wear({3: vec}, prompt_len=PLEN)
    for layer in layers:
        layer(pos, h, None)
    assert torch.equal(layers[1].seen[-1], h)          # no vector for site 1: inert
    assert torch.equal(layers[3].seen[-1], h + vec)    # uniform: every position
    assert st.fired == 1
    for hd in handles:
        hd.remove()
    layers[3](pos, h, None)
    assert torch.equal(layers[3].seen[-1], h)


def test_wear_injection_doses_like_the_lanes() -> None:
    """wear_injection = the lanes' vecs_for: (alpha * rows).astype(float32)
    per site, {} at alpha 0 (the catch arm stays on the inert path); a
    span mismatch between the spec and the installed state is refused."""
    import numpy as np

    rows = np.random.default_rng(0).standard_normal((2, H)).astype(np.float32)
    st = T.steer_vllm.VllmInjectionState(span=UNIF)
    st.wear_injection(T.Injection(sites=(2, 5), vectors=rows, alpha=0.35), prompt_len=PLEN)
    want = (0.35 * rows).astype(np.float32)
    assert set(st.vecs) == {2, 5}
    assert torch.equal(st.vecs[5], torch.from_numpy(want[1]))
    st.wear_injection(T.Injection(sites=(2, 5), vectors=rows, alpha=0.0), prompt_len=PLEN)
    assert st.vecs == {}
    with pytest.raises(T.SteerError, match="installed hooks"):
        st.wear_injection(T.Injection(sites=(2, 5), vectors=rows, span=CONT), prompt_len=PLEN)


def test_receipt_names_the_span() -> None:
    st = T.vllm_state(vecs={}, span=UNIF)  # a dose lane's default span
    assert st.receipt() == {"injection_span": "uniform", "time_profile": "constant",
                            "time_k": 64.0}
    assert T.span_receipt("continuation")["injection_span"] == "continuation"
