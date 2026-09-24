"""tools/fetch_kit.py and the demo kit's manifest + example deployment.

Runs from both trees: in the public repo this file is
`tests/unit/repo/test_fetch_kit.py`, and in the source repo it lives in the
`public/` overlay at the same relative path, so `parents[3]` is the tree that
holds `tools/fetch_kit.py`, `kit-manifest.json` and `examples/` in both.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tomllib
import urllib.error
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

OVERLAY = Path(__file__).resolve().parents[3]
MANIFEST = OVERLAY / "kit-manifest.json"
EXAMPLE_DEPLOY = OVERLAY / "examples" / "deploy-llama31-8b.toml"


def _load_fetch_kit() -> ModuleType:
    path = OVERLAY / "tools" / "fetch_kit.py"
    spec = importlib.util.spec_from_file_location("fetch_kit_under_test", path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # pydantic resolves annotations through it
    spec.loader.exec_module(mod)
    return mod


fk = _load_fetch_kit()


def _repo_file(rel: str) -> Path:
    """`rel` in the public tree, or (source repo) beside the overlay."""
    for base in (OVERLAY, OVERLAY.parent):
        if (base / rel).exists():
            return base / rel
    raise FileNotFoundError(rel)


def _manifest(tmp: Path, assets: list[dict[str, Any]]) -> Path:
    p = tmp / "kit-manifest.json"
    p.write_text(json.dumps({"schema": 1, "kit": "toy", "profile": "profiles/toy.toml",
                             "release": "kit-toy-v1", "assets": assets}))
    return p


def _asset(tmp: Path, name: str, payload: bytes, *, sha: str | None = None,
           dest: str | None = None, role: str = "map") -> dict[str, Any]:
    src = tmp / "release" / name
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(payload)
    return {"name": name, "role": role, "url": src.as_uri(),
            "sha256": sha if sha is not None else hashlib.sha256(payload).hexdigest(),
            "dest": dest or f"kit/toy/{name}", "bytes": len(payload)}


def _run(manifest: Path, root: Path, **kw: Any) -> tuple[int, list[str]]:
    lines: list[str] = []
    rc = fk.run(manifest, root, echo=lines.append, **kw)
    return rc, lines


# ── fetch + verify over file:// ─────────────────────────────────────────────


def test_fetches_verifies_and_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    m = _manifest(tmp_path, [_asset(tmp_path, "map.npz", b"map bytes"),
                             _asset(tmp_path, "pca.pkl", b"pca", dest="kit/toy/calib/pca.pkl",
                                    role="calibration")])
    rc, lines = _run(m, root)
    assert rc == 0, lines
    assert (root / "kit/toy/map.npz").read_bytes() == b"map bytes"
    assert (root / "kit/toy/calib/pca.pkl").read_bytes() == b"pca"
    assert not list(root.rglob("*.part"))
    rc, lines = _run(m, root)
    assert rc == 0 and all("present" in ln for ln in lines[:-1]), lines


def test_sha_mismatch_refuses_and_leaves_nothing(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    m = _manifest(tmp_path, [_asset(tmp_path, "map.npz", b"tampered", sha="0" * 64)])
    rc, lines = _run(m, root)
    assert rc == 1
    assert "MISMATCH" in "\n".join(lines)
    assert not (root / "kit/toy/map.npz").exists()
    assert not list(root.rglob("*.part"))


def test_existing_wrong_file_is_refused_unless_forced(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    m = _manifest(tmp_path, [_asset(tmp_path, "map.npz", b"good")])
    (root / "kit/toy").mkdir(parents=True)
    (root / "kit/toy/map.npz").write_bytes(b"stale")
    rc, lines = _run(m, root)
    assert rc == 1 and "refusing to overwrite" in "\n".join(lines)
    assert (root / "kit/toy/map.npz").read_bytes() == b"stale"
    rc, _ = _run(m, root, force=True)
    assert rc == 0 and (root / "kit/toy/map.npz").read_bytes() == b"good"


def test_verify_only_downloads_nothing(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    m = _manifest(tmp_path, [_asset(tmp_path, "map.npz", b"x")])
    rc, lines = _run(m, root, verify_only=True)
    assert rc == 1 and "missing" in lines[0]
    assert not (root / "kit").exists()


def test_only_selects_by_name_or_role_and_refuses_unknown(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    m = _manifest(tmp_path, [_asset(tmp_path, "map.npz", b"m"),
                             _asset(tmp_path, "band.json", b"{}", role="dose_band")])
    rc, _ = _run(m, root, only=["dose_band"])
    assert rc == 0
    assert (root / "kit/toy/band.json").exists() and not (root / "kit/toy/map.npz").exists()
    rc, lines = _run(m, root, only=["nope"])
    assert rc == 2 and "names no asset" in lines[0]


def test_unfilled_manifest_is_refused_before_any_download(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    m = _manifest(tmp_path, [_asset(tmp_path, "a.npz", b"a"),
                             _asset(tmp_path, "b.npz", b"b", sha="")])
    rc, lines = _run(m, root)
    assert rc == 2 and "not filled in yet" in lines[0] and "b.npz" in lines[0]
    assert not root.exists()


@pytest.mark.parametrize(("field", "value"), [
    ("dest", "../outside.npz"), ("dest", "/abs/outside.npz"), ("dest", "~user/x.npz"),
    ("url", "http://example.org/x.npz"), ("url", "ftp://example.org/x.npz"),
])
def test_manifest_refuses_unsafe_urls_and_dests(tmp_path: Path, field: str, value: str) -> None:
    a = _asset(tmp_path, "x.npz", b"x")
    a[field] = value
    rc, lines = _run(_manifest(tmp_path, [a]), tmp_path / "repo")
    assert rc == 2 and "malformed" in lines[0]


def test_duplicate_dest_is_refused(tmp_path: Path) -> None:
    a = _asset(tmp_path, "x.npz", b"x", dest="kit/same")
    b = _asset(tmp_path, "y.npz", b"y", dest="kit/same")
    rc, lines = _run(_manifest(tmp_path, [a, b]), tmp_path / "repo")
    assert rc == 2 and "share dest" in lines[0]


def test_transient_errors_are_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fk.time, "sleep", lambda _s: None)
    payload = b"eventually"
    asset = fk.Asset.model_validate(_asset(tmp_path, "x.npz", payload))
    calls = {"n": 0}

    def flaky(_url: str) -> io.BytesIO:
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("connection reset")
        return io.BytesIO(payload)

    dst = tmp_path / "out" / "x.npz"
    assert fk.download(asset, dst, opener=flaky, retries=2) == asset.sha256
    assert calls["n"] == 3 and dst.read_bytes() == payload
    calls["n"] = -10  # never succeeds within the retries
    with pytest.raises(fk.KitError, match="download failed"):
        fk.download(asset, tmp_path / "out" / "y.npz", opener=flaky, retries=1)


# ── the shipped manifest and example deployment ─────────────────────────────


def test_shipped_manifest_is_well_formed() -> None:
    m = fk.load_manifest(MANIFEST)
    assert _repo_file(m.profile).is_file()
    prefix = f"kit/{m.kit}/"
    release_base = f"https://github.com/LuxiaSL/pleroma/releases/download/{m.release}/"
    for a in m.assets:
        assert a.dest.startswith(prefix), a.dest
        assert a.url == release_base + a.name, a.url
        assert a.sha256 == "" or a.filled, f"{a.name}: sha256 is neither filled nor empty"
    roles = {a.role for a in m.assets}
    assert {"map", "dose_band", "discriminants", "calibration"} <= roles


def test_example_deploy_points_at_the_kit_and_launches() -> None:
    from pleroma.cli import main as pleroma_main
    from pleroma.config import load_deployment

    m = fk.load_manifest(MANIFEST)
    by_dest = {a.dest: a for a in m.assets}
    dep = load_deployment(EXAMPLE_DEPLOY)
    assert dep.profile == m.kit
    pinned = [dep.map, dep.discriminants] + ([dep.dose_band] if dep.dose_band else [])
    for pp in pinned:
        asset = by_dest.get(pp.path.as_posix())
        assert asset is not None, f"{pp.path} is not a kit asset"
        if pp.sha256 is not None:  # a pin, when written, must be the manifest's
            assert pp.sha256 == asset.sha256, pp.path
    calib = {a.dest for a in m.assets if a.role == "calibration"}
    assert calib and all(Path(d).parent.as_posix() == dep.calib_dir.as_posix() for d in calib)
    with open(EXAMPLE_DEPLOY, "rb") as fh:
        assert "build" not in tomllib.load(fh)  # the demo needs no build inputs
    profile = str(_repo_file(m.profile))
    for verb in ("serve", "worker"):
        assert pleroma_main([verb, "--profile", profile, "--deploy", str(EXAMPLE_DEPLOY),
                             "--dry-run"]) == 0
