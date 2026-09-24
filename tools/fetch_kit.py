"""Fetch a demo kit's release assets and verify every one against its sha256.

    python tools/fetch_kit.py                    # fetch everything kit-manifest.json lists
    python tools/fetch_kit.py --list             # show the manifest, fetch nothing
    python tools/fetch_kit.py --verify           # check what is on disk, fetch nothing
    python tools/fetch_kit.py --only map --only dose_band   # by asset name or role

The manifest (``kit-manifest.json`` at the repo root) is the ONLY source of
truth for what a kit is. Each asset names a URL, the sha256 its bytes must
hash to, and a destination relative to the repo root. Rules:

- a download is streamed to ``<dest>.part``, hashed as it arrives, and moved
  into place only if the hash matches; on a mismatch the partial file is
  deleted and the run fails;
- a destination that already exists with the right hash is left alone; one
  with the WRONG hash is refused (``--force`` replaces it) — this script never
  silently overwrites a file it did not verify;
- a manifest with an unfilled sha256 is refused as a whole, before anything
  is downloaded: an unpinned asset is not a kit;
- only ``https://`` and ``file://`` URLs are accepted, and a destination must
  stay inside the root (no absolute paths, no ``..``).

Some kit assets are pickles (the calibration's PCA model). The sha pin is what
makes loading them safe: the bytes are exactly the bytes the kit was built
with. Do not load a kit file that did not verify.

Exit status: 0 all verified; 1 a download or verification failed; 2 the
manifest or the arguments are unusable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHUNK = 1 << 20
ALLOWED_SCHEMES: tuple[str, ...] = ("https://", "file://")

Status = Literal["fetched", "present", "missing", "mismatch", "failed"]


class KitError(RuntimeError):
    """The manifest or an asset cannot be used; the message says why."""


class Asset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    #: what the asset is for: map, dose_band, discriminants, calibration, cv_report, ...
    role: str = Field(min_length=1)
    url: str = Field(min_length=1)
    #: lowercase hex; "" means NOT FILLED IN YET (the whole manifest is refused)
    sha256: str
    #: repo-root-relative POSIX path
    dest: str = Field(min_length=1)
    #: expected size, when known (checked in addition to the hash)
    bytes: int | None = Field(default=None, ge=0)
    description: str = ""

    @field_validator("url")
    @classmethod
    def _scheme(cls, v: str) -> str:
        if not v.startswith(ALLOWED_SCHEMES):
            raise ValueError(f"url must start with one of {ALLOWED_SCHEMES}, got {v!r}")
        return v

    @field_validator("dest")
    @classmethod
    def _relative_inside(cls, v: str) -> str:
        p = PurePosixPath(v)
        if p.is_absolute() or "\\" in v or ".." in p.parts or v.startswith("~"):
            raise ValueError(f"dest must be a relative path inside the root, got {v!r}")
        return v

    @property
    def filled(self) -> bool:
        return bool(SHA256_RE.fullmatch(self.sha256))


class KitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = Field(alias="schema")
    kit: str = Field(min_length=1)
    #: the model profile this kit serves (repo-relative)
    profile: str = Field(min_length=1)
    #: the GitHub release tag the assets are attached to
    release: str = Field(min_length=1)
    description: str = ""
    assets: tuple[Asset, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> KitManifest:
        for field in ("name", "dest"):
            seen: set[str] = set()
            for a in self.assets:
                value = getattr(a, field)
                if value in seen:
                    raise ValueError(f"two assets share {field} {value!r}")
                seen.add(value)
        return self

    def unfilled(self) -> list[str]:
        return [a.name for a in self.assets if not a.filled]

    def select(self, only: Sequence[str]) -> list[Asset]:
        if not only:
            return list(self.assets)
        wanted = set(only)
        picked = [a for a in self.assets if a.name in wanted or a.role in wanted]
        unknown = wanted - {a.name for a in picked} - {a.role for a in picked}
        if unknown:
            raise KitError(f"--only names no asset: {sorted(unknown)} "
                           f"(assets: {[a.name for a in self.assets]})")
        return picked


def load_manifest(path: Path) -> KitManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise KitError(f"manifest not found: {path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise KitError(f"manifest {path} is not readable JSON: {exc}") from exc
    try:
        return KitManifest.model_validate(raw)
    except ValidationError as exc:
        raise KitError(f"manifest {path} is malformed: {exc}") from exc


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def dest_path(root: Path, asset: Asset) -> Path:
    root = root.resolve()
    dst = (root / asset.dest).resolve()
    if root != dst and root not in dst.parents:
        raise KitError(f"{asset.name}: dest {asset.dest!r} resolves outside {root}")
    return dst


Opener = Callable[[str], BinaryIO]


def _open_url(url: str) -> BinaryIO:
    req = urllib.request.Request(url, headers={"User-Agent": "pleroma-fetch-kit/1"})
    return urllib.request.urlopen(req, timeout=60)  # noqa: S310 (scheme checked in Asset)


def download(asset: Asset, dst: Path, opener: Opener = _open_url, retries: int = 2) -> str:
    """Stream `asset` to `dst` via `<dst>.part`; returns the verified sha256.
    Raises KitError (with the partial file removed) on any failure."""
    part = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(retries + 1):
        h = hashlib.sha256()
        n = 0
        try:
            with opener(asset.url) as src, part.open("wb") as out:
                for chunk in iter(lambda: src.read(CHUNK), b""):
                    h.update(chunk)
                    out.write(chunk)
                    n += len(chunk)
            break
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
            part.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
    else:
        raise KitError(f"{asset.name}: download failed after {retries + 1} attempts: {last}")
    got = h.hexdigest()
    if got != asset.sha256:
        part.unlink(missing_ok=True)
        raise KitError(f"{asset.name}: sha256 MISMATCH — expected {asset.sha256}, got {got} "
                       f"({n} bytes from {asset.url}); nothing was written")
    if asset.bytes is not None and n != asset.bytes:
        part.unlink(missing_ok=True)
        raise KitError(f"{asset.name}: size {n} != manifest bytes {asset.bytes}")
    os.replace(part, dst)
    return got


def fetch_asset(asset: Asset, root: Path, *, verify_only: bool = False, force: bool = False,
                opener: Opener = _open_url) -> tuple[Status, str]:
    """One asset -> (status, detail). Never raises for a per-asset failure."""
    try:
        dst = dest_path(root, asset)
        if dst.exists():
            have = sha256_of(dst)
            if have == asset.sha256:
                return "present", str(dst)
            if verify_only or not force:
                return "mismatch", (f"{dst} exists with sha256 {have}, manifest says "
                                    f"{asset.sha256}" + ("" if verify_only else
                                                          "; refusing to overwrite (--force)"))
        elif verify_only:
            return "missing", str(dst)
        download(asset, dst, opener=opener)
        return "fetched", str(dst)
    except KitError as exc:
        return "failed", str(exc)


def run(manifest_path: Path, root: Path, *, only: Sequence[str] = (), verify_only: bool = False,
        force: bool = False, opener: Opener = _open_url,
        echo: Callable[[str], None] = print) -> int:
    try:
        manifest = load_manifest(manifest_path)
        unfilled = manifest.unfilled()
        if unfilled:
            raise KitError(
                f"manifest {manifest_path} is not filled in yet: no sha256 for "
                f"{unfilled}. The kit for release {manifest.release!r} has not been "
                "published; nothing was downloaded.")
        assets = manifest.select(only)
    except KitError as exc:
        echo(f"fetch_kit: {exc}")
        return 2
    bad = 0
    for a in assets:
        status, detail = fetch_asset(a, root, verify_only=verify_only, force=force,
                                     opener=opener)
        ok = status in ("fetched", "present")
        bad += not ok
        echo(f"{'ok ' if ok else 'ERR'} {status:<8} {a.name:<28} {detail}")
    echo(f"{len(assets) - bad}/{len(assets)} assets verified for kit {manifest.kit!r}")
    return 0 if bad == 0 else 1


def list_manifest(manifest_path: Path, echo: Callable[[str], None] = print) -> int:
    try:
        m = load_manifest(manifest_path)
    except KitError as exc:
        echo(f"fetch_kit: {exc}")
        return 2
    echo(f"kit {m.kit!r}  release {m.release!r}  profile {m.profile}")
    for a in m.assets:
        size = f"{a.bytes} B" if a.bytes is not None else "size ?"
        echo(f"  {a.name:<28} {a.role:<14} {a.sha256 or 'NOT FILLED'}  {size}  -> {a.dest}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parent.parent
    ap.add_argument("--manifest", type=Path, default=here / "kit-manifest.json")
    ap.add_argument("--root", type=Path, default=None,
                    help="where dest paths resolve (default: the manifest's directory)")
    ap.add_argument("--only", action="append", default=[], metavar="NAME_OR_ROLE")
    ap.add_argument("--verify", action="store_true", help="check files on disk; download nothing")
    ap.add_argument("--force", action="store_true",
                    help="replace a destination whose sha256 does not match")
    ap.add_argument("--list", action="store_true", help="print the manifest and exit")
    args = ap.parse_args(argv)
    if args.list:
        return list_manifest(args.manifest)
    root = args.root if args.root is not None else args.manifest.resolve().parent
    return run(args.manifest, root, only=args.only, verify_only=args.verify, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
