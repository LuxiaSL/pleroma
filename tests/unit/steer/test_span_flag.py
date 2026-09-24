"""pleroma.steer's CLI flag and import hygiene (the vLLM backend must import on
a machine without vllm — it is plain torch)."""

from __future__ import annotations

import argparse
import subprocess
import sys

import pytest

from pleroma.steer import InjectionSpan, add_injection_span_arg


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    add_injection_span_arg(ap)
    return ap


def test_flag_defaults_to_uniform_and_accepts_continuation() -> None:
    assert _parser().parse_args([]).injection_span is InjectionSpan.UNIFORM
    ns = _parser().parse_args(["--injection-span", "continuation"])
    assert ns.injection_span is InjectionSpan.CONTINUATION


def test_flag_rejects_an_unknown_span() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["--injection-span", "prompt"])


def test_backends_import_without_vllm() -> None:
    code = ("import sys; import pleroma.steer.vllm, pleroma.steer.hf; "
            "assert 'vllm' not in sys.modules, 'pleroma.steer imported vllm'")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
