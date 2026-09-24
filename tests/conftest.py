"""Repo root on sys.path so ``pleroma.*`` resolves under pytest.

CPU-only: numpy, torch (CPU), pytest."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# In a source checkout, anamnesis may sit at vendor/anamnesis, pinned to the
# commit pyproject.toml names. pytest.ini's `pythonpath` covers this process;
# subprocess-based tests (`python -c ...`, `python -m ... --help`) need it on
# PYTHONPATH too, so the suite runs without installing it. Where that directory
# is absent, the installed anamnesis is what imports.
_ANAMNESIS = str(REPO_ROOT / "vendor" / "anamnesis")
_pp = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
if _ANAMNESIS not in _pp:
    os.environ["PYTHONPATH"] = os.pathsep.join([_ANAMNESIS, *_pp])
