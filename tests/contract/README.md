# Contract tests

These pin the behaviour of the golden path: what each stage reads, computes
and writes. They are the contract the package is held to.

Rules:

- **One area per directory** (`map/`, `build/`, `dose_steer/`,
  `format_judge_stats/`). Each area imports the code under test ONLY through
  its own `targets.py`. Moving a function inside `pleroma/` means editing
  `targets.py` and nothing else. A test that needs a different edit to survive
  the move is testing layout, not behaviour.
- **CPU-only, offline, fast.** Synthetic fixtures built in-test (tiny dims);
  no GPU, network, machine-specific paths or model weights.
- **Pin what IS, flag what's wrong.** Current behaviour is asserted as-is.
  A known defect is pinned as the CORRECT behaviour under
  `@pytest.mark.known_gap` + `pytest.mark.xfail(strict=True, reason=...)`,
  with the reason stating the defect — so the fix flips it to XPASS → strict
  failure → remove the xfail.
- Golden numbers from real artifacts are allowed when the artifact is in the
  repository.
