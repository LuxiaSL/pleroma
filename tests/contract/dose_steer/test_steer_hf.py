"""Contract: the HF steering hooks — ``pleroma.steer.hf`` (the ONE HF
implementation) and the loom's WEAR write.

The loom delegates to ``pleroma.steer.hf.attach``; its thin ``attach``
(which binds the sites and the span) is bound by targets.hf_attach and run on
a tiny CPU residual stack (_toy.py). The loom is the only HF steering caller,
and the map/lever site convention is canonical. We pin: which LAYER receives
the write (site convention), which POSITIONS (span), the SCALING
(normalize=False: alpha x row, verbatim), and that removing the handles
restores the unsteered forward bit-for-bit.

A decided spec: the loom injects UNIFORM over the whole sequence
(start_pos=0), prompt included.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from tests.contract.dose_steer import _toy
from tests.contract.dose_steer import targets as T

SEQ = 7
PROMPT_LEN = 3


def _vectors(n_sites: int, seed: int = 11, norm: float = 3.7) -> np.ndarray:
    """Rows with a deliberately NON-unit norm, so normalize=True would show."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n_sites, _toy.HIDDEN))
    return (v / np.linalg.norm(v, axis=1, keepdims=True) * norm).astype(np.float32)


def _wear(lane: str, model: Any, sites: list[int], vectors: np.ndarray, alpha: float,
          start: int | None = None, end: int | None = None) -> list[Any]:
    """Call `lane`'s attach with its own production signature
    (targets.HF_ATTACH_LANES[lane]['sig'])."""
    attach = T.hf_attach(lane, model, sites)
    if lane == "loom_serve":
        return attach(vectors, alpha)
    raise AssertionError(lane)


def _run(model: Any, seq: int = SEQ) -> tuple[torch.Tensor, list[torch.Tensor]]:
    h, cp = _toy.prefill_input(seq)
    out = model(h, cp)
    return out, model.inputs()


def _remove(handles: list[Any]) -> None:
    for h in handles:
        h.remove()


# ── the loom's WEAR write ────────────────────────────────────────────────────


def test_loom_writes_every_position_including_the_prompt() -> None:
    """DECIDED SPEC: the loom injects UNIFORM — start_pos=0, end None — so on a prefill every
    position, prompt included, receives alpha x row at the site's input."""
    model = _toy.ToyDecoder()
    base_out, base = _run(model)
    vec = _vectors(1)
    handles = T.loom_attach(model, [2])(vec, 0.5)
    try:
        _out, worn = _run(model)
    finally:
        _remove(handles)
    delta = (worn[2] - base[2])[0]
    want = torch.from_numpy(0.5 * vec[0]).expand(SEQ, -1)
    torch.testing.assert_close(delta, want, rtol=0, atol=1e-6)
    assert all(h.spec.start_pos == 0 and h.spec.end_pos is None for h in handles)


def test_loom_writes_incremental_decode_steps_too() -> None:
    """Under HF generate() each decode step is a seq_len-1 forward carrying its
    absolute cache_position; the loom's uniform spec injects there as well
    (anamnesis's residual write gates on cache_position)."""
    model = _toy.ToyDecoder()
    h, cp = _toy.step_input(SEQ + 4)
    model(h, cp)
    base = model.inputs()
    vec = _vectors(1)
    handles = T.loom_attach(model, [3])(vec, 0.25)
    try:
        model(h, cp)
        worn = model.inputs()
    finally:
        _remove(handles)
    torch.testing.assert_close((worn[3] - base[3])[0, 0], torch.from_numpy(0.25 * vec[0]),
                               rtol=0, atol=1e-6)
    assert handles[0].stats["saw_cache_position"] is True


def test_loom_site_is_a_pre_hook_on_layers_site() -> None:
    """Site convention (map/lever): a map site s is a hidden_states index; the
    write is a forward PRE-hook on layers[s], i.e. the INPUT of layer s. Layers
    below s see nothing; layer s's input carries the write."""
    model = _toy.ToyDecoder()
    _o, base = _run(model)
    handles = T.loom_attach(model, [3])(_vectors(1), 1.0)
    try:
        _o, worn = _run(model)
    finally:
        _remove(handles)
    for i in range(3):
        assert torch.equal(worn[i], base[i]), f"layer {i} below the site was touched"
    assert not torch.allclose(worn[3], base[3])


def test_loom_does_not_normalize_the_vector() -> None:
    """normalize=False: the write
    is alpha x row VERBATIM — the ruler and dose already set its norm. A copy
    that forgot the flag would unit-normalize and inject alpha, not
    alpha x norm_ref (a 2-8x dose error on the 70B ruler)."""
    model = _toy.ToyDecoder()
    _o, base = _run(model)
    vec = _vectors(1, norm=3.7)
    handles = T.loom_attach(model, [1])(vec, 0.3)
    try:
        _o, worn = _run(model)
    finally:
        _remove(handles)
    n = torch.linalg.vector_norm((worn[1] - base[1])[0], dim=-1)
    torch.testing.assert_close(n, torch.full((SEQ,), 0.3 * 3.7), rtol=1e-5, atol=1e-6)
    assert handles[0].spec.normalize is False


def test_loom_removal_restores_the_unsteered_forward_bitwise() -> None:
    """Unwear: removing every handle returns the forward to the unsteered
    output bit-for-bit (the unwear/fresh-draw path, so a draw after an unwear
    is never steered by a stale hook)."""
    model = _toy.ToyDecoder()
    base_out, _ = _run(model)
    handles = T.loom_attach(model, [1, 2, 4])(_vectors(3), 0.8)
    worn_out, _ = _run(model)
    assert not torch.equal(worn_out, base_out)
    _remove(handles)
    after, _ = _run(model)
    assert torch.equal(after, base_out)


def test_loom_zero_vector_gate_is_bitwise_inert() -> None:
    """The loom's boot gate (the hook self-test): a
    zero-vector wear at alpha 1 must leave the forward byte-identical to no
    hooks, at every site."""
    model = _toy.ToyDecoder()
    base_out, _ = _run(model)
    handles = T.loom_attach(model, [0, 2, 5])(np.zeros((3, _toy.HIDDEN), np.float32), 1.0)
    try:
        out, _ = _run(model)
    finally:
        _remove(handles)
    assert torch.equal(out, base_out)


def test_loom_attach_is_all_or_none() -> None:
    """If any site fails to attach (out of range) or rows != sites, the loom
    removes the handles it already registered and re-raises: no half-worn
    model (zip strict=True)."""
    model = _toy.ToyDecoder()
    base_out, _ = _run(model)
    with pytest.raises(ValueError, match="out of range"):
        T.loom_attach(model, [1, 99])(_vectors(2), 1.0)
    with pytest.raises(ValueError):
        T.loom_attach(model, [1, 2])(_vectors(3), 1.0)
    out, _ = _run(model)
    assert torch.equal(out, base_out)
    assert all(len(layer._forward_pre_hooks) == 0 for layer in model.layers)


# ── every HF lane, one table (each row must keep its span) ───────────────────

#: lane -> (start, end) passed, and the positions that MUST be written
SPAN_CASES: list[tuple[str, int | None, int | None, list[int]]] = [
    ("loom_serve", None, None, list(range(SEQ))),                  # uniform, hardwired
]


@pytest.mark.parametrize(("lane", "start", "end", "written"), SPAN_CASES,
                         ids=[f"{c[0]}-{c[1]}-{c[2]}" for c in SPAN_CASES])
def test_each_hf_copy_writes_alpha_times_row_at_its_span(lane: str, start: int | None,
                                                         end: int | None, written: list[int]) -> None:
    """Per lane: at the first site's input the
    write is alpha x row verbatim (normalize=False) at exactly the positions the
    copy's span selects, zero elsewhere; removal restores bitwise."""
    model = _toy.ToyDecoder()
    base_out, base = _run(model)
    vec = _vectors(1)
    alpha = 0.6
    handles = _wear(lane, model, [2], vec, alpha, start, end)
    try:
        _o, worn = _run(model)
    finally:
        _remove(handles)
    eff = alpha
    delta = (worn[2] - base[2])[0]
    for p in range(SEQ):
        want = torch.from_numpy(eff * vec[0]) if p in written else torch.zeros(_toy.HIDDEN)
        torch.testing.assert_close(delta[p], want, rtol=0, atol=1e-6, msg=f"{lane} pos {p}")
    assert all(h.spec.normalize is False for h in handles)
    after, _ = _run(model)
    assert torch.equal(after, base_out)


def test_loom_equals_the_shared_hook_called_uniform() -> None:
    """The loom's wear == pleroma.steer.hf.attach at span UNIFORM.
    Bit-identical output."""
    outs = []
    vec = _vectors(2)
    for use_loom in (True, False):
        model = _toy.ToyDecoder()
        if use_loom:
            handles = _wear("loom_serve", model, [1, 3], vec, 0.315)
        else:
            handles = T.steer_hf.attach(model, T.Injection(
                sites=(1, 3), vectors=vec, alpha=0.315, span=T.InjectionSpan.UNIFORM))
        try:
            outs.append(_run(model)[0])
        finally:
            _remove(handles)
    assert torch.equal(outs[0], outs[1])


# ── pleroma.steer.hf directly ────────────────────────────────────────────────


def _legacy_loom_attach(model: Any, sites: list[int], vectors: np.ndarray,
                        alpha: float) -> list[Any]:
    """REFERENCE: the loom's WEAR write as a direct anamnesis residual write
    per site — the write every banked served session was measured under.
    pleroma.steer.hf must reproduce it bit-for-bit."""
    handles: list[Any] = []
    try:
        for row, site in zip(vectors, sites, strict=True):
            handles.append(T.attach_residual_write(model, T.ResidualWriteSpec(
                layer_idx=int(site),
                vector=torch.from_numpy(np.ascontiguousarray(row, dtype=np.float32)),
                alpha=float(alpha), start_pos=0, end_pos=None, normalize=False)))
    except Exception:
        _remove(handles)
        raise
    return handles


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_loom_wear_is_bit_identical_to_the_pre_refactor_loom(dtype: torch.dtype) -> None:
    """THE behaviour-preservation proof for the loom: over sites, alphas
    (incl. 0 and > 1), prefill AND an incremental decode step, in fp32/bf16/
    fp16, the loom's wear through pleroma.steer.hf produces the same bits at
    every layer input and at the output as the reference write."""
    rng = np.random.default_rng(5)
    for sites in ([0], [1, 3], [0, 2, 4, 5]):
        for alpha in (0.0, 0.125, 0.315, 1.0, 2.7, -0.4):
            vec = (rng.standard_normal((len(sites), _toy.HIDDEN)) * 3.1).astype(np.float32)
            outs = []
            for attach in (lambda m: T.loom_attach(m, sites)(vec, alpha),
                           lambda m: _legacy_loom_attach(m, sites, vec, alpha)):
                model = _toy.ToyDecoder().to(dtype)
                handles = attach(model)
                try:
                    h, cp = _toy.prefill_input(SEQ)
                    o = model(h.to(dtype), cp)
                    ins = model.inputs()
                    hs, cps = _toy.step_input(SEQ + 3)
                    o2 = model(hs.to(dtype), cps)
                finally:
                    _remove(handles)
                outs.append([o, o2, *ins])
            for a, b in zip(outs[0], outs[1], strict=True):
                assert a.dtype == dtype
                assert torch.equal(a, b), (sites, alpha, dtype)


def test_hf_continuation_writes_from_prompt_len_and_uniform_ignores_it() -> None:
    """span=CONTINUATION -> start_pos = prompt_len; UNIFORM -> start_pos = 0
    whatever prompt_len says."""
    for span, plen, written in ((T.InjectionSpan.CONTINUATION, PROMPT_LEN,
                                 range(PROMPT_LEN, SEQ)),
                                (T.InjectionSpan.UNIFORM, PROMPT_LEN, range(SEQ)),
                                (T.InjectionSpan.UNIFORM, None, range(SEQ))):
        model = _toy.ToyDecoder()
        _o, base = _run(model)
        vec = _vectors(1)
        handles = T.steer_hf.attach(model, T.Injection(sites=(2,), vectors=vec, alpha=0.6,
                                                       span=span), prompt_len=plen)
        try:
            _o, worn = _run(model)
        finally:
            _remove(handles)
        assert all(h.spec.start_pos == (plen if span is T.InjectionSpan.CONTINUATION else 0)
                   for h in handles)
        delta = (worn[2] - base[2])[0]
        for p in range(SEQ):
            want = torch.from_numpy(0.6 * vec[0]) if p in written else torch.zeros(_toy.HIDDEN)
            torch.testing.assert_close(delta[p], want, rtol=0, atol=1e-6)


def test_hf_refusals_are_actionable_and_register_nothing() -> None:
    """Every refusal happens before any hook is registered: CONTINUATION
    without prompt_len, a vLLM-only time profile, out-of-range or duplicate
    sites, rows != sites, hidden-size mismatch, non-finite vectors/alpha."""
    model = _toy.ToyDecoder()
    vec = _vectors(2)
    cases: list[tuple[type[Exception], str, Any]] = [
        (T.SpanError, "prompt_len", lambda: T.steer_hf.attach(model, T.Injection(
            sites=(1, 2), vectors=vec, span="continuation"))),
        (T.SpanError, "vLLM-only", lambda: T.steer_hf.attach(model, T.Injection(
            sites=(1, 2), vectors=vec, time_profile="ramp"))),
        (T.SiteError, "out of range", lambda: T.steer_hf.attach(model, T.Injection(
            sites=(1, _toy.N_LAYERS), vectors=vec))),
        (T.SiteError, "duplicate", lambda: T.Injection(sites=(2, 2), vectors=vec)),
        (T.SiteError, "negative", lambda: T.Injection(sites=(-1, 2), vectors=vec)),
        (T.VectorShapeError, "one row per site", lambda: T.Injection(sites=(1,), vectors=vec)),
        (T.VectorShapeError, "hidden size", lambda: T.steer_hf.attach(model, T.Injection(
            sites=(1, 2), vectors=np.zeros((2, _toy.HIDDEN + 1), np.float32)))),
        (T.SteerDTypeError, "NaN", lambda: T.Injection(
            sites=(1, 2), vectors=np.full((2, _toy.HIDDEN), np.nan, np.float32))),
        (T.SteerError, "finite", lambda: T.Injection(sites=(1, 2), vectors=vec,
                                                     alpha=float("inf"))),
    ]
    for exc, match, fn in cases:
        with pytest.raises(exc, match=match):
            fn()
        assert all(len(layer._forward_pre_hooks) == 0 for layer in model.layers), match
    assert issubclass(T.SteerError, ValueError)


def test_injection_is_a_frozen_float32_spec() -> None:
    """Sites normalised to a tuple of ints (numpy ints accepted), vectors to
    C-contiguous float32, span/profile coerced from strings; frozen."""
    inj = T.Injection(sites=np.array([3, 1]), vectors=np.ones((2, 4)), alpha=np.float32(0.5),
                      span="continuation", time_profile="step")
    assert inj.sites == (3, 1) and all(type(s) is int for s in inj.sites)
    assert inj.vectors.dtype == np.float32 and inj.vectors.flags.c_contiguous
    assert inj.span is T.InjectionSpan.CONTINUATION and inj.time_profile is T.TimeProfile.STEP
    assert type(inj.alpha) is float and inj.n_sites == 2 and inj.hidden == 4
    assert inj.receipt() == {"injection_span": "continuation", "time_profile": "step",
                             "time_k": 64.0}
    with pytest.raises(AttributeError):
        inj.alpha = 2.0  # type: ignore[misc]
    one = T.Injection(sites=[2], vectors=np.ones(4))
    assert one.vectors.shape == (1, 4) and one.span is T.InjectionSpan.UNIFORM
