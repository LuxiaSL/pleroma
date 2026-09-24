"""The corpus the G3 checkers are held to: strings each one must catch, and must not.

Both halves of G3 were written by reading the code they police, and reading is how
a checker acquires a blind spot: a rule that covers `#` comments looks complete
until somebody puts the same sentence in a docstring, and a rule keyed to the
shape of one document name looks complete until a document is named some other
way. Nothing in a source tree fails when a gate stops seeing — the verdict still
says PASS.

So the rules are pinned by a corpus instead. Every case below is a real string,
taken off the surfaces prose actually lands on: a comment, a docstring, and the
message a `raise` or a logging call says out loud. Each one carries the rule set it
must produce, and half the corpus carries the empty set — legitimate prose that
must stay quiet, because a gate that cries wolf is a gate the next contributor
switches off. The two checkers are run over every case together, since their rule
names are distinct and a violation belongs to whichever one owns it.

Adding a case is how a blind spot gets closed: write the string that slipped
through, give it the rule it should have produced, and the gate is pinned there
from then on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tools import check_referents, check_timelessness

CITATION = check_referents.CITATION_RULE
DEFERRAL = check_referents.DEFERRAL_RULE
PATH = check_referents.PATH_RULE
PRIVATE = check_referents.PRIVATE_RULE
DATED = check_timelessness.DATED_RULE
MARKER = check_timelessness.MARKER_RULE

#: The surfaces a case can be written on. `data` is the control: the same sentence
#: as a value in a payload, which no rule may read.
SURFACES = ("comment", "docstring", "raise", "log", "data")


@dataclass(frozen=True)
class Case:
    """One string, where it sits, and the rules it must produce there."""

    text: str
    surface: str
    rules: frozenset[str] = field(default_factory=frozenset)
    #: True when the line carries a date the allowlist exempts rather than no date.
    exempt: bool = False

    def source(self) -> str:
        """The smallest module that puts `text` on `surface`."""
        # `ensure_ascii=False` so a section mark or a box-drawing rule reaches the
        # generated line as itself rather than as an escape the rules cannot see.
        literal = json.dumps(self.text, ensure_ascii=False)
        if self.surface == "comment":
            return f"# {self.text}\nVALUE = 1\n"
        if self.surface == "docstring":
            return f"{literal}\nVALUE = 1\n"
        if self.surface == "raise":
            return (
                "def run(count: int) -> None:\n"
                "    if not count:\n"
                f"        raise ValueError({literal})\n"
            )
        if self.surface == "log":
            return (
                "import logging\n\n"
                "LOGGER = logging.getLogger(__name__)\n\n\n"
                "def run() -> None:\n"
                f"    LOGGER.warning({literal})\n"
            )
        if self.surface == "data":
            return f"PAYLOAD = {{'note': {literal}}}\n"
        raise AssertionError(f"unknown surface {self.surface!r}; expected one of {SURFACES}")

    def label(self) -> str:
        return f"{self.surface}:{self.text[:60]}"


MUST_CATCH: tuple[Case, ...] = (
    # A date in prose, on each of the three surfaces. The first was always caught;
    # the rest are the shapes a rule keyed to `#` could not see.
    Case("measured 2026-07-12, n=80", "comment", frozenset({DATED})),
    Case("Measured 2026-07-12 on the 3B corpus.", "docstring", frozenset({DATED})),
    Case("Ruling of 2026-07-12.", "docstring", frozenset({DATED, CITATION})),
    Case(
        "validated on the merged v3 corpus (2026-06-14)",
        "docstring",
        frozenset({DATED}),
    ),
    Case("the corpus was rebuilt 2026-07-18", "raise", frozenset({DATED})),
    Case("calibration banked 2026-07-12", "log", frozenset({DATED})),
    # Citations of documents this repository does not contain.
    Case(
        "prereg-vmb-v1 §6b: one analysis template for every arm x model",
        "comment",
        frozenset({CITATION}),
    ),
    Case(
        "SPEC-path-receptacles-and-span-coverage-2026-09-11 §1a",
        "comment",
        frozenset({PRIVATE, CITATION, DATED}),
    ),
    Case(
        "the how-axis LDA directions, spec P-B",
        "docstring",
        frozenset({CITATION}),
    ),
    Case(
        "M5 onboarding audit 2026-07-12; journal W10",
        "comment",
        frozenset({CITATION, DATED}),
    ),
    Case(
        "RoPE gate FAILED (14e): rope_theta not found",
        "raise",
        frozenset({CITATION}),
    ),
    Case(
        "PORT-MANIFEST-anamnesis-2026-09-13.md holds the allow-list.",
        "docstring",
        frozenset({PRIVATE, DATED}),
    ),
    Case("ported at 4acedbf0 from the frozen record", "comment", frozenset({CITATION})),
    # One case per citation idiom, so a pattern that stops working fails on its own
    # rather than hiding behind a neighbour that happens to match the same line.
    Case("The receptacle rule is stated in §4 of the brief.", "docstring", frozenset({CITATION})),
    Case("one analysis template per model, as prereg fixes it", "docstring", frozenset({CITATION})),
    Case("The floor is the one arm A9 reported.", "docstring", frozenset({CITATION})),
    Case("The row is read the way A3 defined it.", "docstring", frozenset({CITATION})),
    Case("M5 left the shared branch gated.", "comment", frozenset({CITATION})),
    Case("kept per the addendum", "comment", frozenset({CITATION})),
    Case("the codicil that followed narrows it", "comment", frozenset({CITATION})),
    Case("the row the journal entry records", "comment", frozenset({CITATION})),
    Case("the ruling that stands over both readouts", "comment", frozenset({CITATION})),
    # The two worked failures in CONTRIBUTING.md's referent section.
    Case(
        "See research/notes/v3-delta-memo for why the localization claim was revised.",
        "comment",
        frozenset({PRIVATE}),
    ),
    Case(
        "Kept for the reason discussed when this was ruled on.",
        "comment",
        frozenset({DEFERRAL}),
    ),
    Case(
        "signatures from two machines must not be joined; see the earlier ruling",
        "raise",
        frozenset({DEFERRAL, CITATION}),
    ),
    Case("falling back to anamnesis/analysis/absent.py", "log", frozenset({PATH})),
    Case("TODO: tighten this", "comment", frozenset({MARKER})),
    # pleroma's private documents: a citation of one resolves nowhere in the public tree.
    Case("The shelf is the registered baseline (REFACTOR-PLAN phase 3).", "comment",
         frozenset({CITATION})),
    Case("See RESULTS 54 for the gauge.", "docstring", frozenset({CITATION})),
    Case("Mirrors LOOM-DRIVE section 7.", "comment", frozenset({CITATION})),
    Case("Measured per JUDGE-POLICY.", "raise", frozenset({CITATION})),
    # A section of a document that is not here stays a citation.
    Case("The gauge is a pick policy (§54).", "comment", frozenset({CITATION})),
)

MUST_NOT_CATCH: tuple[Case, ...] = (
    # The constraint and the reason: what the prose style asks for.
    Case(
        "Flash attention and SDPA return no attention weights, so the eager kernel "
        "is a correctness requirement rather than a preference.",
        "comment",
    ),
    # A numbered section of the analysis is a module here, and opens.
    Case(
        "Section 9 reads the generated text, so a corpus without it fails first.",
        "docstring",
    ),
    # Hyphenated compounds that name a contrast or a category, not a document.
    Case(
        "The ANCHOR-versus-RECENCY contrast is the block-routing echo of the how-axis.",
        "comment",
    ),
    Case("A PORT-as-is item keeps its name and its signature.", "comment"),
    # Quantities that share the shape of an item code without being one.
    Case(
        "Vectorized: no Python loops over heads, ~10-30x faster than the loop it replaces.",
        "docstring",
    ),
    Case("A floor of 1e-12 keeps the ratio finite.", "docstring"),
    Case("The 8b preset and the 3b preset decode at a different nucleus mass.", "comment"),
    Case("gemma3-27b and dsv2-lite sample at 0.95.", "comment"),
    # Identifiers the allowlist exists for: a hub model id, the on-disk schema, a
    # module in this package, an ordinary use of a word a citation idiom also uses.
    Case("Presets cover meta-llama/Llama-3.1-8B-Instruct and Qwen/Qwen2.5-7B.", "comment"),
    Case("Reads signatures_v3/ beside metadata.json.", "docstring"),
    Case("The read-side gate on lane identity is anamnesis.analysis.lane_guard.", "docstring"),
    Case("The spec a cell was generated under, read from its own run metadata.", "docstring"),
    Case("Published as a TABLE over alpha, BH-FDR-corrected at the per-test rate.", "docstring"),
    # The message a run writes for an operator, with the substance stated inline.
    Case(
        "eos ids are model-specific and must be passed explicitly; an empty list "
        "would let generation run to the length cap",
        "raise",
    ),
    # The same sentences as data: a payload value is not a claim.
    Case("the reason is in research/notes/some-memo.md, banked 2026-07-12", "data"),
    Case("we now cache the matrix; the split is no longer identical", "data"),
    # A numbered section of a document that ships in this tree resolves.
    Case("Doses do not transfer between maps (docs/FINDINGS.md §2).", "comment"),
    Case("The route table is in docs/API.md §1.", "docstring"),
    # The HTTP routes the server exposes.
    Case("Every route answers at /api/v1/chat; /loom/progress streams the harvest.", "docstring"),
    # A printf zero-padding conversion in a log call.
    Case("replayed gen_%03d of shard_%02d", "log"),
)

EXEMPTED: tuple[Case, ...] = (
    # The one surviving date exemption: the date is part of the address.
    Case("The frozen record is https://example.com/2026-07-12/anamnesis.", "comment", exempt=True),
)

ALL_CASES = MUST_CATCH + MUST_NOT_CATCH + EXEMPTED


@pytest.fixture(scope="module")
def index() -> check_referents.TreeIndex:
    """One index of this repository, shared by every case."""
    return check_referents.TreeIndex.build(check_referents.DEFAULT_REPO)


@pytest.fixture(scope="module")
def referent_allowlist() -> check_referents.Allowlist:
    return check_referents.load_allowlist(check_referents.DEFAULT_ALLOWLIST)


@pytest.fixture(scope="module")
def date_allowlist() -> list:
    return check_timelessness.load_allowlist(check_timelessness.DEFAULT_ALLOWLIST)


def run_case(
    case: Case,
    index: check_referents.TreeIndex,
    referent_allowlist: check_referents.Allowlist,
    date_allowlist: list,
) -> tuple[set[str], int]:
    """Both checkers over one case; returns (rules fired, dates exempted)."""
    path = Path("case.py")
    source = case.source()
    timeless, exempted = check_timelessness.scan_source(path, source, date_allowlist)
    referents = check_referents.scan_source(path, source, index, referent_allowlist)
    return {v.rule for v in timeless} | {v.rule for v in referents}, len(exempted)


@pytest.mark.parametrize("case", ALL_CASES, ids=[c.label() for c in ALL_CASES])
def test_the_gates_produce_exactly_the_rules_the_corpus_states(
    case: Case,
    index: check_referents.TreeIndex,
    referent_allowlist: check_referents.Allowlist,
    date_allowlist: list,
) -> None:
    fired, exempted = run_case(case, index, referent_allowlist, date_allowlist)
    assert fired == set(case.rules), f"{case.surface}: {case.text!r}"
    assert exempted == (1 if case.exempt else 0)


def test_the_corpus_covers_every_rule_and_every_surface() -> None:
    """A rule with no case behind it, or a surface with none, is the next blind spot."""
    covered = {rule for case in MUST_CATCH for rule in case.rules}
    expected = set(check_referents.rule_names()) | set(check_timelessness.rule_names())
    missing = expected - covered
    assert missing <= {
        check_referents.MODULE_RULE,
        "changelog-prior-state",
        "changelog-changed-in",
        "changelog-we-now",
        "changelog-no-longer",
        "changelog-used-to",
    }, missing
    assert {case.surface for case in ALL_CASES} == set(SURFACES)


def test_every_case_is_written_on_a_surface_the_corpus_knows() -> None:
    for case in ALL_CASES:
        assert case.surface in SURFACES
        assert case.source()
