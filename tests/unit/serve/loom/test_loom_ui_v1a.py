"""The legacy loom page must describe the map the server is actually serving.

The page is `pleroma/serve/static/legacy_ui.html`. These tests pin it against
four false statements a page built for an earlier map would make:

  * a wordmark carrying a literal map generation (`v0`) while the server runs
    v1a;
  * `nAxes()` reading the rank off the ATLAS and falling back to a hardcoded
    8, so the provenance bar announces "code rank 8" over a rank-64 map, the
    code glyph silently truncates 64 coordinates to 8, and the constellation
    scales v1a coordinates against a different map's bank quantiles;
  * consuming `GET /atlas`, which has no map check of any kind and, with no
    `--atlas-report`, falls back to a stale rank-8 report built for a
    different bank;
  * 2-MEANS CAMPS as the page's primary structural language (card spine
    colour, two warp sheds, the constellation legend), although v1a is fit
    one-vs-rest on member-minus-fan-mean displacement and has no partition at
    all, and the signature-space split is uncorrelated with anything a reader
    sees (docs/FINDINGS.md section 9).

Nothing catches those unless something renders the page. These tests do
two things: assert the copy against the record statically, and — when a
headless Chrome is available — actually render the page against
`loom_ui_stub_server` and assert on the resulting DOM.

The static assertions are deliberately about CLAIMS, not formatting: they
pin the facts the page is allowed to state, not the words around them.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.unit.serve.loom.loom_ui_stub_server import (
    GAUGE_REFUSAL,
    LIVE_PROBE_RECEIPT,
    RESOLVED_RESOLUTION,
    STALE_RANK8_ATLAS,
    V1A_INFO,
    Fixture,
    serve,
)

UI = Path(__file__).resolve().parents[4] / "pleroma" / "serve" / "static" / "legacy_ui.html"


@pytest.fixture(scope="module")
def ui_text() -> str:
    return UI.read_text(encoding="utf-8")


# ── static: the page may not hardcode a map generation or a rank ──────────

def test_the_wordmark_is_not_a_hardcoded_map_generation(ui_text: str) -> None:
    """It said `v0` for a day after the server moved to v1a."""
    assert '<span class="ver">v0</span>' not in ui_text
    assert 'id="mapVer"' in ui_text, "the version badge must be filled from /info"
    assert "function mapGeneration(" in ui_text


def test_the_tab_title_is_not_a_hardcoded_map_generation(ui_text: str) -> None:
    assert "<title>LOOM v0</title>" not in ui_text
    assert "<title>LOOM</title>" in ui_text
    assert 'document.title = g ? ("LOOM " + g) : "LOOM"' in ui_text


def test_the_code_rank_comes_from_the_map_not_the_atlas(ui_text: str) -> None:
    assert "const CODE_RANK = 8" not in ui_text, "no hardcoded rank fallback"
    assert "function mapRank(" in ui_text
    # nAxes must delegate to the map, and must not read state.atlas.rank
    body = re.search(r"function nAxes\(\)\{(.*?)\}", ui_text, re.S)
    assert body is not None
    assert "mapRank" in body.group(1)
    assert "atlas" not in body.group(1)


def _prov(ui_text: str, key: str) -> str:
    """The tier-3 provenance body registered under `key`.

    The long copy this suite pins lives once, keyed by concept, in the PROV
    registry rather than inline in the `title` next to the chip it describes,
    so the assertions follow it there instead of searching a proximity window
    around the chip. Covers both the static table and a `prov(k, h, body)`
    call made at render time.
    """
    m = re.search(r'\[\s*"%s"\s*,\s*"(?:[^"\\]|\\.)*"\s*,\s*(.*?)\]\s*,\s*\n\s*\[' % re.escape(key),
                  ui_text, re.S)
    if m is None:
        m = re.search(r'prov\(\s*"%s"\s*,\s*"(?:[^"\\]|\\.)*"\s*,\s*(.*?)\);' % re.escape(key),
                      ui_text, re.S)
    assert m is not None, f"no tier-3 provenance registered under {key!r}"
    return m.group(1)


def _uncommented(src: str) -> str:
    """The page with HTML comments and JS block/line comments removed.

    Crude on purpose: it can only over-remove (a `/*` inside a string would
    swallow code), which makes this a conservative test — it never invents a
    violation, it can only miss one.
    """
    src = re.sub(r"<!--.*?-->", " ", src, flags=re.S)
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", " ", src)


def test_no_live_copy_claims_the_map_itself_is_rank_8(ui_text: str) -> None:
    """The original bug: the page announced the MAP as rank-8 over a rank-64
    map, because nAxes() read the atlas's rank and fell back to a literal 8.

    A blanket "no live 'rank-8'" ban would say the wrong thing: the
    object-view stub states, on its face, that the deleted ATLAS was rank-8 and this map is not — which is the
    opposite claim and the one the page must keep making. So the assertion is
    on the claim, not the substring: every live mention of rank-8 must be
    about the atlas, and must sit next to the map's real rank.
    """
    live = _uncommented(ui_text)
    assert "eight numbers" not in live
    for m in re.finditer(r".{90}rank-8.{90}", live, re.S):
        ctx = m.group(0)
        assert "atlas" in ctx.lower(), f"a live rank-8 that is not about the atlas: {ctx!r}"
        # ...and must disown the served map in the same breath, either by
        # naming the map's real rank or by saying outright that this is
        # another map's report. A bare "rank-8" is the claim that started
        # all this; the first assertion above catches it when it is not even
        # about an atlas, and this one catches it when it is about an atlas
        # but leaves the reader to assume the two are the same object.
        disowned = bool(
            re.search(r"(this|the served) map is rank-\d", ctx)
            or re.search(r"different (map|space)", ctx)
            or "another map" in ctx)
        assert disowned, (
            "a live rank-8 must disown the served map in the same sentence — "
            f"name the map's real rank, or say it is another map's: {ctx!r}")
    # and the narration itself must survive, so the reason is not lost
    assert "rank-8" in ui_text, "keep the comment explaining why this changed"


ATLAS_SYMBOLS = (
    "function atlasUsable(", "function atlasMismatch(", "function atlasRank(",
    "function atlasProvenanceNote(", "function atlasAxis(", "function atlasQuantiles(",
    "function atlasGroups(", "function atlasClasses(", "function inSpaceEnergy(",
    "function twinRange(", "function classHue(", "function loadAtlas(",
    "function buildAxisSelects(", "state.atlas", "state.classColour",
)


def test_the_atlas_path_is_deleted_root_and_branch(ui_text: str) -> None:
    """REPLACES test_the_atlas_is_validated_against_the_map_before_it_is_used.

    That test guarded a subsystem that is now gone: `GET /atlas` served a
    rank-8 w5 lever-bank report under a rank-64 v1a map, the page refused it
    correctly, and then spent ~230 lines saying so. The guarantee it encoded
    — no atlas is consumed unless it squares with the map — is now enforced
    the only way that cannot rot: there is no atlas consumer at all.

    A half-deletion is the dangerous state: an accessor left behind with its
    guard removed is exactly how the rank-8 report gets read again.
    """
    live = _uncommented(ui_text)
    for sym in ATLAS_SYMBOLS:
        assert sym not in live, f"{sym} survived the atlas deletion"
    assert '"/atlas"' not in live and "'/atlas'" not in live, "the page still fetches /atlas"


def test_the_fingerprint_scale_falls_back_to_the_fan_and_not_to_a_bare_one(ui_text: str) -> None:
    """With no bank quantiles there is exactly one honest scale left: this
    fan's own largest coordinate. Not 1.0 — which was neither — and not a
    remembered span from another draw."""
    body = _fn(ui_text, "axisNorm")
    assert "fanCodeScale()" in body
    assert "atlas" not in body.lower()
    assert "function fanCodeScale(" in ui_text


def test_the_reason_a_rank_check_could_never_certify_an_atlas_survives(ui_text: str) -> None:
    """Two rank-64 maps accept each other's codes, and a code reconstructed
    through the wrong one lands at cosine 0.068 — a silent corruption, not an
    error.

    The page has no atlas guard; the REASON one could never certify an atlas
    by rank must stay on it, or the next person to wire an
    atlas up re-derives rank-equality and ships the same footgun. A deleted
    guarantee that is merely untested is how the bug comes back.
    """
    assert "WITHOUT ERRORING" in ui_text, "the same-rank hazard must still be recorded"
    assert "0.068" in ui_text, "state the measurement"
    assert "map fingerprint" in ui_text, (
        "say what a future atlas path must carry INSTEAD of a rank")


# ── static: camp is demoted to a labelled per-draw diagnostic ─────────────

def test_camps_no_longer_colour_the_card_spine(ui_text: str) -> None:
    assert '.card[data-camp="0"]' not in ui_text
    assert '.card[data-camp="1"]' not in ui_text


def test_camps_are_retired_from_the_page(ui_text: str) -> None:
    """The per-draw 2-means camps are a v0 partition readout; v1a has no
    partition, and the signature-space split tracks nothing a reader sees
    (docs/FINDINGS.md section 9). The page neither renders nor reads them, and has no
    `minority` pick policy to consume them."""
    body = re.sub(r"/\*.*?\*/", "", ui_text, flags=re.S)   # comments may keep history
    for gone in ('"data-camp"', "function campOfScores(", 'text:"diag · camp ',
                 'text:"diag · camps"', 'text:"diag · separation"', "campCaveat(",
                 'key:"minority"', '["camp", "CAMP', '["separation", "SEPARATION'):
        assert gone not in body, f"retired camp affordance still present: {gone}"


def test_the_camp_affordances_say_what_the_map_actually_is(ui_text: str) -> None:
    """A camp tooltip must name v1a's construction, not just disclaim itself."""
    assert ui_text.count("one-vs-rest") >= 3
    assert "docs/FINDINGS.md §1" in ui_text


def test_the_warp_no_longer_splits_into_two_sheds(ui_text: str) -> None:
    """The two-shed fan drew a partition the served map does not have."""
    assert "twoSheds" not in ui_text
    assert "SHED_X" not in ui_text


def test_the_object_view_is_built_on_the_served_codes_and_claims_no_axis(ui_text: str) -> None:
    """REPLACES test_the_object_view_is_an_honest_stub_and_not_an_empty_plot.

    The stub existed because the only plot on offer was built from a
    different map's atlas. The object view that replaces it reads nothing but
    the codes of the fan on screen, so the guarantees move: it must not read
    any atlas, it must keep x and y on ONE scale (distances are the only true
    thing in a projection), and it must not label an axis it cannot back —
    no banked length axis exists for these maps, so x is never "length".
    """
    body = _fn(ui_text, "renderConstellation")
    assert "OBJECT VIEW — NOT BUILT" not in body, "the stub is replaced, not kept beside the plot"
    assert "atlas" not in body.lower().replace("atlases", ""), "the object view reads no atlas"
    assert "Math.min(sx, sy)" in body, "x and y must share one scale"
    objs = _fn(ui_text, "codeObjects")
    assert "codeOf(f)" in objs and "harvested === true" in objs, (
        "descriptors come from the harvested futures' own codes")
    assert "k < 3" in objs, "a fan with fewer than three bodies has no shape and must not draw one"
    # the tier-3 entry keeps the refusal to call x 'length'
    m = re.search(r'\["object-view",(.*?)\],', ui_text, re.S)
    assert m and "NOT CLAIMED" in m.group(1) and "length" in m.group(1)


def test_the_object_view_sizes_bodies_by_the_cards_own_pull(ui_text: str) -> None:
    """Two different 'pulls' for one future on one screen is a contradiction.
    The plot takes the card's number (fanPull) and only falls back to its own
    centred-code ratio when the server sent none."""
    objs = _fn(ui_text, "codeObjects")
    assert "fanPull(" in objs


def test_the_object_view_eigensolver_is_exact() -> None:
    """The projection is only as true as its eigenvectors. Run the page's own
    jacobiEig on a random Gram matrix and check G·V = V·diag(λ), VᵀV = I."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("no node on this host")
    src = UI.read_text(encoding="utf-8")
    a = src.index("function jacobiEig(")
    b = src.index("/* the descriptors, once per draw.")
    prog = src[a:b] + r"""
let seed = 11; const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };
const k = 12, R = 64;
const X = [...Array(k)].map(() => [...Array(R)].map(rnd));
const G = X.map(x => X.map(y => x.reduce((s, v, j) => s + v * y[j], 0)));
const e = jacobiEig(G);
let res = 0, orth = 0;
for (let i = 0; i < k; i++) for (let c = 0; c < k; c++){
  let s = 0; for (let j = 0; j < k; j++) s += G[i][j] * e.V[j][c];
  res = Math.max(res, Math.abs(s - e.vals[c] * e.V[i][c]));
}
for (let c = 0; c < k; c++) for (let d = 0; d < k; d++){
  let s = 0; for (let i = 0; i < k; i++) s += e.V[i][c] * e.V[i][d];
  orth = Math.max(orth, Math.abs(s - (c === d ? 1 : 0)));
}
console.log(JSON.stringify({ res, orth }));
"""
    out = subprocess.run([node, "-e", prog], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout)
    assert r["res"] < 1e-9 and r["orth"] < 1e-9, r


# ── static: what v1a computes per candidate is on the page ────────────────

def test_the_fan_relative_contrastive_magnitude_is_a_primary_readout(ui_text: str) -> None:
    """★ The copy lives in tier 3; the READOUT is still primary.

    Finding the words "CONTRASTIVE MAGNITUDE" near the chip is not enough —
    the name promises that `pull` renders as a PRIMARY readout. Both are
    asserted, and the copy is chased to where it went rather than
    dropped.
    """
    assert "function fanPull(" in ui_text
    assert "predicted_dose_scale" in ui_text
    i = ui_text.index('el("span", { text:"pull" })')
    chip = ui_text[i - 700: i]
    assert 'class:"sc primary"' in chip, "pull must render as a primary readout"
    assert '"pull"' in chip, "and must point at its own provenance entry"
    body = _prov(ui_text, "pull")
    assert "CONTRASTIVE MAGNITUDE" in body
    assert "r = .757" in body and "43%" in body


def test_the_magnitude_copy_carries_its_measured_numbers(ui_text: str) -> None:
    """`loudest` on v1a captures 43% of an oracle's picking skill
    (docs/FINDINGS.md section 8), and v1a predicts within-fan magnitude at r = .757
    against the incumbent map's .538."""
    assert "r = .757" in ui_text
    assert ".538" in ui_text
    assert "43%" in ui_text and "33%" in ui_text
    assert "docs/FINDINGS.md §8" in ui_text


def test_distinct_is_no_longer_the_primary_number(ui_text: str) -> None:
    i = ui_text.index('el("span", { text:"distinct" })')
    block = ui_text[i - 1400: i]
    assert 'class:"sc primary"' not in block.rsplit('row.appendChild', 1)[-1]
    assert "docs/FINDINGS.md §9" in block


def test_estimated_uncertified_labelling_is_kept_on_uncertified_numbers(ui_text: str) -> None:
    """A pick policy is not a measurement (docs/FINDINGS.md section 8), so every
    advisory readout is labelled estimated and uncertified."""
    assert ui_text.lower().count("estimated, uncertified") >= 3


# ── static: the fan is where the ceiling is, and the page says so ─────────

def test_the_spread_strip_states_the_ceiling_is_in_the_fan(ui_text: str) -> None:
    assert "+0.136" in ui_text
    assert "docs/FINDINGS.md §11" in ui_text


def test_the_flat_fan_copy_names_the_framing_not_the_model(ui_text: str) -> None:
    """Chat framing, not weights or size, flattens a fan, and 8B chat is
    exactly as flat as 3B chat — so 'go bigger' is not the advice
    (docs/FINDINGS.md section 11)."""
    assert "FRAMING artifact" in ui_text
    assert "assistant costume" in ui_text
    assert "8B chat is exactly as" in ui_text or "flat as 3B chat" in ui_text


def test_the_spread_strip_does_not_claim_its_own_numbers_predict_steerability(ui_text: str) -> None:
    """screen_mpd vs observed Δnr: ρ = +0.17 / −0.01 / −0.04 (n=9/12/10)."""
    i = ui_text.index("function renderSpread(")
    block = ui_text[i - 2600: i + 7000]
    assert "fpcd" in block, "the readout that DOES predict it must be named"
    assert "ρ = +0.833" in block, "state the measured numbers"
    assert "docs/FINDINGS.md §9" in block


def test_separation_is_not_shown_at_all(ui_text: str) -> None:
    """Separation is not displayed: a high separation can be one outlier
    rather than a fork, and signature-space spread does not track a reader
    (docs/FINDINGS.md section 9)."""
    body = re.sub(r"/\*.*?\*/", "", ui_text, flags=re.S)
    assert "sp.separation" not in body


# ── static: the ruler reference is not called a bank median ───────────────

def test_the_loudness_reference_is_described_as_the_ruler(ui_text: str) -> None:
    """`loudness_ref` is mean(norm_ref) — and on v1a norm_ref is the WIDE
    map's ruler, not v1a's own. Calling it a bank median invites reading it
    as a property of the bank rather than as the ruler that gives α its
    meaning (docs/FINDINGS.md section 2)."""
    assert "bank-median reference" not in ui_text
    assert "bank-median norm" not in ui_text
    assert "RULER in force" in ui_text


# ── the dose-band work another agent landed must not have regressed ───────

def test_the_zone_is_still_read_at_the_effective_alpha(ui_text: str) -> None:
    """The zone is read at the EFFECTIVE alpha, never the bare alpha — and at
    the LOUDEST SITE's effective alpha, not the per-site mean. Each
    site is dosed and clamped independently, so `scale_mean` can read nominal
    while one site sits at the 2.0 ceiling, and it is the loudest site that
    damages a generation (loom_serve.effective_alpha_json says the same:
    "the zone an operator should read off the band is the one containing
    alpha * scale[s]")."""
    assert "function effectiveAlpha(" in ui_text
    i = ui_text.index("const eff = effectiveAlpha(w);")
    window = ui_text[i: i + 200]
    assert "zoneFor(eff.alphaWorst)" in window
    assert "zoneFor(eff.alpha)" not in window, "the mean is not the gauge"


def test_every_zone_read_of_a_dosed_wear_uses_the_worst_site(ui_text: str) -> None:
    """There are three of them — the wear bar, the auto-wear log line and the
    hand-wear log line — and a gauge that is honest in one place and stale in
    another is worse than one that is wrong everywhere, because nobody knows
    which reading to believe."""
    for stale in ("zoneFor(eff.alpha)", "zoneFor(autoEff.alpha)"):
        assert stale not in ui_text, f"{stale} reads the per-site MEAN, not the loudest site"
    assert ui_text.count("zoneFor(eff.alphaWorst)") == 2
    assert "zoneFor(autoEff.alphaWorst)" in ui_text


def test_the_cross_model_bound_is_surfaced_when_the_server_sends_one(ui_text: str) -> None:
    """A band rescaled across MODELS carries an assumption its same-model
    validation cannot see. The server ships the measured size of that
    ambiguity on dose_band.derivation.cross_model; the UI must show it, and
    must show nothing at all when it was not sent (never a guessed bound)."""
    assert "derivation.cross_model" in ui_text or "derivation && info.dose_band.derivation.cross_model" in ui_text
    assert "band ×-model" in ui_text
    assert "served_is_conservative" in ui_text
    # The bound rides the rig's `band` gauge
    # cell as a sub-readout rather than being a cell of its own.
    i = ui_text.index('"data-sub":"band ×-model"')
    guard = ui_text[max(0, i - 400): i]
    assert "cm && cm.absolute && cm.relative" in guard, (
        "the cell must be gated on the server actually having sent a bound")


def test_the_dose_band_tier_badge_still_says_provisional_on_derived(ui_text: str) -> None:
    assert "DERIVED · provisional" in ui_text
    assert "UNCALIBRATED" in ui_text
    assert "function renderDoseBandBadge(" in ui_text


# ── the stub itself ──────────────────────────────────────────────────────

def test_the_stub_fixture_reproduces_the_live_trap() -> None:
    """The default fixture must BE the live trap: a rank-64 map served beside
    a stale rank-8 atlas."""
    assert V1A_INFO["map_meta"]["rank"] == 64
    assert STALE_RANK8_ATLAS["rank"] == 8
    fx = Fixture()
    assert fx.atlas is not None and fx.atlas["rank"] != fx.info["map_meta"]["rank"]


def test_the_stub_futures_carry_everything_the_ui_reads() -> None:
    rows = Fixture().futures()
    assert len(rows) == 8
    for r in rows:
        assert len(r["code"]) == 64
        assert {"distinct", "loudness", "predicted_dose_scale"} <= set(r["scores"])
    scales = [r["scores"]["predicted_dose_scale"] for r in rows]
    assert min(scales) < 1.0 < max(scales), "the fan must have real pull range"


def test_the_stub_serves_the_real_page_and_never_the_live_port() -> None:
    srv, url = serve()
    try:
        assert url.startswith("http://127.0.0.1:")
        assert not url.endswith(":8767/")
        import urllib.request
        with urllib.request.urlopen(url, timeout=10) as r:
            body = r.read().decode("utf-8")
        assert "loom deck" in body
        with urllib.request.urlopen(url + "info", timeout=10) as r:
            assert json.loads(r.read())["map_meta"]["rank"] == 64
    finally:
        srv.shutdown()


# ── rendered: the page in a real engine ──────────────────────────────────

CHROME = (shutil.which("google-chrome") or shutil.which("google-chrome-stable")
          or shutil.which("chromium") or shutil.which("chromium-browser"))

HARNESS = """
<script>
(function(){
  window.__errs = [];
  window.addEventListener("error", (e) => window.__errs.push(String(e.message)));
  window.addEventListener("unhandledrejection", (e) => window.__errs.push("rej: " + e.reason));
  setTimeout(function(){
    try {
      const c = document.getElementById("composer");
      c.value = "so why do i abandon things at 70%?";
      c.dispatchEvent(new Event("input", { bubbles: true }));
      // the object view is ON by default now; open it only if it is not
      const ov = document.getElementById("constelBtn");
      if (ov.getAttribute("aria-pressed") !== "true") ov.click();
      setTimeout(() => document.getElementById("loomBtn").click(), 300);
    } catch (e){ window.__errs.push("harness: " + e); }
  }, 600);
  setTimeout(function(){
    const d = document.createElement("div");
    d.id = "__errbox";
    d.setAttribute("data-errs", JSON.stringify(window.__errs));
    document.body.appendChild(d);
  }, 3200);
})();
</script>
"""


def _steps(*steps: str, settle: int = 1000) -> str:
    """A harness that runs `steps` in order, `settle` ms apart, then dumps errors.

    The first step always types into the composer, because nothing else on the
    page is reachable without a message to contemplate. Every step only touches
    controls a person could click — the point of the stub is that the page is
    driven, not stubbed.
    """
    blocks = "\n".join(
        "  setTimeout(function(){ try { %s } catch(e){ window.__errs.push('step%d: ' + e); } }, %d);"
        % (s, i, 600 + settle * i)
        for i, s in enumerate(steps)
    )
    return """
<script>
(function(){
  window.__errs = [];
  window.addEventListener("error", (e) => window.__errs.push(String(e.message)));
  window.addEventListener("unhandledrejection", (e) => window.__errs.push("rej: " + e.reason));
  function type(t){
    const c = document.getElementById("composer");
    c.value = t;
    c.dispatchEvent(new Event("input", { bubbles: true }));
  }
  window.__type = type;
%s
  setTimeout(function(){
    const d = document.createElement("div");
    d.id = "__errbox";
    d.setAttribute("data-errs", JSON.stringify(window.__errs));
    document.body.appendChild(d);
  }, %d);
})();
</script>
""" % (blocks, 600 + settle * (len(steps) + 1))


def _render(fx: Fixture, harness: str | None = None, budget: int = 6000,
            log: list[str] | None = None) -> str:
    """Render the page. `log`, when given, is filled with every request path
    the page made — so a test can assert on what was NOT fetched."""
    srv, url = serve(fx, harness if harness is not None else HARNESS)
    tmp = Path(tempfile.mkdtemp(prefix="loomui-test-"))
    try:
        proc = subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
             "--hide-scrollbars", "--window-size=1680,1100",
             f"--virtual-time-budget={budget}", f"--user-data-dir={tmp / 'p'}",
             "--dump-dom", url],
            capture_output=True, text=True, timeout=240,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        if log is not None:
            log.extend(getattr(srv, "request_log", []))
        return proc.stdout
    finally:
        srv.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)


def _body(dom: str) -> str:
    """The rendered DOM with every <script> AND <style> removed.

    --dump-dom hands back the inline source too, so a naive `"gauge Δnr" not in
    dom` matches the branch of the code that WOULD print it. Assertions about
    what is on screen must be made against this.

    ★ <style> is stripped too: a CSS comment explaining WHY a readout is
    styled quietly ("...next to a large manner move...") would otherwise match
    a test for whether the page displays a manner metric. Same trap as
    <script> — prose about the code is not prose on the screen.
    """
    out = re.sub(r"<script\b.*?</script>", " ", dom, flags=re.S | re.I)
    return re.sub(r"<style\b.*?</style>", " ", out, flags=re.S | re.I)


def _panel(dom: str, el_id: str) -> str:
    """One element's rendered subtree, by id, with script/style gone.

    Whole-document assertions are the wrong instrument for "this PANEL must
    not say X": the page is 5,000 lines and something somewhere will say it.
    """
    body = _body(dom)
    i = body.find('id="%s"' % el_id)
    assert i >= 0, f"no #{el_id} in the rendered page"
    start = body.rfind("<", 0, i)
    depth, j = 0, start
    while j < len(body):
        if body.startswith("</", j):
            depth -= 1
            j = body.index(">", j) + 1
            if depth == 0:
                return body[start: j]
        elif body[j] == "<" and body[j + 1] not in "!/":
            tag_end = body.index(">", j)
            if body[tag_end - 1] != "/":
                depth += 1
            j = tag_end + 1
        else:
            j += 1
    return body[start:]


def _errs(dom: str) -> list[str]:
    m = re.search(r'id="__errbox" data-errs="([^"]*)"', dom)
    assert m is not None, "the harness never ran — the page did not boot"
    return json.loads(m.group(1).replace("&quot;", '"').replace("&amp;", "&"))


def _cells(dom: str) -> dict[str, str]:
    """The top rig's cells, name -> face text.

    Rig cells are gauge cells — a name, a small SVG form, then the face. The
    text-only shape is also matched (the `info unavailable` cell uses it)."""
    out = {
        m.group(1): m.group(2)
        for m in re.finditer(
            r'<div class="cell"[^>]*><span class="k">([^<]*)</span>'
            r'<span class="v[^"]*">([^<]*)</span></div>', dom)
    }
    for name, face, _title in _rig(dom):
        out[name] = face
    return out


def _rig(dom: str) -> list[tuple[str, str, str]]:
    """(name, face, hover) for every gauge cell on the top rig, in order."""
    rows: list[tuple[str, str, str]] = []
    for m in re.finditer(r'<div class="cell g[^"]*" data-g="([^"]+)"([^>]*)>(.*?)</div>', dom, re.S):
        attrs, inner = m.group(2), m.group(3)
        t = re.search(r'title="([^"]*)"', attrs)
        v = re.search(r'<span class="v">([^<]*)</span>', inner)
        rows.append((m.group(1), v.group(1) if v else "", t.group(1) if t else ""))
    return rows


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_the_page_renders_a_full_draw_with_no_javascript_errors() -> None:
    dom = _render(Fixture())
    assert _errs(dom) == []
    assert "8 futures" in dom and "8 harvested" in dom


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_the_rendered_page_names_the_served_map_and_its_real_rank() -> None:
    dom = _render(Fixture())
    assert re.search(r'<span class="ver"[^>]*>v1a</span>', dom), "wordmark must read v1a"
    assert "<title>LOOM v1a</title>" in dom, "the tab title must follow the map too"
    # The rank and the map file are CONSTANTS, so they live in one `map`
    # cell (face "r64") whose hover names the file
    # and whose click opens every constant. Still the MAP's rank, never 8.
    rig = {name: (face, hover) for name, face, hover in _rig(dom)}
    assert rig["map"][0] == "r64"
    assert "v1a_r64" in rig["map"][1] and "rank 64" in rig["map"][1]


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_the_page_never_requests_the_atlas_at_all() -> None:
    """Nothing on this page asks for an atlas.

    This is stronger than pinning how an atlas consumer would treat a stale,
    rank-matching or missing atlas, and it is the guarantee that cannot rot: nothing on this page asks for an atlas. The stub is served WITH the
    live rank-8 trap present and advertised, and the page must still not
    reach for it — a version that fetches it and then discards it would pass
    an "are the bank cells absent" test while leaving the footgun loaded.
    """
    log: list[str] = []
    dom = _render(Fixture(), log=log)          # rank-8 atlas IS available here
    assert log, "the request log never filled — the harness did not drive the page"
    assert "GET /info" in log, "sanity: the page must still read /info"
    assert not [r for r in log if "/atlas" in r], f"the page fetched the atlas: {log}"
    cells = _cells(dom)
    assert cells["map"] == "r64", "the rank is the MAP's and survives"
    for gone in ("atlas", "bank", "in-space"):
        assert gone not in cells, f"the header still publishes an {gone!r} cell"
    body = _body(dom)
    assert "atlas ignored" not in body, (
        "the one-off toast is gone; the stub states it permanently instead")


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_the_object_view_draws_one_body_per_harvested_future() -> None:
    """REPLACES test_the_object_view_stub_states_the_mismatch_on_screen.

    ON by default now (a viewer who switched it off stays off), so no click."""
    # before any draw it says so rather than drawing an empty frame
    empty = _panel(_render(Fixture(), _steps('__type("a contemplated turn");')), "constel")
    assert "draw a fan" in empty and "objv-body" not in empty
    dom = _render(Fixture())
    assert _errs(dom) == []
    panel = _panel(dom, "constel")
    assert "OBJECT VIEW — NOT BUILT" not in panel
    n = len(re.findall(r'class="objv-body', panel))
    assert n == 8, f"one body per harvested future, got {n}"
    assert "ALONE" in panel.upper(), "the most solitary future is named on the face"
    assert "position: the fan" in panel, "the projection is stated on the face"


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_the_cards_carry_their_object_instead_of_64_bars() -> None:
    """REPLACES test_the_code_glyph_shows_all_64_axes_without_blowing_up_the_card.

    With a shape to show, each card carries its object chip and the 64-bar
    comb is gone from the face; every chip's hover fits the tip budget."""
    dom = _render(Fixture())
    chips = re.findall(r'<svg class="objchip"[^>]*aria-label="([^"]*)"', dom)
    assert len(chips) == 8, f"one object chip per harvested card, got {len(chips)}"
    assert all(len(c) <= 100 for c in chips), chips
    assert 'class="fp"' not in _body(dom), "the comb is only the fallback now"


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_the_fan_relative_pull_leads_the_spread_strip_and_the_cards() -> None:
    dom = _render(Fixture())
    groups = re.findall(r'<div class="grp[^"]*"[^>]*><span>([^<]*)</span><b>([^<]*)</b>', dom)
    keys = [g[0] for g in groups]
    assert keys[0] == "pull range", f"spread strip leads with {keys[0]!r}"
    rng = dict(groups)["pull range"]
    assert rng.startswith("×") and "–" in rng
    assert re.search(r'<span>pull</span><b>×\d\.\d\d</b>', dom), "cards must carry pull"


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_no_camp_renders_on_a_real_draw() -> None:
    dom = _body(_render(Fixture()))
    assert "data-camp=" not in dom
    assert "diag · camp" not in dom and "diag · separation" not in dom


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_the_warp_draws_one_shed_for_one_fan() -> None:
    dom = _render(Fixture())
    fan = dom[dom.index('<svg class="fan"'):]
    fan = fan[: fan.index("</svg>")]
    # the take-off ring is the shed marker; one prefix, one shed, one ring
    rings = re.findall(r'<circle cx="([\d.]+)" cy="5.5" r="5.5"', fan)
    assert len(rings) == 1, f"expected one take-off ring, got {rings}"


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_the_dose_gauge_still_reads_derived_as_provisional() -> None:
    dom = _render(Fixture())
    assert "DERIVED · provisional" in dom


# ══ the gauge and the auto-dose must be REACHABLE ══════════════════════════
#
# An operator must be able to use both from the page:
#
#   * the probe is reachable on its own, so a fan already on screen can be
#     scored without paying for a second one — `POST /probe` is live code,
#     not only a rider on auto-wear with policy=gauge;
#   * the /loom response's `probe` receipt is read — the `resolution` block,
#     `ranking`, `pick`, `clean_regime`, `absolute_comparable`, the label and
#     the timing;
#   * /state's `probe` and `probe_note` are read, so a restored snapshot's
#     gauge scores never render as if they had been measured this session;
#   * the per-candidate cell never renders a signed "+0.389" with a
#     zero-centred meter and a "mean base rank − mean steered rank" tooltip,
#     i.e. exactly the Δnr the receipt forbids under
#     `absolute_comparable: false`;
#   * `dose_policy` has a control, not a read-only provenance cell telling the
#     operator to restart the server.
#
# These tests are about CLAIMS and REACHABILITY, not layout.

# ── static ────────────────────────────────────────────────────────────────

def test_the_page_can_post_probe_without_redrawing(ui_text: str) -> None:
    """The whole gap: a probe you can only get by redrawing is not a probe you
    can use on the fan you are looking at."""
    assert '"/probe"' in ui_text, "the page must call POST /probe"
    assert "function doProbe(" in ui_text
    assert 'id="probeBtn"' in ui_text
    live = _uncommented(ui_text)
    assert 'post("/probe"' in live, "the /probe call must be live code, not a comment"


def test_the_probe_is_priced_from_the_server_and_never_hardcoded(ui_text: str) -> None:
    """A fabricated latency estimate is worse than none — the same rule the
    dose band lives by. /info.probe.cost_estimate.basis is the only source."""
    assert "function probeCost(" in ui_text
    body = _fn(ui_text, "probeCost")
    assert "cost_estimate" in body and "per_span_harvest_s" in body
    assert "return null" in body, "an unpriceable probe must be unpriced"
    # no bare seconds constants standing in for the server's basis
    assert "10.5" not in body and "14.6" not in body and "16.2" not in body


def test_the_probe_cost_is_shown_before_the_button_is_pressed(ui_text: str) -> None:
    assert 'id="probeCost"' in ui_text
    block = _fn(ui_text, "renderProbeRow")
    assert "probeCost()" in block
    assert "cost not priced by this server" in block


def _fn(ui_text: str, name: str) -> str:
    """One whole top-level function body, by its own boundaries rather than a
    character count — a window that a later edit can outgrow is a test that
    stops checking without failing."""
    i = ui_text.index("function " + name + "(")
    j = ui_text.find("\nfunction ", i + 1)
    return ui_text[i: j if j > 0 else len(ui_text)]


def test_the_resolution_block_is_rendered_field_by_field(ui_text: str) -> None:
    """`resolution` is the answer to "does this reading mean anything" and it
    was parsed by nothing. Every field the server ships must be on the page."""
    for field in ("resolved", "within_candidate_sd_nr", "between_candidate_sd_cell",
                  "estimated_signal_sd",
                  "noise_share_of_cell_variance_at_this_reps",
                  "reps_needed_to_resolve", "n_candidates_excluded_for_reps"):
        assert field in ui_text, f"the UI never reads resolution.{field}"
    assert "function renderProbeVerdict(" in ui_text


def test_an_unresolved_probe_prints_the_servers_own_why_string(ui_text: str) -> None:
    """Paraphrasing the reason is how a caveat becomes decoration."""
    block = _fn(ui_text, "renderProbeVerdict")
    assert "r.why" in block
    assert '"WHY: "' in block


def test_a_missing_resolution_is_a_third_state_and_not_a_pass(ui_text: str) -> None:
    block = _fn(ui_text, "renderProbeVerdict")
    assert "resolution unknown" in block
    assert "Absent is not a pass" in block


def test_the_gauge_cell_is_not_rendered_as_a_delta_nr(ui_text: str) -> None:
    """base='fan' makes the cells RANKING STATISTICS ONLY
    (`absolute_comparable: false`), and the receipt's own label says so: the
    constant self-distance offset cancels in the order but not in the value."""
    assert "function gaugeRanking(" in ui_text
    assert "absolute_comparable" in ui_text
    i = ui_text.index('el("span", { text: absolute ? "gauge Δnr" : "gauge rank" })')
    # ★ SCOPED to the gauge chip's own construction, by its own boundaries.
    # A fixed character window is only a guess at "the gauge chip": an edit
    # that shortens the copy slides it forward to swallow the NEXT chip
    # (cos→worn), whose signed meter is correct. A window that a later edit
    # can outgrow is a test that fails without a bug.
    start = ui_text.rindex("const bits = [", 0, i)
    end = ui_text.index('row.appendChild(el("div", Object.assign({\n      class:"sc primary"', i)
    block = ui_text[start: end]
    assert "ordering only" in block
    # the meter must be the POSITION in the order, not the magnitude
    assert "rk.n - mine + 1" in ui_text[start - 600: end]
    # and the old zero-centred signed magnitude meter must be gone from it
    assert 'class:"meter signed"' not in block, (
        "the gauge chip must not use a zero-centred magnitude meter")
    # the cell's own provenance must still refuse the Δnr reading
    body = _prov(ui_text, "gauge-cell")
    assert "absolute_comparable: false" in body
    assert "ORDER" in body


def test_the_absolute_reading_is_only_taken_when_the_receipt_grants_it(ui_text: str) -> None:
    """Absent is NOT true: a receipt without the field is read as
    non-comparable, which is the conservative reading."""
    block = _fn(ui_text, "gaugeRanking")
    assert "absolute_comparable === true" in block


def test_the_resolution_nulls_are_not_rendered_as_zeroes(ui_text: str) -> None:
    """★ The Number(null) === 0 trap (a null renders as a measured 0), one
    level in.
    `reps_needed_to_resolve` is null exactly when the fan did NOT resolve, and
    `noise_share_…` is null when the between-candidate SD is 0. A
    Number.isFinite(Number(x)) guard passes both and printed "reps needed null"
    against the live 8B receipt."""
    block = _fn(ui_text, "renderProbeVerdict")
    assert 'const num = (v) => (typeof v === "number" && Number.isFinite(v))' in block
    live = _uncommented(block)
    for field in ("reps_needed_to_resolve",
                  "noise_share_of_cell_variance_at_this_reps",
                  "estimated_signal_sd"):
        j = live.index(field)
        assert "num(r." in live[max(0, j - 60): j], f"{field} is not behind the typeof guard"
    # and a null rep count under resolved:false must SAY there is no such count
    assert "no rep count helps" in block


def test_a_null_gauge_still_cannot_render_as_a_measured_zero(ui_text: str) -> None:
    """Number(null) is 0 and 0 is finite, so the loose idiom would print a dead candidate as
    '+0.000' under a tooltip claiming it was probed."""
    assert 'typeof s.gauge === "number"' in ui_text
    i = ui_text.index('typeof s.gauge === "number"')
    assert "Number.isFinite(s.gauge)" in ui_text[i: i + 120]


def test_a_probe_receipt_from_loom_is_adopted_rather_than_dropped(ui_text: str) -> None:
    """/loom degrades a FAILED probe to {error, note} inside a 200 — it will not
    throw away a paid-for draw. A swallowed receipt reads exactly like a probe
    that never ran."""
    assert "function adoptProbe(" in ui_text
    live = _uncommented(ui_text)
    assert "adoptProbe(r.probe" in live
    assert "p.error" in _fn(ui_text, "renderProbeVerdict"), "the error shape must render too"


def test_the_state_probe_note_is_surfaced(ui_text: str) -> None:
    """Gauge scores persist across a restart; the receipt does NOT, and the
    server ships `probe_note` saying to re-probe before trusting them."""
    assert "probe_note" in ui_text
    assert "probeNote" in ui_text
    assert "probeStale" in ui_text


def test_the_gauge_refusal_is_surfaced_and_never_retried_under_another_policy(
        ui_text: str) -> None:
    """auto.policy='gauge' on an unprobed fan is a 400 with instructions.
    Substituting a free policy under gauge's name would pass a pick policy
    off as the probe's measurement (docs/FINDINGS.md section 8)."""
    assert "gauge refused — not degraded" in ui_text
    i = ui_text.index("gauge refused — not degraded")
    block = ui_text[i - 900: i + 900]
    assert "refused rather than" in block
    assert "PROBE FAN" in block


def test_the_dose_policy_has_a_control_built_from_the_servers_list(ui_text: str) -> None:
    assert 'id="dosePolicy"' in ui_text
    assert "function buildDosePolicySelect(" in ui_text
    assert "function dosePolicies(" in ui_text
    live = _uncommented(ui_text)
    assert "body.dose_policy = dp" in live, "/wear must carry the chosen policy"
    assert "autoReq.dose_policy = dp" in live, "auto-wear must carry it too"


def test_the_dose_policy_default_comes_only_from_dose_policy_default(ui_text: str) -> None:
    """★ The per-entry `dose_policies[].default` says `flat` on a server whose
    RUNNING default is `predicted` (--dose-policy-default). Believing it is how
    the page comes to describe a server it is not talking to."""
    block = _fn(ui_text, "serverDosePolicy")
    assert "dose_policy_default" in block
    assert ".default" not in block.replace("dose_policy_default", ""), (
        "serverDosePolicy must not read the per-entry default flag")


def test_the_page_no_longer_tells_the_operator_to_restart_the_server(ui_text: str) -> None:
    assert "Restart with --dose-policy-default predicted" not in ui_text
    assert "no restart needed" in ui_text


def test_a_dose_policy_refusal_is_not_quietly_retried_as_flat(ui_text: str) -> None:
    """`predicted` refuses rather than degrading, and the UI must not undo that."""
    i = ui_text.index("dose policy refused")
    block = ui_text[i - 900: i + 900]
    assert "refused this dose policy rather than" in block
    assert "nothing has been worn" in block


def test_the_servers_own_effective_alpha_receipt_is_read(ui_text: str) -> None:
    """The page computes effective alpha from `worn.dose`; the server also
    reports it outright. Showing both means a drift is visible, not silent."""
    assert "function renderEffAlpha(" in ui_text
    assert "alpha_effective_per_site" in ui_text
    assert "alpha_effective_range" in ui_text
    block = _fn(ui_text, "renderEffAlpha")
    assert "DISAGREES WITH THIS PAGE" in block
    # and it must still read the LOUDEST site, not the mean
    assert "Math.max.apply" in block


def test_the_probe_price_survives_the_repaint_that_used_to_erase_it(ui_text: str) -> None:
    """buildPolicySelect put the price in the gauge option's text and renderAuto
    — which runs on every slider nudge — overwrote every option with a bare
    label. The one option that costs tens of seconds therefore read as free."""
    assert "function policyOptionLabel(" in ui_text
    block = _fn(ui_text, "renderAuto")
    assert "policyOptionLabel(p, blocked)" in block
    assert "opt.textContent = p.label" not in block


# ── the stub's own new fixtures ───────────────────────────────────────────

def test_the_stub_reproduces_the_live_dose_policy_disagreement() -> None:
    """The live /info says `dose_policies[0].default: true` for flat while
    `dose_policy_default` is `predicted`. The fixture must carry that trap."""
    flat = next(p for p in V1A_INFO["dose_policies"] if p["key"] == "flat")
    assert flat["default"] is True
    assert V1A_INFO["dose_policy_default"] == "predicted"


def test_the_stub_probe_receipt_is_the_live_unresolved_one() -> None:
    """The fans this server can currently draw do not resolve. That is the
    default fixture, because it is the case the UI must not launder."""
    r = LIVE_PROBE_RECEIPT["resolution"]
    assert r["resolved"] is False
    assert r["estimated_signal_sd"] == 0.0
    assert r["reps_needed_to_resolve"] is None
    assert LIVE_PROBE_RECEIPT["absolute_comparable"] is False
    assert RESOLVED_RESOLUTION["resolved"] is True


def test_the_stub_ranks_by_cell_and_leaves_a_null_when_asked() -> None:
    fx = Fixture()
    r = fx.probe_receipt()
    cells = {c["index"]: c["gauge"] for c in r["candidates"]}
    assert r["ranking"] == sorted(cells, key=lambda i: (-cells[i], i))
    assert r["pick"] == r["ranking"][0]
    dead = Fixture(unscored_candidate=3).probe_receipt()
    assert dead["candidates"][3]["gauge"] is None
    assert 3 not in dead["ranking"]
    assert dead["dead_rows"]


def test_the_stub_refuses_gauge_without_a_probe_the_way_the_server_does() -> None:
    assert "needs probe scores" in GAUGE_REFUSAL
    assert "POST /probe" in GAUGE_REFUSAL


# ── rendered ─────────────────────────────────────────────────────────────

PROBE_STEPS = (
    '__type("so why do i abandon things at 70%?");',
    'document.getElementById("loomBtn").click();',
    'document.getElementById("probeBtn").click();',
)


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_the_probe_row_is_visible_and_priced_after_a_draw() -> None:
    dom = _render(Fixture(), _steps(*PROBE_STEPS[:2]), budget=9000)
    assert _errs(dom) == []
    row = re.search(r'<div class="proberow" id="probeRow"([^>]*)>', dom)
    assert row is not None, "the probe row is not in the DOM"
    assert "hidden" not in row.group(1), "the probe row must be shown when /info offers a probe"
    cost = re.search(r'<span class="cost" id="probeCost"[^>]*>(.*?)</span>\s*<span class="spacer">', dom, re.S)
    assert cost is not None
    # k=8 (the fan on screen) x reps=2, priced off the server's own basis:
    # 8*0.99 + 16*1.33 = 29.2 s over 16 spans
    assert "29 s" in cost.group(1), cost.group(1)
    assert "16 spans" in cost.group(1)
    shown = _body(dom)
    assert 'id="probeBtn"' in shown and 'id="probeBtn" disabled' not in shown


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_an_unresolved_probe_says_so_next_to_the_numbers() -> None:
    dom = _render(Fixture(), _steps(*PROBE_STEPS), budget=12000)
    assert _errs(dom) == []
    box = re.search(r'<div class="probeverdict ([a-z]+)" id="probeVerdict"[^>]*>(.*?)</div>', dom, re.S)
    assert box is not None, "no verdict line rendered after a probe"
    assert box.group(1) == "unresolved", box.group(1)
    body = box.group(2)
    assert "NOT resolved" in body
    # the server's own sentence, verbatim
    assert "entirely explained by measurement noise" in body
    assert "signal sd" in body and ">0.000<" in body
    assert "noise share" in body and "46%" in body
    # the live receipt's reps_needed_to_resolve is null and must NOT print as one
    assert ">null<" not in body, body
    assert "no rep count helps" in body


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_a_probed_fan_shows_ranks_and_never_a_delta_nr() -> None:
    dom = _render(Fixture(), _steps(*PROBE_STEPS), budget=12000)
    assert _errs(dom) == []
    shown = _body(dom)
    ranks = re.findall(r'<span>gauge rank</span><b>#(\d+) of (\d+)</b>', shown)
    assert len(ranks) == 8, f"expected a rank on every card, got {ranks}"
    assert {r[1] for r in ranks} == {"8"}
    assert sorted(int(r[0]) for r in ranks) == list(range(1, 9))
    # the cell rides along, labelled, and never as the headline
    assert "(ordering only)" in shown
    assert "gauge Δnr" not in shown, "base='fan' must not be rendered as Δnr"
    # every rank is marked provisional, because this fan did not resolve
    assert shown.count("NOT RESOLVED") >= 8


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_a_resolved_probe_reads_as_resolved_and_keeps_the_uncertified_label() -> None:
    fx = Fixture(probe_resolution=dict(RESOLVED_RESOLUTION))
    dom = _render(fx, _steps(*PROBE_STEPS), budget=12000)
    assert _errs(dom) == []
    box = re.search(r'<div class="probeverdict ([a-z]+)" id="probeVerdict"[^>]*>(.*?)</div>', dom, re.S)
    assert box is not None and box.group(1) == "resolved", box and box.group(1)
    assert "ranking resolved" in box.group(2)
    assert "reps needed" in box.group(2)
    # resolved is not certified: the gauge failed its test as an instrument
    # (docs/FINDINGS.md section 8)
    assert "ESTIMATED AND UNCERTIFIED" in box.group(2)
    assert "NOT RESOLVED" not in _body(dom)


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_a_fresh_base_earns_the_absolute_reading() -> None:
    """base='fresh' makes `absolute_comparable` true, and only then is the Δnr
    reading allowed."""
    steps = (
        '__type("so why do i abandon things at 70%?");',
        'const s = document.getElementById("probeBase"); s.value = "fresh";'
        ' s.dispatchEvent(new Event("change", { bubbles: true }));',
        'document.getElementById("loomBtn").click();',
        'document.getElementById("probeBtn").click();',
    )
    dom = _render(Fixture(), _steps(*steps), budget=14000)
    assert _errs(dom) == []
    shown = _body(dom)
    assert "gauge Δnr" in shown, "a fresh base must restore the absolute reading"
    assert "gauge rank</span>" not in shown
    assert "(ordering only)" not in shown


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_a_candidate_the_probe_could_not_score_is_not_a_measured_zero() -> None:
    """An unscored candidate never renders as a measured "+0.000", through a
    real render."""
    dom = _render(Fixture(unscored_candidate=3), _steps(*PROBE_STEPS), budget=12000)
    assert _errs(dom) == []
    shown = _body(dom)
    ranks = re.findall(r'<span>gauge rank</span><b>#(\d+) of (\d+)</b>', shown)
    assert len(ranks) == 7, f"the unscored candidate must carry no rank: {ranks}"
    assert {r[1] for r in ranks} == {"7"}
    assert "+0.000" not in shown, "a null gauge rendered as a measured zero"
    assert "1 dead probe row" in shown


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_a_failed_probe_is_reported_and_does_not_look_like_no_probe() -> None:
    dom = _render(Fixture(probe_fails=True), _steps(*PROBE_STEPS), budget=12000)
    # the failure is surfaced, not thrown: no uncaught rejection
    assert _errs(dom) == []
    shown = _body(dom)
    assert "/probe failed" in shown
    assert "no z vector" in shown


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_a_loom_side_probe_failure_still_serves_the_draw_and_says_why() -> None:
    """/loom degrades the probe INSIDE a 200. Both halves must be visible."""
    steps = (
        '__type("so why do i abandon things at 70%?");',
        'document.getElementById("autoArm").click();',
        'const s = document.getElementById("autoPolicy"); s.value = "gauge";'
        ' s.dispatchEvent(new Event("change", { bubbles: true }));',
        'document.getElementById("loomBtn").click();',
    )
    dom = _render(Fixture(probe_fails=True), _steps(*steps), budget=14000)
    assert _errs(dom) == []
    assert "8 futures" in _body(dom), "the draw must survive a failed probe"
    box = re.search(r'<div class="probeverdict ([a-z]+)" id="probeVerdict"[^>]*>(.*?)</div>', dom, re.S)
    assert box is not None and box.group(1) == "unresolved"
    assert "probe failed" in box.group(2)
    assert "POST /probe to retry without redrawing" in box.group(2)


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_the_gauge_policy_refusal_reaches_the_screen() -> None:
    """The stub refuses `auto.policy=gauge` with no probe exactly as pick_auto
    does. The refusal must arrive verbatim, with a route out, and nothing must
    be worn by a substituted policy."""
    steps = (
        '__type("so why do i abandon things at 70%?");',
        'document.getElementById("autoArm").click();',
        'const s = document.getElementById("autoPolicy"); s.value = "gauge";'
        ' s.dispatchEvent(new Event("change", { bubbles: true }));'
        ' window.__noProbe = true;',
        # send the draw with the rider suppressed, the way an older client or a
        # hand-built request would: the server must refuse and the page must say so
        'const p = window.fetch; window.fetch = function(u, o){'
        ' if (o && o.body && String(u).indexOf("/loom") >= 0){'
        '   const b = JSON.parse(o.body); delete b.probe; o.body = JSON.stringify(b); }'
        ' return p.apply(this, arguments); };'
        ' document.getElementById("loomBtn").click();',
    )
    dom = _render(Fixture(), _steps(*steps), budget=14000)
    shown = _body(dom)
    assert "needs probe scores" in shown, "the server's own refusal must be shown"
    assert "gauge refused" in shown
    assert "PROBE FAN" in shown
    assert "WEARING" not in shown, "nothing may be worn under a refused policy"


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_the_dose_policy_control_offers_the_servers_list_and_its_real_default() -> None:
    dom = _render(Fixture(), _steps('__type("hi");'), budget=6000)
    assert _errs(dom) == []
    sel = re.search(r'<select id="dosePolicy"[^>]*>(.*?)</select>', dom, re.S)
    assert sel is not None, "no dose-policy control in the DOM"
    opts = re.findall(r'<option value="([a-z]+)"[^>]*>([^<]*)</option>', sel.group(1))
    assert [o[0] for o in opts] == ["flat", "predicted"]
    # ★ the RUNNING default, not the per-entry `default: true` on flat
    marked = [o[0] for o in opts if "server default" in o[1]]
    assert marked == ["predicted"], opts
    assert 'id="dosePolWrap" hidden' not in dom


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_a_predicted_wear_shows_the_servers_effective_alpha_at_the_loudest_site() -> None:
    steps = (
        '__type("so why do i abandon things at 70%?");',
        'document.getElementById("loomBtn").click();',
        'document.getElementById("wearBtn").click();',
    )
    dom = _render(Fixture(), _steps(*steps), budget=12000)
    assert _errs(dom) == []
    out = re.search(r'<span class="effalpha([^"]*)" id="effAlphaOut"[^>]*>(.*?)</span>\s*<span class="why',
                    dom, re.S)
    assert out is not None, "the server's effective_alpha receipt is not rendered"
    body = out.group(2)
    assert "α_eff" in body and "loudest" in body
    # the zone is read at the loudest site and the two arithmetics must agree
    assert "DISAGREES WITH THIS PAGE" not in body, body
    assert re.search(r"\[(SUBLIMINAL|THRESHOLD|AUDIBLE|OVERDRIVEN|UNCALIBRATED)\]", body), body


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_a_refused_dose_policy_is_reported_and_nothing_is_worn() -> None:
    steps = (
        '__type("so why do i abandon things at 70%?");',
        'document.getElementById("loomBtn").click();',
        'const s = document.getElementById("dosePolicy"); s.value = "flat";'
        ' s.dispatchEvent(new Event("change", { bubbles: true }));',
        'document.getElementById("wearBtn").click();',
    )
    dom = _render(Fixture(dose_policy_refuses=True), _steps(*steps), budget=14000)
    shown = _body(dom)
    assert "/wear failed" in shown
    assert "dose policy refused" in shown
    assert "nothing has been worn" in shown
    assert "NO CODE WORN" in shown


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_both_new_controls_vanish_against_a_server_that_lacks_them() -> None:
    """An offered control the server will refuse is worse than no control. A
    server whose /info advertises no probe and no dose_policies, and the page
    must then show neither — and still draw."""
    old = json.loads(json.dumps(V1A_INFO))
    old.pop("probe")
    old.pop("dose_policies")
    old.pop("dose_policy_default")
    old["auto_policies"] = [p for p in old["auto_policies"] if p["key"] != "gauge"]
    dom = _render(Fixture(info=old),
                  _steps('__type("hi");',
                         'document.getElementById("loomBtn").click();'),
                  budget=9000)
    assert _errs(dom) == []
    shown = _body(dom)
    assert 'id="probeRow" aria-label="gauge probe" hidden' in shown
    assert 'id="dosePolWrap" hidden' in shown
    sel = re.search(r'<select id="autoPolicy"[^>]*>(.*?)</select>', shown, re.S)
    assert sel is not None and "gauge" not in sel.group(1)
    assert "8 futures" in shown, "the page must still work without either feature"


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_restored_gauge_scores_are_labelled_as_having_no_receipt() -> None:
    """Gauge scores persist; the receipt does not. A rank whose verdict was lost
    must not look like one that was measured."""
    dom = _render(Fixture(state_gauge_scores=True), _steps('__type("hi");'), budget=8000)
    assert _errs(dom) == []
    box = re.search(r'<div class="probeverdict ([a-z]+)" id="probeVerdict"[^>]*>(.*?)</div>', dom, re.S)
    assert box is not None, "a fan with gauge scores and no receipt said nothing"
    assert box.group(1) == "unknown"
    assert "no receipt" in box.group(2)
    assert "Re-probe before trusting them" in box.group(2)
    assert "unverified" in _body(dom), "the cards must carry the same warning"


# ══ PHASE B — length, and what changed ════════════════════════════════════
#
# The two items share one failure mode and the tests are built around it:
# both are ways of putting a NUMBER next to a reply, and the number is
# smaller than what it is standing next to.
#
#   · every future in both captured draws hit the horizon, so its word count
#     is a RATE and not a length it chose;
#   · on the captured worn turn the reply moved -17 words (-6.6%) while the
#     manner moved a great deal, and the EARLIER, unsteered pair moved +28
#     words (+12.5%) the other way.
#
# A UI that quoted either number bare would misreport the turn. These pin
# that it does not.

def _words(n: int, salt: str = "alpha") -> str:
    """A reply of exactly n words, distinctive per salt."""
    return " ".join(f"{salt}{i}" for i in range(n))


#: The captured shape, reproduced: an earlier pair that differs MORE than the
#: later one. Both are restored, so neither carries a wear record.
PHASEB_HISTORIES = {
    "loom": [
        {"role": "user", "content": "what is your genre?"},
        {"role": "assistant", "content": _words(252, "lw")},
        {"role": "user", "content": "say more, in that voice."},
        {"role": "assistant", "content": _words(239, "lx")},
    ],
    "base": [
        {"role": "user", "content": "what is your genre?"},
        {"role": "assistant", "content": _words(224, "bw")},
        {"role": "user", "content": "say more, in that voice."},
        {"role": "assistant", "content": _words(256, "bx")},
    ],
}
PHASEB_WORN = {"index": 3, "alpha": 0.5, "loom_id": "stub-loom-1",
               "per_site_norms_at_alpha1": [1.9, 2.0, 1.8, 2.1]}


def test_a_future_that_hit_the_horizon_is_not_quoted_as_a_length(ui_text: str) -> None:
    """★ Every future in both captured draws has n_tokens == horizon: the
    span was CUT OFF, not finished. "159 words" on a truncated generation is
    a rate, not a length the model chose; the page describes what is actually
    served."""
    body = _fn(ui_text, "spanEnd")
    assert '"cut"' in body and '"own"' in body and '"flat"' in body, (
        "how a span ended is three states plus unknown, never a boolean — a "
        "boolean reads 'not cut' on every restored session, which is where "
        "the horizon is not on the wire")
    strip = _fn(ui_text, "scoresStrip")
    assert '"rate"' in strip, "a cut span's number must be labelled a rate"
    assert '"length"' in strip, "and a span that stopped on its own, a length"
    # the page must never assume it finished when it cannot tell
    assert "how this span ended is not knowable here" in strip


def test_the_length_copy_carries_the_measurement_and_its_two_traps(ui_text: str) -> None:
    body = _prov(ui_text, "length")
    assert "+0.2571" in body and "+0.0848" in body, "cite both rankers"
    assert "12 of 13" in body or "12/13" in body
    assert "RATE" in body or "rate" in body
    assert "NOT VOICE" in body.upper()


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_length_is_legible_but_never_the_primary_readout() -> None:
    """The design constraint, made a test. Length must be visible enough that
    a length-only effect cannot masquerade as voice, and quiet enough that a
    real manner shift is not buried under a 6% number."""
    dom = _render(Fixture(), _steps('__type("a contemplated turn");',
                                    'document.getElementById("loomBtn").click();'))
    assert _errs(dom) == []
    body = _body(dom)
    assert re.search(r'class="sc len"', dom), "the length unit must render"
    # ...and must NOT be dressed as a primary readout
    assert not re.search(r'class="sc len primary"', dom)
    assert not re.search(r'class="sc primary len"', dom)
    # the stub's futures are all n_tokens == horizon == 96, so every one is cut
    assert "cut at 96 tok" in body, "a horizon-capped span must say so on its face"
    assert "rate" in body


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_what_changed_shows_every_aligned_pair_and_not_just_the_latest() -> None:
    """★ THE HONEST FRAME IS THE SERIES. The steered pair moved -17 w (-6.6%)
    and the earlier pair moved +28 w (+12.5%) the other way — two independent
    generations of the same prompt differ by MORE than the steered turn did.
    Showing only the latest delta invites reading it as the effect of the
    wear; showing the series puts it next to its own null."""
    dom = _render(Fixture(histories=PHASEB_HISTORIES, worn=PHASEB_WORN),
                  _steps('document.getElementById("tabBoth").click();'))
    assert _errs(dom) == []
    body = _body(dom)
    assert "WHAT CHANGED" in body
    assert "2 turns on both branches" in body
    # both deltas, both directions
    assert "−17 w" in body and "−6.6%" in body, "the steered pair"
    assert "+28 w" in body and "+12.5%" in body, "and the null it must be read against"


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_what_changed_invents_no_score_and_claims_no_wear_it_was_not_told() -> None:
    """Invariant 4, at the place Phase B was most likely to break it.

    Nothing on the wire supports 'more steered' or a change score, and GET
    /state sends histories as {role, content} with NO per-turn wear record —
    so a restored turn must read 'wear unknown' rather than being attributed
    to the code that happens to be worn NOW."""
    dom = _render(Fixture(histories=PHASEB_HISTORIES, worn=PHASEB_WORN),
                  _steps('document.getElementById("tabBoth").click();'))
    panel = _panel(dom, "changed")
    assert "wear unknown" in panel, (
        "a restored turn carries no wear record and must say so — the page "
        "must not attribute it to whatever is worn now")
    # scoped to the PANEL: a whole-document scan would catch the page's own
    # comments about why it does NOT do these things
    for invented in ("more steered", "less steered", "steered more", "change score",
                     "similarity", "manner", "voice score", "effect of"):
        assert invented not in panel.lower(), f"the panel invented {invented!r}"
    # what it MAY say, and does
    assert "word types" in panel, "the lexical diff is a set difference, stated as one"
    assert "neither measures voice" in panel


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_pairs_are_joined_on_the_user_turn_and_never_on_position() -> None:
    """`mirror to base` is optional, so the branches can hold different
    numbers of exchanges. A positional join would compare a reply to the
    answer to a DIFFERENT question — the one way this panel could lie
    without inventing a number. Unmatched turns are dropped."""
    h = json.loads(json.dumps(PHASEB_HISTORIES))
    # base answered an extra, earlier question the loom branch never saw
    h["base"] = ([{"role": "user", "content": "an unmirrored question"},
                  {"role": "assistant", "content": _words(400, "bz")}] + h["base"])
    dom = _render(Fixture(histories=h, worn=PHASEB_WORN),
                  _steps('document.getElementById("tabBoth").click();'))
    assert _errs(dom) == []
    body = _body(dom)
    assert "2 turns on both branches" in body, (
        "the unmirrored turn must be dropped, not paired with something else")
    # the deltas must be the SAME as without the extra turn — proof the join
    # did not slide by one
    assert "−17 w" in body and "+28 w" in body
    assert "+400" not in body and "400 w" not in body


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_the_page_stays_correct_when_nothing_is_worn_and_nothing_is_paired() -> None:
    """The scenario the fixture does NOT cover, asserted anyway: empty
    branches, nothing worn. Every Phase B affordance must degrade to saying
    it has nothing, rather than to a zero."""
    dom = _render(Fixture(), _steps('document.getElementById("tabBoth").click();'))
    assert _errs(dom) == []
    body = _body(dom)
    assert "nothing to compare yet" in body
    assert "Δ +0 w" not in body and "0.0%" not in body, (
        "no pairs means no delta — not a delta of zero")


def test_an_uncalibrated_band_does_not_pin_the_dose_knob_at_zero() -> None:
    """★ A 70B /info can send `dose_band: {tier: "none",
    alpha_max: null}`. `Number(null)` is 0 and 0 is finite, so the slider's max
    would become 0 and every wear — auto-wear included — would go out at alpha
    0.000.
    Run the page's own alphaMax() on each shape a band can arrive in."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("no node on this host")
    src = UI.read_text(encoding="utf-8")
    a = src.index("const DEFAULT_ALPHA_MAX")
    b = src.index("function doseZones(")
    prog = "let doseBand = null;\n" + src[a:b] + r"""
const out = {};
for (const [k, band] of Object.entries({
    none_null: { tier: "none", alpha_max: null },
    none_missing: { tier: "none" },
    zero: { tier: "none", alpha_max: 0 },
    measured: { tier: "measured", alpha_max: 1.422757 },
    no_info: null })){
  doseBand = band; out[k] = alphaMax();
}
console.log(JSON.stringify(out));
"""
    r = subprocess.run([node, "-e", prog], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert got["measured"] == pytest.approx(1.422757)
    for k in ("none_null", "none_missing", "zero", "no_info"):
        assert got[k] == 1.5, f"{k}: an absent ceiling must not become {got[k]}"


def test_the_big_dose_number_is_the_dose_applied_not_the_knob(ui_text: str) -> None:
    """★ After an auto-wear, the biggest number on screen is the dose worn,
    not the knob's 0.5. Under
    `predicted` the knob is only the base. The big readout must state the dose
    that IS applied (the worn receipt) or WILL be (knob × the selected future's
    predicted_dose_scale), say which, and quote the knob beside it."""
    g = _fn(ui_text, "renderGauge")
    assert "doseReadout()" in g and 'fmt(r.v, 3)' in g, "the big number is the readout, not state.alpha"
    assert '$("alphaOut").textContent = fmt(a, 3)' not in g
    assert "zoneFor(r.worst)" in g, "the zone is read at the applied dose's loudest site"
    r = _fn(ui_text, "doseReadout")
    assert "effectiveAlpha(w)" in r, "a worn dose comes from the wear's own receipt"
    assert "predicted_dose_scale" in r, "a preview multiplies by the future's own scale"
    for kind in ('"worn"', '"preview"', '"knob"'):
        assert kind in r
    assert 'id="doseKind"' in ui_text and 'id="doseBase"' in ui_text, "it says what the number is"


def test_a_new_draw_takes_the_previous_wear_off_first(ui_text: str) -> None:
    """★ A new draw is not taken under the previous future's wear: that
    imprint is already in the text the future generated. With
    the fresh-draw toggle on (the default), doLoom unwears BEFORE /loom; a
    policy that picks relative to the wear (needsWorn) draws with the hook
    detached instead; a failed unwear falls back to detach, never to drawing
    under the old wear."""
    assert re.search(r'<input type="checkbox" id="unwearOnLoom" checked>', ui_text), "on by default"
    body = _fn(ui_text, "doLoom")
    i_unwear = body.index('post("/unwear"')
    i_loom = body.index('post("/loom"')
    assert i_unwear < i_loom, "the unwear must precede the draw"
    assert 'unwearOnLoom' in body and ".checked" in body
    assert "pol.needsWorn" in body, "stay/swerve still see the wear they pick against"
    assert body.count("detachWear = true") >= 3, "needsWorn, a refused unwear and a failed one all detach"
    assert "if (detachWear) body.detach_wear = true;" in body


def test_the_knob_starts_mid_threshold_of_the_served_band(ui_text: str) -> None:
    """★ On the 70B, α=.5 measures as already damage, so no fixed start is
    safe across models. The knob's
    start is derived from the served band (middle of its threshold zone) unless
    the viewer has moved it; no band → the fixed default."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("no node on this host")
    src = UI.read_text(encoding="utf-8")
    a = src.index("function bandStartAlpha(")
    b = src.index("function doseTicks(")
    prog = "let doseBand = null;\n" + src[a:b] + r"""
const out = {};
doseBand = { zones: [{key:"subliminal",lo:0,hi:.125},{key:"threshold",lo:.125,hi:.5},{key:"overdriven",lo:.5,hi:1}] };
out.modelc = bandStartAlpha();
doseBand = { zones: [{key:"subliminal",lo:0,hi:.125},{key:"threshold",lo:.125,hi:.75}] };
out.eightb = bandStartAlpha();
doseBand = { tier:"none", alpha_max:null }; out.none = bandStartAlpha();
console.log(JSON.stringify(out));
"""
    r = subprocess.run([node, "-e", prog], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert got["modelc"] == pytest.approx(0.315)   # (.125+.5)/2 = .3125, snapped to the 0.005 step
    assert got["eightb"] == pytest.approx(0.44)
    assert got["none"] is None
    load = _fn(ui_text, "loadInfo")
    assert 'bandStartAlpha()' in load and 'dataset.touched' in load, "only an untouched knob moves"
