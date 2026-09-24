"""The legacy loom page's lever control: absolute | contrast.

The page is `pleroma/serve/static/legacy_ui.html`. The server takes
`lever_kind` on /wear and /loom's `auto` (see
test_loom_lever_kind.py for why). The page must:

  * build the control from /info.lever_kinds + /info.lever_kind_default and
    hardcode neither; hide it against a server that never published them;
  * remember the viewer's pick per browser (localStorage, try/catch);
  * SEND `lever_kind` on WEAR and on auto-wear;
  * preview `LANDS AT` with `predicted_dose_scale_contrast` when contrast is
    selected under `predicted` dosing — the server divides each kind by the
    fan mean of THAT kind, so the absolute scale would be a wrong gauge;
  * say which lever is worn, in one word, with hovers inside the tip budget.

Static assertions pin the wiring; the rendered ones (headless Chrome against
`loom_ui_stub_server`, same harness as test_loom_ui_v1a.py) pin behaviour.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests.unit.serve.loom.loom_ui_stub_server import (
    LEVER_KIND_INFO_FIELDS,
    V1A_INFO,
    Fixture,
    serve,
)

UI = Path(__file__).resolve().parents[4] / "pleroma" / "serve" / "static" / "legacy_ui.html"


@pytest.fixture(scope="module")
def ui_text() -> str:
    return UI.read_text(encoding="utf-8")


def _fn(src: str, name: str) -> str:
    m = re.search(r"function " + re.escape(name) + r"\(\)\{(.*?)\n\}", src, re.S)
    assert m is not None, f"no function {name}"
    return m.group(1)


# ── static ──────────────────────────────────────────────────────────────────

def test_the_control_sits_beside_the_dose_policy(ui_text: str) -> None:
    i = ui_text.index('id="dosePolWrap"')
    j = ui_text.index('id="leverKindWrap"')
    k = ui_text.index('id="effAlphaOut"')
    assert i < j < k, "the lever control belongs in the gauge actions row"
    assert '<select id="leverKind" aria-label="lever kind"></select>' in ui_text
    assert 'id="leverKindWrap" hidden' in ui_text, "hidden until /info offers it"


def test_the_list_and_default_come_from_info(ui_text: str) -> None:
    assert "state.info.lever_kinds" in _fn(ui_text, "leverKinds")
    assert "state.info.lever_kind_default" in _fn(ui_text, "serverLeverKind")
    build = _fn(ui_text, "buildLeverKindSelect")
    assert '"absolute"' not in build and '"contrast"' not in build, \
        "the control must not hardcode the kinds"


def test_the_pick_is_remembered_per_browser_behind_try_catch(ui_text: str) -> None:
    assert 'const LEVER_KEY = "loom.leverKind";' in ui_text
    assert "try { return localStorage.getItem(LEVER_KEY); } catch(_){ return null; }" in ui_text
    assert "try { localStorage.setItem(LEVER_KEY, String(key)); } catch(_){}" in ui_text
    eff = _fn(ui_text, "effectiveLeverKind")
    # a remembered value is re-validated against the server's list
    assert "list.indexOf(mine) >= 0" in eff


def test_wear_and_auto_wear_send_lever_kind(ui_text: str) -> None:
    assert "if (lk) body.lever_kind = lk;" in ui_text
    assert "if (lk) autoReq.lever_kind = lk;" in ui_text
    i = ui_text.index("if (lk) body.lever_kind = lk;")
    assert ui_text.index('post("/wear", body', i) - i < 200, "set right before POST /wear"


def test_the_preview_switches_to_the_contrast_scale(ui_text: str) -> None:
    body = _fn(ui_text, "doseReadout")
    assert 'effectiveLeverKind() === "contrast"' in body
    assert "f.scores.predicted_dose_scale_contrast" in body
    assert "f.scores.predicted_dose_scale" in body


def test_the_worn_banner_names_the_lever(ui_text: str) -> None:
    body = _fn(ui_text, "renderWorn")
    assert 'id:"wornLeverKind"' in body
    assert "text:w.leverKind" in body
    assert "w.lever_kind" in _fn_args(ui_text, "adoptWorn")


def _fn_args(src: str, name: str) -> str:
    m = re.search(r"function " + re.escape(name) + r"\([^)]*\)\{(.*?)\n\}", src, re.S)
    assert m is not None, f"no function {name}"
    return m.group(1)


def test_the_tier3_entry_carries_the_measurement(ui_text: str) -> None:
    body = _fn(ui_text, "registerLeverKindProv")
    for fact in ("0.55", "0.87", "0.45–0.85", "−0.1", "UNVALIDATED"):
        assert fact in body, fact


def test_a_lever_refusal_is_surfaced_not_retried(ui_text: str) -> None:
    assert "/lever_kind/i.test(String(e.message" in ui_text
    assert "lever refused" in ui_text


# ── rendered ────────────────────────────────────────────────────────────────

CHROME = (shutil.which("google-chrome") or shutil.which("google-chrome-stable")
          or shutil.which("chromium") or shutil.which("chromium-browser"))
needs_chrome = pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")


def _steps(*steps: str, settle: int = 1000) -> str:
    """test_loom_ui_v1a._steps, plus a readout of the remembered lever kind.
    Tests that must seed localStorage before /info lands (the control is built
    from it) splice a line in
    at the top of the IIFE (see test_a_remembered_pick_survives_a_reload)."""
    blocks = "\n".join(
        "  setTimeout(function(){ try { %s } catch(e){ window.__errs.push('step%d: ' + e); } }, %d);"
        % (s, i, 600 + settle * i)
        for i, s in enumerate(steps))
    return """
<script>
(function(){
  window.__errs = [];
  window.addEventListener("error", (e) => window.__errs.push(String(e.message)));
  window.addEventListener("unhandledrejection", (e) => window.__errs.push("rej: " + e.reason));
  window.__type = function(t){
    const c = document.getElementById("composer");
    c.value = t; c.dispatchEvent(new Event("input", { bubbles: true }));
  };
%s
  setTimeout(function(){
    const d = document.createElement("div");
    d.id = "__errbox";
    d.setAttribute("data-errs", JSON.stringify(window.__errs));
    d.setAttribute("data-store", String((function(){ try { return localStorage.getItem("loom.leverKind"); } catch(_){ return "ERR"; } })()));
    document.body.appendChild(d);
  }, %d);
})();
</script>
""" % (blocks, 600 + settle * (len(steps) + 1))


def _render(fx: Fixture, harness: str, budget: int = 9000
            ) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    srv, url = serve(fx, harness)
    tmp = Path(tempfile.mkdtemp(prefix="loomui-lever-"))
    try:
        proc = subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
             "--hide-scrollbars", "--window-size=1680,1100",
             f"--virtual-time-budget={budget}", f"--user-data-dir={tmp / 'p'}",
             "--dump-dom", url],
            capture_output=True, text=True, timeout=240)
        assert proc.returncode == 0, proc.stderr[-2000:]
        return proc.stdout, list(getattr(srv, "post_bodies", []))
    finally:
        srv.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)


def _errs(dom: str) -> list[str]:
    m = re.search(r'id="__errbox" data-errs="([^"]*)"', dom)
    assert m is not None, "the harness never ran — the page did not boot"
    return json.loads(m.group(1).replace("&quot;", '"').replace("&amp;", "&"))


def _stored(dom: str) -> str:
    m = re.search(r'id="__errbox"[^>]*data-store="([^"]*)"', dom)
    assert m is not None
    return m.group(1)


def _body(dom: str) -> str:
    out = re.sub(r"<script\b.*?</script>", " ", dom, flags=re.S | re.I)
    return re.sub(r"<style\b.*?</style>", " ", out, flags=re.S | re.I)


def _info(**over: Any) -> dict[str, Any]:
    return {**json.loads(json.dumps(V1A_INFO)), **LEVER_KIND_INFO_FIELDS, **over}


SET_CONTRAST = ('const s = document.getElementById("leverKind"); s.value = "contrast";'
                ' s.dispatchEvent(new Event("change", { bubbles: true }));')


@needs_chrome
def test_the_control_offers_the_servers_kinds_and_its_default() -> None:
    dom, _ = _render(Fixture(info=_info()), _steps('__type("hi");'), budget=6000)
    assert _errs(dom) == []
    assert 'id="leverKindWrap" hidden' not in dom
    sel = re.search(r'<select id="leverKind"[^>]*>(.*?)</select>', dom, re.S)
    assert sel is not None
    opts = re.findall(r'<option value="([a-z]+)"[^>]*>([^<]*)</option>', sel.group(1))
    assert [o[0] for o in opts] == ["absolute", "contrast"]
    assert [o[0] for o in opts if "server default" in o[1]] == ["absolute"]


@needs_chrome
def test_the_control_is_hidden_against_a_server_without_it() -> None:
    """The live 70B server predates the toggle: no control, no field sent."""
    steps = ('__type("so why do i abandon things at 70%?");',
             'document.getElementById("loomBtn").click();',
             'document.getElementById("wearBtn").click();')
    dom, posts = _render(Fixture(), _steps(*steps), budget=12000)
    assert _errs(dom) == []
    assert 'id="leverKindWrap" hidden' in dom
    wears = [b for p, b in posts if p == "/wear"]
    assert wears, "the harness never wore"
    assert all("lever_kind" not in b for b in wears)
    assert 'id="wornLeverKind"' not in _body(dom)


@needs_chrome
def test_wear_sends_the_default_kind_and_the_banner_names_it() -> None:
    steps = ('__type("so why do i abandon things at 70%?");',
             'document.getElementById("loomBtn").click();',
             'document.getElementById("wearBtn").click();')
    dom, posts = _render(Fixture(info=_info()), _steps(*steps), budget=12000)
    assert _errs(dom) == []
    wears = [b for p, b in posts if p == "/wear"]
    assert wears and wears[-1]["lever_kind"] == "absolute"
    m = re.search(r'<span class="leverkind" id="wornLeverKind"([^>]*)>([^<]*)</span>',
                  _body(dom))
    assert m is not None and m.group(2) == "absolute"
    title = re.search(r'title="([^"]*)"', m.group(1))
    assert title is not None and len(title.group(1)) <= 100


@needs_chrome
def test_choosing_contrast_is_sent_remembered_and_shown() -> None:
    steps = ('__type("so why do i abandon things at 70%?");',
             SET_CONTRAST,
             'document.getElementById("loomBtn").click();',
             'document.getElementById("wearBtn").click();')
    dom, posts = _render(Fixture(info=_info()), _steps(*steps), budget=14000)
    assert _errs(dom) == []
    wears = [b for p, b in posts if p == "/wear"]
    assert wears and wears[-1]["lever_kind"] == "contrast"
    assert _stored(dom) == "contrast", "the pick must be remembered in this browser"
    m = re.search(r'id="wornLeverKind"[^>]*>([^<]*)<', _body(dom))
    assert m is not None and m.group(1) == "contrast"
    assert "(contrast) at α" in _body(dom), "the wear line in the chat names it"


@needs_chrome
def test_auto_wear_sends_the_kind() -> None:
    steps = ('__type("so why do i abandon things at 70%?");',
             SET_CONTRAST,
             'document.getElementById("autoArm").click();',
             'document.getElementById("loomBtn").click();')
    dom, posts = _render(Fixture(info=_info()), _steps(*steps), budget=12000)
    assert _errs(dom) == []
    looms = [b for p, b in posts if p == "/loom"]
    assert looms and isinstance(looms[-1].get("auto"), dict)
    assert looms[-1]["auto"]["lever_kind"] == "contrast"


def _lands_at(dom: str) -> tuple[str, str, str]:
    body = _body(dom)
    out = re.search(r'id="alphaOut"[^>]*>([^<]*)<', body)
    kind = re.search(r'id="doseKind"[^>]*>([^<]*)<', body)
    base = re.search(r'id="doseBase"[^>]*>(.*?)</span>\s*(?:<|$)', body, re.S)
    assert out is not None and kind is not None
    return out.group(1), kind.group(1), re.sub(r"<[^>]+>", "", base.group(1)) if base else ""


@needs_chrome
@pytest.mark.parametrize("kind", ["absolute", "contrast"])
def test_lands_at_uses_the_selected_kinds_scale(kind: str) -> None:
    """★ Under `predicted` (the stub server's default) with a future selected
    and nothing worn, LANDS AT = knob × that future's scale OF THE SELECTED
    KIND. The stub's contrast scales (0.62 + 0.11·i) are far from the
    absolute ones, so the wrong key cannot pass."""
    steps = ['__type("so why do i abandon things at 70%?");',
             'document.getElementById("loomBtn").click();',
             # dump what the page is previewing, to compare against the fixture
             # (the page's `state` is inside an IIFE; the knob is on the slider,
             # and after a draw the page selects the FIRST harvested future —
             # index 0 on the stub, where every future harvests)
             'const d = document.createElement("div"); d.id = "__sel";'
             ' d.setAttribute("data-sel", "0");'
             ' d.setAttribute("data-alpha", String(document.getElementById("alpha").value));'
             ' document.body.appendChild(d);']
    if kind == "contrast":
        steps.insert(1, SET_CONTRAST)
    dom, _ = _render(Fixture(info=_info()), _steps(*steps), budget=12000)
    assert _errs(dom) == []
    sel = re.search(r'id="__sel" data-sel="([^"]*)" data-alpha="([^"]*)"', dom)
    assert sel is not None
    idx, alpha = int(sel.group(1)), float(sel.group(2))
    row = Fixture().futures()[idx]
    key = "predicted_dose_scale_contrast" if kind == "contrast" else "predicted_dose_scale"
    scale = float(row["scores"][key])
    out, face, _base = _lands_at(dom)
    assert face == "lands at"
    assert out == "≈" + f"{alpha * scale:.3f}", (out, alpha, scale, kind)


@needs_chrome
def test_a_remembered_pick_survives_a_reload() -> None:
    """Seed the browser the way a returning viewer's would hold it: the page
    must come up on `contrast` without a click, and send it."""
    pre = 'try { localStorage.setItem("loom.leverKind", "contrast"); } catch(_){}'
    steps = ('__type("so why do i abandon things at 70%?");',
             'document.getElementById("loomBtn").click();',
             'document.getElementById("wearBtn").click();')
    harness = _steps(*steps).replace("(function(){\n  window.__errs = [];",
                                     "(function(){\n  " + pre + "\n  window.__errs = [];", 1)
    dom, posts = _render(Fixture(info=_info()), harness, budget=12000)
    assert _errs(dom) == []
    wears = [b for p, b in posts if p == "/wear"]
    assert wears and wears[-1]["lever_kind"] == "contrast"


@needs_chrome
def test_a_stale_remembered_kind_falls_back_to_the_server_default() -> None:
    pre = 'try { localStorage.setItem("loom.leverKind", "differential"); } catch(_){}'
    steps = ('__type("so why do i abandon things at 70%?");',
             'document.getElementById("loomBtn").click();',
             'document.getElementById("wearBtn").click();')
    harness = _steps(*steps).replace("(function(){\n  window.__errs = [];",
                                     "(function(){\n  " + pre + "\n  window.__errs = [];", 1)
    dom, posts = _render(Fixture(info=_info()), harness, budget=12000)
    assert _errs(dom) == []
    wears = [b for p, b in posts if p == "/wear"]
    assert wears and wears[-1]["lever_kind"] == "absolute"


@needs_chrome
def test_every_new_hover_fits_the_tip_budget() -> None:
    """Every hover on the lever control, its options and the worn label is
    ≤ 100 chars (TIP_BUDGET) — read off the rendered DOM, after a contrast
    wear so every lever hover the page can produce is present."""
    steps = ('__type("so why do i abandon things at 70%?");',
             SET_CONTRAST,
             'document.getElementById("loomBtn").click();',
             'document.getElementById("wearBtn").click();')
    dom, _ = _render(Fixture(info=_info()), _steps(*steps), budget=14000)
    assert _errs(dom) == []
    body = _body(dom)
    titles = [html_unescape(t) for t in re.findall(r'title="([^"]*)"', body)
              if "lever" in t or "absolute" in t or "contrast" in t]
    assert len(titles) >= 3, titles     # wrap + two options, at least
    assert any(t.startswith("lever worn: contrast") for t in titles), titles
    assert [t for t in titles if len(t) > 100] == []


def html_unescape(t: str) -> str:
    import html
    return html.unescape(t)
