"""Contract tests for the loom server's durable sessions.

The GPU half of ``loom_serve`` lives behind deferred imports inside ``main()``;
the persistence layer is module-level and torch-free on purpose, so the thing
that must not lose a conversation is testable on a laptop with numpy + pytest.

What matters and is asserted here:

  * a session tag can NEVER become a path — ``../../etc/passwd`` and friends
    land on one file inside the sessions directory or raise, and the escape is
    injective so two tags never share a file;
  * round trip keeps what was promised: every turn of both branches, the worn
    lever's index/dose/loom_id/code/per-site norms, the loom log with picks and
    spread, and NOT the big arrays (a snapshot you can read with ``jq``);
  * a restored wear is inert until rehydrated — ``worn.active`` is false and no
    caller can attach ``None``;
  * a corrupt, truncated or future-schema file is SKIPPED loudly, never fatal:
    one bad snapshot must not take the instrument down at boot;
  * writes are atomic (temp + replace, no leftovers) and failures are swallowed
    with the previous snapshot left intact — the server stays up, the file goes
    stale, in that order of priority;
  * a snapshot this process never read is moved aside, not clobbered.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pleroma.serve import legacy as ls


# ── helpers ────────────────────────────────────────────────────────────────────


def make_session(n_sites: int = 3, hidden: int = 8) -> ls.LoomSession:
    """A session with everything a real one carries: both branches, a worn
    lever with actual vectors, a loom log, a live fan."""
    sess = ls.LoomSession()
    sess.histories["loom"] = [
        {"role": "user", "content": "hey — pick a thing to think about"},
        {"role": "assistant", "content": "the way a fan collapses as it commits"},
        {"role": "user", "content": "say more, but slower"},
        {"role": "assistant", "content": "slower, then: each turn spends spread"},
    ]
    sess.histories["base"] = [
        {"role": "user", "content": "hey — pick a thing to think about"},
        {"role": "assistant", "content": "an untouched control reply"},
    ]
    rng = np.random.default_rng(20260917)
    vectors = rng.normal(size=(n_sites, hidden)).astype(np.float64)
    sess.worn = {
        "index": 2, "alpha": 0.35, "loom_id": "abc0123456-001",
        "vectors": vectors, "code": [0.1, -0.2, 0.3, 0.0, 1.5, -1.25, 0.75, -0.5],
        "loom_dir": "/x/looms/abc0123456/loom_001",
    }
    sess.n_looms = 2
    sess.loom_id = "abc0123456-001"
    sess.last_loom_dir = "/x/looms/abc0123456/loom_001"
    sess.reroll_n = {"loom": 3}
    sess.last_futures = [
        {"index": 0, "n_tokens": 190, "text": "a future that says one thing",
         "harvested": True, "note": None, "scores": {"distinct": 0.41, "camp": 0},
         "code": [0.2] * 8},
        {"index": 2, "n_tokens": 171, "text": "a future that says another",
         "harvested": True, "note": None, "scores": {"distinct": 0.55, "camp": 1},
         "code": [0.3] * 8},
    ]
    sess.looms = [
        ls.LoomSnapshot(
            loom_id="abc0123456-000", k=6, horizon=192, created_at=1_758_000_000.0,
            loom_dir="/x/looms/abc0123456/loom_000",
            spread={"mean_pairwise_distance": 0.48, "n_scored": 6},
            futures=[ls.LoomSnapshot.future_meta(f) for f in sess.last_futures],
        ),
        ls.LoomSnapshot(
            loom_id="abc0123456-001", k=6, horizon=192, created_at=1_758_000_400.0,
            loom_dir="/x/looms/abc0123456/loom_001",
            spread={"mean_pairwise_distance": 0.31, "n_scored": 6},
            futures=[ls.LoomSnapshot.future_meta(f) for f in sess.last_futures],
            auto_selected={"policy": "distinct", "effective_policy": "distinct",
                           "index": 2, "alpha": 0.35},
            worn_index=2, worn_alpha=0.35,
        ),
    ]
    return sess


def store_at(tmp_path: Path) -> ls.SessionStore:
    store = ls.SessionStore(tmp_path)
    assert store.ensure_dir()
    return store


# ── tag sanitisation: the hard requirement ─────────────────────────────────────


@pytest.mark.parametrize("tag", [
    "../../etc/passwd",
    "../sibling",
    "..",
    ".",
    ".hidden",
    "a/b/c",
    "a\\b",
    "/absolute",
    "trailing/",
    "nul\x00byte",
    "new\nline",
    "sp ace",
    "sémantique",
    "emoji-🧵",
    "%2E%2E",
    "-",
])
def test_tag_never_escapes_the_sessions_directory(tmp_path: Path, tag: str) -> None:
    store = store_at(tmp_path)
    path = store.path_for(tag)
    name = ls.session_basename(tag)
    assert name.endswith(".json")
    assert "/" not in name and "\\" not in name and "\x00" not in name
    assert not name.startswith(".")
    assert name not in (".", "..", ".json", "..json")
    # The only thing that matters: the resolved file is a direct child of the
    # sessions directory, whatever the tag tried to be.
    assert path.resolve().parent == store.dir.resolve()
    assert path.name == name


@pytest.mark.parametrize("tag", ["", "   ", "\t\n"])
def test_empty_tags_are_refused_not_guessed_at(tag: str) -> None:
    with pytest.raises(ValueError):
        ls.session_basename(tag)


def test_non_string_tag_is_refused() -> None:
    with pytest.raises(ValueError):
        ls.session_basename(None)  # type: ignore[arg-type]


def test_escape_is_injective_so_two_tags_never_share_a_file() -> None:
    tags = ["a/b", "a%2Fb", "a_b", "a.b", ".a", "%2Ea", "A", "a",
            "../x", "..%2Fx", "x" * 300, "x" * 300 + "y",
            "sémantique", "s%C3%A9mantique"]
    names = [ls.session_basename(t) for t in tags]
    assert len(set(names)) == len(names), "collision: one tag would eat another"


def test_over_long_tags_stay_a_legal_file_name() -> None:
    name = ls.session_basename("z" * 5000)
    assert len(name.encode()) < 255
    assert name.endswith(".json")


def test_traversal_tag_writes_inside_the_directory(tmp_path: Path) -> None:
    """The end-to-end version of the requirement: a malicious tag saves, and
    the bytes land in the sessions dir with nothing created outside it."""
    store = store_at(tmp_path)
    sentinel = tmp_path / "passwd"
    assert store.save("../../passwd", ls.LoomSession()) is True
    assert not sentinel.exists()
    assert not (tmp_path.parent / "passwd").exists()
    written = list(store.dir.glob("*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text())["session"] == "../../passwd"


# ── round trip ─────────────────────────────────────────────────────────────────


def test_round_trip_keeps_turns_worn_and_loom_history(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    sess = make_session()
    assert store.save("vps-drift-0917", sess) is True

    fresh = ls.State()
    report = ls.SessionStore(tmp_path).restore_all(fresh)
    assert (report.restored, report.failed) == (1, 0)
    back = fresh.get("vps-drift-0917")

    assert back.histories == sess.histories
    assert back.n_looms == sess.n_looms
    assert back.loom_id == sess.loom_id
    assert back.last_loom_dir == sess.last_loom_dir
    assert back.reroll_n == sess.reroll_n
    assert back.last_futures == sess.last_futures  # text of the live fan included

    assert back.worn is not None
    assert back.worn["index"] == 2
    assert back.worn["alpha"] == pytest.approx(0.35)
    assert back.worn["loom_id"] == "abc0123456-001"
    assert back.worn["code"] == sess.worn["code"]
    assert back.worn["loom_dir"] == sess.worn["loom_dir"]
    expected_norms = [float(np.linalg.norm(r)) for r in sess.worn["vectors"]]
    assert back.worn["per_site_norms_at_alpha1"] == pytest.approx(
        expected_norms, abs=1e-5)

    assert [lm.loom_id for lm in back.looms] == [lm.loom_id for lm in sess.looms]
    assert back.looms[1].worn_index == 2
    assert back.looms[1].worn_alpha == pytest.approx(0.35)
    assert back.looms[1].auto_selected == sess.looms[1].auto_selected
    assert back.looms[0].spread == sess.looms[0].spread
    assert back.looms[1].futures == sess.looms[1].futures


def test_snapshot_holds_no_big_arrays_and_reads_with_jq(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    store.save("s", make_session(n_sites=3, hidden=4096))
    raw = store.path_for("s").read_text()
    blob = json.loads(raw)
    assert "vectors" not in json.dumps(blob["worn"])
    assert len(blob["worn"]["per_site_norms_at_alpha1"]) == 3  # norms, not rows
    assert "\n" in raw and raw.startswith("{")  # indented: human-readable
    assert len(raw) < 20_000, "a 4096-d lever leaked into the snapshot"
    # the shape an operator greps for
    assert blob["kind"] == "loom-session"
    assert blob["schema"] == ls.SESSION_SCHEMA_VERSION
    assert blob["n_turns"] == {"base": 1, "loom": 2}
    assert blob["saved_at_iso"].endswith("Z")


def test_restored_wear_is_inert_until_rehydrated(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    store.save("s", make_session())
    back = store.restore_one("s", ls.State())
    assert back.worn is not None
    assert back.worn["vectors"] is None
    assert back.worn_vectors() is None          # no caller can attach None
    public = back.worn_public()
    assert public is not None
    assert public["active"] is False            # honest about what is attached
    assert public["index"] == 2 and public["alpha"] == pytest.approx(0.35)
    assert public["code"] == make_session().worn["code"]
    assert len(public["per_site_norms_at_alpha1"]) == 3


def test_live_wear_reports_active(tmp_path: Path) -> None:
    sess = make_session()
    public = sess.worn_public()
    assert public is not None and public["active"] is True
    assert sess.worn_vectors() is not None


def test_unicode_and_awkward_tags_round_trip(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    for tag in ["sémantique-🧵", "../../etc/passwd", ".dotty", "a b/c"]:
        sess = ls.LoomSession()
        sess.histories["loom"] = [{"role": "user", "content": tag}]
        assert store.save(tag, sess) is True
    state = ls.State()
    report = ls.SessionStore(tmp_path).restore_all(state)
    assert report.restored == 4 and report.failed == 0
    for tag in ["sémantique-🧵", "../../etc/passwd", ".dotty", "a b/c"]:
        assert state.get(tag).histories["loom"][0]["content"] == tag


def test_empty_session_round_trips_as_a_reset(tmp_path: Path) -> None:
    """/reset writes the cleared session; it must come back cleared, not
    resurrected from an older snapshot."""
    store = store_at(tmp_path)
    store.save("s", make_session())
    store.save("s", ls.LoomSession())
    back = store.restore_one("s", ls.State())
    assert back.histories == {"base": [], "loom": []}
    assert back.worn is None and back.n_looms == 0 and back.looms == []


def test_loom_log_is_capped(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    sess = ls.LoomSession()
    sess.looms = [
        ls.LoomSnapshot(loom_id=f"x-{i:04d}", k=6, horizon=192, created_at=float(i))
        for i in range(ls.MAX_PERSISTED_LOOMS + 50)
    ]
    store.save("s", sess)
    back = store.restore_one("s", ls.State())
    assert len(back.looms) == ls.MAX_PERSISTED_LOOMS
    assert back.looms[-1].loom_id == f"x-{ls.MAX_PERSISTED_LOOMS + 49:04d}"


# ── damaged files: skipped loudly, never fatal ─────────────────────────────────


@pytest.mark.parametrize("body", [
    "",                                    # empty (crashed mid-write, pre-atomic)
    "{",                                   # truncated
    "not json at all",
    "[]",                                  # wrong top-level type
    '{"session": "", "histories": {}}',    # no usable tag
    '{"session": "s"}',                    # no histories key -> empty, still fine
    '{"session": "s", "histories": {"loom": [{"role": "wizard", "content": "x"}]}}',
    '{"session": "s", "histories": {"loom": [{"role": "user"}]}}',
    '{"session": "s", "histories": {}, "worn": {"alpha": 0.5}}',
    '{"session": "s", "histories": {}, "looms": [{"k": 6}]}',
    '{"session": "s", "histories": {}, "schema": 99}',
    '{"session": "s", "histories": {}, "reroll_n": {"loom": "many"}}',
])
def test_one_broken_snapshot_never_takes_the_server_down(
    tmp_path: Path, body: str
) -> None:
    store = store_at(tmp_path)
    store.save("healthy", make_session())
    (store.dir / "broken.json").write_text(body, encoding="utf-8")

    state = ls.State()
    report = ls.SessionStore(tmp_path).restore_all(state)
    assert "healthy" in state.sessions
    assert state.get("healthy").histories["loom"][0]["content"].startswith("hey")
    # '{"session": "s"}' is legal-but-sparse; everything else must be rejected.
    assert report.restored + report.failed == 2
    if report.failed:
        assert report.errors and "broken.json" in report.errors[0]


def test_restore_all_on_a_missing_directory_is_empty_not_fatal(tmp_path: Path) -> None:
    store = ls.SessionStore(tmp_path / "nope")
    report = store.restore_all(ls.State())
    assert (report.restored, report.failed) == (0, 0)


def test_explicit_restore_of_a_missing_session_raises(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    with pytest.raises(ValueError, match="no snapshot"):
        store.restore_one("never-existed", ls.State())


def test_explicit_restore_of_a_corrupt_session_raises(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    store.path_for("s").write_text("{ nope", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        store.restore_one("s", ls.State())


def test_newer_schema_is_refused_rather_than_misread() -> None:
    with pytest.raises(ValueError, match="newer than this server"):
        ls.SessionSnapshot.from_json(
            {"schema": ls.SESSION_SCHEMA_VERSION + 1, "session": "s",
             "histories": {}}
        )


# ── write discipline ───────────────────────────────────────────────────────────


def test_write_is_atomic_and_leaves_no_temp_files(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    for _ in range(3):
        assert store.save("s", make_session()) is True
    assert sorted(p.name for p in store.dir.iterdir()) == ["s.json"]
    json.loads((store.dir / "s.json").read_text())  # always complete JSON


def test_a_failed_write_keeps_the_previous_snapshot_and_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = store_at(tmp_path)
    store.save("s", make_session())
    good = store.path_for("s").read_text()

    def boom(path: Path, body: str) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ls.SessionStore, "_atomic_write", staticmethod(boom))
    sess = make_session()
    sess.histories["loom"].append({"role": "user", "content": "lost turn"})
    assert store.save("s", sess) is False          # reported, not raised
    assert store.write_failures == 1
    assert store.last_error is not None and "No space left" in store.last_error
    assert store.path_for("s").read_text() == good  # old snapshot intact


def test_a_temp_file_is_cleaned_up_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = store_at(tmp_path)

    def boom(src: Any, dst: Any) -> None:
        raise OSError("replace refused")

    monkeypatch.setattr(ls.os, "replace", boom)
    assert store.save("s", make_session()) is False
    assert list(store.dir.iterdir()) == []


def test_an_unserialisable_snapshot_is_reported_not_raised(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    sess = ls.LoomSession()
    sess.last_futures = [{"index": 0, "oops": object()}]
    assert store.save("s", sess) is False
    assert store.last_error is not None
    assert not store.path_for("s").exists()


def test_numpy_scalars_survive_the_encoder(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    sess = ls.LoomSession()
    sess.last_futures = [{"index": 0, "scores": {"distinct": np.float64(0.42),
                                                 "camp": np.int64(1)}}]
    assert store.save("s", sess) is True
    blob = json.loads(store.path_for("s").read_text())
    assert blob["last_futures"][0]["scores"] == {"distinct": 0.42, "camp": 1}


def test_persistence_can_be_switched_off(tmp_path: Path) -> None:
    store = ls.SessionStore(tmp_path, enabled=False)
    assert store.save("s", make_session()) is False
    assert not (tmp_path / ls.SESSIONS_DIRNAME).exists()


def test_a_snapshot_this_process_never_read_is_moved_aside(tmp_path: Path) -> None:
    """--no-restore, or a file skipped as corrupt: the old bytes must not be
    silently overwritten by an empty session."""
    first = store_at(tmp_path)
    first.save("s", make_session())
    old = first.path_for("s").read_text()

    second = store_at(tmp_path)               # fresh process, no restore
    assert second.save("s", ls.LoomSession()) is True
    orphans = list(second.dir.glob("s.json.orphaned-*"))
    assert len(orphans) == 1
    assert orphans[0].read_text() == old
    assert json.loads(second.path_for("s").read_text())["n_turns"]["loom"] == 0
    # An orphan is not a session: it must not come back at the next boot.
    state = ls.State()
    report = ls.SessionStore(tmp_path).restore_all(state)
    assert report.restored == 1 and report.failed == 0


def test_a_restored_session_is_written_back_without_an_orphan(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    store.save("s", make_session())
    state = ls.State()
    reloaded = ls.SessionStore(tmp_path)
    reloaded.restore_all(state)
    assert reloaded.save("s", state.get("s")) is True
    assert list(reloaded.dir.glob("*.orphaned-*")) == []


# ── GET /sessions listing ──────────────────────────────────────────────────────


def test_listing_reports_tags_turns_worn_and_mtime(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    store.save("with-wear", make_session())
    store.save("empty", ls.LoomSession())
    (store.dir / "junk.json").write_text("{ broken", encoding="utf-8")

    state = ls.State()
    state.put("ram-only", ls.LoomSession())
    rows = {r.session: r for r in store.list_sessions(state)}

    assert rows["with-wear"].ok and rows["with-wear"].n_turns == {"base": 1, "loom": 2}
    assert rows["with-wear"].n_looms == 2
    assert rows["with-wear"].worn is not None
    assert rows["with-wear"].worn["index"] == 2
    assert rows["with-wear"].modified_unix > 0
    assert rows["with-wear"].modified_iso.endswith("Z")
    assert rows["with-wear"].size_bytes > 0
    assert rows["with-wear"].in_memory is False

    assert rows["empty"].n_turns == {"base": 0, "loom": 0}
    assert rows["junk"].ok is False and rows["junk"].error
    assert rows["ram-only"].in_memory is True and rows["ram-only"].file == ""
    assert all(isinstance(r.to_json(), dict) for r in rows.values())


def test_listing_marks_what_is_in_memory(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    store.save("s", make_session())
    state = ls.State()
    store.restore_all(state)
    rows = {r.session: r for r in store.list_sessions(state)}
    assert rows["s"].in_memory is True


# ── rehydrating a restored wear from the loom dir ──────────────────────────────


def write_loom_dir(root: Path, indices: list[int], n_feat: int = 5,
                   n_bins: int = 4) -> Path:
    (root / "signatures").mkdir(parents=True, exist_ok=True)
    for j in indices:
        np.savez(root / "signatures" / f"gen_{j:03d}.npz",
                 features=np.full(n_feat, float(j) + 1.0))
    np.savez(root / "bins.npz",
             features=np.stack([np.full(n_bins, float(j) + 0.5) for j in indices]),
             generation_id=np.array(indices))
    return root


def fake_lever_of(sig: np.ndarray, brow: np.ndarray) -> tuple[np.ndarray, list[float],
                                                              list[float]]:
    lever = np.outer(np.array([1.0, 2.0, 3.0]), np.concatenate([sig, brow]))
    return lever, [1.0, 2.0, 3.0], [0.0] * 8


def test_rehydrate_recomputes_the_lever_from_the_loom_dir(tmp_path: Path) -> None:
    loom_dir = write_loom_dir(tmp_path / "loom_000", [0, 2, 5])
    worn = {"index": 2, "alpha": 0.35, "loom_dir": str(loom_dir)}
    lever = ls.rehydrate_worn_vectors(worn, fake_lever_of)
    expected, _, _ = fake_lever_of(np.full(5, 3.0), np.full(4, 2.5))
    assert lever.shape == (3, 9)
    assert np.allclose(lever, expected)


@pytest.mark.parametrize("mutate,match", [
    (lambda w, d: w.update(loom_dir=None), "no loom_dir"),
    (lambda w, d: w.update(loom_dir="/nonexistent/loom"), "no signature"),
    (lambda w, d: w.update(index=99), "no signature"),
])
def test_rehydrate_raises_instead_of_guessing(tmp_path: Path, mutate: Any,
                                              match: str) -> None:
    loom_dir = write_loom_dir(tmp_path / "loom_000", [0, 2])
    worn: dict[str, Any] = {"index": 2, "alpha": 0.35, "loom_dir": str(loom_dir)}
    mutate(worn, loom_dir)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        ls.rehydrate_worn_vectors(worn, fake_lever_of)


def test_rehydrate_refuses_a_lever_from_a_different_map(tmp_path: Path) -> None:
    """The silent-science failure this guard exists for: same shapes, other
    map, a lever nobody picked."""
    loom_dir = write_loom_dir(tmp_path / "loom_000", [0, 2])
    worn = {"index": 2, "alpha": 0.35, "loom_dir": str(loom_dir),
            "map_fingerprint": "old3bmap00000000"}
    with pytest.raises(ValueError, match="refusing to recompute"):
        ls.rehydrate_worn_vectors(worn, fake_lever_of, expect_map="w4map0000000000")
    same = ls.rehydrate_worn_vectors(worn, fake_lever_of,
                                     expect_map="old3bmap00000000")
    assert same.shape == (3, 9)
    # A snapshot from before fingerprints existed still rehydrates (loudly).
    worn.pop("map_fingerprint")
    assert ls.rehydrate_worn_vectors(worn, fake_lever_of,
                                     expect_map="w4map0000000000").shape == (3, 9)


def test_map_fingerprint_is_stable_and_discriminating() -> None:
    sites, norm_ref, mu_y = [8, 16, 24], np.array([1.0, 2.0, 3.0]), np.arange(9.0)
    meta = {"rank": 8, "discriminants_sha256": "deadbeef"}
    base = ls.map_fingerprint(sites, norm_ref, mu_y, meta)
    assert base == ls.map_fingerprint(sites, norm_ref, mu_y, dict(meta))
    assert base != ls.map_fingerprint([8, 16, 25], norm_ref, mu_y, meta)
    assert base != ls.map_fingerprint(sites, norm_ref * 1.01, mu_y, meta)
    assert base != ls.map_fingerprint(sites, norm_ref, mu_y + 1e-6, meta)
    assert base != ls.map_fingerprint(sites, norm_ref, mu_y,
                                      {**meta, "rank": 12})
    assert len(base) == 16


def test_map_fingerprint_round_trips_in_the_snapshot(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    sess = make_session()
    sess.worn["map_fingerprint"] = "abc123def4567890"
    store.save("s", sess)
    back = store.restore_one("s", ls.State())
    assert back.worn is not None
    assert back.worn["map_fingerprint"] == "abc123def4567890"


def test_rehydrate_refuses_a_dir_whose_bins_lack_the_future(tmp_path: Path) -> None:
    loom_dir = write_loom_dir(tmp_path / "loom_000", [0, 2])
    np.savez(loom_dir / "bins.npz", features=np.zeros((1, 4)),
             generation_id=np.array([0]))
    worn = {"index": 2, "alpha": 0.35, "loom_dir": str(loom_dir)}
    with pytest.raises(ValueError, match="no generation_id 2"):
        ls.rehydrate_worn_vectors(worn, fake_lever_of)


def test_rehydrate_refuses_non_finite_features(tmp_path: Path) -> None:
    loom_dir = write_loom_dir(tmp_path / "loom_000", [0, 2])
    np.savez(loom_dir / "signatures" / "gen_002.npz",
             features=np.array([1.0, np.nan, 2.0, 3.0, 4.0]))
    worn = {"index": 2, "alpha": 0.35, "loom_dir": str(loom_dir)}
    with pytest.raises(ValueError, match="non-finite"):
        ls.rehydrate_worn_vectors(worn, fake_lever_of)


def test_rehydrated_session_wears_again_end_to_end(tmp_path: Path) -> None:
    """The whole point, in one test: wear -> snapshot -> restart -> the lever
    that comes back is the lever that was worn."""
    loom_dir = write_loom_dir(tmp_path / "loom_001", [0, 2])
    original, _, _ = fake_lever_of(np.full(5, 3.0), np.full(4, 2.5))
    sess = ls.LoomSession()
    sess.worn = {"index": 2, "alpha": 0.35, "loom_id": "abc-001",
                 "vectors": original, "code": [0.5] * 8,
                 "loom_dir": str(loom_dir)}
    store = store_at(tmp_path)
    store.save("s", sess)

    back = ls.SessionStore(tmp_path).restore_one("s", ls.State())
    assert back.worn_vectors() is None
    back.worn["vectors"] = ls.rehydrate_worn_vectors(back.worn, fake_lever_of)
    assert back.worn_vectors() is not None
    assert np.allclose(back.worn_vectors(), original)
    public = back.worn_public()
    assert public is not None and public["active"] is True
    assert public["per_site_norms_at_alpha1"] == pytest.approx(
        [round(float(np.linalg.norm(r)), 3) for r in original])


# ── state bookkeeping ──────────────────────────────────────────────────────────


def test_state_put_and_names(tmp_path: Path) -> None:
    state = ls.State()
    state.put("b", ls.LoomSession())
    state.put("a", ls.LoomSession())
    assert state.snapshot_names() == ["a", "b"]
    assert state.get("a") is state.sessions["a"]


def test_store_reports_itself_for_info(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    store.save("s", make_session())
    blob = store.to_json()
    assert blob["enabled"] is True
    assert blob["writes"] == 1 and blob["write_failures"] == 0
    assert blob["dir"].endswith(os.path.join("", ls.SESSIONS_DIRNAME))
    assert blob["last_error"] is None
