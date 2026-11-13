"""The TLE-history world: loader, truth vs prediction, measured error,
manoeuvres, the clock's end, and both app modes."""

import json
import math

import numpy as np
import pytest

from satcollision.history_world import apply_maneuvers, jd_of, maneuver_offset_rtn, plan_burn
from satcollision.tle_history import TLEHistory, detect_jumps, fit_tle_errors, parse_satnum, to_rtn
from satcollision.views import operator_view
from satcollision.world import World, WorldConfig
from tests.history_fixture import BASE, FAST_TCA_AFTER_S, N_SETS, TCA_AFTER_S, config, write_history

ENC = f"ENC-90001-90011-{int(TCA_AFTER_S)}"


@pytest.fixture(scope="module")
def hist_file(tmp_path_factory):
    return write_history(tmp_path_factory.mktemp("hist") / "history.tle")


def _world(hist_file, tmp_path, **over):
    cfg = WorldConfig.from_dict(config(hist_file, **over))
    return World.open(cfg, tmp_path / "run", tmp_path, reset=True)


# ---------------------------------------------------------------------------
# The history itself
# ---------------------------------------------------------------------------

def test_alpha5_numbers():
    assert parse_satnum("25544") == 25544
    assert parse_satnum("   11") == 11
    assert parse_satnum("A0000") == 100000
    assert parse_satnum("Z9999") == 339999


def test_loader_groups_sorts_and_drops_reissued_sets(hist_file):
    h = TLEHistory.load(hist_file)
    assert h.n_objects == 7
    assert h.n_sets == 7 * N_SETS                 # the re-issued last set is not counted twice
    assert h.names[h.row_of[90011]] == "BETA-1"
    for row in range(h.n_objects):
        ep = h.epochs[h.start[row]:h.start[row + 1]]
        assert np.all(np.diff(ep) > 0)
    k = int(h.start[0])
    s = h.satrec(k)
    assert h.epochs[k] == pytest.approx(s.jdsatepoch + s.jdsatepochF, abs=1e-8)


def test_asof_never_uses_a_set_from_the_future(hist_file):
    h = TLEHistory.load(hist_file)
    rows = np.arange(h.n_objects)
    t = jd_of(BASE) + 0.1
    asof = h.asof(rows, t)
    assert np.all(h.epochs[asof] <= t)
    nearest = h.nearest(rows, jd_of(BASE) + 0.24)    # just before the next 6-hourly set
    assert np.all(h.epochs[nearest] > jd_of(BASE) + 0.24)
    before = h.asof(rows, h.first_epoch - 1)
    assert np.all(before == -1)


def test_truth_is_continuous_across_a_set_switch(hist_file):
    h = TLEHistory.load(hist_file)
    row = h.row_of[90003]                       # ALPHA-3: its sets jump 30 km here
    k0 = int(h.start[row]) + 13
    e0, e1 = h.epochs[k0], h.epochs[k0 + 1]
    mid = 0.5 * (e0 + e1)
    t = mid + np.arange(-900, 901, 1.0) / 86400.0
    r, v, ok = h.truth_states(np.array([row]), t, blend_s=600.0)
    assert ok.all()
    # Truth - set A grows smoothly from 0 to (set B - set A) over the 600 s
    # blend: never a jump, and never more than its share per second.
    from sgp4.api import SatrecArray
    whole = np.floor(t - 0.5) + 0.5
    _, rr, _ = SatrecArray([h.satrec(k0), h.satrec(k0 + 1)]).sgp4(whole, t - whole)
    gap = np.linalg.norm(rr[1] - rr[0], axis=1)
    off = np.linalg.norm(r[0] - rr[0], axis=1)
    assert 25.0 < gap.max() < 35.0
    assert off[0] < 1e-9 and np.linalg.norm(r[0][-1] - rr[1][-1]) < 1e-9
    assert np.abs(np.diff(off)).max() <= gap.max() / 600.0 * 1.05
    # Away from the switch the truth is exactly the nearest set.
    far = np.array([e0 + 0.01])
    r_far, _, _ = h.truth_states(np.array([row]), far, blend_s=600.0)
    s = h.satrec(k0)
    _, r_set, _ = s.sgp4(math.floor(far[0] - 0.5) + 0.5, far[0] - (math.floor(far[0] - 0.5) + 0.5))
    assert np.allclose(r_far[0, 0], r_set, atol=1e-6)


def test_error_fit_measures_the_planted_noise_and_flags_the_jump(hist_file):
    h = TLEHistory.load(hist_file)
    model = fit_tle_errors(h, max_objects_per_group=50)
    rows = [r for r in model.table["other"] if r["n"]]
    # ~100 m along-track per set, two sets per difference: ~0.14 km.
    for r in rows:
        assert 0.09 < r["sigma_t_km"] < 0.2
        assert r["sigma_r_km"] < 0.02 and r["sigma_c_km"] < 0.02
    sig = model.sigma_rtn("other", 30.0)
    assert sig[1] > sig[0] and sig[1] > sig[2]
    jumps = detect_jumps(h, model, 10.0)
    assert [j["name"] for j in jumps] == ["ALPHA-3"]
    assert jumps[0]["axis"] == "T" and abs(jumps[0]["rtn_km"][1]) == pytest.approx(30.0, rel=0.05)


def test_burn_moves_the_satellite_by_the_planned_amount():
    n = 2 * math.pi / 5700.0
    lead = 5400.0
    dv = plan_burn(n, lead, 1.0, "+")
    assert dv > 0 and plan_burn(n, lead, 1.0, "-") == pytest.approx(-dv)
    m = {"burn_jd": 0.0, "tca_jd": lead / 86400.0, "dv_km_s": dv, "n_rad_s": n}
    off = maneuver_offset_rtn(m, np.array([0.0, lead / 86400.0, 2 * lead / 86400.0, 3 * lead / 86400.0]))
    assert np.linalg.norm(off[0]) == 0.0
    assert np.linalg.norm(off[1]) == pytest.approx(1.0, rel=1e-6)
    assert np.linalg.norm(off[2]) == pytest.approx(0.0, abs=1e-9)     # the return burn brings it back
    assert np.linalg.norm(off[3]) == 0.0
    r = np.array([[[7000.0, 0, 0]] * 4])
    v = np.array([[[0, 7.5, 0]] * 4])
    apply_maneuvers(r, v, np.array([5]), np.array([0.0, lead / 86400.0, 2 * lead / 86400.0, 0.5]),
                    [{**m, "norad_id": 5}])
    assert np.linalg.norm(r[0, 1] - [7000.0, 0, 0]) == pytest.approx(1.0, rel=1e-6)
    assert np.allclose(to_rtn(r[0, 1] - [7000.0, 0, 0], np.array([7000.0, 0, 0]), np.array([0, 7.5, 0])), off[1])


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------

def test_config_rules_for_history_worlds(hist_file):
    with pytest.raises(ValueError, match="foster"):
        WorldConfig.from_dict(config(hist_file, pc_model="circular"))
    with pytest.raises(ValueError, match="cycle_days"):
        WorldConfig.from_dict(config(hist_file, cycle_days=2))
    with pytest.raises(ValueError, match="at_end"):
        WorldConfig.from_dict(config(hist_file, fleet={"at_end": "wrap"}))
    with pytest.raises(FileNotFoundError, match="gp_history"):
        World(WorldConfig.from_dict(config("no/such/file.tle")), "unused")


def test_shipped_history_world_config_loads():
    cfg = WorldConfig.load("worlds/real_leo_history.json")
    assert cfg.fleet["type"] == "history" and cfg.fleet["at_end"] == "stop"


def test_engineered_crossing_is_assessed_signed_and_avoided(hist_file, tmp_path):
    w = _world(hist_file, tmp_path)
    assert w.start_time == BASE and w.operators == ["Alpha", "Beta"]
    for _ in range(20):
        w.tick()
    e = next(e for e in w.encounters if e["id"] == ENC)
    assert e["handling"] == "federated" and e["signals_sent"] == ["Alpha", "Beta"]
    for op in ("Alpha", "Beta"):
        side = e["sides"][op]
        assert side["pc"] >= w.cfg.pc_threshold and side["handling"] == "federated"
        assert side["other_tle_age_h"] > 0 and len(side["sigma_other_rtn_km"]) == 3
    assert e["plan"] is not None
    # The truth got both burns and the miss grew past a kilometre.
    assert {m["norad_id"] for m in w.maneuvers} == {90001, 90011}
    assert e["truth"]["miss_km_without_maneuver"] < 0.5 < 1.0 < e["truth"]["miss_km"]
    kinds = [ev["kind"] for ev in w.recent_events]
    assert "maneuver" in kinds and "Applied to the truth" in next(ev["text"] for ev in w.recent_events
                                                                  if ev["kind"] == "maneuver")


def test_fast_crossing_between_samples_is_found(hist_file, tmp_path):
    """BETA-3 crosses ALPHA-2 at 8.5 km/s 15 s after a screening
    sample, ~127 km apart at both neighbouring samples: invisible to a grid
    screen, found by the refined one, with its TCA to the second."""
    from satcollision.history_world import dt_of
    from satcollision.twin import PropagationResult, screen_conjunctions
    w = _world(hist_file, tmp_path)
    while w.elapsed_s < FAST_TCA_AFTER_S - w.cfg.lookahead_s + w.cfg.tick_sim_s:
        w.tick()
    e = next(e for e in w.encounters if e["pair"] == [90002, 90013])
    assert abs(e["tca_s"] - FAST_TCA_AFTER_S) < 1.0
    assert e["rel_speed_km_s"] > 8.0
    assert e["handling"] == "federated" and e["plan"] is not None
    # The old grid screen over the same samples does not see it.
    h = w.history.hist
    rows = np.array([h.row_of[90002], h.row_of[90013]])
    t_off = np.arange(-60.0, 61.0, 30.0)
    jd = jd_of(BASE) + (7200.0 + t_off) / 86400.0          # the world's own sample times
    r, v, ok, _ = h.predicted_states(rows, jd_of(BASE), jd)
    objs = [o for o in w._objects if o.norad_id in (90002, 90013)]
    objs.sort(key=lambda o: o.norad_id)
    prop = PropagationResult(t_offsets_s=t_off, object_ids=[90002, 90013], positions=r, velocities=v, valid=ok)
    assert screen_conjunctions(objs, prop) == []


def test_operator_view_shows_its_own_assessment_and_never_observer_only_records(hist_file, tmp_path):
    w = _world(hist_file, tmp_path)
    for _ in range(20):
        w.tick()
    w.encounters.append({"id": "TRUTH-90002-90012-99", "pair": [90002, 90012], "labels": ["Alpha-90002", "Beta-90012"],
                         "operators": ["Alpha", "Beta"], "tca_s": 9e9, "tca": "x", "status": "upcoming",
                         "miss_km": 0.3, "pc": 0.0, "observer_only": True, "handling": "unseen"})
    view = operator_view(w, "Alpha")
    ids = [e["id"] for e in view["encounters"]]
    assert ENC in ids and not any(i.startswith("TRUTH-") for i in ids)
    mine = next(e for e in view["encounters"] if e["id"] == ENC)
    assert mine["pc"] == next(e for e in w.encounters if e["id"] == ENC)["sides"]["Alpha"]["pc"]
    assert "public_pc" in mine and "other_tle_age_h" in mine
    assert not any("truth" in json.dumps(e) for e in view["encounters"])


def test_jumps_are_reported_once_per_hour(hist_file, tmp_path):
    w = _world(hist_file, tmp_path, tick_sim_s=1800, lookahead_s=1800, federation_period_s=3600)
    # ALPHA-3's jump is in the set at FIRST + 14 * 6 h = BASE + 2.5 days.
    while w.sim_time < BASE.replace(day=BASE.day + 2, hour=13):
        w.tick()
    jumps = [e for e in w.recent_events if e["kind"] == "tle_jump"]
    assert len(jumps) == 1 and "ALPHA-3" in jumps[0]["text"] and jumps[0]["n"] == 1
    assert w.counters["tle_jumps"] == 1


def test_clock_stops_at_the_end_of_the_file_and_says_so(hist_file, tmp_path):
    w = _world(hist_file, tmp_path, tick_sim_s=3600, lookahead_s=1800, federation_period_s=3600)
    for _ in range(200):
        w.tick()
    assert w.finished
    stopped_at = w.tick_index
    w.tick()
    assert w.tick_index == stopped_at
    assert [e["kind"] for e in w.recent_events].count("end") == 1
    state = w.state()
    assert state["clock"]["finished"] is True and state["clock"]["at_end"] == "stop"
    assert state["world"]["history"]["end_time"] == state["clock"]["end_time"]


def test_loop_mode_replays_the_file(hist_file, tmp_path):
    w = _world(hist_file, tmp_path, tick_sim_s=3600, lookahead_s=1800, federation_period_s=3600,
               fleet={"at_end": "loop"})
    for _ in range(80):
        w.tick()
    assert not w.finished and w.cycle >= 1
    assert any(e["kind"] == "cycle" and "replaying" in e["text"] for e in w.recent_events)


def test_restart_reproduces_the_same_ledger(hist_file, tmp_path):
    cfg = WorldConfig.from_dict(config(hist_file))
    a = World.open(cfg, tmp_path / "a", tmp_path, reset=True)
    for _ in range(24):
        a.tick()
    a.save()
    b = World.open(cfg, tmp_path / "b", tmp_path, reset=True)
    for _ in range(12):
        b.tick()
    b.save()
    b = World.open(cfg, tmp_path / "b", tmp_path)
    for _ in range(12):
        b.tick()
    b.save()
    assert a.ledger.tail_hash() == b.ledger.tail_hash()
    assert a.maneuvers == b.maneuvers
    assert [e["id"] for e in a.encounters] == [e["id"] for e in b.encounters]


def test_operator_twin_finds_the_same_encounter_with_only_its_own_side(hist_file, tmp_path):
    from satcollision.identity import OperatorIdentity
    cfg = WorldConfig.from_dict(config(hist_file))
    twin = World.open(cfg, tmp_path / "twin", tmp_path, reset=True, mode="operator", operator="Beta",
                      identity=OperatorIdentity.generate("Beta"))
    ref = World.open(cfg, tmp_path / "all", tmp_path, reset=True)
    for _ in range(20):
        twin.tick()
        ref.tick()
    e = next(e for e in twin.encounters if e["id"] == ENC)
    assert set(e["sides"]) == {"Beta"} and "truth" not in e
    assert e["sides"]["Beta"] == next(r for r in ref.encounters if r["id"] == ENC)["sides"]["Beta"]
    assert len(twin.outbox) >= 1 and twin.maneuvers == []
    # The twin draws other operators' satellites from published sets only.
    jd = jd_of(twin.sim_time)
    h = twin.history.hist
    for o in twin._objects:
        k = [k for k in h.sets_of(h.row_of[o.norad_id]) if h.satrec(k) is o.satrec]
        if o.operator != "Beta":
            assert k and h.epochs[k[0]] <= jd


def test_separate_processes_match_one_process(hist_file, tmp_path):
    from pathlib import Path
    from satcollision.processes import ProcessWorld
    repo = Path(__file__).resolve().parents[1]
    cfg = WorldConfig.from_dict(config(hist_file))
    group = ProcessWorld.start(cfg, tmp_path / "proc", repo, reset=True)
    try:
        ref = World.open(cfg, tmp_path / "ref", repo, reset=True)
        for _ in range(35):
            group.tick()
            ref.tick()
        for op in ("Alpha", "Beta"):
            mine = group.operator_view(op)
            theirs = operator_view(ref, op)
            assert [e["id"] for e in mine["encounters"]] == [e["id"] for e in theirs["encounters"]]
            assert [e["plan"] for e in mine["encounters"]] == [e["plan"] for e in theirs["encounters"]]
            assert [e["pc"] for e in mine["encounters"]] == [e["pc"] for e in theirs["encounters"]]
            assert mine["reputation"] == theirs["reputation"]
        fed = group.federation_view()
        assert fed["ledger"]["entries"] == len(ref.ledger.entries)
        assert group.sky.maneuvers == ref.maneuvers
        obs = group.observer_view()
        assert any(e["id"] == ENC and e["truth"]["miss_km"] > 1.0 for e in obs["upcoming"] + obs["passed"])
    finally:
        group.close()
