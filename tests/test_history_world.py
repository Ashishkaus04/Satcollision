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
    assert e["truth"]["miss_km"] > 2 * e["truth"]["miss_km_without_maneuver"]
    chk = e["avoidance"]["check"]
    assert chk["met"] and chk["pc_precise_after"] <= e["avoidance"]["target_pc"] < chk["pc_precise_before"]
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


def mine_view_paths(view):
    return [(p["satellite"], p["burn_time"], p["maneuvered"][-1]) for p in view["burn_paths"]]


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
        assert any(e["id"] == ENC and e["truth"]["miss_km"] > 2 * e["truth"]["miss_km_without_maneuver"]
                   for e in obs["upcoming"] + obs["passed"])
        # Each twin worked out its own burn alone; it is the one the sky applied.
        for op in ("Alpha", "Beta"):
            mine = next(e for e in group.operator_view(op)["encounters"] if e["id"] == ENC)
            theirs = next(e for e in operator_view(ref, op)["encounters"] if e["id"] == ENC)
            assert mine["my_burn"] == theirs["my_burn"] and mine["my_burn"]["dv_m_s"] > 0
            assert mine_view_paths(group.operator_view(op)) == mine_view_paths(operator_view(ref, op))
    finally:
        group.close()


# ---------------------------------------------------------------------------
# Optimised avoidance (avoidance.py)
# ---------------------------------------------------------------------------

def _run_to_both_encounters(hist_file, tmp_path, **over):
    w = _world(hist_file, tmp_path, **over)
    while w.elapsed_s < FAST_TCA_AFTER_S - w.cfg.lookahead_s + w.cfg.tick_sim_s:
        w.tick()
    return w, [e for e in w.encounters if e.get("plan")]


def test_cw_displacement_matches_the_burn_model():
    from satcollision.avoidance import cw_matrix, displacement_rtn
    n = 2 * math.pi / 5700.0
    dv = np.array([0.0001, 0.0002, -0.00005])
    m = {"burn_jd": 0.0, "tca_jd": 3000.0 / 86400.0, "n_rad_s": n, "dv_rtn_km_s": list(dv)}
    off = displacement_rtn(m, np.array([1500.0, 3000.0, 6000.0, 7000.0]) / 86400.0)
    assert np.allclose(off[0], cw_matrix(n, 1500.0) @ dv)
    assert np.allclose(off[1], cw_matrix(n, 3000.0) @ dv)
    assert np.allclose(off[2], 0.0) and np.allclose(off[3], 0.0)       # returned to its track
    # A finite (low-thrust) burn spreads the same dv over minutes: close to,
    # not the same as, the impulse.
    finite = displacement_rtn({**m, "accel_m_s2": 1e-4}, np.array([3000.0 / 86400.0]))[0]
    assert 0.0 < np.linalg.norm(finite - off[1]) < 0.2 * np.linalg.norm(off[1])


def test_optimiser_meets_the_target_with_less_fuel_than_the_fixed_rule(hist_file, tmp_path):
    w, plans = _run_to_both_encounters(hist_file, tmp_path / "opt")
    assert len(plans) == 2
    for e in plans:
        a = e["avoidance"]
        assert a["mode"] == "optimal" and a["check"]["met"] and a["check"]["pc_precise_after"] <= a["target_pc"]
        assert set(e["burns"]) == {"Alpha", "Beta"}
        for b in e["burns"].values():
            assert b["dv_m_s"] == pytest.approx(b["planned_dv_m_s"], rel=0.10)   # own orbit vs public plan
            assert not b["secondary_risk"]
    # Against the fixed 1 km rule, on the encounter where the fixed rule also
    # meets the target (its 1.9 km): the same outcome for less fuel.
    fast = "ENC-90002-90013-"
    optimal = next(e for e in plans if e["id"].startswith(fast))["avoidance"]["total_dv_m_s"]
    f, fixed_plans = _run_to_both_encounters(hist_file, tmp_path / "fixed",
                                             fleet={"avoidance": {"avoidance": "fixed"}})
    fixed_enc = next(e for e in fixed_plans if e["id"].startswith(fast))
    fixed = sum(abs(m["dv_km_s"]) * 1000.0 for m in f.maneuvers if m["encounter"] == fixed_enc["id"])
    assert fixed_enc["truth"]["miss_km"] >= next(e for e in plans if e["id"].startswith(fast))["truth"]["miss_km"] - 0.01
    assert optimal < fixed
    events = [ev["text"] for ev in w.recent_events if ev["kind"] == "maneuver"]
    assert events and all("optimal burns" in t and "precise covariance" in t for t in events)


def test_a_satellite_without_propulsion_does_not_burn(hist_file, tmp_path):
    w, plans = _run_to_both_encounters(hist_file, tmp_path,
                                       fleet={"avoidance": {"no_propulsion": {"Beta": ["BETA-1"]}}})
    e = next(e for e in plans if e["id"] == ENC)
    assert e["avoidance"]["capability"] == {"Alpha": 1.0, "Beta": 0.0}
    assert set(e["burns"]) == {"Alpha"}
    assert e["avoidance"]["check"]["met"]
    assert not any(m["norad_id"] == 90011 for m in w.maneuvers)


def test_when_neither_can_burn_the_world_says_so(hist_file, tmp_path):
    w, plans = _run_to_both_encounters(hist_file, tmp_path,
                                       fleet={"avoidance": {"no_propulsion": {"Alpha": ["ALPHA"], "Beta": ["BETA"]}}})
    assert plans and all(e["avoidance"].get("cannot_avoid") for e in plans)
    assert w.maneuvers == []
    assert any(ev["kind"] == "cannot_avoid" for ev in w.recent_events)


def test_secondary_check_finds_an_approach_the_burn_creates():
    from satcollision.avoidance import displacement_rtn, secondary_approaches
    from satcollision.tle_history import from_rtn
    n = 2 * math.pi / 5700.0
    jd0 = 2461351.5
    m = {"burn_jd": jd0, "tca_jd": jd0 + 1800.0 / 86400.0, "n_rad_s": n, "dv_rtn_km_s": [0.0, 0.001, 0.0]}
    t = np.arange(0.0, 3601.0, 30.0)
    r0 = np.stack([7000.0 + 0 * t, 7.5 * t, 0 * t], axis=1)          # a straight track is enough here
    v0 = np.tile([0.0, 7.5, 0.0], (len(t), 1))
    i = 40                                                          # 1200 s after the burn
    moved = r0[i] + from_rtn(displacement_rtn(m, np.array([jd0 + t[i] / 86400.0]))[0], r0[i], v0[i])
    # crosses the manoeuvred track there at 10 km/s, out of plane; the
    # unmanoeuvred track is 1.4 km away (the burn's displacement)
    other_r = np.tile(moved, (len(t), 1)) + np.outer(t - t[i], [0.0, 0.0, 10.0])
    other_v = np.tile([0.0, 0.0, 10.0], (len(t), 1))

    def own(jd):
        return r0, v0, np.ones(len(t), bool)

    def others(jd):
        return np.array([555]), other_r[None], other_v[None], np.ones((1, len(t)), bool)

    hits = secondary_approaches(own, others, m, (jd0, jd0 + 3600.0 / 86400.0), 30.0, 1.0, set(),
                                lambda *a: 1e-3, 1e-5)
    assert [h["other"] for h in hits] == [555] and hits[0]["miss_km"] < 0.01


def test_a_burn_time_that_makes_a_secondary_approach_is_skipped(hist_file, tmp_path, monkeypatch):
    import satcollision.avoidance as av
    real = av.secondary_approaches
    calls = []

    def first_one_bad(own, others, m, horizon, *a, **k):
        calls.append(m["burn_jd"])
        if len(calls) == 1:
            return [{"other": 90020, "miss_km": 0.1, "pc": 1e-3, "jd": horizon[0]}]
        return real(own, others, m, horizon, *a, **k)

    monkeypatch.setattr(av, "secondary_approaches", first_one_bad)
    w, plans = _run_to_both_encounters(hist_file, tmp_path)
    e = next(e for e in plans if e["id"] == ENC)
    retried = next(b for b in e["burns"].values() if b["candidates_tried"] == 2)
    assert retried["secondary_avoided"] == 1 and not retried["secondary_risk"]
    assert calls[0] != calls[1]                     # a different burn time was used
    assert e["avoidance"]["check"]["met"]


def test_avoidance_paths_are_drawn_for_the_owner_and_the_observer_only(hist_file, tmp_path):
    from satcollision.views import observer_view
    w = _world(hist_file, tmp_path)
    for _ in range(18):                              # past the decision, before TCA
        w.tick()
    obs = observer_view(w)["burn_paths"]
    assert {p["satellite"] for p in obs if p["encounter"] == ENC} == {"Alpha-90001", "Beta-90011"}
    p = next(p for p in obs if p["satellite"] == "Alpha-90001")
    assert len(p["original"]) == len(p["maneuvered"]) > 30 and len(p["original"][0]) == 3
    assert p["shift_at_tca_km"] > 0.5                # the burn moves it at TCA
    assert p["original"][0] == pytest.approx(p["maneuvered"][0], abs=1e-6)   # same before the burn
    assert p["miss_after_km"] > p["miss_before_km"]
    for op, other in (("Alpha", "Beta"), ("Beta", "Alpha")):
        mine = operator_view(w, op)["burn_paths"]
        assert mine and all(q["operator"] == op for q in mine)
        assert not any(q["satellite"].startswith(other) for q in mine)
        assert "miss_after_km" not in mine[0]        # the truth is the observer's only
