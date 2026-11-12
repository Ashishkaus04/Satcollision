"""Tests for the always-running worlds (world.py, scripts/run_world.py)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from satcollision.signal import _ALLOWED_SIGNAL_FIELDS
from satcollision.world import World, WorldConfig

ROOT = Path(__file__).resolve().parent.parent
WORLDS = ROOT / "worlds"


def _cfg(name: str, **overrides) -> WorldConfig:
    data = json.loads((WORLDS / f"{name}.json").read_text(encoding="utf-8"))
    data.update(overrides)
    return WorldConfig.from_dict(data)


def _run(cfg: WorldConfig, run_dir: Path, ticks: int, reset: bool = True) -> World:
    world = World.open(cfg, run_dir, ROOT, reset=reset)
    for _ in range(ticks):
        world.tick()
    world.save()
    return world


def test_shipped_configs_load():
    names = {WorldConfig.load(p).name for p in WORLDS.glob("*.json")}
    assert {"sandbox", "adversarial", "real_leo"} <= names


@pytest.mark.parametrize("bad", [
    {"tick_sim_s": 100},             # not a multiple of the 30 s screening step
    {"pc_model": "magic"},
    {"behaviours": {"Gamma": "saboteur"}},
    {"no_such_setting": 1},
])
def test_config_validation(bad):
    data = json.loads((WORLDS / "sandbox.json").read_text(encoding="utf-8"))
    data.update(bad)
    with pytest.raises(ValueError):
        WorldConfig.from_dict(data)


def test_every_engineered_encounter_is_found(tmp_path):
    world = _run(_cfg("sandbox"), tmp_path / "w", 240)
    found = world.counters["encounters_shared"] + world.counters["private_watch"]
    due = [r for r in world.injections if r["tca_s"] <= world.elapsed_s + world.cfg.lookahead_s]
    retired = world.counters["injections"] - len(world.injections)
    assert world.counters["injections"] >= 8
    assert found == retired + len(due)
    assert world.counters["maneuvers_agreed"] == world.counters["encounters_shared"]  # all honest


def test_restart_reproduces_the_same_ledger(tmp_path):
    cfg = _cfg("adversarial")
    straight = _run(cfg, tmp_path / "a", 200)
    half = _run(cfg, tmp_path / "b", 100)
    assert half.tick_index == 100
    resumed = _run(cfg, tmp_path / "b", 100, reset=False)
    assert resumed.tick_index == 200
    assert resumed.ledger_verified
    assert resumed.ledger.tail_hash() == straight.ledger.tail_hash()
    assert resumed.counters == straight.counters
    lines = (tmp_path / "b" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(straight.ledger.entries)


def test_unsaved_tick_is_redone_not_duplicated(tmp_path):
    cfg = _cfg("adversarial")
    world = _run(cfg, tmp_path / "w", 60)
    # A tick that writes its ledger lines but dies before the checkpoint.
    for _ in range(40):
        world.tick()
    with open(tmp_path / "w" / "ledger.jsonl", "a", encoding="utf-8") as fh:
        fh.write("\n".join(world._pending_ledger_lines) + "\n")
    resumed = World.open(cfg, tmp_path / "w", ROOT)
    assert resumed.tick_index == 60 and resumed.ledger_verified
    for _ in range(40):
        resumed.tick()
    assert resumed.ledger.tail_hash() == world.ledger.tail_hash()


def test_editing_the_ledger_file_is_detected(tmp_path):
    cfg = _cfg("adversarial")
    world = _run(cfg, tmp_path / "w", 120)
    assert len(world.ledger.entries) > 3
    path = tmp_path / "w" / "ledger.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[1])
    if rec["kind"] == "signal":
        rec["payload"]["severity_tier"] = "critical" if rec["payload"]["severity_tier"] != "critical" else "threshold"
    else:
        rec["payload"]["counts"][0] += 5
    lines[1] = json.dumps(rec, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not World.open(cfg, tmp_path / "w", ROOT).ledger_verified


def test_nothing_but_abstracted_fields_cross_the_boundary(tmp_path):
    _run(_cfg("adversarial"), tmp_path / "w", 120)
    for line in (tmp_path / "w" / "ledger.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["kind"] == "signal":
            assert set(rec["payload"]) == _ALLOWED_SIGNAL_FIELDS
        else:
            assert set(rec["payload"]) == {"operator", "counts", "is_adversarial"}


def test_spoofer_loses_reputation_and_influence(tmp_path):
    world = _run(_cfg("adversarial"), tmp_path / "w", 360)
    scores = world.reputation.scores
    assert scores["Gamma"] < 0.45 < min(scores["Alpha"], scores["Beta"])
    assert world.counters["phantom_signals"] > 0 and world.counters["unilateral"] > 0
    last = world.rounds[-1]["errors"]
    assert last["reputation_weighted"] <= last["mean"]


def test_config_change_blocks_resume_but_speed_does_not(tmp_path):
    _run(_cfg("sandbox"), tmp_path / "w", 5)
    World.open(_cfg("sandbox", speed=600), tmp_path / "w", ROOT)  # pacing only: fine
    with pytest.raises(RuntimeError, match="--reset"):
        World.open(_cfg("sandbox", lookahead_s=3600), tmp_path / "w", ROOT)


def test_cycle_restarts_the_clock(tmp_path):
    world = _run(_cfg("sandbox", cycle_days=0.05), tmp_path / "w", 40)  # 72 min cycles, 80 min run
    assert world.cycle == 1 and world.counters["cycles"] == 1
    assert world.elapsed_s < world.total_sim_s
    assert world.sim_time < world.start_time + __import__("datetime").timedelta(hours=1)


def test_state_file_is_what_the_dashboard_reads(tmp_path):
    _run(_cfg("sandbox"), tmp_path / "w", 30)
    state = json.loads((tmp_path / "w" / "state.json").read_text(encoding="utf-8"))
    for key in ("world", "clock", "counters", "upcoming", "events", "reputation", "rounds", "ledger", "tracks"):
        assert key in state
    pts = state["tracks"]["points"]
    assert len(pts) == sum(state["world"]["fleet"].values())
    assert all(-90 <= p[0] <= 90 and -180 <= p[1] <= 180 for p in pts)


@pytest.mark.skipif(not (ROOT / "data" / "kuiper.tle").exists(), reason="real catalogue not downloaded")
def test_real_catalogue_world_runs(tmp_path):
    cfg = _cfg("real_leo", fleet={"type": "tle", "files": {"Kuiper": "data/kuiper.tle"}})
    world = _run(cfg, tmp_path / "w", 3)
    assert world.start_time.year >= 2025 and world.tick_index == 3
    assert world.operators == ["Kuiper"]


def test_runner_script_smoke(tmp_path):
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_world.py"), "--world", str(WORLDS / "sandbox.json"),
         "--runs", str(tmp_path), "--ticks", "5", "--fast", "--reset"],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert "stopped at tick 5" in out.stdout
    assert (tmp_path / "sandbox" / "state.json").exists()


def test_group_catalogue_splits_a_mixed_catalogue_by_name(tmp_path):
    from satcollision.fleets import load_tle_file
    from satcollision.world import NON_MEMBER, group_catalogue
    mixed = tmp_path / "mixed.tle"
    mixed.write_text((ROOT / "data" / "starlink.tle").read_text() + (ROOT / "data" / "oneweb.tle").read_text())
    objs = load_tle_file(str(mixed))
    both = group_catalogue(objs, {"Starlink": ["STARLINK"], "OneWeb": ["ONEWEB"]})
    assert {o.operator for o in both} == {"Starlink", "OneWeb"} and not any(o.is_debris for o in both)
    only = group_catalogue(objs, {"Starlink": ["STARLINK"]})
    others = [o for o in only if o.operator == NON_MEMBER]
    assert others and all(o.is_debris for o in others)            # in the sky, but in nobody's federation
    low = group_catalogue(objs, {"Starlink": ["STARLINK"], "OneWeb": ["ONEWEB"]}, max_altitude_km=800)
    n_oneweb = sum(o.operator == "OneWeb" for o in both)
    n_low = sum(o.operator == "OneWeb" for o in low)              # OneWeb flies at ~1,200 km;
    assert n_low < 0.1 * n_oneweb                                  # only a few are raising or deorbiting
    assert sum(o.operator == "Starlink" for o in low) == sum(o.operator == "Starlink" for o in both)


def test_catalogue_world_counts_non_members_as_nobodys(tmp_path):
    cfg = WorldConfig.from_dict({
        "name": "mixed", "fleet": {"type": "catalogue", "file": str(ROOT / "data" / "oneweb.tle"),
                                   "operators": {"OneWeb": ["ONEWEB-00"]}},
        "tick_sim_s": 150, "screening_step_s": 30, "lookahead_s": 600, "federation_period_s": 3600})
    w = World(cfg, tmp_path, ROOT)
    assert w.operators == ["OneWeb"]
    assert any(o.is_debris for o in w._objects)
    w.tick()
