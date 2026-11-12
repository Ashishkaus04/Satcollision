"""Tests for running a world as separate OS processes (satcollision.processes):
operator twins and the federation in their own processes, and what each
one can and cannot get at."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from satcollision.identity import OperatorIdentity, sign_signal
from satcollision.node_http import NodeError, call
from satcollision.processes import ProcessWorld
from satcollision.signal import AbstractedSignal
from satcollision.views import operator_view
from satcollision.wire import encode_signed_signal
from satcollision.world import World, WorldConfig

ROOT = Path(__file__).resolve().parents[1]
WORLD_FILE = ROOT / "worlds" / "adversarial.json"
TICKS = 260   # ~8.7 simulated hours: four federation rounds, Gamma suspended


@pytest.fixture(scope="module")
def cfg():
    return WorldConfig.load(WORLD_FILE)


@pytest.fixture(scope="module")
def reference(cfg, tmp_path_factory):
    """The same world in one process, for comparison."""
    w = World.open(cfg, tmp_path_factory.mktemp("all"), ROOT)
    for _ in range(TICKS):
        w.tick()
    return w


@pytest.fixture(scope="module")
def group(cfg, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("split")
    g = ProcessWorld.start(cfg, run_dir, ROOT, reset=True, world_file=WORLD_FILE)
    g.run_dir = run_dir
    try:
        for _ in range(TICKS):
            g.tick()
        g.save()
        yield g
    finally:
        g.close()


def _fed(group, method, path, body=None, token=None):
    return call(group.federation.url, method, path, body, token=group.token if token is None else token)


def test_every_party_is_its_own_process(group):
    pids = group.processes()
    assert len(set(pids.values())) == len(pids) == 2 + len(group.sky.operators)
    assert all(t.alive() for t in group.twins.values()) and group.federation.alive()
    for op in group.sky.operators:
        v = group.operator_view(op)
        assert v["process"] == {"pid": pids[op], "separate": True}


def test_federation_process_never_loads_orbital_code(group):
    health = _fed(group, "GET", "/health")
    assert health["heavy_modules_loaded"] == []
    # And importing it from scratch pulls none of them in either.
    code = ("import sys; import satcollision.federation_node; "
            "bad = [m for m in ('sgp4', 'scipy', 'satcollision.twin', 'satcollision.fleets', "
            "'satcollision.world', 'satcollision.views') if m in sys.modules]; print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, check=True)
    assert out.stdout.strip() == "[]"


def test_separate_processes_reach_the_same_outcome(group, reference):
    fed = group.federation_view()
    assert fed["reputation"]["scores"] == {op: round(reference.reputation.score(op), 3)
                                           for op in reference.operators}
    assert fed["ledger"]["entries"] == len(reference.ledger.entries)
    assert [r["reports"] for r in fed["rounds"]][::-1] == [r["reports"] for r in reference.rounds]
    for op in reference.operators:
        mine = group.operator_view(op)
        ref = operator_view(reference, op)
        assert [e["id"] for e in mine["encounters"]] == [e["id"] for e in ref["encounters"]]
        assert [bool(e["plan"]) for e in mine["encounters"]] == [bool(e["plan"]) for e in ref["encounters"]]
        assert [e["plan"] for e in mine["encounters"]] == [e["plan"] for e in ref["encounters"]]
        assert sorted(a["encounter_id"] for a in mine["alerts"]) == sorted(a["encounter_id"] for a in ref["alerts"])
        assert mine["reputation"] == ref["reputation"]


def test_observer_still_gets_errors_the_federation_cannot_compute(group, reference):
    fed_rounds = group.federation_view()["rounds"]
    assert all(r["errors"] == {} for r in fed_rounds)          # it doesn't know who is honest
    obs_rounds = group.observer_view()["rounds"]
    assert obs_rounds[0]["errors"] == reference.rounds[-1]["errors"]


def test_private_keys_exist_only_in_each_operators_folder(group):
    run_dir = Path(group.run_dir)
    fed_files = [p for p in (run_dir / "federation").rglob("*") if p.is_file()]
    registry = json.loads((run_dir / "federation" / "checkpoint.json").read_text())["registry"]
    for op in group.sky.operators:
        key_file = run_dir / "nodes" / op / "identity.key"
        raw = key_file.read_bytes()
        identity = OperatorIdentity.from_private_key_bytes(op, raw)
        assert registry[op] == identity.public_key_bytes().hex()   # only the public half went over
        for p in fed_files + [p for p in (run_dir / "sky").rglob("*") if p.is_file()]:
            data = p.read_bytes()
            assert raw not in data and raw.hex().encode() not in data, p


def test_federation_refuses_forgeries_and_strangers(group):
    # A message signed with any key other than the one Alpha pinned.
    impostor = OperatorIdentity.generate("Alpha")
    sig = AbstractedSignal(encounter_id="ENC-1-2-3", sender_operator="Alpha", local_object_label="Alpha-1",
                           tca_offset_s=600.0, geometry_class="head-on", severity_tier="critical")
    with pytest.raises(NodeError) as exc:
        _fed(group, "POST", "/message", encode_signed_signal(sign_signal(impostor, sig)))
    assert exc.value.status == 403
    # Re-registering Alpha with that key (to take over its identity).
    with pytest.raises(NodeError) as exc:
        _fed(group, "POST", "/register", {"operator": "Alpha", "url": "http://127.0.0.1:1",
                                          "public_key": impostor.public_key_bytes().hex()})
    assert exc.value.status == 403
    # Asking for Alpha's alerts without Alpha's key.
    ts = time.time()
    with pytest.raises(NodeError) as exc:
        _fed(group, "POST", "/feed", {"operator": "Alpha", "ts": ts,
                                      "signature": impostor.sign_bytes(f"feed|Alpha|{ts}".encode()).hex()})
    assert exc.value.status == 403
    # Anything without the internal token.
    with pytest.raises(NodeError) as exc:
        _fed(group, "GET", "/view", token="wrong")
    assert exc.value.status == 401


def test_a_twin_only_answers_for_its_own_satellites(group):
    alpha = group.twins["Alpha"]
    view = group.operator_view("Alpha")
    other = next(o for o in view["public_catalogue"] if o["operator"] != "Alpha")
    with pytest.raises(PermissionError):
        group.track(other["id"], 60, "Alpha")
    assert group.track(view["own_satellites"][0]["id"], 60, "Alpha")["points"]
    own_id = view["own_satellites"][0]["id"]
    mine = call(alpha.url, "POST", "/attest", {"ids": [f"ENC-{own_id}-999999-10", "ENC-999998-999999-10", "junk"]},
                token=group.token)["mine"]
    assert mine == [f"ENC-{own_id}-999999-10"]


def test_group_resumes_and_a_lagging_twin_catches_up(cfg):
    with tempfile.TemporaryDirectory() as d:
        g = ProcessWorld.start(cfg, d, ROOT, reset=True, world_file=WORLD_FILE)
        for _ in range(30):
            g.tick()
        g.save()
        g.close()
        # The sky moves on while every twin is down (as if they had crashed).
        sky = World.open(cfg, Path(d) / "sky", ROOT, mode="sky")
        for _ in range(5):
            sky.tick()
        sky.save()
        g = ProcessWorld.start(cfg, d, ROOT, world_file=WORLD_FILE)
        try:
            assert g.sky.tick_index == 35
            for op, twin in g.twins.items():
                assert call(twin.url, "GET", "/health", token=g.token)["tick"] == 35
            g.tick()
            fed = g.federation_view()
            assert fed["ledger"]["verified_on_load"] is True
        finally:
            g.close()
