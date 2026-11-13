"""Tests for the full-stack app's backend (app/backend) and the role views
(satcollision.views): above all, that each party gets only what it may see."""

import json

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("jwt")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from app.backend.main import Settings, create_app  # noqa: E402

FORBIDDEN_FOR_SERVER = ("lat", "lon", "alt_km", "speed_km_s", "miss_km", "pc", "position", "velocity")


# Every test below runs twice: with one process playing every part, and with
# each operator's twin and the federation as separate OS processes. The
# privacy rules must hold the same way in both.
@pytest.fixture(scope="module", params=["0", "1"], ids=["one-process", "separate-processes"])
def client(tmp_path_factory, request):
    runs = tmp_path_factory.mktemp("runs")
    settings = Settings(APP_RUNS_DIR=str(runs), APP_RUN_WORLDS="0", APP_RESET="1",
                        APP_WORLDS="adversarial", APP_SECRET="t" * 40, APP_PROCESSES=request.param)
    app = create_app(settings)
    with TestClient(app) as c:
        app.state.app_state.runners["adversarial"].step(260)  # ~8.7 simulated hours
        yield c


def _headers(client, user):
    r = client.post("/api/login", json={"username": user, "password": "demo"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_login_rejects_bad_password_and_missing_token(client):
    assert client.post("/api/login", json={"username": "alpha", "password": "nope"}).status_code == 401
    assert client.get("/api/worlds").status_code == 401
    assert client.get("/api/worlds", headers={"Authorization": "Bearer forged"}).status_code == 401


def test_operator_sees_own_fleet_in_full_and_others_coarse(client):
    v = client.get("/api/worlds/adversarial/view", headers=_headers(client, "alpha")).json()
    assert v["role"] == "operator" and v["operator"] == "Alpha"
    assert v["own_satellites"] and all(s["label"].startswith("Alpha-") for s in v["own_satellites"])
    assert all("speed_km_s" in s for s in v["own_satellites"])
    others = v["public_catalogue"]
    assert others and not any(o["operator"] == "Alpha" for o in others)
    for o in others:
        assert "speed_km_s" not in o
        assert abs(o["lat"] / 0.5 - round(o["lat"] / 0.5)) < 1e-6   # rounded to the public grid
        assert abs(o["lon"] / 0.5 - round(o["lon"] / 0.5)) < 1e-6


def test_operator_only_sees_encounters_it_is_party_to(client):
    v = client.get("/api/worlds/adversarial/view", headers=_headers(client, "beta")).json()
    assert v["encounters"]
    truth = client.get("/api/worlds/adversarial/view", headers=_headers(client, "observer")).json()
    all_ids = {e["id"] for e in truth["upcoming"] + truth["passed"]}
    mine = {e["id"] for e in v["encounters"]}
    assert mine <= all_ids
    for e in truth["upcoming"] + truth["passed"]:
        assert (e["id"] in mine) == ("Beta" in e["operators"])


def test_federation_view_contains_no_positions_or_exact_numbers(client):
    v = client.get("/api/worlds/adversarial/view", headers=_headers(client, "federation")).json()
    assert v["role"] == "federation" and v["positions"] == []
    assert v["signals"], "the adversarial world should have produced signed signals by now"
    leaked = set(_walk_keys({k: v[k] for k in ("signals", "encounters", "reports", "rounds")})) & set(FORBIDDEN_FOR_SERVER)
    assert not leaked, leaked


def test_alerts_are_cut_to_the_receivers_tier(client):
    v = client.get("/api/worlds/adversarial/view", headers=_headers(client, "gamma")).json()
    assert v["reputation"]["tier"] in ("suspended", "degraded")
    for a in v["alerts"]:
        if not a.get("safety_override"):
            assert "geometry_class" not in a and "counterparty_operator" not in a


def test_operator_cannot_reach_other_views_or_orbits(client):
    alpha = _headers(client, "alpha")
    assert client.get("/api/worlds/real_leo/view", headers=_headers(client, "starlink")).status_code == 404  # not running here
    truth = client.get("/api/worlds/adversarial/view", headers=_headers(client, "observer")).json()
    enc = (truth["passed"] + truth["upcoming"])[0]["id"]
    assert client.get(f"/api/worlds/adversarial/compare/{enc}", headers=alpha).status_code == 403
    v = client.get("/api/worlds/adversarial/view", headers=alpha).json()
    own_id = v["own_satellites"][0]["id"]
    other_id = v["public_catalogue"][0]["id"]
    assert client.get(f"/api/worlds/adversarial/satellites/{own_id}/track", headers=alpha).status_code == 200
    assert client.get(f"/api/worlds/adversarial/satellites/{other_id}/track", headers=alpha).status_code == 403


def test_starlink_login_cannot_open_a_world_it_is_not_in(tmp_path):
    settings = Settings(APP_RUNS_DIR=str(tmp_path), APP_RUN_WORLDS="0", APP_RESET="1",
                        APP_WORLDS="sandbox", APP_SECRET="t" * 40, APP_PROCESSES="0")
    with TestClient(create_app(settings)) as c:
        h = _headers(c, "starlink")
        assert c.get("/api/worlds", headers=h).json() == []
        assert c.get("/api/worlds/sandbox/view", headers=h).status_code == 403


def test_compare_is_built_from_the_role_views(client):
    obs = _headers(client, "observer")
    truth = client.get("/api/worlds/adversarial/view", headers=obs).json()
    fed_enc = next(e for e in truth["passed"] + truth["upcoming"] if e["handling"] == "federated")
    cmp = client.get(f"/api/worlds/adversarial/compare/{fed_enc['id']}", headers=obs).json()
    assert {s["party"] for s in cmp["operators"]} == set(fed_enc["operators"])
    assert cmp["server"]["positions"] == []
    assert not set(_walk_keys(cmp["server"]["signals"])) & set(FORBIDDEN_FOR_SERVER)
    for side in cmp["operators"]:
        assert side["encounter"]["miss_km"] == fed_enc["miss_km"]      # each owner computed it itself
        if side["partner_as_seen"]:
            assert "speed_km_s" not in side["partner_as_seen"]


def test_websocket_pushes_the_role_view(client):
    token = _headers(client, "federation")["Authorization"].split()[1]
    with client.websocket_connect(f"/api/worlds/adversarial/live?token={token}") as ws:
        msg = ws.receive_json()
    assert msg["role"] == "federation" and msg["positions"] == []


def test_users_file_disables_demo_accounts(tmp_path):
    users = tmp_path / "users.json"
    users.write_text(json.dumps({"ops": {"role": "observer", "password": "s3cret"}}))
    settings = Settings(APP_RUNS_DIR=str(tmp_path / "r"), APP_RUN_WORLDS="0", APP_WORLDS="sandbox",
                        APP_USERS_FILE=str(users), APP_SECRET="t" * 40, APP_PROCESSES="0")
    with TestClient(create_app(settings)) as c:
        assert c.get("/api/demo-users").json() == {"enabled": False, "users": []}
        assert c.post("/api/login", json={"username": "alpha", "password": "demo"}).status_code == 401
        assert c.post("/api/login", json={"username": "ops", "password": "s3cret"}).status_code == 200


@pytest.fixture(scope="module")
def two_worlds(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs2")
    settings = Settings(APP_RUNS_DIR=str(runs), APP_RUN_WORLDS="0", APP_RESET="1",
                        APP_WORLDS="sandbox,adversarial", APP_SECRET="t" * 40, APP_PROCESSES="0")
    app = create_app(settings)
    with TestClient(app) as c:
        for r in app.state.app_state.runners.values():
            r.step(40)
        yield c, app


def test_every_view_is_the_callers_own_in_the_world_it_asked_for(two_worlds):
    c, _ = two_worlds
    for user, role, op in (("alpha", "operator", "Alpha"), ("beta", "operator", "Beta"),
                           ("federation", "federation", None), ("observer", "observer", None)):
        h = _headers(c, user)
        token = h["Authorization"].split()[1]
        for world in ("sandbox", "adversarial"):
            v = c.get(f"/api/worlds/{world}/view", headers=h).json()
            assert v["role"] == role and v["world"]["name"] == world and v.get("operator") == op
            with c.websocket_connect(f"/api/worlds/{world}/live?token={token}") as ws:
                m = ws.receive_json()
            assert m["role"] == role and m["world"]["name"] == world and m.get("operator") == op


def test_an_operator_outside_a_world_gets_nothing_from_it(two_worlds):
    c, _ = two_worlds
    h = _headers(c, "kuiper")
    assert c.get("/api/worlds/sandbox/view", headers=h).status_code == 403
    assert c.get("/api/worlds", headers=h).json() == []
    token = h["Authorization"].split()[1]
    with pytest.raises(Exception):
        with c.websocket_connect(f"/api/worlds/sandbox/live?token={token}") as ws:
            ws.receive_json()


def test_server_refuses_to_send_someone_elses_view(two_worlds):
    """If a bug ever handed Alpha's request Beta's view, nothing leaves."""
    c, app = two_worlds
    runner = app.state.app_state.runners["sandbox"]
    real = runner.operator_view
    runner.operator_view = lambda op: real("Beta")
    runner._cache.clear()
    try:
        h = _headers(c, "alpha")
        assert c.get("/api/worlds/sandbox/view", headers=h).status_code == 500
        token = h["Authorization"].split()[1]
        with pytest.raises(Exception):
            with c.websocket_connect(f"/api/worlds/sandbox/live?token={token}") as ws:
                ws.receive_json()
    finally:
        runner.operator_view = real
        runner._cache.clear()


def test_server_never_sends_an_operator_another_operators_path():
    from fastapi import HTTPException
    from app.backend.main import User, _check_view_is_for
    alpha = User(username="alpha", role="operator", operator="Alpha")
    view = {"role": "operator", "operator": "Alpha", "world": {"name": "w"},
            "burn_paths": [{"operator": "Alpha", "satellite": "Alpha-1"}]}
    _check_view_is_for(view, alpha, "w")
    view["burn_paths"].append({"operator": "Beta", "satellite": "Beta-2"})
    with pytest.raises(HTTPException):
        _check_view_is_for(view, alpha, "w")
