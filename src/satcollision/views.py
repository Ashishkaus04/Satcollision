"""
Role-based views of a running world: what each party is allowed to see.

The federation's whole claim is that collision avoidance works without
anyone handing over their orbits. A dashboard that shows one omniscient map
cannot demonstrate that, so the app built on this module serves three
different pictures of the same world, and the filtering happens here, on
the server, before anything is serialised:

* :func:`operator_view` -- one operator's digital twin. Its own satellites in
  full (position, velocity, altitude, every close approach with the exact
  miss distance and Pc). Everyone else's objects only as the coarse public
  catalogue shows them (positions rounded to about 50 km, no velocities).
  The cross-operator alerts it has received, cut down to what its
  reputation tier has earned (reputation.tailor_alert).
* :func:`federation_view` -- the central server. It holds no positions at all,
  by construction: only the signed abstracted signals and count reports
  that operators sent it (read straight from the ledger), the reputation it
  computes, and the aggregate. Its "map" is empty, and the view says so.
* :func:`observer_view` -- simulation ground truth, for the demo and the
  examiners. Clearly labelled as something no real party has.

:func:`compare_encounter` puts one encounter side by side from both
operators and the server, built from the same functions, so the comparison
cannot show anything the separate views would not.

Every view carries a ``withheld`` list naming what that party does *not*
get, so the frontend can show the gap instead of silently leaving it out.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional

import numpy as np
from sgp4.api import SatrecArray

from .deconfliction import negotiate_maneuver
from .federation_service import (
    FEDERATION_WITHHELD,
    encounter_rows,
    ledger_report_rows,
    ledger_signal_rows,
    reputation_block,
)
from .reputation import tailor_alert
from .signal import AbstractedSignal
from .world import World, _gmst_rad, _iso, _jd_fr

R_EARTH_KM = 6378.137
PUBLIC_GRID_DEG = 0.5          # public-catalogue positions are rounded to this
PUBLIC_ALT_KM = 10.0
MAX_OTHER_POINTS = 1500         # cap on coarse "everyone else" dots per view
MAX_OWN_POINTS = 2500           # cap on an operator's own dots (Starlink has 10,713)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _states(world: World, objs: List) -> List[Optional[dict]]:
    """Lat/lon/alt/speed of ``objs`` at the world's current instant."""
    if not objs:
        return []
    jd, fr = _jd_fr(world.sim_time)
    err, r, v = SatrecArray([o.satrec for o in objs]).sgp4(np.array([jd]), np.array([fr]))
    gmst = _gmst_rad(jd, fr)
    out: List[Optional[dict]] = []
    for e, pos, vel in zip(err[:, 0], r[:, 0, :], v[:, 0, :]):
        if e != 0:
            out.append(None)
            continue
        x, y, z = (float(c) for c in pos)
        rad = math.sqrt(x * x + y * y + z * z)
        out.append({
            "lat": math.degrees(math.asin(z / rad)),
            "lon": (math.degrees(math.atan2(y, x) - gmst) + 540.0) % 360.0 - 180.0,
            "alt_km": rad - R_EARTH_KM,
            "speed_km_s": float(np.linalg.norm(vel)),
        })
    return out


def _thin(objs: List, limit: int, keep: Iterable[int] = ()) -> List:
    """Evenly spaced subset of at most ``limit`` objects, always keeping ``keep``."""
    keep = set(keep)
    if len(objs) <= limit:
        return list(objs)
    stride = math.ceil(len(objs) / limit)
    chosen = objs[::stride]
    ids = {o.norad_id for o in chosen}
    chosen += [o for o in objs if o.norad_id in keep and o.norad_id not in ids]
    return chosen


def _round_to(value: float, grid: float) -> float:
    return round(round(value / grid) * grid, 3)


def _world_header(world: World) -> dict:
    return {
        "name": world.cfg.name,
        "title": world.cfg.title or world.cfg.name,
        "description": world.cfg.description,
        "sim_time": _iso(world.sim_time),
        "tick": world.tick_index,
        "speed": world.cfg.speed,
        "operators": world.operators,
        "source": "synthetic fleets" if world.cfg.fleet["type"] == "synthetic" else "real TLE catalogue",
        "pc_model": world.cfg.pc_model,
        "pc_threshold": world.cfg.pc_threshold,
        "lookahead_s": world.cfg.lookahead_s,
        "federation_period_s": world.cfg.federation_period_s,
    }


def _ledger_signals(world: World, limit: int = 400) -> List[dict]:
    """The signed signals the federation actually holds, newest first."""
    return ledger_signal_rows(world.ledger.entries, limit)


def _public_encounter(e: dict) -> dict:
    return {k: e[k] for k in ("id", "tca", "status") if k in e}


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

OPERATOR_WITHHELD = [
    "Other operators' precise positions and velocities (only public-catalogue quality, rounded to ~50 km)",
    "Other operators' orbital elements and covariances",
    "Close approaches that don't involve your satellites",
    "Other operators' reputation scores and reports",
    "The exact Pc and miss distance the other side computed (only its coarse severity tier)",
]


def operator_view(world: World, operator: str, feed: Optional[dict] = None) -> dict:
    """``feed`` is what the federation process told this operator (see
    FederationService.feed) when the twin runs as its own process; without
    it, alerts, reputation and reports are read from the in-process world."""
    if operator not in world.operators:
        raise KeyError(f"{operator} is not an operator in world {world.cfg.name}")

    own_objs = sorted((o for o in world._objects if o.operator == operator), key=lambda o: o.norad_id)
    other_objs = sorted((o for o in world._objects if o.operator != operator), key=lambda o: o.norad_id)
    involved_ids = set()
    mine = []
    for e in world.encounters:
        if operator in e["operators"]:
            mine.append(e)
            involved_ids.update(e["pair"])

    own_shown = _thin(own_objs, MAX_OWN_POINTS, keep=involved_ids)
    own_states = _states(world, own_shown)
    own = []
    for o, st in zip(own_shown, own_states):
        if st is None:
            continue
        own.append({
            "id": o.norad_id, "label": o.label,
            "lat": round(st["lat"], 3), "lon": round(st["lon"], 3),
            "alt_km": round(st["alt_km"], 2), "speed_km_s": round(st["speed_km_s"], 4),
            "in_encounter": o.norad_id in involved_ids,
            "engineered": o.is_actor,
        })

    others_shown = _thin(other_objs, MAX_OTHER_POINTS, keep=involved_ids)
    others = []
    for o, st in zip(others_shown, _states(world, others_shown)):
        if st is None:
            continue
        others.append({
            "id": o.norad_id, "label": o.label, "operator": o.operator,
            "lat": _round_to(st["lat"], PUBLIC_GRID_DEG), "lon": _round_to(st["lon"], PUBLIC_GRID_DEG),
            "alt_km": _round_to(st["alt_km"], PUBLIC_ALT_KM),
            "in_encounter": o.norad_id in involved_ids,
        })

    # My own twin's encounters, in full. The partner is named only by its
    # public catalogue label; its state stays with its owner.
    senders = (feed or {}).get("senders", {})
    encounters = []
    for e in sorted(mine, key=lambda e: (e["status"] != "upcoming", e["tca_s"] if e["status"] == "upcoming" else -e["tca_s"])):
        mine_idx = 0 if e["operators"][0] == operator else 1
        encounters.append({
            "id": e["id"], "status": e["status"], "tca": e["tca"],
            "time_to_tca_s": e["tca_s"] - world.elapsed_s,
            "my_satellite": e["labels"][mine_idx],
            "other_object": e["labels"][1 - mine_idx],
            "other_operator": e["operators"][1 - mine_idx],
            "miss_km": e["miss_km"], "pc": e["pc"], "geometry": e["geometry"],
            "rel_speed_km_s": e["rel_speed_km_s"], "handling": e.get("handling"),
            "i_signalled": e.get("i_signalled", operator in e.get("signals_sent", [])),
            "plan": _my_plan(e.get("plan") or _plan_from_feed(e, senders), operator),
            "engineered": e.get("engineered", False),
        })

    # Alerts: signals other operators sent about encounters I'm party to,
    # tailored to the tier my reputation has earned.
    if feed is not None:
        # Separate processes: the federation already cut these down to this
        # operator's tier before they left its process.
        alerts = feed["alerts"]
        reputation = feed["reputation"]
        my_reports = feed["my_reports"]
    else:
        tier = world.reputation.tier(operator)
        reputation = {"score": round(world.reputation.score(operator), 3), "tier": tier}
        my_ids = {e["id"] for e in mine}
        alerts = []
        for s in _ledger_signals(world):
            if s["encounter_id"] in my_ids and s["sender_operator"] != operator:
                sig = AbstractedSignal(
                    encounter_id=s["encounter_id"], sender_operator=s["sender_operator"],
                    local_object_label=s["local_object_label"], tca_offset_s=s["tca_offset_s"],
                    geometry_class=s["geometry_class"], severity_tier=s["severity_tier"])
                alert = tailor_alert(sig, tier)
                alert["received"] = s["received"]
                alerts.append(alert)
        my_reports = [{"received": r["received"], "counts": r["counts"]}
                      for r in ledger_report_rows(world.ledger.entries, operator, limit=10)]

    return {
        "role": "operator",
        "operator": operator,
        "world": _world_header(world),
        "own_satellites": own,
        "own_count": len(own_objs),
        "public_catalogue": others,
        "public_count": len(other_objs),
        "public_grid_deg": PUBLIC_GRID_DEG,
        "encounters": encounters[:60],
        "alerts": alerts[:40],
        "reputation": reputation,
        "my_reports": my_reports,
        "behaviour": world.cfg.behaviours.get(operator, "honest"),
        "process": {"pid": os.getpid(), "separate": feed is not None},
        "withheld": OPERATOR_WITHHELD,
    }


def _plan_from_feed(e: dict, senders: Dict[str, List[str]]) -> Optional[dict]:
    """A twin in its own process learns from the federation whether the
    partner signed too; if so it derives the joint plan itself, from the
    public encounter id, exactly as the partner does."""
    pair = e.get("pair_signals")
    if not pair or len(set(senders.get(e["id"], []))) < 2:
        return None
    plan = negotiate_maneuver(AbstractedSignal(**pair[0]), AbstractedSignal(**pair[1]))
    return {"primary": plan.primary_object, "primary_operator": plan.primary_operator,
            "secondary": plan.secondary_object, "secondary_operator": plan.secondary_operator}


def _my_plan(plan: Optional[dict], operator: str) -> Optional[dict]:
    if not plan:
        return None
    if plan["primary_operator"] == operator:
        return {"my_satellite": plan["primary"], "direction": "+", "partner_direction": "-"}
    if plan["secondary_operator"] == operator:
        return {"my_satellite": plan["secondary"], "direction": "-", "partner_direction": "+"}
    return None


# ---------------------------------------------------------------------------
# Federation server
# ---------------------------------------------------------------------------

def federation_view(world: World) -> dict:
    signals = _ledger_signals(world)
    return {
        "role": "federation",
        "world": _world_header(world),
        "positions": [],     # deliberately empty: the server never receives one
        "signals": signals[:120],
        "encounters": encounter_rows(signals)[:60],
        "reports": ledger_report_rows(world.ledger.entries),
        "rounds": world.rounds[-20:][::-1],
        "reputation": reputation_block(world.reputation, world.operators),
        "ledger": {"entries": len(world.ledger.entries), "tail_hash": world.ledger.tail_hash(),
                   "verified_on_load": world.ledger_verified},
        "process": {"pid": os.getpid(), "separate": False},
        "withheld": FEDERATION_WITHHELD,
    }


# ---------------------------------------------------------------------------
# Observer (ground truth)
# ---------------------------------------------------------------------------

def observer_view(world: World, federation: Optional[dict] = None) -> dict:
    """``federation``: the federation process's own view, when it runs
    separately; reputation, rounds and the ledger then come from there."""
    state = world.state()
    if federation is not None:
        state["reputation"] = federation["reputation"]
        state["rounds"] = [dict(r, errors=_errors_vs_honest(world, r)) for r in federation["rounds"]]
        state["ledger"] = federation["ledger"]
    return {
        "role": "observer",
        "world": _world_header(world),
        "note": "Simulation ground truth. No real party has this view: it exists so the "
                "operator and federation views can be checked against it.",
        "tracks": state["tracks"],
        "fleet": state["world"]["fleet"],
        "behaviours": state["world"]["behaviours"],
        "counters": state["counters"],
        "upcoming": state["upcoming"],
        "passed": state["passed"],
        "events": state["events"][:100],
        "reputation": state["reputation"],
        "rounds": state["rounds"][:10],
        "ledger": state["ledger"],
        "withheld": [],
    }


# ---------------------------------------------------------------------------
# One encounter, three perspectives
# ---------------------------------------------------------------------------

def _behaviour_in_round(world: World, operator: str, round_no: int) -> str:
    b = world.cfg.behaviours.get(operator, "honest")
    if b == "on_off":
        return "spoofer" if round_no >= world.cfg.defect_after_round else "honest"
    return b


def _errors_vs_honest(world: World, summary: dict) -> dict:
    """How far each aggregate landed from the honest operators' mean. Only
    the simulation knows who is honest, so this is added on the observer's
    side, never computed by the federation."""
    reports = summary.get("reports", {})
    honest = [np.array(c, dtype=float) for op, c in reports.items()
              if _behaviour_in_round(world, op, summary["round"]) == "honest"]
    if not honest or len(honest) == len(reports):
        return {}
    truth = np.mean(honest, axis=0)
    return {k: round(float(np.linalg.norm(np.array(v) - truth)), 3)
            for k, v in summary.get("aggregates", {}).items()}


def compare_encounter(world: World, encounter_id: str) -> dict:
    """The same encounter as each party holds it, assembled only from the
    role views above (so nothing appears here that a view would not show)."""
    truth = next((e for e in world.encounters if e["id"] == encounter_id), None)
    if truth is None:
        raise KeyError(encounter_id)
    views = {op: operator_view(world, op) for op in dict.fromkeys(truth["operators"]) if op in world.operators}
    return compare_from_views(world, truth, views, federation_view(world))


def compare_from_views(world: World, truth: dict, op_views: Dict[str, dict], fed: dict) -> dict:
    encounter_id = truth["id"]
    sides = []
    for op in dict.fromkeys(truth["operators"]):
        if op not in op_views:
            continue  # debris has no operator
        view = op_views[op]
        sides.append({
            "party": op,
            "encounter": next((e for e in view["encounters"] if e["id"] == encounter_id), None),
            "alerts": [a for a in view["alerts"] if a["encounter_id"] == encounter_id],
            "own_satellite": next((s for s in view["own_satellites"]
                                   if s["label"] in truth["labels"]), None),
            "partner_as_seen": next((s for s in view["public_catalogue"]
                                     if s["label"] in truth["labels"]), None),
            "reputation": view["reputation"],
            "withheld": view["withheld"],
        })
    server = {
        "party": "Federation server",
        "encounter": next((e for e in fed["encounters"] if e["encounter_id"] == encounter_id), None),
        "signals": [s for s in fed["signals"] if s["encounter_id"] == encounter_id],
        "positions": [],
        "withheld": fed["withheld"],
    }
    return {
        "encounter_id": encounter_id,
        "world": _world_header(world),
        "truth": {k: truth[k] for k in ("labels", "operators", "tca", "miss_km", "pc", "geometry",
                                         "rel_speed_km_s", "handling", "status")},
        "operators": sides,
        "server": server,
    }


# ---------------------------------------------------------------------------
# Orbits
# ---------------------------------------------------------------------------

def ground_track(world: World, norad_id: int, minutes: int = 100) -> dict:
    """Lat/lon/alt every minute for the next ``minutes`` (raises KeyError)."""
    obj = next((o for o in world._objects if o.norad_id == norad_id), None)
    if obj is None:
        raise KeyError(norad_id)
    jd0, fr0 = _jd_fr(world.sim_time)
    t = np.arange(0, minutes * 60 + 1, 60.0)
    fr = fr0 + t / 86400.0
    jd = jd0 + np.floor(fr)
    fr = fr - np.floor(fr)
    err, r, _ = SatrecArray([obj.satrec]).sgp4(jd, fr)
    pts = []
    for k in range(len(t)):
        if err[0, k]:
            continue
        x, y, z = r[0, k]
        rad = float(np.linalg.norm(r[0, k]))
        g = _gmst_rad(jd[k], fr[k])
        pts.append([round(math.degrees(math.asin(z / rad)), 3),
                    round((math.degrees(math.atan2(y, x) - g) + 540) % 360 - 180, 3),
                    round(rad - R_EARTH_KM, 2)])
    return {"id": norad_id, "label": obj.label, "operator": obj.operator, "points": pts}
