"""
What a world driven by a TLE history (fleet type ``"history"``) knows, and
how it judges an encounter. world.py calls into this; nothing here keeps
state that a checkpoint needs except the manoeuvre list, which World saves.

Three pictures of every satellite, each used by the party that would have it:

* **truth** (the sky, the observer): blended nearest-epoch ephemeris from the
  history (tle_history.TLEHistory.truth_states), plus any avoidance
  manoeuvre the world has applied;
* **owner's own ephemeris**: an operator knows its own satellites as well as
  the truth (it flies them), with a small ``precise_sigma_km`` per object.
  It does *not* include the world's simulated avoidance manoeuvres: those
  are a demonstration layer on the truth, and an operator twin in its own
  process never hears about them, so both app modes stay identical;
* **public prediction**: the newest element set with epoch at or before
  "now", propagated, with an RTN sigma measured from the file for that
  set's age at TCA (tle_history.ErrorModel).

Detection is public. Every party screens the same public catalogue (as-of
sets for every object), so every party finds the same encounters with the
same id, whichever process it runs in. Each operator then assesses its own
side: its satellite from its own ephemeris, the other object from the
public prediction, a Foster Pc with both covariances. Whether it shares is
decided on its own assessment, so the two operators of one encounter can
disagree, as they do in reality.

Avoidance manoeuvres: when both operators sign and a plan is agreed, the
world gives each satellite a small along-track burn at detection time
(primary prograde, secondary retrograde), sized so the satellite is
``maneuver_shift_km`` away from where it would have been at TCA, with the
displacement from the Clohessy-Wiltshire solution for a tangential burn:

    radial(t) = (2 dv / n) (1 - cos n t)
    along(t)  = (4 dv / n) sin n t - 3 dv t

After TCA the displacement is mirrored back to zero over the same time (a
return burn), so a manoeuvred satellite rejoins its real track.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pc import DEFAULT_HBR_KM, foster_pc, rtn_covariance
from .tle_history import (
    DEFAULT_BINS_H,
    SECONDS_PER_DAY,
    ErrorModel,
    TLEHistory,
    cached_analysis,
    from_rtn,
    group_of,
)
from .twin import compute_pc

JD_UNIX = 2440587.5

DEFAULTS = {
    "blend_s": 600.0,           # truth: blend across an element-set switch over this long
    "warmup_h": 24.0,           # start this long after the first epoch, so most objects have a set
    "at_end": "stop",           # "stop" or "loop" when the file runs out
    "precise_sigma_km": 0.354,  # an owner's own satellite, per object (two of them: ~0.5 km combined)
    "refine_window_s": 300.0,   # search this far either side of the public TCA
    "candidate_km": 5.0,        # public refined screen: assess approaches predicted closer than this
    "jump_sigma": 10.0,         # element set this many robust sigmas off the previous: a manoeuvre
    "jump_min_km": 1.0,         # ... and at least this far off
    "maneuver_shift_km": 1.0,   # avoidance burn sized to move each satellite this far by TCA
    "fit_objects_per_group": 1500,
    "error_bins_h": list(DEFAULT_BINS_H),
}

_LOADED: Dict[Tuple[str, float, int], TLEHistory] = {}


def jd_of(when: datetime) -> float:
    return when.timestamp() / SECONDS_PER_DAY + JD_UNIX


def dt_of(jd: float) -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=(jd - JD_UNIX) * SECONDS_PER_DAY)


def load_history(path: Path) -> TLEHistory:
    """One parsed copy per file per process (tests open many worlds)."""
    st = path.stat()
    key = (str(path.resolve()), st.st_mtime, st.st_size)
    if key not in _LOADED:
        _LOADED.clear()
        _LOADED[key] = TLEHistory.load(path)
    return _LOADED[key]


@dataclass
class Assessment:
    miss_km: float
    pc: float
    tca_jd: float
    sigma_other_rtn_km: Optional[Tuple[float, float, float]] = None
    other_tle_age_h: Optional[float] = None

    def as_dict(self) -> dict:
        d = {"miss_km": round(self.miss_km, 4), "pc": float(self.pc)}
        if self.sigma_other_rtn_km is not None:
            d["sigma_other_rtn_km"] = [round(s, 3) for s in self.sigma_other_rtn_km]
            d["other_tle_age_h"] = round(self.other_tle_age_h, 1)
        return d


class HistoryModel:
    """The history, its error model and jump list, and the per-world settings."""

    def __init__(self, fleet: dict, repo_root: Path):
        self.settings = {**DEFAULTS, **{k: v for k, v in fleet.items() if k in DEFAULTS}}
        path = Path(fleet["file"])
        if not path.is_absolute():
            path = repo_root / path
        if not path.exists():
            raise FileNotFoundError(
                f"needs {path}: a Space-Track gp_history file (3LE). It is not in git; "
                f"download it first (see README)")
        self.hist = load_history(path)
        self.model, self.jumps = cached_analysis(
            self.hist, repo_root / "cache", self.settings["error_bins_h"], 5.0,
            seed=0, max_objects_per_group=int(self.settings["fit_objects_per_group"]))
        self.jump_epochs = np.array([j["epoch_jd"] for j in self.jumps])
        self.blend_s = float(self.settings["blend_s"])
        self.precise = float(self.settings["precise_sigma_km"])

    # ---- clock ----

    def start_time(self) -> datetime:
        t = dt_of(self.hist.first_epoch + self.settings["warmup_h"] / 24.0)
        return t.replace(second=0, microsecond=0)

    def end_time(self) -> datetime:
        return dt_of(self.hist.last_epoch)

    def source_text(self) -> str:
        return (f"Space-Track TLE history: {self.hist.n_sets:,} element sets for {self.hist.n_objects:,} objects, "
                f"{dt_of(self.hist.first_epoch):%d %b %Y %H:%M} to {dt_of(self.hist.last_epoch):%d %b %Y %H:%M} UTC")

    # ---- states ----

    def truth(self, rows: np.ndarray, jd: np.ndarray, maneuvers: Optional[List[dict]] = None,
              norad_ids: Optional[np.ndarray] = None):
        r, v, ok = self.hist.truth_states(rows, jd, self.blend_s)
        if maneuvers:
            apply_maneuvers(r, v, np.asarray(norad_ids), np.asarray(jd, dtype=float), maneuvers)
        return r, v, ok

    def predicted(self, rows: np.ndarray, jd_now: float, jd: np.ndarray):
        return self.hist.predicted_states(rows, jd_now, jd)

    def sigma_public(self, row: int, set_index: int, tca_jd: float) -> Tuple[Tuple[float, float, float], float]:
        age_h = max(0.0, (tca_jd - self.hist.epochs[set_index]) * 24.0)
        return self.model.sigma_rtn(group_of(self.hist.names[row]), age_h), age_h

    def jumps_between(self, jd_lo: float, jd_hi: float, row_ids: set) -> List[dict]:
        lo = np.searchsorted(self.jump_epochs, jd_lo, side="left")
        hi = np.searchsorted(self.jump_epochs, jd_hi, side="left")
        out = []
        for j in self.jumps[lo:hi]:
            if j["norad_id"] not in row_ids:
                continue
            if j["n_sigma"] < self.settings["jump_sigma"]:
                continue
            if max(abs(x) for x in j["rtn_km"]) < self.settings["jump_min_km"]:
                continue
            out.append(j)
        return out


# ---------------------------------------------------------------------------
# Closest approach, Pc
# ---------------------------------------------------------------------------

def _min_distance(states_a, states_b, jd: np.ndarray):
    (ra, va, oka), (rb, vb, okb) = states_a, states_b
    d = np.linalg.norm(ra - rb, axis=-1)
    d = np.where(oka & okb, d, np.inf)
    i = int(np.argmin(d))
    return i, float(d[i])


def refine_tca(state_a, state_b, tca_jd: float, window_s: float):
    """Closest approach of two objects near ``tca_jd``. ``state_x(jd_array)``
    returns (r, v, ok) for one object, each with a leading time axis.
    Returns (miss_km, tca_jd, ra, va, rb, vb) or None if never valid."""
    coarse = tca_jd + np.arange(-window_s, window_s + 0.5, 2.0) / SECONDS_PER_DAY
    a, b = state_a(coarse), state_b(coarse)
    i, d = _min_distance(a, b, coarse)
    if not math.isfinite(d):
        return None
    fine = coarse[i] + np.arange(-2.0, 2.0001, 0.05) / SECONDS_PER_DAY
    a, b = state_a(fine), state_b(fine)
    j, d = _min_distance(a, b, fine)
    if not math.isfinite(d):
        return None
    return d, float(fine[j]), a[0][j], a[1][j], b[0][j], b[1][j]


def pc_with(ra, va, rb, vb, sig_a: Tuple[float, float, float], sig_b: Tuple[float, float, float],
            hbr_km: float = DEFAULT_HBR_KM) -> float:
    """Foster Pc with an RTN covariance per object; circular with the
    combined total sigma if the geometry is too slow for the encounter-plane
    method (co-orbital neighbours)."""
    try:
        cov_a = rtn_covariance(ra, va, *sig_a)
        cov_b = rtn_covariance(rb, vb, *sig_b)
        return foster_pc(np.asarray(ra) - np.asarray(rb), np.asarray(va) - np.asarray(vb),
                         cov_a, cov_b, hbr_km=hbr_km)
    except ValueError:
        total = math.sqrt(sum(s * s for s in sig_a) / 3.0 + sum(s * s for s in sig_b) / 3.0)
        return compute_pc(float(np.linalg.norm(np.asarray(ra) - np.asarray(rb))), total, hbr_km)


# ---------------------------------------------------------------------------
# Avoidance manoeuvres on the truth
# ---------------------------------------------------------------------------

def _cw(dv: float, n: float, tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    radial = (2.0 * dv / n) * (1.0 - np.cos(n * tau))
    along = (4.0 * dv / n) * np.sin(n * tau) - 3.0 * dv * tau
    return radial, along


def plan_burn(mean_motion_rad_s: float, lead_s: float, shift_km: float, direction: str) -> float:
    """Delta-v (km/s, signed: + prograde) that displaces the satellite by
    ``shift_km`` after ``lead_s`` seconds."""
    radial, along = _cw(1.0, mean_motion_rad_s, np.array([lead_s]))
    per_unit = float(math.hypot(radial[0], along[0]))
    dv = shift_km / per_unit if per_unit > 0 else 0.0
    return dv if direction == "+" else -dv


def maneuver_offset_rtn(m: dict, jd: np.ndarray) -> np.ndarray:
    """(len(jd), 3) RTN displacement of one manoeuvre at each time."""
    t = (jd - m["burn_jd"]) * SECONDS_PER_DAY
    lead = (m["tca_jd"] - m["burn_jd"]) * SECONDS_PER_DAY
    tau = np.where(t <= lead, t, 2.0 * lead - t)      # mirrored back after TCA
    active = (tau > 0)
    radial, along = _cw(m["dv_km_s"], m["n_rad_s"], np.where(active, tau, 0.0))
    out = np.zeros((len(jd), 3))
    out[:, 0] = np.where(active, radial, 0.0)
    out[:, 1] = np.where(active, along, 0.0)
    return out


def apply_maneuvers(r: np.ndarray, v: np.ndarray, norad_ids: np.ndarray, jd: np.ndarray,
                    maneuvers: List[dict]) -> None:
    """Add each manoeuvre's displacement to the rows it applies to, in place."""
    for m in maneuvers:
        rows = np.nonzero(norad_ids == m["norad_id"])[0]
        if rows.size == 0:
            continue
        off = maneuver_offset_rtn(m, jd)
        if not off.any():
            continue
        for i in rows:
            r[i] += from_rtn(off, r[i], v[i])


def maneuver_end_jd(m: dict) -> float:
    return m["tca_jd"] + (m["tca_jd"] - m["burn_jd"])
