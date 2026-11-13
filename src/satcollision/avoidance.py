"""
Optimised collision-avoidance burns for both satellites of an agreed plan.

The federation's plan (deconfliction.negotiate_maneuver) only says which two
satellites move and that their moves are opposite. This module decides *when*
each one burns, *which way* and *how much*: the least weighted fuel that brings
the Pc under a target, without either operator learning the other's precise
orbit.

The problem, in the encounter plane at TCA
------------------------------------------
Let m be the miss vector (A minus B) in the plane normal to the relative
velocity, C the 2x2 combined position covariance there, and w = C^(-1/2) m the
whitened miss. In the short-encounter model

    Pc ~= HBR^2 / (2 sqrt(det C)) * exp(-|w|^2 / 2)

so Pc <= target means |w'| >= k*, k*^2 = 2 ln(HBR^2 / (2 sqrt(det C) target)).
A burn dv (RTN) at time t_b moves a satellite at TCA by R^T Phi(n, TCA - t_b) dv
(Clohessy-Wiltshire), so its whitened in-plane effect is M dv, with
M = C^(-1/2) P^T R^T Phi (2x3). Both burns give w' = w + M_A dv_A - M_B dv_B.

For a required whitened push p the cheapest split of
sum_X |dv_X|^2 / cap_X is closed-form: with Q = sum_X cap_X M_X M_X^T it costs
p^T Q^-1 p, and side X delivers p_X = cap_X M_X M_X^T Q^-1 p. So the search is
over the burn time and over the point w + p on the circle of radius k*
(scanned in 0.5 degree steps). cap_X is 1 / weight (equal by default), and
0 for a satellite that cannot burn.

Why this needs no private data (option (a))
-------------------------------------------
M depends on the orbit's geometry (mean motion, RTN frame at TCA), which the
public element sets give to far better than the plan needs; C, m and the
target come from public data too. So both operators compute the *same* plan
from the public catalogue and agree on each side's push p_X without exchanging
anything. Each then delivers its own p_X from its own precise orbit, at the
burn time that suits it: the least-norm dv with M_X dv = p_X, over its own
grid of burn times, cheapest first, subject to its own checks.

Things it accounts for
----------------------
* the measured covariance (TLE age), plus a burn-execution error (a fraction
  of the planned shift, widening C along the push) and a small margin in k*
  for linearisation error;
* min/max delta-v per burn, earliest and latest burn time;
* low-thrust (electric) propulsion: a burn of acceleration a lasts |dv| / a
  and is modelled as small impulses spread over that time;
* secondary conjunctions: the manoeuvred trajectory is re-screened against the
  public catalogue from the burn until the return burn brings it back to its
  track (after that the orbit is unchanged). A burn time that creates an
  approach over ``secondary_pc`` is dropped for the next cheapest.

The displacement after TCA is mirrored back to zero over the same time (a
return burn), as the fixed rule in history_world did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pc import DEFAULT_HBR_KM, encounter_plane_basis, foster_pc
from .tle_history import SECONDS_PER_DAY, from_rtn

DEFAULTS = {
    "avoidance": "optimal",          # or "fixed": the earlier 1 km along-track rule
    "target_pc": 1.0e-6,
    "fairness": "equal",             # "equal", "capability" or "weights"
    "weights": {},                   # operator -> weight (higher = moves less)
    "no_propulsion": {},             # operator -> [name prefixes] that cannot burn
    "low_thrust_accel_m_s2": {"Starlink": 1.7e-4},   # electric propulsion, finite burns
    "min_dv_m_s": 0.001,
    "max_dv_m_s": 1.0,
    "min_lead_s": 0.0,               # earliest burn: this long after the decision
    "latest_before_tca_s": 1200.0,   # latest burn: this long before TCA
    "burn_step_s": 60.0,             # burn-time grid
    "exec_error_frac": 0.05,         # 1-sigma execution error, fraction of the planned shift
    "k_margin": 0.1,                 # extra whitened separation, for linearisation error
    "secondary_pc": 1.0e-5,
    "secondary_step_s": 30.0,
}


# ---------------------------------------------------------------------------
# Clohessy-Wiltshire and burns
# ---------------------------------------------------------------------------

def cw_matrix(n: float, tau: float) -> np.ndarray:
    """RTN displacement after ``tau`` seconds per unit RTN delta-v (km per km/s)."""
    if tau <= 0:
        return np.zeros((3, 3))
    s, c = math.sin(n * tau), math.cos(n * tau)
    return np.array([
        [s / n, 2.0 * (1.0 - c) / n, 0.0],
        [2.0 * (c - 1.0) / n, 4.0 * s / n - 3.0 * tau, 0.0],
        [0.0, 0.0, s / n],
    ])


def burn_segments(dv_km_s, accel_m_s2: Optional[float], max_segments: int = 12):
    """(offset_s, dv) impulses that approximate one burn. With a finite
    acceleration the burn lasts |dv| / a, centred on the burn time."""
    dv_km_s = np.asarray(dv_km_s, dtype=float)
    mag = float(np.linalg.norm(dv_km_s))
    if not accel_m_s2 or mag == 0.0:
        return [(0.0, dv_km_s)]
    duration = mag * 1000.0 / accel_m_s2
    k = max(1, min(max_segments, int(math.ceil(duration / 30.0))))
    offsets = (np.arange(k) + 0.5) / k * duration - duration / 2.0
    return [(float(o), dv_km_s / k) for o in offsets]


def displacement_rtn(m: dict, jd) -> np.ndarray:
    """(len(jd), 3) RTN displacement of one planned burn (a manoeuvre dict as
    World stores it) at each time, mirrored back to zero after TCA."""
    jd = np.asarray(jd, dtype=float)
    lead = (m["tca_jd"] - m["burn_jd"]) * SECONDS_PER_DAY
    t = (jd - m["burn_jd"]) * SECONDS_PER_DAY
    tau_eff = np.where(t <= lead, t, 2.0 * lead - t)
    out = np.zeros((len(jd), 3))
    for off, part in burn_segments(m["dv_rtn_km_s"], m.get("accel_m_s2")):
        for i, tau in enumerate(tau_eff - off):
            if tau > 0:
                out[i] += cw_matrix(m["n_rad_s"], float(tau)) @ part
    return out


def _sensitivity(n: float, tau_s: float, accel: Optional[float] = None, dv_guess_km_s: float = 0.0) -> np.ndarray:
    """RTN displacement at TCA per unit dv, including the finite-burn spread
    for a burn of roughly ``dv_guess`` (the spread depends on its size)."""
    if not accel or dv_guess_km_s <= 0:
        return cw_matrix(n, tau_s)
    segs = burn_segments(np.array([0.0, dv_guess_km_s, 0.0]), accel)
    return sum(cw_matrix(n, tau_s - off) for off, _ in segs) / len(segs)


def _rtn_rows(r, v) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    rh = r / np.linalg.norm(r)
    ch = np.cross(r, v)
    ch /= np.linalg.norm(ch)
    return np.stack([rh, np.cross(ch, rh), ch])


def _inv_sqrt_2x2(c: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(0.5 * (c + c.T))
    return v @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-18))) @ v.T


def required_k(c2: np.ndarray, target_pc: float, hbr_km: float = DEFAULT_HBR_KM) -> float:
    det = float(np.linalg.det(c2))
    arg = hbr_km ** 2 / (2.0 * math.sqrt(max(det, 1e-30)) * target_pc)
    return math.sqrt(2.0 * math.log(arg)) if arg > 1.0 else 0.0


def plane_gain(basis, c_inv_half, r_tca, v_tca, n: float, tau_s: float,
               accel: Optional[float] = None, dv_guess_km_s: float = 0.0) -> np.ndarray:
    """2x3: RTN delta-v at the burn -> whitened in-plane shift at TCA."""
    return c_inv_half @ basis.T @ _rtn_rows(r_tca, v_tca).T @ _sensitivity(n, tau_s, accel, dv_guess_km_s)


def capability(sides: Dict[str, bool], rule: str, weights: Dict[str, float]) -> Dict[str, float]:
    """1 / weight per operator, 0 if it cannot burn. "equal" and
    "capability" weigh everyone who can burn alike; "weights" reads them from
    the config (a higher weight moves less)."""
    out = {}
    for op, can in sides.items():
        if not can:
            out[op] = 0.0
        elif rule == "weights":
            out[op] = 1.0 / max(float(weights.get(op, 1.0)), 1e-9)
        else:
            out[op] = 1.0
    return out


def burn_grid(now_jd: float, tca_jd: float, settings: dict) -> List[float]:
    first = now_jd + settings["min_lead_s"] / SECONDS_PER_DAY
    last = tca_jd - settings["latest_before_tca_s"] / SECONDS_PER_DAY
    step = settings["burn_step_s"] / SECONDS_PER_DAY
    out = []
    t = first
    while t <= last + 1e-12:
        out.append(t)
        t += step
    return out


# ---------------------------------------------------------------------------
# The joint plan, from public data
# ---------------------------------------------------------------------------

@dataclass
class Agreement:
    """What both operators compute from public data and so agree on: the
    whitened in-plane push each satellite must add (A's sense, r_A - r_B)."""
    basis: np.ndarray
    c_inv_half: np.ndarray
    w: np.ndarray
    k_req: float
    push: Dict[str, np.ndarray]
    caps: Dict[str, float]
    pc_before: float
    burn_jd: Optional[float] = None
    planned_dv_m_s: Dict[str, float] = field(default_factory=dict)

    @property
    def k_now(self) -> float:
        return float(np.linalg.norm(self.w))

    @property
    def needed(self) -> bool:
        return self.k_now < self.k_req

    @property
    def possible(self) -> bool:
        return bool(self.push)


def agree(states: Dict[str, Tuple[np.ndarray, np.ndarray]], order: Tuple[str, str],
          mean_motion: Dict[str, float], covs: Dict[str, np.ndarray], caps: Dict[str, float],
          tca_jd: float, grid: Sequence[float], target_pc: float, exec_error_frac: float = 0.0,
          k_margin: float = 0.1, hbr_km: float = DEFAULT_HBR_KM, n_dirs: int = 720) -> Agreement:
    """The joint minimum of sum_X |dv_X|^2 / cap_X with Pc <= target, from
    public states at TCA. ``order`` = (operator of A, operator of B)."""
    op_a, op_b = order
    (ra, va), (rb, vb) = states[op_a], states[op_b]
    r_rel = np.asarray(ra) - np.asarray(rb)
    v_rel = np.asarray(va) - np.asarray(vb)
    basis = encounter_plane_basis(v_rel)
    c2 = basis.T @ (covs[op_a] + covs[op_b]) @ basis
    cih = _inv_sqrt_2x2(c2)
    c_half = np.linalg.inv(cih)
    w = cih @ (basis.T @ r_rel)
    pc = foster_pc(r_rel, v_rel, covs[op_a], covs[op_b], hbr_km=hbr_km)
    k_base = required_k(c2, target_pc, hbr_km)
    agreement = Agreement(basis=basis, c_inv_half=cih, w=w, k_req=k_base + k_margin, push={}, caps=caps,
                          pc_before=pc)
    if not agreement.needed or not any(caps.values()):
        return agreement

    def k_with_exec_error(p: np.ndarray) -> float:
        # The execution error widens C along the planned shift; find the
        # radius (in the agreed metric, along the result's direction) that
        # still meets the target in the widened metric.
        shift = c_half @ p
        c_tot = c2 + exec_error_frac ** 2 * np.outer(shift, shift)
        target = w + p
        direction = target / max(float(np.linalg.norm(target)), 1e-12)
        stretch = float(np.linalg.norm(_inv_sqrt_2x2(c_tot) @ c_half @ direction))
        return required_k(c_tot, target_pc, hbr_km) / max(stretch, 1e-12) + k_margin

    thetas = np.linspace(0.0, 2 * np.pi, n_dirs, endpoint=False)
    circle = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    best = None
    for tb in grid:
        tau = (tca_jd - tb) * SECONDS_PER_DAY
        ms = {op: (1.0 if op == op_a else -1.0) * plane_gain(basis, cih, *states[op], mean_motion[op], tau)
              for op in order}
        q = sum(caps[op] * ms[op] @ ms[op].T for op in order)
        if np.linalg.cond(q) > 1e12:
            continue
        qi = np.linalg.inv(q)
        k_req = agreement.k_req
        for _ in range(4):   # the required radius depends weakly on the push (execution error)
            p_all = k_req * circle - w
            i = int(np.argmin(np.einsum("ij,jk,ik->i", p_all, qi, p_all)))
            k_new = max(k_with_exec_error(p_all[i]), agreement.k_req)
            if abs(k_new - k_req) < 1e-3:
                break
            k_req = k_new
        p = k_req * circle[i] - w
        cost = float(p @ qi @ p)
        if best is None or cost < best[0]:
            lam = qi @ p
            push = {op: caps[op] * ms[op] @ ms[op].T @ lam for op in order}
            dv = {op: float(np.linalg.norm(caps[op] * ms[op].T @ lam)) * 1000.0 for op in order}
            best = (cost, tb, push, dv, k_req)
    if best is not None:
        _, agreement.burn_jd, agreement.push, agreement.planned_dv_m_s, agreement.k_req = best
    return agreement


# ---------------------------------------------------------------------------
# One side's burn, from its own orbit
# ---------------------------------------------------------------------------

@dataclass
class SideBurn:
    operator: str
    norad_id: int
    burn_jd: float
    tca_jd: float
    dv_rtn_km_s: np.ndarray
    n_rad_s: float
    accel_m_s2: Optional[float]
    planned_dv_m_s: float = 0.0       # what the public joint plan expected for this side
    shortfall: float = 0.0            # fraction of the push max_dv could not deliver
    tried: int = 1                    # burn times tried (negative: all made a secondary approach)
    secondary_avoided: List[dict] = field(default_factory=list)

    @property
    def dv_m_s(self) -> float:
        return float(np.linalg.norm(self.dv_rtn_km_s)) * 1000.0

    def as_maneuver(self, encounter: str, label: str) -> dict:
        return {"norad_id": self.norad_id, "encounter": encounter, "label": label,
                "burn_jd": self.burn_jd, "tca_jd": self.tca_jd, "n_rad_s": self.n_rad_s,
                "dv_rtn_km_s": [float(x) for x in self.dv_rtn_km_s], "accel_m_s2": self.accel_m_s2}


def candidate_burns(agreement: Agreement, op: str, sign: float, state_at_tca, n: float, tca_jd: float,
                    grid: Sequence[float], settings: dict, accel: Optional[float]):
    """(burn_jd, dv_rtn, shortfall) giving this side's agreed push from its
    own state, at each burn time on the grid, cheapest first. ``sign``: +1
    for object A, -1 for object B."""
    push = agreement.push.get(op)
    if push is None or not np.any(push):
        return []
    out = []
    for t in grid:
        tau = (tca_jd - t) * SECONDS_PER_DAY
        dv = np.zeros(3)
        guess = 0.0
        for _ in range(3):   # the finite-burn spread depends on the burn's own size
            m = sign * plane_gain(agreement.basis, agreement.c_inv_half, *state_at_tca, n, tau, accel, guess)
            mm = m @ m.T
            if np.linalg.cond(mm) > 1e12:
                dv = None
                break
            dv = m.T @ np.linalg.solve(mm, push)       # least-norm dv giving exactly this push
            guess = float(np.linalg.norm(dv))
        if dv is None:
            continue
        mag = float(np.linalg.norm(dv)) * 1000.0
        shortfall = 0.0
        if mag > settings["max_dv_m_s"]:
            shortfall = 1.0 - settings["max_dv_m_s"] / mag
            dv = dv * settings["max_dv_m_s"] / mag
        elif 0 < mag < settings["min_dv_m_s"]:
            dv = dv * settings["min_dv_m_s"] / mag
        out.append((t, dv, shortfall))
    out.sort(key=lambda x: (x[2], float(np.linalg.norm(x[1])), x[0]))
    return out


# ---------------------------------------------------------------------------
# Secondary conjunctions
# ---------------------------------------------------------------------------

def secondary_approaches(own_states: Callable, others: Callable, maneuver: dict, horizon_jd: Tuple[float, float],
                         step_s: float, gate_km: float, exclude: set, pc_fn: Callable, threshold: float) -> List[dict]:
    """Approaches of the manoeuvred satellite that the burn creates or worsens.

    ``own_states(jd) -> (r, v, ok)`` is the satellite's own unmanoeuvred
    track, ``others(jd) -> (ids, r, v, ok)`` the public catalogue near it.
    Between samples the closest approach is solved for linearly, as the
    refined screen does (screening.screen_conjunctions_refined: candidate
    radius max relative speed x step / 2 + gate). Approaches under
    ``gate_km`` are scored with ``pc_fn(other_id, r, v, r_other, v_other)``
    and returned when over ``threshold`` and worse than without the burn."""
    lo, hi = horizon_jd
    if hi <= lo:
        return []
    jd = lo + np.arange(0.0, (hi - lo) * SECONDS_PER_DAY + step_s, step_s) / SECONDS_PER_DAY
    r0, v0, ok0 = own_states(jd)
    off = displacement_rtn(maneuver, jd)
    r1 = r0 + np.stack([from_rtn(off[i], r0[i], v0[i]) for i in range(len(jd))])
    ids, ro, vo, oko = others(jd)
    if len(ids) == 0:
        return []
    radius = 16.0 * step_s / 2.0 + gate_km
    worst: Dict[Tuple[int, str], dict] = {}
    for track, which in ((r1, "with"), (r0, "without")):
        d = np.linalg.norm(ro - track[None], axis=2)              # (objects, times)
        for j, i in zip(*np.nonzero((d < radius) & oko & ok0[None])):
            oid = int(ids[j])
            if oid in exclude:
                continue
            dr = track[i] - ro[j, i]
            dv = v0[i] - vo[j, i]
            tau = float(np.clip(-np.dot(dr, dv) / max(np.dot(dv, dv), 1e-12), -step_s / 2, step_s / 2))
            miss = float(np.linalg.norm(dr + dv * tau))
            if miss >= gate_km:
                continue
            pc = pc_fn(oid, track[i] + v0[i] * tau, v0[i], ro[j, i] + vo[j, i] * tau, vo[j, i])
            key = (oid, which)
            if key not in worst or pc > worst[key]["pc"]:
                worst[key] = {"other": oid, "miss_km": round(miss, 4), "pc": pc,
                              "jd": float(jd[i] + tau / SECONDS_PER_DAY)}
    bad = []
    for (oid, which), h in worst.items():
        if which != "with" or h["pc"] <= threshold:
            continue
        base = worst.get((oid, "without"))
        if base is not None and base["pc"] >= h["pc"]:
            continue   # there anyway, not caused by the burn
        bad.append(h)
    return bad
