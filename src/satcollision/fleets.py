"""
Orbital simulation layer.

The sandbox this project is currently being developed in has no outbound
network access to celestrak.org (only pypi/npm/etc. package registries are
reachable), so this module cannot download real TLE data directly. Two
paths are provided:

1. ``build_synthetic_fleets()`` — constructs physically realistic satellite
   populations directly from Keplerian elements (altitude, inclination,
   number of planes/satellites) that mirror the *publicly documented* shell
   parameters of real constellations (Starlink, OneWeb, Iridium NEXT).
   These are not scraped/real TLEs; they are legitimate synthetic orbits
   built the same way a mission-design tool would, from public specs.
2. ``load_tle_file()`` — reads a real two-line-element file (e.g. one
   downloaded from https://celestrak.org/NORAD/elements/gp.php on a machine
   that *does* have internet access, or the CelesTrak "supplemental data"
   CSV/TLE bundles). Drop a ``.tle`` file into ``data/`` and pass its path
   here to switch from synthetic to real data with no other code changes,
   which satisfies the historical-backtesting requirement in Section 7 of
   the project definition.

Both paths return the same internal representation: a list of
``TrackedObject`` records, each wrapping an ``sgp4.api.Satrec`` that can be
propagated with :func:`propagate`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from sgp4.api import Satrec, WGS72, jday

MU_EARTH_KM3_S2 = 398600.4418  # Earth's gravitational parameter
R_EARTH_KM = 6378.137

# Publicly documented, approximate shell parameters for real constellations.
# altitude_km / inclination_deg are representative of each operator's
# published orbital shells; not exact per-satellite figures.
OPERATOR_SHELLS = {
    "Alpha": {  # Starlink-like: low altitude, ~53 deg inclination, many planes
        "altitude_km": 550.0,
        "inclination_deg": 53.0,
        "n_planes": 4,
        "sats_per_plane": 6,
        "eccentricity": 0.0001,
    },
    "Beta": {  # OneWeb-like: higher altitude, near-polar
        "altitude_km": 1200.0,
        "inclination_deg": 87.4,
        "n_planes": 3,
        "sats_per_plane": 5,
        "eccentricity": 0.0001,
    },
    "Gamma": {  # Iridium-like: mid altitude, polar
        "altitude_km": 780.0,
        "inclination_deg": 86.4,
        "n_planes": 3,
        "sats_per_plane": 4,
        "eccentricity": 0.0001,
    },
}

# Ambient debris: scattered across a band overlapping all three shells so
# that conjunctions with debris (not just operator-vs-operator) also occur.
DEBRIS_ALTITUDE_RANGE_KM = (500.0, 1300.0)
DEBRIS_INCLINATION_RANGE_DEG = (40.0, 98.0)
DEFAULT_N_DEBRIS = 25


@dataclass
class TrackedObject:
    """One tracked object (an operator satellite or a debris fragment)."""

    norad_id: int
    operator: str  # one of OPERATOR_SHELLS keys, or "DEBRIS"
    satrec: Satrec
    is_debris: bool = False
    is_actor: bool = False  # flagged as a scenario protagonist (demo hook)
    meta: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.operator}-{self.norad_id}"


def _mean_motion_rad_per_min(altitude_km: float) -> float:
    """Two-body mean motion for a near-circular orbit at the given altitude.

    This ignores J2/Kozai corrections (a standard simplification for
    synthetic scenario generation); SGP4 itself still applies full
    perturbation physics during propagation.
    """
    a = R_EARTH_KM + altitude_km
    n_rad_s = math.sqrt(MU_EARTH_KM3_S2 / a**3)
    return n_rad_s * 60.0


def _epoch_from_calendar(year: int, month: int, day: int, hour: float = 0.0) -> float:
    jd, fr = jday(year, month, day, hour, 0, 0.0)
    return (jd - 2433281.5) + fr, jd, fr


def _make_satrec(
    norad_id: int,
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    mean_anomaly_deg: float,
    eccentricity: float,
    epoch_days: float,
    bstar: float = 1.0e-5,
) -> Satrec:
    satrec = Satrec()
    satrec.sgp4init(
        WGS72,
        "i",  # improved ('afspc') operation mode
        norad_id,
        epoch_days,
        bstar,
        0.0,  # ndot (unused by SGP4, kept for TLE compatibility)
        0.0,  # nddot (unused by SGP4)
        eccentricity,
        math.radians(0.0),  # argument of perigee
        math.radians(inclination_deg),
        math.radians(mean_anomaly_deg),
        _mean_motion_rad_per_min(altitude_km),
        math.radians(raan_deg),
    )
    return satrec


def build_synthetic_fleets(
    epoch=(2026, 9, 9, 0.0),
    n_debris: int = DEFAULT_N_DEBRIS,
    seed: int = 42,
) -> list[TrackedObject]:
    """Build 3 synthetic operator fleets plus an ambient debris population.

    Returns a flat list of :class:`TrackedObject`. Deterministic given the
    same ``seed`` so results are reproducible across runs/tests.
    """
    import random

    rng = random.Random(seed)
    epoch_days, _, _ = _epoch_from_calendar(*epoch)

    objects: list[TrackedObject] = []
    norad_id = 90000  # synthetic ID range, clearly out of real NORAD catalog space

    for operator, shell in OPERATOR_SHELLS.items():
        n_planes = shell["n_planes"]
        sats_per_plane = shell["sats_per_plane"]
        for plane in range(n_planes):
            raan = (360.0 / n_planes) * plane
            for slot in range(sats_per_plane):
                mean_anom = (360.0 / sats_per_plane) * slot + rng.uniform(-2.0, 2.0)
                satrec = _make_satrec(
                    norad_id=norad_id,
                    altitude_km=shell["altitude_km"] + rng.uniform(-1.5, 1.5),
                    inclination_deg=shell["inclination_deg"] + rng.uniform(-0.05, 0.05),
                    raan_deg=raan,
                    mean_anomaly_deg=mean_anom,
                    eccentricity=shell["eccentricity"],
                    epoch_days=epoch_days,
                )
                objects.append(
                    TrackedObject(norad_id=norad_id, operator=operator, satrec=satrec)
                )
                norad_id += 1

    for _ in range(n_debris):
        alt = rng.uniform(*DEBRIS_ALTITUDE_RANGE_KM)
        inc = rng.uniform(*DEBRIS_INCLINATION_RANGE_DEG)
        satrec = _make_satrec(
            norad_id=norad_id,
            altitude_km=alt,
            inclination_deg=inc,
            raan_deg=rng.uniform(0, 360),
            mean_anomaly_deg=rng.uniform(0, 360),
            eccentricity=rng.uniform(0.0001, 0.01),
            epoch_days=epoch_days,
            bstar=rng.uniform(1e-5, 5e-4),  # debris tends to be draggier
        )
        objects.append(
            TrackedObject(norad_id=norad_id, operator="DEBRIS", satrec=satrec, is_debris=True)
        )
        norad_id += 1

    return objects


def load_tle_file(path: str, operator_name: str = "REAL") -> list[TrackedObject]:
    """Load real TLEs (e.g. downloaded from CelesTrak) from a 2- or 3-line file.

    Use this in place of :func:`build_synthetic_fleets` once running outside
    this network-restricted sandbox, to satisfy the historical-backtesting
    requirement against genuine historical orbital data.
    """
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]

    objects: list[TrackedObject] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            l1, l2 = lines[i], lines[i + 1]
            name = None
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
            i += 3
        else:
            i += 1
            continue
        satrec = Satrec.twoline2rv(l1, l2)
        objects.append(
            TrackedObject(
                norad_id=satrec.satnum,
                operator=operator_name,
                satrec=satrec,
                meta={"name": name} if name else {},
            )
        )
    return objects


def inject_actor_encounter(
    objects: list[TrackedObject],
    epoch=(2026, 9, 9, 0.0),
    watch_operator_a: str = "Alpha",
    new_operator_b: str = "Beta",
    tca_offset_min: float = 30.0,
    crossing_angle_deg: float = 55.0,
    seed: int = 7,
) -> tuple[TrackedObject, TrackedObject]:
    """Deterministically engineer one genuine close approach for the demo.

    Real synthetic fleets built from independent random shell parameters
    essentially never produce a sub-kilometer conjunction by chance in a
    short propagation window (by design — that's what "safe" orbital
    spacing looks like). To exercise the full pipeline end-to-end
    (screening -> Pc -> abstraction -> federation -> deconfliction) without
    waiting for a rare random event, this picks one existing satellite from
    ``watch_operator_a`` as "actor A", and *constructs* a new "actor B"
    satellite whose orbit is solved (in closed form, then refined with a
    1-D numerical correction against the real SGP4 propagation) to pass
    within a few hundred meters of actor A at ``tca_offset_min`` minutes
    after epoch. Both new objects are flagged ``is_actor=True``.

    This mirrors how the illustrative HTML demo ("Conjunction Watch") sets
    up its scripted scenario, except here the close approach is a genuine
    property of two SGP4-propagated orbits rather than a scripted animation.
    """
    from scipy.optimize import minimize

    epoch_days, jd0, fr0 = _epoch_from_calendar(*epoch)
    actor_a = next(o for o in objects if o.operator == watch_operator_a and not o.is_debris)
    actor_a.is_actor = True

    # advance (jd0, fr0) by tca_offset_min minutes
    fr_t = fr0 + tca_offset_min / (24.0 * 60.0)
    jd_t = jd0 + math.floor(fr_t)
    fr_t = fr_t - math.floor(fr_t)

    pos_a, _ = propagate(actor_a, jd_t, fr_t)
    P = pos_a
    r_target = math.sqrt(sum(c * c for c in P))
    inc_a_deg = OPERATOR_SHELLS.get(watch_operator_a, {"inclination_deg": 53.0})["inclination_deg"]
    inc_b_deg = (inc_a_deg + crossing_angle_deg) % 180.0
    i_rad = math.radians(inc_b_deg)

    sin_i = math.sin(i_rad)
    ratio = max(-1.0, min(1.0, P[2] / (r_target * sin_i))) if sin_i != 0 else 0.0
    u0 = math.asin(ratio)
    # asin has a two-branch ambiguity (u0 and pi - u0 share the same sine,
    # i.e. the same z, but flip the sign of cos(u)); try both and keep
    # whichever actually reconstructs the target x/y, rather than assuming
    # the principal branch is correct.
    best = None
    for u_candidate in (u0, math.pi - u0):
        a_coef = math.cos(u_candidate)
        b_coef = math.sin(u_candidate) * math.cos(i_rad)
        raan_candidate = math.atan2(P[1], P[0]) - math.atan2(b_coef, a_coef)
        x_check = r_target * (
            math.cos(raan_candidate) * a_coef - math.sin(raan_candidate) * b_coef
        )
        y_check = r_target * (
            math.sin(raan_candidate) * a_coef + math.cos(raan_candidate) * b_coef
        )
        residual = math.hypot(x_check - P[0], y_check - P[1])
        if best is None or residual < best[0]:
            best = (residual, u_candidate, raan_candidate)
    _, u, raan = best

    altitude_km = r_target - R_EARTH_KM
    raan_deg = math.degrees(raan) % 360.0
    # `u` above is the argument of latitude (~mean anomaly, argp=0) AT THE
    # TARGET TIME, not at epoch. sgp4init wants the mean anomaly AT EPOCH,
    # so back-propagate by the elapsed time using this shell's two-body
    # mean motion before handing it to _make_satrec — otherwise the
    # satellite ends up roughly (mean_motion * elapsed_time) radians away
    # from the intended target point instead of at it.
    mean_motion_deg_per_min = math.degrees(_mean_motion_rad_per_min(altitude_km))
    mean_anomaly_deg = (math.degrees(u) - mean_motion_deg_per_min * tca_offset_min) % 360.0

    # The closed-form two-body inversion above ignores SGP4's own J2/drag
    # initialization corrections (which shift the *osculating* epoch state
    # a few km from the naive two-body point even at t=0, and add RAAN
    # precession over the elapsed time), which is why it lands within a
    # few km rather than exactly on target. A 3-parameter local refinement
    # (nudging epoch mean anomaly, RAAN, and altitude — matching the 3
    # position components exactly) against the *real* SGP4-propagated
    # position closes that residual to sub-meter.
    def _miss_distance(delta: "tuple[float, float, float]") -> float:
        delta_ma_deg, delta_raan_deg, delta_alt_km = delta
        satrec = _make_satrec(
            norad_id=99000,
            altitude_km=altitude_km + delta_alt_km,
            inclination_deg=inc_b_deg,
            raan_deg=raan_deg + delta_raan_deg,
            mean_anomaly_deg=mean_anomaly_deg + delta_ma_deg,
            eccentricity=0.0001,
            epoch_days=epoch_days,
        )
        _, pos_b, _ = satrec.sgp4(jd_t, fr_t)
        return math.sqrt(sum((pb - pa) ** 2 for pb, pa in zip(pos_b, P)))

    res = minimize(_miss_distance, x0=[0.0, 0.0, 0.0], method="Nelder-Mead",
                    options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 5000})
    best_delta_ma, best_delta_raan, best_delta_alt = res.x

    satrec_b = _make_satrec(
        norad_id=99000,
        altitude_km=altitude_km + best_delta_alt,
        inclination_deg=inc_b_deg,
        raan_deg=raan_deg + best_delta_raan,
        mean_anomaly_deg=mean_anomaly_deg + best_delta_ma,
        eccentricity=0.0001,
        epoch_days=epoch_days,
    )
    actor_b = TrackedObject(
        norad_id=99000,
        operator=new_operator_b,
        satrec=satrec_b,
        is_actor=True,
        meta={"engineered_tca_offset_min": tca_offset_min},
    )
    objects.append(actor_b)
    return actor_a, actor_b


def propagate(obj: TrackedObject, jd: float, fr: float):
    """Propagate one object to the given Julian date; returns (position_km, velocity_km_s).

    ``position``/``velocity`` are 3-tuples in the TEME frame. Raises
    ``RuntimeError`` if SGP4 reports an error code (e.g. decayed orbit).
    """
    error_code, position, velocity = obj.satrec.sgp4(jd, fr)
    if error_code != 0:
        raise RuntimeError(f"SGP4 error code {error_code} propagating {obj.label}")
    return position, velocity


def propagate_all(objects: Iterable[TrackedObject], jd: float, fr: float) -> dict:
    """Propagate every object to one instant; returns {norad_id: (pos, vel)}.

    Objects that error out (e.g. decayed) are silently skipped and absent
    from the returned dict — callers should treat a missing id as untracked
    at that instant rather than crash the whole scenario.
    """
    out = {}
    for obj in objects:
        try:
            out[obj.norad_id] = propagate(obj, jd, fr)
        except RuntimeError:
            continue
    return out
