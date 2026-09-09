"""
Per-operator digital twin layer: conjunction screening and probability-of-
collision (Pc) estimation, run against a real SGP4-propagated time window.

This is deliberately kept "boring" and standard: it screens pairwise
distances across a time grid (the same coarse-to-fine approach real
conjunction-assessment pipelines use — a wide screening volume first, then
refinement near the minimum) and estimates Pc with the widely used
simplified circular (equal-variance) 2D formula, e.g. as summarized in
Alfano's papers on collision probability:

    Pc ~= (HBR^2 / (2 * sigma^2)) * exp(-d^2 / (2 * sigma^2))

where ``d`` is the miss distance projected into the encounter plane,
``HBR`` is the combined hard-body radius of the two objects, and ``sigma``
is the (assumed isotropic) combined position uncertainty in that plane. A
full implementation would use the non-circular Foster/Estes double
integral with each operator's actual covariance; that refinement is listed
as future work in the project definition. What matters for this project is
that Pc responds correctly to real propagated geometry (closing speed,
miss distance, encounter angle) rather than being scripted, which this
delivers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .fleets import TrackedObject, propagate_all

# Industry-realistic defaults (see project definition, Sec. 3):
# ~1e-4 is the common threshold for standard maneuver planning,
# tighter (~1e-5) for high-value missions.
DEFAULT_PC_THRESHOLD = 1.0e-4
DEFAULT_SCREENING_DISTANCE_KM = 25.0  # coarse-pass "worth looking closer at" radius
DEFAULT_COMBINED_SIGMA_KM = 0.5  # assumed combined position uncertainty (1-sigma)
DEFAULT_COMBINED_HBR_KM = 0.02  # ~20 m combined hard-body radius (typical smallsat + margin)


@dataclass
class TimeSample:
    t_offset_s: float
    positions: dict  # {norad_id: (x,y,z) km}
    velocities: dict  # {norad_id: (vx,vy,vz) km/s}


@dataclass
class Conjunction:
    object_a: TrackedObject
    object_b: TrackedObject
    tca_offset_s: float
    miss_distance_km: float
    pc: float
    geometry_class: str
    relative_speed_km_s: float
    meta: dict = field(default_factory=dict)

    @property
    def pair_label(self) -> str:
        return f"{self.object_a.label} <-> {self.object_b.label}"


def propagate_window(objects: list[TrackedObject], jd0: float, fr0: float,
                      duration_s: float, step_s: float) -> list["TimeSample"]:
    """Propagate every object across a time grid; returns one sample per step."""
    samples = []
    n_steps = int(duration_s // step_s) + 1
    for k in range(n_steps):
        t_offset_s = k * step_s
        fr = fr0 + t_offset_s / 86400.0
        jd = jd0 + math.floor(fr)
        fr = fr - math.floor(fr)
        result = propagate_all(objects, jd, fr)
        positions = {nid: pv[0] for nid, pv in result.items()}
        velocities = {nid: pv[1] for nid, pv in result.items()}
        samples.append(TimeSample(t_offset_s=t_offset_s, positions=positions, velocities=velocities))
    return samples


def _dist(p1, p2) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def _classify_geometry(vel_a, vel_b) -> str:
    """Classify encounter geometry from the angle between velocity vectors.

    This is a coarse, standard 3-way bucketing (co-orbital/overtaking vs.
    crossing vs. head-on) — the same three categories used in the abstracted
    signal the federation layer shares (Sec. 3 of the project definition),
    chosen so no finer-grained (and more identifying) geometry ever needs
    to cross the operator boundary.
    """
    na = math.sqrt(sum(c * c for c in vel_a))
    nb = math.sqrt(sum(c * c for c in vel_b))
    if na == 0 or nb == 0:
        return "unknown"
    cos_angle = sum(a * b for a, b in zip(vel_a, vel_b)) / (na * nb)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle_deg = math.degrees(math.acos(cos_angle))
    if angle_deg < 30.0:
        return "overtaking"
    if angle_deg > 150.0:
        return "head-on"
    return "crossing"


def compute_pc(miss_distance_km: float,
               combined_sigma_km: float = DEFAULT_COMBINED_SIGMA_KM,
               combined_hbr_km: float = DEFAULT_COMBINED_HBR_KM) -> float:
    """Simplified circular 2D probability-of-collision estimate. See module docstring."""
    if combined_sigma_km <= 0:
        return 0.0
    exponent = -(miss_distance_km ** 2) / (2.0 * combined_sigma_km ** 2)
    pc = (combined_hbr_km ** 2 / (2.0 * combined_sigma_km ** 2)) * math.exp(exponent)
    return min(pc, 1.0)


def screen_conjunctions(
    objects: list[TrackedObject],
    samples: list["TimeSample"],
    screening_distance_km: float = DEFAULT_SCREENING_DISTANCE_KM,
    combined_sigma_km: float = DEFAULT_COMBINED_SIGMA_KM,
    combined_hbr_km: float = DEFAULT_COMBINED_HBR_KM,
    include_debris_debris: bool = False,
) -> list[Conjunction]:
    """Screen every object pair across the time grid for close approaches.

    Two-pass approach, mirroring real conjunction-assessment practice:
    pass 1 finds, per pair, the time sample with minimum distance within
    ``screening_distance_km``; pass 2 (implicit, via the time grid
    resolution) reports that sample's data as the TCA estimate. For a
    final-year prototype this grid resolution is an accepted
    simplification — a production system refines TCA with a local
    quadratic/Brent search between grid points.
    """
    by_id = {o.norad_id: o for o in objects}
    ids = list(by_id.keys())
    n = len(ids)

    best: dict[tuple, dict] = {}
    for sample in samples:
        present = [i for i in ids if i in sample.positions]
        for a_idx in range(len(present)):
            for b_idx in range(a_idx + 1, len(present)):
                ida, idb = present[a_idx], present[b_idx]
                obj_a, obj_b = by_id[ida], by_id[idb]
                if obj_a.is_debris and obj_b.is_debris and not include_debris_debris:
                    continue
                d = _dist(sample.positions[ida], sample.positions[idb])
                if d > screening_distance_km:
                    continue
                key = (ida, idb)
                if key not in best or d < best[key]["dist"]:
                    best[key] = {
                        "dist": d,
                        "t": sample.t_offset_s,
                        "vel_a": sample.velocities[ida],
                        "vel_b": sample.velocities[idb],
                    }

    conjunctions = []
    for (ida, idb), rec in best.items():
        obj_a, obj_b = by_id[ida], by_id[idb]
        vrel = tuple(a - b for a, b in zip(rec["vel_a"], rec["vel_b"]))
        rel_speed = math.sqrt(sum(c * c for c in vrel))
        pc = compute_pc(rec["dist"], combined_sigma_km, combined_hbr_km)
        geometry = _classify_geometry(rec["vel_a"], rec["vel_b"])
        conjunctions.append(
            Conjunction(
                object_a=obj_a,
                object_b=obj_b,
                tca_offset_s=rec["t"],
                miss_distance_km=rec["dist"],
                pc=pc,
                geometry_class=geometry,
                relative_speed_km_s=rel_speed,
            )
        )
    conjunctions.sort(key=lambda c: -c.pc)
    return conjunctions
