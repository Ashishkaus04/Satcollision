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

import numpy as np
from scipy.spatial import cKDTree
from sgp4.api import SatrecArray

from .fleets import TrackedObject

# Industry-realistic defaults (see project definition, Sec. 3):
# ~1e-4 is the common threshold for standard maneuver planning,
# tighter (~1e-5) for high-value missions.
DEFAULT_PC_THRESHOLD = 1.0e-4
DEFAULT_SCREENING_DISTANCE_KM = 25.0  # coarse-pass "worth looking closer at" radius
DEFAULT_COMBINED_SIGMA_KM = 0.5  # assumed combined position uncertainty (1-sigma)
DEFAULT_COMBINED_HBR_KM = 0.02  # ~20 m combined hard-body radius (typical smallsat + margin)


@dataclass
class PropagationResult:
    """Vectorized propagation output: every object, every timestep, in one shot.

    ``positions``/``velocities`` have shape (n_objects, n_times, 3);
    ``valid`` is a (n_objects, n_times) boolean mask (False where SGP4
    reported an error at that instant, e.g. a decayed orbit) — callers must
    check it before trusting a row/column, since invalid entries hold
    whatever garbage SGP4 returned rather than being pre-zeroed.
    """
    t_offsets_s: np.ndarray       # shape (n_times,)
    object_ids: list              # length n_objects, row order matches positions/velocities
    positions: np.ndarray         # shape (n_objects, n_times, 3) km
    velocities: np.ndarray        # shape (n_objects, n_times, 3) km/s
    valid: np.ndarray             # shape (n_objects, n_times) bool


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

    # Full ECI state of both objects at the TCA sample (km, km/s). Populated by
    # ``screen_conjunctions``; ``None`` on a Conjunction built by hand from a
    # scalar miss distance.
    #
    # The scalar miss distance is enough for the circular Pc model and for
    # nothing else. A non-circular model needs the relative position and
    # velocity (the velocity defines the encounter plane) and each object's own
    # position and velocity (each error ellipsoid is built in that object's own
    # RTN frame) — all of which the screening pass has in hand at the moment it
    # records the closest sample, and all of which were being thrown away.
    position_a_km: np.ndarray | None = None
    position_b_km: np.ndarray | None = None
    velocity_a_km_s: np.ndarray | None = None
    velocity_b_km_s: np.ndarray | None = None

    @property
    def pair_label(self) -> str:
        return f"{self.object_a.label} <-> {self.object_b.label}"

    @property
    def has_state_vectors(self) -> bool:
        return all(v is not None for v in (self.position_a_km, self.position_b_km,
                                           self.velocity_a_km_s, self.velocity_b_km_s))

    @property
    def r_rel_km(self) -> np.ndarray:
        """Relative position at TCA, object A minus object B."""
        return np.asarray(self.position_a_km, dtype=float) - np.asarray(self.position_b_km, dtype=float)

    @property
    def v_rel_km_s(self) -> np.ndarray:
        """Relative velocity at TCA, object A minus object B."""
        return np.asarray(self.velocity_a_km_s, dtype=float) - np.asarray(self.velocity_b_km_s, dtype=float)


def propagate_window(objects: list[TrackedObject], jd0: float, fr0: float,
                      duration_s: float, step_s: float) -> "PropagationResult":
    """Propagate every object across a time grid in one vectorized call.

    Uses ``sgp4.api.SatrecArray``, which propagates every satellite at
    every requested time inside compiled code in a single call, instead of
    looping one (object, timestep) pair at a time in Python. That loop is
    fine for the ~76-object synthetic demo fleet but becomes the dominant
    cost against a real multi-thousand-satellite catalog (a 24h/30s-step
    backtest against ~8,000 real Starlink satellites is ~23 million
    individual propagations) — this is what makes that case tractable.
    """
    n_steps = int(duration_s // step_s) + 1
    t_offsets_s = np.arange(n_steps) * step_s
    fr_raw = fr0 + t_offsets_s / 86400.0
    jd_arr = jd0 + np.floor(fr_raw)
    fr_arr = fr_raw - np.floor(fr_raw)

    satrec_array = SatrecArray([o.satrec for o in objects])
    errors, r, v = satrec_array.sgp4(jd_arr, fr_arr)  # shapes: (n_obj,n_times), (n_obj,n_times,3) x2

    return PropagationResult(
        t_offsets_s=t_offsets_s,
        object_ids=[o.norad_id for o in objects],
        positions=r,
        velocities=v,
        valid=(errors == 0),
    )


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
    propagation: "PropagationResult",
    screening_distance_km: float = DEFAULT_SCREENING_DISTANCE_KM,
    combined_sigma_km: float = DEFAULT_COMBINED_SIGMA_KM,
    combined_hbr_km: float = DEFAULT_COMBINED_HBR_KM,
    include_debris_debris: bool = False,
    pc_model=None,
) -> list[Conjunction]:
    """Screen every object pair across the time grid for close approaches.

    Broad-phase pass: at each timestep, a KD-tree (``scipy.spatial.cKDTree``)
    finds every pair within ``screening_distance_km`` in O(n log n) instead
    of brute-force O(n^2) — the standard "broad phase" trick borrowed from
    collision detection, applied here because it is exactly the problem
    conjunction screening has. Brute force is fine for the ~76-object
    synthetic demo fleet (a few thousand pairs) but explodes against a real
    multi-thousand-satellite catalog (~8,000 objects is ~32 million pairs
    PER timestep, times thousands of timesteps); with sparse real
    conjunctions, the KD-tree pass finds the same handful of close pairs in
    a small fraction of the time.

    Narrow-phase: per candidate pair, keep whichever timestep had the
    smallest distance as the TCA estimate — for a final-year prototype this
    grid resolution is an accepted simplification; a production system
    would refine TCA with a local quadratic/Brent search between grid
    points.

    ``pc_model`` selects the probability-of-collision estimator. ``None`` —
    the default — is the circular ``compute_pc`` on exactly the code path it
    used before, so every existing result is unchanged. Passing a model from
    ``pc.py`` swaps only the estimator; the screening, the TCA search and the
    geometry classification are identical either way, which is what makes a
    both-ways comparison a comparison rather than two experiments.

    Either way the TCA state vectors are now recorded on each Conjunction, so
    an encounter screened under the default model can still be re-scored under
    a non-circular one afterwards without re-propagating.
    """
    by_id = {o.norad_id: o for o in objects}
    object_ids = propagation.object_ids
    n_times = propagation.positions.shape[1]

    best: dict[tuple, dict] = {}
    for t_idx in range(n_times):
        valid_idx = np.nonzero(propagation.valid[:, t_idx])[0]
        if valid_idx.size < 2:
            continue
        coords = propagation.positions[valid_idx, t_idx, :]
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=screening_distance_km, output_type="ndarray")
        if pairs.size == 0:
            continue

        t_offset_s = float(propagation.t_offsets_s[t_idx])
        for a_local, b_local in pairs:
            a_glob, b_glob = int(valid_idx[a_local]), int(valid_idx[b_local])
            ida, idb = object_ids[a_glob], object_ids[b_glob]
            obj_a, obj_b = by_id[ida], by_id[idb]
            if obj_a.is_debris and obj_b.is_debris and not include_debris_debris:
                continue
            d = float(np.linalg.norm(coords[a_local] - coords[b_local]))
            key = (min(ida, idb), max(ida, idb))
            if key not in best or d < best[key]["dist"]:
                # Copied rather than stored as views: ``coords`` is rebuilt
                # every timestep, and a view would keep the whole timestep's
                # position array alive for as long as the pair stays the best
                # one. Three floats each is cheaper than that.
                best[key] = {
                    "dist": d,
                    "t": t_offset_s,
                    "pos_a": np.array(coords[a_local]),
                    "pos_b": np.array(coords[b_local]),
                    "vel_a": np.array(propagation.velocities[a_glob, t_idx, :]),
                    "vel_b": np.array(propagation.velocities[b_glob, t_idx, :]),
                    "obj_a": obj_a,
                    "obj_b": obj_b,
                }

    if pc_model is not None:
        # Imported here rather than at module scope: ``pc`` imports this module
        # for the circular formula, so a top-level import either way would be a
        # cycle. The default path never reaches this line.
        from .pc import EncounterGeometry

    conjunctions = []
    for rec in best.values():
        vrel = rec["vel_a"] - rec["vel_b"]
        rel_speed = float(np.linalg.norm(vrel))
        if pc_model is None:
            pc = compute_pc(rec["dist"], combined_sigma_km, combined_hbr_km)
        else:
            pc = pc_model.pc(
                EncounterGeometry(
                    miss_distance_km=rec["dist"],
                    position_a_km=rec["pos_a"], position_b_km=rec["pos_b"],
                    velocity_a_km_s=rec["vel_a"], velocity_b_km_s=rec["vel_b"],
                ),
                combined_sigma_km,
            )
        geometry = _classify_geometry(rec["vel_a"], rec["vel_b"])
        conjunctions.append(
            Conjunction(
                object_a=rec["obj_a"],
                object_b=rec["obj_b"],
                tca_offset_s=rec["t"],
                miss_distance_km=rec["dist"],
                pc=pc,
                geometry_class=geometry,
                relative_speed_km_s=rel_speed,
                position_a_km=rec["pos_a"],
                position_b_km=rec["pos_b"],
                velocity_a_km_s=rec["vel_a"],
                velocity_b_km_s=rec["vel_b"],
            )
        )
    conjunctions.sort(key=lambda c: -c.pc)
    return conjunctions
