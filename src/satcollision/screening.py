r"""
Refined conjunction screening: the closest approach *between* time samples.

Why this exists
---------------
``twin.screen_conjunctions`` keeps a pair only if the two objects are within
the screening distance *at one of the grid instants*, and takes the smallest
sampled distance as the miss distance. That is fine when relative motion is
slow, and badly wrong when it is fast. Two LEO satellites crossing at
7 to 10 km/s move 200 to 300 km relative to each other between two 30-second
samples, so they pass through a 5 km screening sphere in about a second and
are usually never sampled inside it.

Measured on the 10 Sep 2026 Starlink catalogue (10,713 objects, a 2-hour
window, 30 s step): the grid screen found 106 pairs inside 5 km and 1 inside
1 km; this screen found 3,144 and 42. The sub-kilometre approaches it adds
have a median relative speed of 7.6 km/s; the two the grid did find were
moving at 0.2 and 1.3 km/s. So the grid screen is not just incomplete, it is
biased towards slow, overtaking geometries. (``research/screen_check.py``
reproduces the comparison.)

Method
------
1. Propagate on the same coarse grid (default 30 s), in time chunks so a
   week-long run over ~11,000 objects fits in memory.
2. At each sample, find every pair within a *candidate* radius large enough
   that no approach can slip between samples:
   ``max_rel_speed * step / 2 + screening_km``.
3. For each candidate, assume straight-line relative motion over half a step
   either side and solve for the closest approach. Over 15 s the *relative*
   trajectory of two nearby objects is very close to a straight line, because
   both feel almost the same gravity.
4. Group a pair's hits into separate approaches (a pair in neighbouring
   orbits meets about every half orbit, and each meeting is its own event),
   keep the closest per approach, and refine it with direct SGP4 calls and a
   bounded 1-D minimisation. The refined state vectors are what the Pc models
   receive.

On the 2-hour check, the linear estimate and the SGP4-refined miss distance
agreed to within 0.3 m on every sub-kilometre approach.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

from .fleets import TrackedObject
from .twin import (
    DEFAULT_COMBINED_HBR_KM, DEFAULT_COMBINED_SIGMA_KM, Conjunction,
    _classify_geometry, compute_pc, propagate_window,
)

# Upper bound on the relative speed of two LEO objects (head-on at ~7.8 km/s
# each). Only used to size the candidate radius, so erring high costs time,
# not correctness.
MAX_REL_SPEED_KM_S = 16.0

# Hits of the same pair less than this far apart in time are one approach.
# Well under half an orbit (~47 min in LEO), well over one step.
SAME_APPROACH_GAP_S = 600.0


def _sgp4_state(obj: TrackedObject, jd0: float, fr0: float, t_s: float):
    fr = fr0 + t_s / 86400.0
    day = np.floor(fr)
    err, r, v = obj.satrec.sgp4(jd0 + day, fr - day)
    return err, np.asarray(r), np.asarray(v)


def _refine(obj_a, obj_b, jd0, fr0, t_guess, half_width_s):
    """Exact closest approach near ``t_guess`` using direct SGP4 calls."""

    def dist(t):
        ea, ra, _ = _sgp4_state(obj_a, jd0, fr0, t)
        eb, rb, _ = _sgp4_state(obj_b, jd0, fr0, t)
        if ea or eb:
            return 1.0e9
        return float(np.linalg.norm(ra - rb))

    res = minimize_scalar(dist, bounds=(t_guess - half_width_s, t_guess + half_width_s),
                          method="bounded", options={"xatol": 1.0e-3})
    t = float(res.x)
    _, ra, va = _sgp4_state(obj_a, jd0, fr0, t)
    _, rb, vb = _sgp4_state(obj_b, jd0, fr0, t)
    return t, float(np.linalg.norm(ra - rb)), ra, rb, va, vb


def screen_conjunctions_refined(
    objects: list[TrackedObject],
    jd0: float,
    fr0: float,
    duration_s: float,
    step_s: float = 30.0,
    screening_distance_km: float = 5.0,
    combined_sigma_km: float = DEFAULT_COMBINED_SIGMA_KM,
    combined_hbr_km: float = DEFAULT_COMBINED_HBR_KM,
    include_debris_debris: bool = False,
    pc_model=None,
    chunk_s: float = 6 * 3600.0,
    pair_filter=None,
    refine: bool = True,
) -> list[Conjunction]:
    """Screen every pair for approaches closer than ``screening_distance_km``.

    Returns one :class:`twin.Conjunction` per *approach* (not per pair), with
    TCA, miss distance and state vectors from SGP4 refinement. The arguments
    mirror ``twin.screen_conjunctions``; ``pair_filter(obj_a, obj_b) -> bool``
    can drop pairs early (e.g. keep only cross-operator pairs).
    """
    n_obj = len(objects)
    candidate_r = MAX_REL_SPEED_KM_S * step_s / 2.0 + screening_distance_km
    half = step_s / 2.0

    is_debris = np.array([o.is_debris for o in objects])
    hits: dict[tuple[int, int], list[tuple[float, float]]] = {}

    n_total = int(duration_s // step_s) + 1
    steps_per_chunk = max(1, int(chunk_s // step_s))
    for c0 in range(0, n_total, steps_per_chunk):
        c1 = min(n_total, c0 + steps_per_chunk)
        t_start = c0 * step_s
        frc = fr0 + t_start / 86400.0
        prop = propagate_window(objects, jd0 + np.floor(frc), frc - np.floor(frc),
                                duration_s=(c1 - c0 - 1) * step_s, step_s=step_s)
        r, v, ok = prop.positions, prop.velocities, prop.valid
        for k in range(r.shape[1]):
            idx = np.nonzero(ok[:, k])[0]
            if idx.size < 2:
                continue
            pairs = cKDTree(r[idx, k]).query_pairs(candidate_r, output_type="ndarray")
            if pairs.size == 0:
                continue
            a, b = idx[pairs[:, 0]], idx[pairs[:, 1]]
            if not include_debris_debris:
                keep = ~(is_debris[a] & is_debris[b])
                a, b = a[keep], b[keep]
            dr = r[a, k] - r[b, k]
            dv = v[a, k] - v[b, k]
            tau = -np.einsum("ij,ij->i", dr, dv) / np.maximum(np.einsum("ij,ij->i", dv, dv), 1e-12)
            tau = np.clip(tau, -half, half)
            dmin = np.linalg.norm(dr + dv * tau[:, None], axis=1)
            m = dmin < screening_distance_km
            t_k = t_start + k * step_s
            for i, j, d, tt in zip(a[m], b[m], dmin[m], tau[m]):
                key = (int(i), int(j)) if i < j else (int(j), int(i))
                hits.setdefault(key, []).append((t_k + float(tt), float(d)))
        del prop, r, v, ok

    conjunctions: list[Conjunction] = []
    if pc_model is not None:
        from .pc import EncounterGeometry
    for (i, j), lst in hits.items():
        obj_a, obj_b = objects[i], objects[j]
        if pair_filter is not None and not pair_filter(obj_a, obj_b):
            continue
        lst.sort()
        # split into approaches, keep the closest hit of each
        groups, cur = [], [lst[0]]
        for h in lst[1:]:
            if h[0] - cur[-1][0] > SAME_APPROACH_GAP_S:
                groups.append(cur)
                cur = [h]
            else:
                cur.append(h)
        groups.append(cur)
        for g in groups:
            t_lin, d_lin = min(g, key=lambda h: h[1])
            if refine:
                t, d, ra, rb, va, vb = _refine(obj_a, obj_b, jd0, fr0, t_lin, step_s)
            else:
                t, d = t_lin, d_lin
                _, ra, va = _sgp4_state(obj_a, jd0, fr0, t)
                _, rb, vb = _sgp4_state(obj_b, jd0, fr0, t)
            if d >= screening_distance_km or t < 0 or t > duration_s:
                continue
            if pc_model is None:
                pc = compute_pc(d, combined_sigma_km, combined_hbr_km)
            else:
                pc = pc_model.pc(EncounterGeometry(miss_distance_km=d, position_a_km=ra,
                                                   position_b_km=rb, velocity_a_km_s=va,
                                                   velocity_b_km_s=vb), combined_sigma_km)
            conjunctions.append(Conjunction(
                object_a=obj_a, object_b=obj_b, tca_offset_s=t, miss_distance_km=d, pc=pc,
                geometry_class=_classify_geometry(va, vb),
                relative_speed_km_s=float(np.linalg.norm(va - vb)),
                meta={"screening": "refined", "linear_miss_km": d_lin},
                position_a_km=ra, position_b_km=rb, velocity_a_km_s=va, velocity_b_km_s=vb,
            ))
    conjunctions.sort(key=lambda c: -c.pc)
    return conjunctions
