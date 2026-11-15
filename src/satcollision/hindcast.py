r"""
Hindcast on a TLE history: what each operator's twin would have predicted,
scored against what really happened.

The research scripts in ``research/`` (two_stage_mc.py, sweep.py,
measured.py) draw the prediction error from a Gaussian with an assumed or
measured sigma. This module removes that assumption. It uses a Space-Track
``gp_history`` file the same way the TLE-history world does
(``tle_history.py``):

* **truth**: the first element set published *after* the moment,
  propagated back to it (blended across switches; see
  :func:`truth_weights` for why this differs from the world's
  nearest-epoch truth);
* **public prediction at an update time u**: the newest element set with
  epoch <= u, propagated with SGP4;
* **an operator's own satellite**: known as well as the truth, with a small
  isotropic ``precise_sigma_km``.

So the prediction error of every object at every update is the *real*
error of the public catalogue, including manoeuvres between sets, drag
mis-modelling and the occasional bad fit, not a draw from a model of it.

Pipeline
--------
1. :func:`screen_truth`: every true close approach between a member
   operator's satellite and any object not flown by the same operator
   (other members, other payloads, debris), refined screen on a 30 s grid
   (closest approach solved between samples, as ``screening.py`` does).
2. :func:`approach_geometry`: exact truth states at each TCA, the 2-D miss
   vector in the encounter plane and the plane projections of each
   object's radial / along-track / cross-track axes.
3. :func:`public_errors`: for each approach, object and update time, the
   RTN error of the public prediction at the true TCA, plus the features of
   the element set used (its age at TCA, mean motion, eccentricity, ...).
4. The Pc of every assessment under any covariance model is then one
   vectorised expression (small-hard-body Foster, ``research/aniso.py``):

       Pc = R^2 / (2 sqrt(det C)) exp(-m^T C^-1 m / 2)

   with C = sum over axes of sigma_axis^2 P_axis for each object, and m the
   *predicted* miss: the true miss minus the in-plane part of the
   prediction error (linear relative motion near TCA, the same assumption
   Foster's method makes), optionally after subtracting a predicted bias.

The linear step is what makes a week of all-LEO hindcasts cheap: one SGP4
call per object and update instead of a re-screen. It is exact to first
order in error / (relative speed x screening window); approaches slower
than ``MIN_VREL_KM_S`` are dropped, as they are in aniso.py.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .tle_history import SECONDS_PER_DAY, TLEHistory, _split_jd, to_rtn

MAX_REL_SPEED_KM_S = 16.0
SAME_APPROACH_GAP_S = 600.0
MIN_VREL_KM_S = 0.01
EARTH_RADIUS_KM = 6378.137
MU_KM3_S2 = 398600.4418
HBR_KM = 0.02


# ---------------------------------------------------------------------------
# Element-wise states: (row_k, jd_k) pairs, each with its own time
# ---------------------------------------------------------------------------

def _keys(hist: TLEHistory, rows: np.ndarray, jd: np.ndarray) -> np.ndarray:
    return rows * hist._span + (jd - hist.jd0)


def asof_sets(hist: TLEHistory, rows: np.ndarray, jd_now: np.ndarray) -> np.ndarray:
    """Newest set with epoch <= jd_now, element-wise (-1: none yet)."""
    rows = np.asarray(rows, dtype=np.int64)
    pos = np.searchsorted(hist._keys, _keys(hist, rows, np.asarray(jd_now, float)), side="right")
    return np.where(pos - 1 >= hist.start[rows], pos - 1, -1)


def propagate_pairs(hist: TLEHistory, sets: np.ndarray, jd: np.ndarray):
    """Propagate set ``sets[k]`` to time ``jd[k]`` for every k. Returns r, v
    (n, 3) and ok (n,). One SGP4 array call per distinct set."""
    sets = np.asarray(sets, dtype=np.int64)
    jd = np.asarray(jd, dtype=float)
    n = len(sets)
    R = np.full((n, 3), np.nan)
    V = np.full((n, 3), np.nan)
    ok = np.zeros(n, dtype=bool)
    good = np.nonzero(sets >= 0)[0]
    if good.size == 0:
        return R, V, ok
    order = good[np.argsort(sets[good], kind="stable")]
    s_sorted = sets[order]
    cuts = np.nonzero(np.diff(s_sorted))[0] + 1
    whole, frac = _split_jd(jd)
    for grp in np.split(order, cuts):
        sat = hist.satrec(int(sets[grp[0]]))
        e, r, v = sat.sgp4_array(whole[grp], frac[grp])
        R[grp] = r
        V[grp] = v
        ok[grp] = e == 0
    return R, V, ok


def truth_weights(hist: TLEHistory, rows: np.ndarray, jd: np.ndarray, blend_s: float = 600.0):
    """Element-wise truth as a blend ``(1 - w) set_a + w set_b``.

    The truth at time t is the *first element set published after t*,
    propagated back to t. This differs from the TLE-history world's
    nearest-epoch truth on purpose: a prediction made at an update u uses
    sets with epoch <= u <= t, so it never shares its element set with the
    truth it is scored against, and its error at age a is exactly the
    TLE-vs-later-TLE difference the public error table measures. (With a
    nearest-epoch truth, every prediction within half a publication gap of
    its own epoch would have zero error by construction.)

    Around each epoch e the truth switches from that set to the next one;
    the switch is blended linearly over ``blend_s`` centred on e, so a
    screened trajectory has no jumps. Where no later set exists (the end of
    the file) the newest set is used."""
    rows = np.asarray(rows, dtype=np.int64)
    jd = np.asarray(jd, dtype=float)
    h = blend_s / 2.0 / SECONDS_PER_DAY
    pos = np.searchsorted(hist._keys, _keys(hist, rows, jd), side="right")
    lo, hi = hist.start[rows], hist.start[rows + 1]
    prev = np.where(pos - 1 >= lo, pos - 1, -1)
    nxt = np.where(pos < hi, pos, -1)
    nxt2 = np.where((nxt >= 0) & (nxt + 1 < hi), nxt + 1, -1)
    ep_p = np.where(prev >= 0, hist.epochs[np.maximum(prev, 0)], -np.inf)
    ep_n = np.where(nxt >= 0, hist.epochs[np.maximum(nxt, 0)], np.inf)
    a = np.where(nxt >= 0, nxt, prev)
    b = a.copy()
    w = np.zeros(len(rows))
    # just after an epoch: still blending from that set into the next one
    after = (prev >= 0) & (nxt >= 0) & (jd - ep_p < h)
    a = np.where(after, prev, a)
    b = np.where(after, nxt, b)
    w = np.where(after, (jd - (ep_p - h)) / (2 * h), w)
    # just before the next epoch: start blending into the set after it
    before = ~after & (nxt2 >= 0) & (ep_n - jd < h)
    b = np.where(before, nxt2, b)
    w = np.where(before, (jd - (ep_n - h)) / (2 * h), w)
    return a, b, np.clip(w, 0.0, 1.0)


def truth_pairs(hist: TLEHistory, rows: np.ndarray, jd: np.ndarray, blend_s: float = 600.0):
    """Truth (see :func:`truth_weights`) of ``rows[k]`` at ``jd[k]``.
    Returns r, v (n, 3) and ok (n,); invalid more than half a day outside
    the object's element sets."""
    rows = np.asarray(rows, dtype=np.int64)
    jd = np.asarray(jd, dtype=float)
    a, b, w = truth_weights(hist, rows, jd, blend_s)
    ra, va, oka = propagate_pairs(hist, a, jd)
    simple = (a == b) | (w <= 0)
    rb, vb, okb = ra.copy(), va.copy(), oka.copy()
    nz = np.nonzero(~simple)[0]
    if nz.size:
        rb[nz], vb[nz], okb[nz] = propagate_pairs(hist, b[nz], jd[nz])
    w3 = np.where(simple, 0.0, w)[:, None]
    first = hist.epochs[hist.start[rows]]
    last = hist.epochs[hist.start[rows + 1] - 1]
    ok = oka & okb & (jd >= first - 0.5) & (jd <= last + 0.5)
    return (1 - w3) * ra + w3 * rb, (1 - w3) * va + w3 * vb, ok


def truth_grid(hist: TLEHistory, rows: np.ndarray, jd: np.ndarray, blend_s: float = 600.0):
    """Truth of every row at every time: r, v (n, m, 3), ok (n, m)."""
    rows = np.asarray(rows, dtype=np.int64)
    n, m = len(rows), len(jd)
    rr = np.repeat(rows, m)
    tt = np.tile(np.asarray(jd, float), n)
    r, v, ok = truth_pairs(hist, rr, tt, blend_s)
    return r.reshape(n, m, 3), v.reshape(n, m, 3), ok.reshape(n, m)


def load_histories(paths: Sequence[str], cache_dir: str = "cache") -> TLEHistory:
    """One TLEHistory from several gp_history files (e.g. consecutive
    weeks). Duplicate element sets collapse in TLEHistory.load."""
    from pathlib import Path
    paths = [Path(p) for p in paths]
    if len(paths) == 1:
        return TLEHistory.load(paths[0])
    import hashlib
    key = hashlib.sha256("|".join(f"{p.resolve()}:{p.stat().st_size}:{p.stat().st_mtime}" for p in paths).encode()).hexdigest()[:16]
    out = Path(cache_dir) / f"merged_history_{key}.tle"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="ascii", errors="replace") as f:
            for p in paths:
                f.write(p.read_text(encoding="ascii", errors="replace").rstrip("\n") + "\n")
    return TLEHistory.load(out)


def subset_history(hist: TLEHistory, jd_max: float) -> TLEHistory:
    """The history as it stood at ``jd_max``: only sets with epoch < jd_max
    (objects with none are dropped)."""
    keep = hist.epochs < jd_max
    rows = hist.row_of_set[keep]
    present = np.unique(rows)
    counts = np.bincount(rows, minlength=hist.n_objects)[present]
    start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return TLEHistory(ids=hist.ids[present], names=[hist.names[i] for i in present], start=start,
                      epochs=hist.epochs[keep], line1=hist.line1[keep], line2=hist.line2[keep],
                      source=hist.source)


# ---------------------------------------------------------------------------
# 1. True close approaches
# ---------------------------------------------------------------------------

def owners_of(names: Sequence[str], operators: Dict[str, Sequence[str]]) -> Tuple[np.ndarray, List[str]]:
    """Operator index per object (-1: not a member), from name prefixes."""
    ops = list(operators)
    own = np.full(len(names), -1, dtype=np.int64)
    for i, n in enumerate(names):
        u = (n or "").upper()
        for k, op in enumerate(ops):
            if any(u.startswith(p.upper()) for p in operators[op]):
                own[i] = k
                break
    return own, ops


def screen_truth(hist: TLEHistory, owner: np.ndarray, jd_start: float, jd_end: float,
                 step_s: float = 30.0, screen_km: float = 10.0, blend_s: float = 600.0,
                 chunk_steps: int = 120, dense_op: Optional[int] = None, progress=None,
                 strata: Optional[Sequence[Tuple[float, float]]] = None) -> Dict[str, np.ndarray]:
    """Every true approach closer than ``screen_km`` between a member's
    satellite and an object the same operator does not fly.

    ``strata``: optional ((edge_km, keep_rate), ...), increasing edges, the
    last edge replacing ``screen_km``. An approach whose miss falls in a
    stratum is kept with that rate (decided by a hash of the pair, so all
    hits of one approach are kept or dropped together) and carries weight
    1 / rate. Wide approaches only matter for false alarms of an
    over-confident model, so they can be thinned heavily.

    ``dense_op``: index of an operator whose satellites are never paired with
    each other (they are, anyway, by the rule above); its rows get their own
    KD-tree so the dense shell's internal pairs are never enumerated.
    Returns arrays row_a, row_b (row_a owned), tca_jd, miss_lin_km."""
    rows = np.arange(hist.n_objects)
    if strata:
        edges = np.array([e for e, _ in strata], float)
        rates = np.array([r for _, r in strata], float)
        screen_km = float(edges[-1])
    else:
        edges, rates = np.array([screen_km]), np.array([1.0])

    def rate_of(d):
        return rates[np.minimum(np.searchsorted(edges, d, side="right"), len(rates) - 1)]

    def u_of(a, b):
        k = (np.minimum(a, b) * 1_000_003 + np.maximum(a, b)).astype(np.uint64)
        k = (k * np.uint64(0x9E3779B97F4A7C15)) ^ (k >> np.uint64(29))
        k = k * np.uint64(0xBF58476D1CE4E5B9)
        return (k >> np.uint64(11)).astype(np.float64) / float(1 << 53)

    cand_r = MAX_REL_SPEED_KM_S * step_s / 2.0 + screen_km
    half = step_s / 2.0
    n_steps = int(round((jd_end - jd_start) * SECONDS_PER_DAY / step_s)) + 1
    dense = (owner == dense_op) if dense_op is not None else np.zeros(len(owner), bool)
    I, J, T, D = [], [], [], []
    for c0 in range(0, n_steps, chunk_steps):
        c1 = min(n_steps, c0 + chunk_steps)
        jd = jd_start + np.arange(c0, c1) * step_s / SECONDS_PER_DAY
        R, V, ok = truth_grid(hist, rows, jd, blend_s)
        for k in range(c1 - c0):
            valid = ok[:, k]
            id_d = np.nonzero(valid & dense)[0]
            id_r = np.nonzero(valid & ~dense)[0]
            pa, pb = [], []
            tr = cKDTree(R[id_r, k])
            p = tr.query_pairs(cand_r, output_type="ndarray")
            if p.size:
                pa.append(id_r[p[:, 0]]); pb.append(id_r[p[:, 1]])
            if id_d.size:
                sm = cKDTree(R[id_d, k]).sparse_distance_matrix(tr, cand_r, output_type="ndarray")
                if sm.size:
                    pa.append(id_d[sm["i"]]); pb.append(id_r[sm["j"]])
            if not pa:
                continue
            a = np.concatenate(pa); b = np.concatenate(pb)
            oa, ob = owner[a], owner[b]
            keep = ((oa >= 0) | (ob >= 0)) & (oa != ob)
            a, b = a[keep], b[keep]
            dr = R[a, k] - R[b, k]
            dv = V[a, k] - V[b, k]
            tau = -np.einsum("ij,ij->i", dr, dv) / np.maximum(np.einsum("ij,ij->i", dv, dv), 1e-12)
            tau = np.clip(tau, -half, half)
            d = np.linalg.norm(dr + dv * tau[:, None], axis=1)
            m = d < screen_km
            if strata:
                m &= u_of(a, b) < rate_of(d)
            if not m.any():
                continue
            a, b, d, tau = a[m], b[m], d[m], tau[m]
            # Put the member first; if both are members, the lower row first.
            swap = (owner[a] < 0) | ((owner[b] >= 0) & (b < a))
            a, b = np.where(swap, b, a), np.where(swap, a, b)
            I.append(a); J.append(b); D.append(d)
            T.append(jd[k] + tau / SECONDS_PER_DAY)
        if progress:
            progress(c1, n_steps)
    if not I:
        z = np.zeros(0)
        return {"row_a": z.astype(np.int64), "row_b": z.astype(np.int64), "tca_jd": z, "miss_lin_km": z,
                "weight": z}
    I = np.concatenate(I); J = np.concatenate(J); T = np.concatenate(T); D = np.concatenate(D)
    # Group hits into approaches: sort by pair then time, split at gaps.
    key = I * (hist.n_objects + 1) + J
    order = np.lexsort((T, key))
    I, J, T, D, key = I[order], J[order], T[order], D[order], key[order]
    new = np.ones(len(I), bool)
    new[1:] = (key[1:] != key[:-1]) | ((T[1:] - T[:-1]) * SECONDS_PER_DAY > SAME_APPROACH_GAP_S)
    gid = np.cumsum(new) - 1
    # index of the minimum distance within each group
    o2 = np.lexsort((D, gid))
    first = np.ones(len(o2), bool)
    first[1:] = gid[o2][1:] != gid[o2][:-1]
    best = o2[first]
    return {"row_a": I[best], "row_b": J[best], "tca_jd": T[best], "miss_lin_km": D[best],
            "weight": 1.0 / rate_of(D[best])}


# ---------------------------------------------------------------------------
# 2. Encounter-plane geometry
# ---------------------------------------------------------------------------

def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def plane_basis(vrel: np.ndarray) -> np.ndarray:
    """(n, 3, 2): two orthonormal vectors spanning the plane normal to vrel."""
    vh = _unit(vrel)
    seed = np.zeros_like(vh)
    seed[np.arange(len(vh)), np.argmin(np.abs(vh), axis=1)] = 1.0
    u = _unit(np.cross(vh, seed))
    w = _unit(np.cross(vh, u))
    return np.stack([u, w], axis=-1)


def rtn_axes(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(n, 3, 3): rows R, T, C unit vectors."""
    rh = _unit(r)
    ch = _unit(np.cross(r, v))
    th = np.cross(ch, rh)
    return np.stack([rh, th, ch], axis=1)


def approach_geometry(hist: TLEHistory, enc: Dict[str, np.ndarray], blend_s: float = 600.0,
                      refine_s: float = 1.0) -> Dict[str, np.ndarray]:
    """Exact truth states at each TCA (re-solved on a +-``refine_s`` linear
    step from SGP4 truth at the screened TCA), the in-plane miss vector and
    each object's RTN axes projected into the plane. Drops approaches whose
    truth is invalid or whose relative speed is below MIN_VREL_KM_S."""
    a, b, t = enc["row_a"], enc["row_b"], enc["tca_jd"]
    ra, va, oka = truth_pairs(hist, a, t, blend_s)
    rb, vb, okb = truth_pairs(hist, b, t, blend_s)
    dr, dv = ra - rb, va - vb
    tau = -np.einsum("ij,ij->i", dr, dv) / np.maximum(np.einsum("ij,ij->i", dv, dv), 1e-12)
    tau = np.clip(np.nan_to_num(tau), -30.0, 30.0)
    t = t + tau / SECONDS_PER_DAY
    ra, va, oka2 = truth_pairs(hist, a, t, blend_s)
    rb, vb, okb2 = truth_pairs(hist, b, t, blend_s)
    ok = oka & okb & oka2 & okb2
    vrel = va - vb
    ok &= np.linalg.norm(np.nan_to_num(vrel), axis=1) >= MIN_VREL_KM_S
    keep = np.nonzero(ok)[0]
    ra, va, rb, vb, t = ra[keep], va[keep], rb[keep], vb[keep], t[keep]
    B = plane_basis(va - vb)
    m = np.einsum("nij,ni->nj", B, ra - rb)
    Pa = np.einsum("nki,nij->nkj", rtn_axes(ra, va), B)    # (n, 3 axes, 2)
    Pb = np.einsum("nki,nij->nkj", rtn_axes(rb, vb), B)
    out = {k: v[keep] for k, v in enc.items()}
    out.update(tca_jd=t, ra=ra, va=va, rb=rb, vb=vb, B=B, m_true=m, Pa=Pa, Pb=Pb,
               miss_km=np.linalg.norm(m, axis=1), rel_speed=np.linalg.norm(va - vb, axis=1))
    return out


# ---------------------------------------------------------------------------
# 3. Public prediction errors at the true TCA
# ---------------------------------------------------------------------------

def set_features(hist: TLEHistory, sets: np.ndarray) -> Dict[str, np.ndarray]:
    """Orbit features of element sets, read straight from the TLE lines."""
    l1 = hist.line1[sets]
    l2 = hist.line2[sets]

    def num(arr, a, b):
        return np.array([float(x[a:b]) for x in arr.astype(str)])

    def bstar(arr):
        out = []
        for x in arr.astype(str):
            f = x[53:61]
            try:
                mant = float(f[0] + "." + f[1:6].strip().replace(" ", "0"))
                out.append(mant * 10.0 ** int(f[6:8]))
            except ValueError:
                out.append(0.0)
        return np.array(out)

    n_rev_day = num(l2, 52, 63)
    n_rad_s = n_rev_day * 2 * math.pi / SECONDS_PER_DAY
    a_km = (MU_KM3_S2 / n_rad_s ** 2) ** (1.0 / 3.0)
    ecc = np.array([float("0." + x[26:33]) for x in l2.astype(str)])
    return {"incl_deg": num(l2, 8, 16), "ecc": ecc, "mean_motion": n_rev_day,
            "alt_km": a_km - EARTH_RADIUS_KM, "perigee_km": a_km * (1 - ecc) - EARTH_RADIUS_KM,
            "bstar": bstar(l1), "ndot": num(l1, 33, 43)}


def update_times(jd_start: float, jd_end: float, every_h: float) -> np.ndarray:
    """Update instants on a fixed clock (whole multiples of ``every_h`` since
    the Unix epoch, so every run uses the same ones)."""
    step = every_h / 24.0
    k0 = math.ceil((jd_start - 2440587.5) / step)
    k1 = math.floor((jd_end - 2440587.5) / step)
    return 2440587.5 + np.arange(k0, k1 + 1) * step


def assessments(enc: Dict[str, np.ndarray], owner: np.ndarray, updates: np.ndarray,
                horizon_h: float = 72.0) -> Dict[str, np.ndarray]:
    """One row per (approach, assessing operator, update): the assessor
    flies object ``own`` (the truth, precise) and predicts ``other`` from the
    public catalogue. Both sides of a member-member approach assess."""
    n = len(enc["tca_jd"])
    rows = []
    for side in (0, 1):
        own_row = enc["row_a"] if side == 0 else enc["row_b"]
        oth_row = enc["row_b"] if side == 0 else enc["row_a"]
        has = owner[own_row] >= 0
        idx = np.nonzero(has)[0]
        rows.append((idx, np.full(idx.size, side)))
    enc_idx = np.concatenate([r[0] for r in rows])
    side = np.concatenate([r[1] for r in rows])
    tca = enc["tca_jd"][enc_idx]
    # every update in (tca - horizon, tca)
    lo = np.searchsorted(updates, tca - horizon_h / 24.0, side="right")
    hi = np.searchsorted(updates, tca, side="left")
    cnt = np.maximum(hi - lo, 0)
    rep = np.repeat(np.arange(len(enc_idx)), cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    u = updates[lo[rep] + off]
    e = enc_idx[rep]
    s = side[rep]
    own = np.where(s == 0, enc["row_a"][e], enc["row_b"][e])
    oth = np.where(s == 0, enc["row_b"][e], enc["row_a"][e])
    return {"enc": e, "side": s, "own_row": own, "other_row": oth, "operator": owner[own],
            "update_jd": u, "lead_h": (enc["tca_jd"][e] - u) * 24.0}


def public_errors(hist: TLEHistory, enc: Dict[str, np.ndarray], asm: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """For each assessment: the public prediction of the *other* object at
    the true TCA, its RTN error (prediction - truth, in the truth's RTN
    frame), the age of the set at TCA and the set's features."""
    e = asm["enc"]
    tca = enc["tca_jd"][e]
    oth_is_b = asm["side"] == 0
    r_true = np.where(oth_is_b[:, None], enc["rb"][e], enc["ra"][e])
    v_true = np.where(oth_is_b[:, None], enc["vb"][e], enc["va"][e])
    sets = asof_sets(hist, asm["other_row"], asm["update_jd"])
    r_pred, _, ok = propagate_pairs(hist, sets, tca)
    err = to_rtn(np.nan_to_num(r_pred - r_true), r_true, v_true)
    age_h = np.where(sets >= 0, (tca - hist.epochs[np.maximum(sets, 0)]) * 24.0, np.nan)
    # Where no set was published after TCA (the end of the file) the truth
    # falls back to the newest set, which may be the very set predicted
    # from: such an assessment has zero error by construction, so drop it.
    ta, tb, _ = truth_weights(hist, asm["other_row"], tca)
    independent = (ta != sets) & (tb != sets)
    return {"set": sets, "ok": ok & (sets >= 0) & independent, "err_rtn": err, "age_h": age_h}


# ---------------------------------------------------------------------------
# 4. Pc under a covariance model
# ---------------------------------------------------------------------------

def plane_cov(P: np.ndarray, sig_rtn: np.ndarray) -> np.ndarray:
    """(n, 2, 2) = sum_axis sigma_axis^2 p_axis p_axis^T for P (n, 3, 2)."""
    return np.einsum("nk,nka,nkb->nab", sig_rtn ** 2, P, P)


def pc_small(m: np.ndarray, C: np.ndarray, hbr_km: float = HBR_KM) -> np.ndarray:
    a, b, d = C[..., 0, 0], C[..., 0, 1], C[..., 1, 1]
    det = np.maximum(a * d - b * b, 1e-30)
    x, y = m[..., 0], m[..., 1]
    q = (d * x * x - 2 * b * x * y + a * y * y) / det
    return np.minimum(hbr_km ** 2 / (2 * np.sqrt(det)) * np.exp(-0.5 * q), 1.0)


def predicted_miss(enc, asm, err, bias_rtn: Optional[np.ndarray] = None) -> np.ndarray:
    """In-plane miss the assessor predicts: own (true) minus other (public,
    optionally bias-corrected), linearised at the true TCA."""
    e = asm["enc"]
    oth_is_b = asm["side"] == 0
    P_oth = np.where(oth_is_b[:, None, None], enc["Pb"][e], enc["Pa"][e])
    de = err["err_rtn"] if bias_rtn is None else err["err_rtn"] - bias_rtn
    shift = np.einsum("nk,nkj->nj", de, P_oth)     # in-plane error of the other object
    m_true = enc["m_true"][e]
    # m = r_a - r_b. Other is b (side 0): m_pred = m_true - shift; other is a: + shift.
    return np.where(oth_is_b[:, None], m_true - shift, m_true + shift)


def assess_pc(enc, asm, err, sigma_other_rtn: np.ndarray, precise_sigma_km: float = 0.354,
              bias_rtn: Optional[np.ndarray] = None, circular_sigma_km: Optional[np.ndarray] = None) -> np.ndarray:
    """Pc of every assessment. ``sigma_other_rtn`` (n, 3) per-axis sigma of
    the other object's public prediction. If ``circular_sigma_km`` is given
    (scalar or (n,)), the circular model with that combined sigma is used
    instead (the covariance arguments are then ignored)."""
    m = predicted_miss(enc, asm, err, bias_rtn)
    if circular_sigma_km is not None:
        s2 = np.broadcast_to(np.asarray(circular_sigma_km, float) ** 2, (len(m),))
        C = s2[:, None, None] * np.eye(2)[None]
        return pc_small(m, C)
    e = asm["enc"]
    oth_is_b = asm["side"] == 0
    P_oth = np.where(oth_is_b[:, None, None], enc["Pb"][e], enc["Pa"][e])
    C = plane_cov(P_oth, sigma_other_rtn) + precise_sigma_km ** 2 * np.eye(2)[None]
    return pc_small(m, C)


def true_pc(enc, precise_sigma_km: float = 0.354) -> np.ndarray:
    """Pc on the true miss with both orbits precise (the full-sharing
    reference that defines a dangerous encounter)."""
    C = 2 * precise_sigma_km ** 2 * np.eye(2)[None]
    return pc_small(enc["m_true"], np.broadcast_to(C, (len(enc["m_true"]), 2, 2)))


# ---------------------------------------------------------------------------
# 5. Scoring
# ---------------------------------------------------------------------------

def score(asm, pc: np.ndarray, dangerous_enc: np.ndarray, n_days: float, threshold: float = 1e-4,
          mask: Optional[np.ndarray] = None, enc_weight: Optional[np.ndarray] = None) -> dict:
    """Per (approach, assessor) decision: alert if Pc >= threshold at any
    update. Caught = dangerous and alerted; warning = lead of the first
    alert. False alarm = alerted but not dangerous. ``enc_weight`` (per
    approach) undoes the thinning of wide approaches in screen_truth."""
    sel = np.ones(len(pc), bool) if mask is None else mask
    key = asm["enc"][sel] * 2 + asm["side"][sel]
    lead = asm["lead_h"][sel]
    hit = pc[sel] >= threshold
    uk, inv = np.unique(key, return_inverse=True)
    first_lead = np.full(len(uk), -1.0)
    np.maximum.at(first_lead, inv, np.where(hit, lead, -1.0))
    alerted = first_lead > 0
    dang = dangerous_enc[uk // 2]
    w = np.ones(len(uk)) if enc_weight is None else enc_weight[uk // 2]
    n_d = float(w[dang].sum())
    caught = alerted & dang
    wl = first_lead[caught]
    fa = float(w[alerted & ~dang].sum())
    n_alert = float(w[alerted].sum())
    return {
        "assessed": float(w.sum()), "dangerous": n_d,
        "caught": float(w[caught].sum()), "caught_frac": float(w[caught].sum() / max(n_d, 1e-9)),
        "caught_24h_frac": float(w[caught & (first_lead >= 24)].sum() / max(n_d, 1e-9)),
        "median_warning_h": float(np.median(wl)) if wl.size else 0.0,
        "warning_hours_per_dangerous": float((w[caught] * wl).sum() / max(n_d, 1e-9)),
        "false_alarms": fa, "false_alarms_per_day": fa / n_days,
        "alerts_per_day": n_alert / n_days,
        "precision": float(w[caught].sum() / max(n_alert, 1e-9)),
    }
