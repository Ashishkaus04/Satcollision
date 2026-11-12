"""Vectorised encounter-plane geometry for the anisotropic (Foster) Monte Carlo.

For each encounter: the 2-D miss vector m in the plane normal to v_rel, and the
2x2 projections Sa, Sb of each object's RTN covariance at unit sigma (the same
shape as pc.rtn_covariance_from_isotropic). A regime with per-object sigmas
(sa, sb) then has combined plane covariance C = sa^2 Sa + sb^2 Sb.

Pc uses the small-hard-body form of Foster's integral,
    Pc ~= pi R^2 * N(m; 0, C) = R^2 / (2 sqrt(det C)) * exp(-m^T C^-1 m / 2),
valid when R (20 m) is far below the sigmas; checked against pc.foster_pc in
check_against_foster().
"""
import math
import numpy as np

MIN_VREL = 0.01


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def rtn_unit_cov(r, v, k):
    rh = _unit(r)
    ch = _unit(np.cross(r, v))
    th = np.cross(ch, rh)
    s = math.sqrt(3.0 / (2.0 + k * k))
    M = np.stack([rh, th, ch], axis=-1)              # (n,3,3) columns RTN
    D = np.array([s * s, (k * s) ** 2, s * s])
    return np.einsum("nij,j,nkj->nik", M, D, M)


def plane_basis(vrel):
    vh = _unit(vrel)
    seed = np.zeros_like(vh)
    seed[np.arange(len(vh)), np.argmin(np.abs(vh), axis=1)] = 1.0
    u = _unit(np.cross(vh, seed))
    w = _unit(np.cross(vh, u))
    return np.stack([u, w], axis=-1)                 # (n,3,2)


def geometry(enc, k, mask=None):
    ra, rb, va, vb = enc["ra"], enc["rb"], enc["va"], enc["vb"]
    if mask is not None:
        ra, rb, va, vb = ra[mask], rb[mask], va[mask], vb[mask]
    vrel = va - vb
    ok = np.linalg.norm(vrel, axis=1) >= MIN_VREL
    B = plane_basis(np.where(ok[:, None], vrel, np.array([1.0, 0, 0])))
    m = np.einsum("nij,ni->nj", B, ra - rb)
    Sa = np.einsum("nij,nik,nkl->njl", B, rtn_unit_cov(ra, va, k), B)
    Sb = np.einsum("nij,nik,nkl->njl", B, rtn_unit_cov(rb, vb, k), B)
    return m, Sa, Sb, ok


def pc_small(m, C, R=0.02):
    """m (...,2), C (...,2,2) -> Pc."""
    a, b, d = C[..., 0, 0], C[..., 0, 1], C[..., 1, 1]
    det = a * d - b * b
    x, y = m[..., 0], m[..., 1]
    q = (d * x * x - 2 * b * x * y + a * y * y) / det
    return np.minimum(R * R / (2 * np.sqrt(det)) * np.exp(-0.5 * q), 1.0)


def check_against_foster(enc, k=8.0, n=200, seed=0):
    from satcollision.pc import FosterPcModel, EncounterGeometry
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.nonzero(enc["miss_km"] < 2.0)[0], n, replace=False)
    mask = np.zeros(len(enc["miss_km"]), bool); mask[idx] = True
    m, Sa, Sb, ok = geometry(enc, k, mask)
    model = FosterPcModel(along_track_ratio=k)
    errs = []
    for t, i in enumerate(np.nonzero(mask)[0]):
        if not ok[t]:
            continue
        for sa, sb in ((0.354, 0.354), (0.354, 2.12)):
            g = EncounterGeometry(miss_distance_km=float(enc["miss_km"][i]), position_a_km=enc["ra"][i],
                                  position_b_km=enc["rb"][i], velocity_a_km_s=enc["va"][i],
                                  velocity_b_km_s=enc["vb"][i])
            ref = model.pc(g, 0, sigma_a_km=sa, sigma_b_km=sb)
            mine = pc_small(m[t], sa * sa * Sa[t] + sb * sb * Sb[t])
            if ref > 1e-12:
                errs.append(abs(mine / ref - 1))
    return float(np.max(errs)), float(np.median(errs)), len(errs)
