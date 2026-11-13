"""Two-stage result under the *measured* public-catalogue error (no assumed sigmas).

Same Monte Carlo as sweep.py, with one change: the public 1-sigma of each object
is no longer the assumed 3 km -> 0.5 km linear ramp, but the RTN sigma measured
on Space-Track history (research/tle_error_fit.py on 6-12 Nov 2026, 11,204
Starlink objects, TLE vs later TLE, robust 1.4826*MAD), as a function of the
age the latest public element set will have at TCA.

Age at TCA = lead + AGE0, AGE0 = 4 h (the mean age of the newest set at an
update, with ~2.8 sets per object per day). Sigmas are interpolated log-log
between bin centres. Two variants:
  spread  : robust sigma only
  biased  : sqrt(sigma^2 + median^2) per axis (the systematic along-track drift
            seen in Starlink sets, treated as unknown to the user)
Precise (operator's own ephemeris): isotropic per-object sigma sp/sqrt(2),
constant to TCA (an assumption; swept over sp).

Usage: python research/measured.py research/cache/<cache>.npz [--sp 0.25,0.5,1.0]
"""
import argparse, json, math, pathlib
import numpy as np
from two_stage import load, H_S, T, MPC_PAIRS_PER_HOUR
import aniso
from sweep import TAUS, UPDATE_S, chol2

AGE0_H = 4.0
# Starlink, gp_history 6-12 Nov 2026: bin centre (h), sigma R/T/N (km), median T (km)
BIN_H = np.array([1.5, 4.5, 9.0, 18.0, 30.0, 42.0, 60.0])
SIG = np.array([[0.11, 0.82, 0.05], [0.15, 0.99, 0.09], [0.20, 2.04, 0.14], [0.28, 7.88, 0.19],
                [0.38, 20.3, 0.24], [0.61, 41.2, 0.33], [1.24, 87.0, 0.45]])
MED_T = np.array([-0.11, 0.05, 0.49, 3.73, 9.34, 21.8, 44.4])


def measured_sigma(age_h, biased=False):
    s = SIG.copy()
    if biased:
        s[:, 1] = np.sqrt(s[:, 1] ** 2 + MED_T ** 2)
    la = np.log(np.clip(age_h, BIN_H[0], None))
    out = []
    for j in range(3):
        ls = np.log(s[:, j])
        slope = (ls[-1] - ls[-2]) / (np.log(BIN_H[-1]) - np.log(BIN_H[-2]))
        v = np.interp(la, np.log(BIN_H), ls)
        v = np.where(la > np.log(BIN_H[-1]), ls[-1] + slope * (la - np.log(BIN_H[-1])), v)
        out.append(np.exp(v))
    return np.stack(out, -1)                     # (..., 3)


def axis_projections(r, v, B):
    rh = aniso._unit(r); ch = aniso._unit(np.cross(r, v)); th = np.cross(ch, rh)
    P = []
    for e in (rh, th, ch):
        q = np.einsum("nij,ni->nj", B, e)        # (n,2)
        P.append(q[:, :, None] * q[:, None, :])
    return np.stack(P, 1)                         # (n,3,2,2)


def geometry(enc, mask):
    ra, rb, va, vb = (enc[x][mask] for x in ("ra", "rb", "va", "vb"))
    vrel = va - vb
    ok = np.linalg.norm(vrel, axis=1) >= aniso.MIN_VREL
    B = aniso.plane_basis(np.where(ok[:, None], vrel, np.array([1.0, 0, 0])))
    m = np.einsum("nij,ni->nj", B, ra - rb)
    Pa, Pb = axis_projections(ra, va, B), axis_projections(rb, vb, B)
    Iso = np.eye(2)[None]                        # isotropic sigma projects to identity in-plane
    return m[ok], Pa[ok], Pb[ok], ok


def run_case(m, Pa, Pb, max_lead, n_days, sp, biased, seed=0):
    rng = np.random.default_rng(seed)
    leads = np.arange(H_S, 0, -UPDATE_S); nl = len(leads)
    p2 = (sp / math.sqrt(2)) ** 2
    pub = measured_sigma(leads / 3600 + AGE0_H, biased) ** 2      # (nl,3) variances
    live = leads[None, :] <= max_lead[:, None]
    n = len(m)
    Ip = p2 * np.broadcast_to(np.eye(2), (n, nl, 2, 2))
    Ca_pub = np.einsum("lj,njab->nlab", pub, Pa)
    Cb_pub = np.einsum("lj,njab->nlab", pub, Pb)

    def est_pc(C):
        l11, l21, l22 = chol2(C)
        z = rng.standard_normal((2,) + l11.shape)
        e = np.stack([l11 * z[0], l21 * z[0] + l22 * z[1]], -1)
        return np.where(live, aniso.pc_small(m[:, None, :] + e, C), 0.0)

    C_full = 2 * Ip
    danger = aniso.pc_small(m, C_full[:, 0]) >= T
    pc_full = est_pc(C_full)
    pc_nc = est_pc(Ca_pub + Cb_pub)
    pc_s1 = np.maximum(est_pc(Ip + Cb_pub), est_pc(Ca_pub + Ip))

    def first(mask):
        a = mask.any(1); return np.where(a, np.argmax(mask, 1), -1)

    def r24(act):
        lead = np.where(act >= 0, leads[np.maximum(act, 0)], 0.0)
        return float((lead[danger] >= 24 * 3600).mean())

    kk = np.arange(nl)[None, :]
    out = {"sigma_precise_km": sp, "biased": biased, "dangerous_per_day": float(danger.sum() / n_days),
           "full_r24": r24(first(pc_full >= T)), "nocoop_r24": r24(first(pc_nc >= T)),
           "nocoop_any": float((first(pc_nc >= T)[danger] >= 0).mean())}
    for name, trig_pc in (("two_stage", pc_s1), ("public", pc_nc)):
        rows = []
        for tau in TAUS:
            trig = first(trig_pc >= tau)
            after = (trig[:, None] >= 0) & (kk >= trig[:, None]) & live
            act = first(after & (pc_full >= T))
            rows.append({"tau": tau, "r24": r24(act), "mpc_h_day": float(after.sum() / n_days / MPC_PAIRS_PER_HOUR)})
        out[name] = rows
        target, cost = 0.99 * out["full_r24"], None
        for lo, hi in zip(rows, rows[1:]):
            if lo is rows[0] and lo["r24"] >= target:
                cost = lo["mpc_h_day"]; break
            if lo["r24"] < target <= hi["r24"]:
                f = (target - lo["r24"]) / (hi["r24"] - lo["r24"])
                cost = math.exp(math.log(max(lo["mpc_h_day"], 1e-9)) + f * (math.log(hi["mpc_h_day"]) - math.log(max(lo["mpc_h_day"], 1e-9))))
                break
        out[f"{name}_mpc_h_for_99pct"] = cost
        out[f"{name}_r24_at_3e-5"] = next(r["r24"] for r in rows if r["tau"] == 3e-5)
        out[f"{name}_mpc_at_3e-5"] = next(r["mpc_h_day"] for r in rows if r["tau"] == 3e-5)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--sp", default="0.25,0.5,1.0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    enc = load(a.cache)
    scor = enc["tca_s"] >= H_S
    n_days = (enc["meta"]["hours"] * 3600 - H_S) / 86400
    m, Pa, Pb, ok = geometry(enc, scor)
    max_lead = np.minimum(enc["tca_s"][scor][ok], H_S)
    res = []
    for biased in (False, True):
        for sp in [float(x) for x in a.sp.split(",")]:
            r = run_case(m, Pa, Pb, max_lead, n_days, sp, biased, a.seed)
            res.append(r)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                              if k not in ("two_stage", "public")}), flush=True)
    p = pathlib.Path(__file__).resolve().parent / "results" / (pathlib.Path(a.cache).stem + "__measured.json")
    p.write_text(json.dumps({"cache_meta": enc["meta"], "age0_h": AGE0_H, "cases": res}, indent=1))
    print(p)
