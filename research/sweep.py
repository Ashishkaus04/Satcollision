"""Robustness sweep of the two-stage result over Pc model and tracking-uncertainty assumptions.

One Monte Carlo implementation for every case: per-object RTN covariances with
along-track ratio k projected into each encounter plane (k = 1 is exactly the
circular model). Per-object sigmas: precise SIG_P = sp/sqrt(2); public falls
linearly from far/sqrt(2) at >= 72 h to sp/sqrt(2) at TCA (sp, far are the
combined values quoted). Estimates are redrawn at each 8-hourly update.

Headline metric per case: the secure compute (MPC cluster-hours/day) each
trigger needs to warn 99% of the dangerous encounters that full sharing warns
at least 24 h ahead, interpolated over the tau grid.
"""
import argparse, json, math, pathlib, itertools, time
import numpy as np
from two_stage import load, H_S, T, MPC_PAIRS_PER_HOUR
import aniso

UPDATE_S = 8 * 3600.0
TAUS = [1e-4, 5e-5, 3e-5, 2e-5, 1e-5, 5e-6, 3e-6, 1e-6, 3e-7, 1e-7]


def chol2(C):
    a, b, d = C[..., 0, 0], C[..., 0, 1], C[..., 1, 1]
    l11 = np.sqrt(a); l21 = b / l11; l22 = np.sqrt(np.maximum(d - l21 ** 2, 1e-30))
    return l11, l21, l22


def run_case(geo, n_days, k, sp, far, seed=0):
    m, Sa, Sb, max_lead = geo
    rng = np.random.default_rng(seed)
    leads = np.arange(H_S, 0, -UPDATE_S)
    nl = len(leads)
    p = sp / math.sqrt(2)
    pub = p + np.clip(leads / H_S, 0, 1) * (far / math.sqrt(2) - p)       # (nl,)
    live = leads[None, :] <= max_lead[:, None]                            # update before TCA and inside window

    def cov(sa, sb):            # sa, sb scalars or (nl,) -> (n, nl, 2, 2)
        sa = np.broadcast_to(np.asarray(sa, float), (nl,)); sb = np.broadcast_to(np.asarray(sb, float), (nl,))
        return Sa[:, None] * (sa ** 2)[None, :, None, None] + Sb[:, None] * (sb ** 2)[None, :, None, None]

    def est_pc(C):
        l11, l21, l22 = chol2(C)
        z = rng.standard_normal((2,) + l11.shape)
        e = np.stack([l11 * z[0], l21 * z[0] + l22 * z[1]], axis=-1)
        return np.where(live, aniso.pc_small(m[:, None, :] + e, C), 0.0)

    C_full = cov(p, p)
    danger = aniso.pc_small(m, C_full[:, 0]) >= T
    pc_full = est_pc(C_full)
    pc_nc = est_pc(cov(pub, pub))
    pc_s1 = np.maximum(est_pc(cov(p, pub)), est_pc(cov(pub, p)))

    def first(mask):
        a = mask.any(1); return np.where(a, np.argmax(mask, 1), -1)

    def r24(act):
        lead = np.where(act >= 0, leads[np.maximum(act, 0)], 0.0)
        return float((lead[danger] >= 24 * 3600).mean())

    kk = np.arange(nl)[None, :]
    out = {"k": k, "sigma_precise_km": sp, "sigma_public_far_km": far,
           "dangerous_per_day": float(danger.sum() / n_days),
           "full_r24": r24(first(pc_full >= T)), "nocoop_r24": r24(first(pc_nc >= T))}
    for name, trig_pc in (("two_stage", pc_s1), ("public", pc_nc)):
        rows = []
        for tau in TAUS:
            trig = first(trig_pc >= tau)
            after = (trig[:, None] >= 0) & (kk >= trig[:, None]) & live
            act = first(after & (pc_full >= T))
            rows.append({"tau": tau, "r24": r24(act), "mpc_h_day": float(after.sum() / n_days / MPC_PAIRS_PER_HOUR)})
        out[name] = rows
        target = 0.99 * out["full_r24"]
        cost = None
        for lo, hi in zip(rows, rows[1:]):
            if lo["r24"] < target <= hi["r24"]:
                f = (target - lo["r24"]) / (hi["r24"] - lo["r24"])
                cost = math.exp(math.log(lo["mpc_h_day"]) + f * (math.log(hi["mpc_h_day"]) - math.log(lo["mpc_h_day"])))
                break
            if lo is rows[0] and lo["r24"] >= target:
                cost = lo["mpc_h_day"]; break
        out[f"{name}_mpc_h_for_99pct"] = cost
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--k", default="1,8")
    ap.add_argument("--sp", default="0.5")
    ap.add_argument("--far", default="3.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="sweep")
    a = ap.parse_args()
    enc = load(a.cache)
    scor = enc["tca_s"] >= H_S
    n_days = (enc["meta"]["hours"] * 3600 - H_S) / 86400
    results, geos = [], {}
    for k, sp, far in itertools.product(*[[float(x) for x in s.split(",")] for s in (a.k, a.sp, a.far)]):
        if k not in geos:
            m, Sa, Sb, ok = aniso.geometry(enc, k, scor)
            geos[k] = (m[ok], Sa[ok], Sb[ok], np.minimum(enc["tca_s"][scor][ok], H_S))
        t0 = time.time()
        r = run_case(geos[k], n_days, k, sp, far, a.seed)
        r["runtime_s"] = round(time.time() - t0, 1)
        results.append(r)
        print(json.dumps({x: (round(v, 4) if isinstance(v, float) else v) for x, v in r.items() if x not in ("two_stage", "public")}), flush=True)
    p = pathlib.Path(__file__).resolve().parent / "results" / (pathlib.Path(a.cache).stem + f"__{a.tag}.json")
    p.write_text(json.dumps({"cache_meta": enc["meta"], "cases": results}, indent=1))
    print(p)
