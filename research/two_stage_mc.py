r"""Monte Carlo version of the two-stage experiment (circular Pc model).

The deterministic analysis feeds every regime the *true* miss distance and
only changes the sigma, so nobody ever misses a dangerous encounter or raises
a false alarm because of a bad estimate: recall is 100% by construction. Here
each regime instead sees an *estimated* encounter-plane miss vector, the true
one plus a Gaussian error with that regime's combined sigma, redrawn at every
screening update (every 8 h from 72 h before TCA, like CDM updates).

Regimes (combined 1-sigma, km): full sharing 0.5; no cooperation sqrt(2)*SIG_PUB(L);
stage-1 alert by one operator sqrt(SIG_P^2 + SIG_PUB(L)^2) (each operator draws
its own estimate); public-only trigger sqrt(2)*SIG_PUB(L). In the two-stage
regimes the secure step returns a full-sharing-quality estimate at the alert
update and every later one, and the operators act when it reaches T.

"Dangerous" = true miss below the distance where the full-sharing Pc on the
true miss is T (1.02 km for HBR 20 m). "Acted" = some estimate before TCA
reached T. Reported per day over the scored window, with bootstrap CIs.
"""
import argparse, json, math, pathlib
import numpy as np
from two_stage import load, sig_pub, SIG_P, SIG_FAR, H_S, T, HBR, MPC_PAIRS_PER_HOUR, TAUS

UPDATE_S = 8 * 3600.0


def pc_circ(d, s):
    return np.minimum(HBR ** 2 / (2 * s * s) * np.exp(-d * d / (2 * s * s)), 1.0)


def run(enc, seed=0, far=SIG_FAR, taus=TAUS, n_boot=1000):
    rng = np.random.default_rng(seed)
    scor = enc["tca_s"] >= H_S
    d_true = enc["miss_km"][scor]
    n = d_true.size
    n_days = (enc["meta"]["hours"] * 3600.0 - H_S) / 86400.0
    leads = np.arange(H_S, 0, -UPDATE_S)        # 72, 64, ..., 8 h before TCA
    s_full = math.sqrt(2) * SIG_P
    d_thr = math.sqrt(-2 * s_full ** 2 * math.log(T * 2 * s_full ** 2 / HBR ** 2))
    danger = d_true < d_thr

    def est(sig):                                # (n, len(leads)) estimated miss
        sig = np.broadcast_to(sig, (len(leads),))
        ex = rng.normal(size=(n, len(leads))) * sig
        ey = rng.normal(size=(n, len(leads))) * sig
        return np.hypot(d_true[:, None] + ex, ey), sig

    sp = sig_pub(leads, far=far)
    full_d, _ = est(np.full(len(leads), s_full))
    pc_full = pc_circ(full_d, s_full)
    nc_d, nc_s = est(math.sqrt(2) * sp)
    pc_nc = pc_circ(nc_d, nc_s)
    s1 = np.sqrt(SIG_P ** 2 + sp ** 2)
    a_d, _ = est(s1); b_d, _ = est(s1)
    pc_s1 = np.maximum(pc_circ(a_d, s1), pc_circ(b_d, s1))

    def first_idx(mask):                        # first update index with mask, or -1
        anyv = mask.any(axis=1)
        return np.where(anyv, np.argmax(mask, axis=1), -1)

    def summarise(act_idx, secure_evals=None):
        acted = act_idx >= 0
        lead = np.where(acted, leads[np.maximum(act_idx, 0)], 0.0) / 3600.0
        di = np.nonzero(danger)[0]
        bi = rng.integers(0, len(di), size=(n_boot, len(di)))
        rec_b = acted[di][bi].mean(axis=1)
        out = {"recall": float(acted[danger].mean()),
               "recall_ci95": [float(np.percentile(rec_b, 2.5)), float(np.percentile(rec_b, 97.5))],
               "recall_24h": float((lead[danger] >= 24).mean()),
               "median_lead_h": float(np.median(lead[danger])),
               "false_actions_per_day": float((acted & ~danger).sum() / n_days),
               "missed_per_day": float((~acted & danger).sum() / n_days)}
        if secure_evals is not None:
            out["secure_evals_per_day"] = float(secure_evals / n_days)
            out["mpc_hours_per_day"] = float(secure_evals / n_days / MPC_PAIRS_PER_HOUR)
        return out

    res = {"n_scored": int(n), "dangerous_per_day": float(danger.sum() / n_days),
           "danger_miss_km": d_thr, "updates_h": (leads / 3600).tolist(),
           "full_sharing": summarise(first_idx(pc_full >= T)),
           "no_cooperation": summarise(first_idx(pc_nc >= T)),
           "two_stage": [], "public_trigger": []}
    k_idx = np.arange(len(leads))[None, :]
    for tau in taus:
        for name, pc_trig in (("two_stage", pc_s1), ("public_trigger", pc_nc)):
            trig = first_idx(pc_trig >= tau)
            after = (trig[:, None] >= 0) & (k_idx >= trig[:, None])
            act = first_idx(after & (pc_full >= T))
            evals = after.sum()
            row = summarise(act, evals)
            row["tau_pre"] = tau
            row["alerts_per_day"] = float((trig >= 0).sum() / n_days)
            res[name].append(row)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--far", type=float, default=3.0)
    a = ap.parse_args()
    enc = load(a.cache)
    runs = [run(enc, seed=s, far=a.far / math.sqrt(2)) for s in range(a.seeds)]
    p = pathlib.Path(__file__).resolve().parent / "results" / (pathlib.Path(a.cache).stem + f"__mc_circular_far{a.far:g}.json")
    p.write_text(json.dumps({"cache_meta": enc["meta"], "seeds": runs}, indent=1))
    r = runs[0]
    print(p)
    f = lambda d: {k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in d.items() if k != "recall_ci95"}
    print("dangerous/day", r["dangerous_per_day"])
    print("full", f(r["full_sharing"])); print("nocoop", f(r["no_cooperation"]))
    for key in ("two_stage", "public_trigger"):
        print(key)
        for row in r[key]:
            print(f(row))
    sd = lambda key, i, m: np.std([x[key][i][m] for x in runs])
    print("seed std of two_stage recall at each tau:", [round(sd("two_stage", i, "recall"), 4) for i in range(len(TAUS))])
