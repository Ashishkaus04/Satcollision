"""Hybrid design: public filter + secure Pc for member pairs, with each pair's
filter set by error models.

    PYTHONPATH=src python research/hindcast_hybrid.py <gp_history files...>

Section 16's filter sends a member pair to the secure Pc when its miss,
predicted from both public element sets, is under a fixed radius R. Here the
filter adapts to how wrong each public prediction is likely to be: a pair
passes when the Mahalanobis distance of the public predicted miss, under the
two public predictions' combined covariance (and after their predicted bias),
is at most k, or the miss is under 1 km. The covariance comes from
  * the public error table (section 9),
  * the public model (public TLE labels, every object),
  * owners' models (each member's model of its own satellites),
  * owners' models trained with DP-SGD at epsilon 2.
Sweeping k (or R) traces caught vs secure evaluations a day; schemes are
compared at equal budgets. The secure step itself is as in section 16 (each
owner's precise orbit with 0.354 km of error, 3 draws).
Writes research/results/hindcast_hybrid.json.
"""
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from satcollision import error_learning as el  # noqa: E402
from satcollision import hindcast as hc  # noqa: E402
from satcollision.tle_history import fit_tle_errors, group_of  # noqa: E402
from hindcast_fl import table_arrays  # noqa: E402

PRECISE = 0.354
HBR = 0.02
T = 1e-4
WINDOWS = {"w4": None, "w4_t7.4": 7.4, "w4_t11.4": 11.4, "w4_t15.4": 15.4}
RADII = [2, 3, 5, 7, 10, 15, 20, 30, 50]
KS = [0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4, 5, 7, 10, 15]
BUDGETS = [10_000, 40_000, 180_000, 400_000]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pc_true(m):
    s2 = 2 * PRECISE ** 2
    return np.minimum(HBR ** 2 / (2 * s2) * np.exp(-0.5 * (m ** 2).sum(-1) / s2), 1.0)


def score(passed, ee, L, dangerous, w, n_days, noisy_acts):
    """caught, caught >= 24 h, false alarms/day (secure step with noisy precise orbits), evals/day."""
    evals = float((w[ee] * passed).sum()) / n_days
    nd = max(float(w[dangerous].sum()), 1e-9)
    c, c24, fa = [], [], []
    for act_noisy in noisy_acts:
        act = passed & act_noisy
        lead = np.full(len(w), -1.0)
        np.maximum.at(lead, ee, np.where(act, L, -1.0))
        c.append(float(w[dangerous & (lead > 0)].sum() / nd))
        c24.append(float(w[dangerous & (lead >= 24)].sum() / nd))
        fa.append(float(w[~dangerous & (lead > 0)].sum()) / n_days)
    return {"caught": float(np.mean(c)), "caught_24h": float(np.mean(c24)), "false_alarms_per_day": float(np.mean(fa)),
            "secure_evals_per_day": evals}


def at_budget(curve, budget, key):
    x = np.log([max(p["secure_evals_per_day"], 1) for p in curve])
    o = np.argsort(x)
    if not (x[o][0] <= np.log(budget) <= x[o][-1]):
        return float("nan")
    return float(np.interp(np.log(budget), x[o], np.array([p[key] for p in curve])[o]))


def main(files):
    cfg = json.loads((ROOT / "worlds" / "real_leo_history.json").read_text())["fleet"]["operators"]
    hist = hc.load_histories(files, cache_dir=str(ROOT / "cache"))
    owner, ops = hc.owners_of(hist.names, cfg)
    first, last = hist.first_epoch, hist.last_epoch
    is_sl_row = np.array([group_of(n) == "starlink" for n in hist.names])
    out = {"windows": {}}
    for name, td in WINDOWS.items():
        split = max(first + 1.5, last - 8.0) if td is None else first + 0.5 + td
        f = sorted(glob.glob(str(ROOT / "research" / "cache" / f"hindcast_{name}_*_{split:.3f}_*.npz")))
        z = np.load(f[0])
        w = z["weight"]
        n_days = float(min(z["tca_jd"].max() - z["tca_jd"].min(), 4.0))
        e, side, u, lead = z["asm_enc"], z["asm_side"], z["asm_update_jd"], z["asm_lead_h"]
        key = e.astype(np.int64) * 100000 + np.round((u - u.min()) * 3).astype(np.int64)
        k0 = {k: i for i, k in enumerate(key) if side[i] == 0}
        pairs = [(k0[k], i) for i, k in enumerate(key) if side[i] == 1 and k in k0]
        i0 = np.array([p[0] for p in pairs]); i1 = np.array([p[1] for p in pairs])
        ee = e[i0]; L = lead[i0]
        Pb = z["Pb"][ee]; Pa = z["Pa"][ee]
        err_b = z["err_err_rtn"][i0]; err_a = z["err_err_rtn"][i1]     # side 0 sees b's public error, side 1 a's
        m_true = z["m_true"][ee]
        m_pub = m_true - np.einsum("nk,nkj->nj", err_b, Pb) + np.einsum("nk,nkj->nj", err_a, Pa)
        member_pair = np.zeros(len(w), bool)
        member_pair[np.unique(ee)] = True
        dangerous = (pc_true(z["m_true"]) >= T) & member_pair      # scored over member-pair encounters only
        rng = np.random.default_rng(0)
        noisy = [pc_true(m_true + rng.normal(0, PRECISE * np.sqrt(2), m_true.shape)) >= T for _ in range(3)]
        set_b = z["err_set"][i0]; set_a = z["err_set"][i1]
        age_b = z["err_age_h"][i0]; age_a = z["err_age_h"][i1]
        row_b = z["asm_other_row"][i0]; row_a = z["asm_other_row"][i1]
        log(f"{name}: {len(i0)} member-pair updates, {int(w[np.unique(ee)][dangerous[np.unique(ee)]].sum())} dangerous")

        # ---- covariance / bias per object under each scheme ----
        t0 = time.time()
        table = fit_tle_errors(hc.subset_history(hist, split), max_objects_per_group=1500, seed=0)
        lo = first + 0.5
        pub = el.train_local(el.build_samples(hist, np.arange(hist.n_objects), lo, split, seed=2, ages_per_set=2,
                                              max_samples=300_000), epochs=30, seed=0)
        local, local_dp = {}, {}
        for k, op in enumerate(ops):
            d = el.build_samples(hist, np.nonzero(owner == k)[0], lo, split, seed=0, max_samples=300_000)
            local[k] = el.train_local(d, epochs=30, seed=0)
            local_dp[k] = el.train_owner_dp(d, target_epsilon=2.0, seed=0)
        log(f"  models {time.time() - t0:.0f} s")
        Xb = el.features(hist, set_b, age_b); Xa = el.features(hist, set_a, age_a)

        def sig_bias(scheme):
            outs = []
            for X, age, rows in ((Xb, age_b, row_b), (Xa, age_a, row_a)):
                if scheme == "table":
                    sp, md = table_arrays(table, is_sl_row[rows], age)
                    outs.append((np.zeros_like(sp), np.sqrt(sp ** 2 + md ** 2)))
                elif scheme == "public_model":
                    outs.append(el.predict(pub, X, age))
                else:
                    reg = el.OwnerModels(local if scheme == "owner_models" else local_dp, None)
                    outs.append(reg.predict(X, age, owner[rows]))
            return outs

        res = {"fixed_radius": [score(np.linalg.norm(m_pub, axis=1) < R, ee, L, dangerous, w, n_days, noisy) | {"R": R}
                                for R in RADII]}
        for scheme in ("table", "public_model", "owner_models", "owner_models_dp2"):
            (bb, sb), (ba, sa) = sig_bias(scheme)
            m_c = m_pub + np.einsum("nk,nkj->nj", bb, Pb) - np.einsum("nk,nkj->nj", ba, Pa)   # bias-corrected
            C = hc.plane_cov(Pb, sb) + hc.plane_cov(Pa, sa)
            a, b, d = C[:, 0, 0], C[:, 0, 1], C[:, 1, 1]
            det = np.maximum(a * d - b * b, 1e-12)
            q = (d * m_c[:, 0] ** 2 - 2 * b * m_c[:, 0] * m_c[:, 1] + a * m_c[:, 1] ** 2) / det
            near = np.linalg.norm(m_c, axis=1) < 1.0
            res[scheme] = [score((np.sqrt(q) <= k) | near, ee, L, dangerous, w, n_days, noisy) | {"k": k} for k in KS]
        out["windows"][name] = res
        for s, curve in res.items():
            log(f"  {s:18s} at budgets {BUDGETS}: caught " +
                ", ".join(f"{at_budget(curve, B, 'caught') * 100:.1f}" for B in BUDGETS) + " | >=24h " +
                ", ".join(f"{at_budget(curve, B, 'caught_24h') * 100:.1f}" for B in BUDGETS))
    # ---- summary over windows at equal budgets ----
    out["summary"] = {}
    schemes = list(next(iter(out["windows"].values())))
    for s in schemes:
        out["summary"][s] = {}
        for B in BUDGETS:
            vals = {k: [at_budget(out["windows"][wn][s], B, k) for wn in out["windows"]]
                    for k in ("caught", "caught_24h", "false_alarms_per_day")}
            out["summary"][s][B] = {k: [float(np.nanmean(v)), float(np.nanstd(v))] for k, v in vals.items()}
    print(f"\n{'scheme':18s} " + " ".join(f"{'budget ' + format(B, ','):>26s}" for B in BUDGETS))
    for s in schemes:
        print(f"{s:18s} " + " ".join(
            f"{out['summary'][s][B]['caught'][0] * 100:5.1f}% / {out['summary'][s][B]['caught_24h'][0] * 100:5.1f}% / "
            f"{out['summary'][s][B]['false_alarms_per_day'][0]:4.0f}" for B in BUDGETS))
    print("(caught / caught >= 24 h / false alarms a day, at that many secure evaluations a day; mean of 4 windows)")
    p = ROOT / "research" / "results" / "hindcast_hybrid.json"
    p.write_text(json.dumps(out, indent=1))
    log("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
