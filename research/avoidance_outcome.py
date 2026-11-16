"""Closed loop: what each design's decisions achieve once operators act on them.

    PYTHONPATH=src python research/avoidance_outcome.py <gp_history files...>

Member-vs-member encounters, the four rolling windows. For each design, the
first update at which it decides to act triggers one avoidance burn of one of
the two satellites:
  * size: the planner wants the miss to reach D_SAFE = 1.83 km (the miss at
    which the precise-covariance Pc falls to 1e-6) and plans from what the
    design believes at that moment (its own miss estimate); a tangential burn
    of dv at lead L moves the satellite about 3 dv L along-track by TCA
    (Clohessy-Wiltshire secular term), so dv = shift / (3 L), with a floor
    shift of 0.2 km;
  * effect: the true miss vector moves by that shift in the direction the
    planner believed was "away" (its estimated miss direction). A wrong
    estimate can therefore push the satellites closer.
Outcome per design: expected collisions per day before and after (sum of the
precise-covariance Pc of the true miss, HBR 20 m), burns per day, total
delta-v per day (m/s).

Designs:
  baseline      each operator alone, circular Pc (0.5 km), either side acts
  two_stage     each operator alone with owners' models (Foster Pc), either side acts
  filter20      fixed 20 km public filter + secure Pc (precise orbits with 0.354 km error)
  adaptive      model-set filter (owners' models, DP eps 2) + secure Pc, at ~180,000 evaluations/day
Writes research/results/avoidance_outcome.json.
"""
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satcollision import error_learning as el  # noqa: E402
from satcollision import hindcast as hc  # noqa: E402

PRECISE = 0.354
HBR = 0.02
T = 1e-4
D_SAFE = 1.83
MIN_SHIFT = 0.2
WINDOWS = {"w4": None, "w4_t7.4": 7.4, "w4_t11.4": 11.4, "w4_t15.4": 15.4}
BUDGET = 180_000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pc_precise(m):
    s2 = 2 * PRECISE ** 2
    return np.minimum(HBR ** 2 / (2 * s2) * np.exp(-0.5 * (m ** 2).sum(-1) / s2), 1.0)


def outcome(act, lead_h, m_est, m_true_upd, ee, n_enc, m_true_enc, w, n_days):
    """act, lead_h, m_est, m_true_upd: per update row; ee: approach index of the row.
    Uses each approach's first act (largest lead)."""
    first = np.full(n_enc, -1)
    best_lead = np.full(n_enc, -1.0)
    rows = np.nonzero(act)[0]
    order = rows[np.argsort(-lead_h[rows])]
    for r in order:
        e = ee[r]
        if first[e] < 0:
            first[e] = r
            best_lead[e] = lead_h[r]
    acted = np.nonzero(first >= 0)[0]
    r = first[acted]
    est = m_est[r]
    dist = np.linalg.norm(est, axis=1)
    shift = np.maximum(D_SAFE - dist, MIN_SHIFT)
    direction = est / np.maximum(dist, 1e-9)[:, None]
    lead_s = np.maximum(lead_h[r], 0.5) * 3600.0
    dv = shift / (3.0 * lead_s) * 1000.0                      # m/s
    m_after = m_true_enc.copy()
    m_after[acted] = m_true_enc[acted] + shift[:, None] * direction
    risk_before = float((w * pc_precise(m_true_enc)).sum()) / n_days
    risk_after = float((w * pc_precise(m_after)).sum()) / n_days
    return {"risk_before_per_day": risk_before, "risk_after_per_day": risk_after,
            "risk_removed_frac": 1 - risk_after / max(risk_before, 1e-30),
            "burns_per_day": float(w[acted].sum()) / n_days,
            "delta_v_m_s_per_day": float((w[acted] * dv).sum()) / n_days,
            "median_dv_mm_s": float(np.median(dv) * 1000) if len(dv) else 0.0,
            "made_worse_per_day": float(w[acted][(pc_precise(m_after[acted]) > pc_precise(m_true_enc[acted]))].sum()) / n_days}


def main(files):
    cfg = json.loads((ROOT / "worlds" / "real_leo_history.json").read_text())["fleet"]["operators"]
    hist = hc.load_histories(files, cache_dir=str(ROOT / "cache"))
    owner, ops = hc.owners_of(hist.names, cfg)
    first, last = hist.first_epoch, hist.last_epoch
    out = {"D_SAFE_km": D_SAFE, "windows": {}}
    for name, td in WINDOWS.items():
        split = max(first + 1.5, last - 8.0) if td is None else first + 0.5 + td
        z = np.load(sorted(glob.glob(str(ROOT / "research" / "cache" / f"hindcast_{name}_*_{split:.3f}_*.npz")))[0])
        enc = {k: z[k] for k in z.files if not k.startswith(("asm_", "err_"))}
        asm = {k[4:]: z[k] for k in z.files if k.startswith("asm_")}
        err = {k[4:]: z[k] for k in z.files if k.startswith("err_")}
        w_all = enc["weight"]
        n_enc = len(w_all)
        n_days = float(min(enc["tca_jd"].max() - enc["tca_jd"].min(), 4.0))
        # member pairs: both sides present at the same update
        e, side, u, lead = asm["enc"], asm["side"], asm["update_jd"], asm["lead_h"]
        key = e.astype(np.int64) * 100000 + np.round((u - u.min()) * 3).astype(np.int64)
        k0 = {k: i for i, k in enumerate(key) if side[i] == 0}
        pairs = [(k0[k], i) for i, k in enumerate(key) if side[i] == 1 and k in k0]
        i0 = np.array([p[0] for p in pairs]); i1 = np.array([p[1] for p in pairs])
        ee = e[i0]; L = lead[i0]
        mm = np.zeros(n_enc, bool); mm[np.unique(ee)] = True
        w = np.where(mm, w_all, 0.0)
        m_true = enc["m_true"]
        Pb, Pa = enc["Pb"][ee], enc["Pa"][ee]
        m_pub = m_true[ee] - np.einsum("nk,nkj->nj", err["err_rtn"][i0], Pb) + np.einsum("nk,nkj->nj", err["err_rtn"][i1], Pa)
        # ---- models ----
        t0 = time.time()
        lo = first + 0.5
        local, local_dp = {}, {}
        for k, op in enumerate(ops):
            d = el.build_samples(hist, np.nonzero(owner == k)[0], lo, split, seed=0, max_samples=300_000)
            local[k] = el.train_local(d, epochs=30, seed=0)
            local_dp[k] = el.train_owner_dp(d, target_epsilon=2.0, seed=0)
        pub = el.train_local(el.build_samples(hist, np.arange(hist.n_objects), lo, split, seed=2, ages_per_set=2,
                                              max_samples=300_000), epochs=30, seed=0)
        log(f"{name}: models {time.time() - t0:.0f} s")
        Xo = el.features(hist, err["set"], err["age_h"])
        oth = asm["other_row"]
        b_own, s_own = el.OwnerModels(local, pub).predict(Xo, err["age_h"], owner[oth])
        res = {}
        # baseline and two-stage: per side, either side acts; estimate = that side's predicted miss
        sides = (i0, i1)
        for dname, kwargs in (("baseline", dict(sig=None, bias=None, circ=0.5)),
                              ("two_stage", dict(sig=s_own, bias=b_own, circ=None))):
            pc = hc.assess_pc(enc, asm, err, kwargs["sig"], PRECISE, bias_rtn=kwargs["bias"],
                              circular_sigma_km=kwargs["circ"])
            m_est_all = hc.predicted_miss(enc, asm, err, kwargs["bias"])
            act = np.concatenate([pc[i0] >= T, pc[i1] >= T])
            res[dname] = outcome(act, np.concatenate([L, L]), np.concatenate([m_est_all[i0], m_est_all[i1]]),
                                 None, np.concatenate([ee, ee]), n_enc, m_true, w, n_days)
        # secure designs: decision and estimate from noisy precise orbits
        rng = np.random.default_rng(0)
        m_prec = m_true[ee] + rng.normal(0, PRECISE * np.sqrt(2), (len(ee), 2))
        secure_act = pc_precise(m_prec) >= T
        res["filter20"] = outcome(secure_act & (np.linalg.norm(m_pub, axis=1) < 20.0), L, m_prec, None, ee, n_enc,
                                  m_true, w, n_days)
        # adaptive filter with DP owner models, k chosen for ~BUDGET evaluations/day
        reg = el.OwnerModels(local_dp, None)
        bb, sb = reg.predict(el.features(hist, err["set"][i0], err["age_h"][i0]), err["age_h"][i0], owner[oth[i0]])
        ba, sa = reg.predict(el.features(hist, err["set"][i1], err["age_h"][i1]), err["age_h"][i1], owner[oth[i1]])
        m_c = m_pub + np.einsum("nk,nkj->nj", bb, Pb) - np.einsum("nk,nkj->nj", ba, Pa)
        C = hc.plane_cov(Pb, sb) + hc.plane_cov(Pa, sa)
        a_, b_, d_ = C[:, 0, 0], C[:, 0, 1], C[:, 1, 1]
        q = np.sqrt((d_ * m_c[:, 0] ** 2 - 2 * b_ * m_c[:, 0] * m_c[:, 1] + a_ * m_c[:, 1] ** 2) / np.maximum(a_ * d_ - b_ * b_, 1e-12))
        near = np.linalg.norm(m_c, axis=1) < 1.0
        ks = np.linspace(0.2, 15, 300)
        evals = np.array([float((w_all[ee] * ((q <= k) | near)).sum()) / n_days for k in ks])
        k_sel = ks[np.argmin(np.abs(np.log(np.maximum(evals, 1)) - np.log(BUDGET)))]
        passed = (q <= k_sel) | near
        res["adaptive"] = outcome(secure_act & passed, L, m_prec, None, ee, n_enc, m_true, w, n_days)
        res["adaptive"]["k"] = float(k_sel)
        res["adaptive"]["evals_per_day"] = float((w_all[ee] * passed).sum()) / n_days
        res["no_action"] = {"risk_before_per_day": res["baseline"]["risk_before_per_day"]}
        out["windows"][name] = res
        for dname, r in res.items():
            if dname == "no_action":
                continue
            log(f"  {dname:10s} risk/day {r['risk_before_per_day']:.2e} -> {r['risk_after_per_day']:.2e} "
                f"(removed {r['risk_removed_frac'] * 100:.1f}%) burns/day {r['burns_per_day']:.1f} "
                f"dv/day {r['delta_v_m_s_per_day']:.3f} m/s median {r['median_dv_mm_s']:.2f} mm/s "
                f"made worse/day {r['made_worse_per_day']:.2f}")
    out["summary"] = {}
    for dname in ("baseline", "two_stage", "filter20", "adaptive"):
        vals = {k: [out["windows"][wn][dname][k] for wn in out["windows"]]
                for k in ("risk_removed_frac", "burns_per_day", "delta_v_m_s_per_day", "risk_after_per_day", "made_worse_per_day")}
        out["summary"][dname] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in vals.items()}
    out["summary"]["risk_before_per_day"] = float(np.mean([out["windows"][wn]["baseline"]["risk_before_per_day"] for wn in out["windows"]]))
    print(json.dumps(out["summary"], indent=1))
    p = ROOT / "research" / "results" / "avoidance_outcome.json"
    p.write_text(json.dumps(out, indent=1))
    log("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
