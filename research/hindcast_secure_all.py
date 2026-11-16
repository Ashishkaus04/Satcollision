"""Alternative design: public filter + secure Pc for every close member pair.

    python research/hindcast_secure_all.py

For member-vs-member approaches in the cached hindcasts (the four rolling
windows of section 10c), at every 8-hourly update in the 72 h before TCA:

1. Public filter: the miss predicted from both objects' *public* element sets
   (true miss minus the in-plane part of both public prediction errors) is
   below R km. Everyone can compute this; no private data is used.
2. Every pair that passes goes to the secure step, where both owners input
   their precise orbit and covariance (section 13). Its decision is the
   full-sharing decision: Pc on the precise miss with both precise
   covariances >= 1e-4, which is exactly how "dangerous" is defined here.

So an encounter is caught if it is dangerous and passes the filter at some
update, with the warning time of the first such update. Reports, per R: share of
dangerous member-pair encounters caught and caught >= 24 h ahead, median
warning, and secure evaluations per day (one per pair and update). For
comparison, the two-stage design with owners' models from
results/hindcast_manoeuvre_w4*.json (both operators' alerts).

The secure step's own false alarms (precise-covariance noise) are not modelled
here: with the truth as its input it makes none. Writes
research/results/hindcast_secure_all.json.
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RADII = [1, 2, 3, 5, 10, 20, 30, 50]
PRECISE = 0.354
HBR = 0.02
T = 1e-4
THROUGHPUT_PER_H = {"semi-honest, laptop (measured)": 3600 / 0.00044, "semi-honest, LAN (modelled)": 4.4e6,
                    "semi-honest, WAN (modelled)": 4.5e5, "malicious, laptop (measured)": 3600 / 0.0139}


def pc_true(m):
    s2 = 2 * PRECISE ** 2
    return np.minimum(HBR ** 2 / (2 * s2) * np.exp(-0.5 * (m ** 2).sum(1) / s2), 1.0)


def analyse(path):
    z = np.load(path)
    enc_tca = z["tca_jd"]
    w = z["weight"]
    test_lo = enc_tca.min()
    n_days = float(min(enc_tca.max() - test_lo, 4.0))
    e, side, u, lead = z["asm_enc"], z["asm_side"], z["asm_update_jd"], z["asm_lead_h"]
    err = z["err_err_rtn"]
    P = np.where((side == 0)[:, None, None], z["Pb"][e], z["Pa"][e])
    shift = np.einsum("nk,nkj->nj", err, P)        # in-plane public error of the *other* object
    # member pairs have both sides: join side 0 (error of b) and side 1 (error of a) at the same update
    key = e.astype(np.int64) * 100000 + np.round((u - u.min()) * 3).astype(np.int64)
    k0 = {k: i for i, k in enumerate(key) if side[i] == 0}
    pairs = [(k0[k], i) for i, k in enumerate(key) if side[i] == 1 and k in k0]
    i0 = np.array([p[0] for p in pairs]); i1 = np.array([p[1] for p in pairs])
    ee = e[i0]
    m_true = z["m_true"][ee]
    # m = r_a - r_b; public prediction of both: m_pub = m_true + (err_a) - (err_b) in the plane
    m_pub = m_true - shift[i0] + shift[i1]
    miss_pub = np.linalg.norm(m_pub, axis=1)
    L = lead[i0]
    dangerous = pc_true(z["m_true"]) >= T
    mm_enc = np.unique(ee)
    nd = float(w[mm_enc][dangerous[mm_enc]].sum())
    res = {"test_days": n_days, "member_pair_approaches": float(w[mm_enc].sum()), "dangerous": nd, "radii": {}}
    # the secure step's input: each owner's precise orbit, itself uncertain (0.354 km per object,
    # isotropic): its decision on a noisy precise miss, redrawn at every update (3 seeds)
    rng = np.random.default_rng(0)
    for R in RADII:
        passed = miss_pub < R
        first = np.full(len(enc_tca), -1.0)
        np.maximum.at(first, ee, np.where(passed, L, -1.0))
        d = dangerous & (first > 0)
        evals = float((w[ee] * passed).sum()) / n_days
        c_n, c24_n, fa_n = [], [], []
        for _ in range(3):
            noisy = m_true + rng.normal(0, PRECISE * np.sqrt(2), m_true.shape)   # two objects, 0.354 km each
            act = passed & (pc_true(noisy) >= T)
            fa_lead = np.full(len(enc_tca), -1.0)
            np.maximum.at(fa_lead, ee, np.where(act, L, -1.0))
            c_n.append(float(w[dangerous & (fa_lead > 0)].sum() / max(nd, 1e-9)))
            c24_n.append(float(w[dangerous & (fa_lead >= 24)].sum() / max(nd, 1e-9)))
            fa_n.append(float(w[~dangerous & (fa_lead > 0)].sum()) / n_days)
        res["radii"][R] = {"caught_ideal": float(w[d].sum() / max(nd, 1e-9)),
                           "caught_24h_ideal": float(w[d & (first >= 24)].sum() / max(nd, 1e-9)),
                           "median_warning_h": float(np.median(first[d])) if d.any() else 0.0,
                           "caught": float(np.mean(c_n)), "caught_24h": float(np.mean(c24_n)),
                           "false_alarms_per_day": float(np.mean(fa_n)),
                           "secure_evals_per_day": evals}
    # upper bound: secure on every member-pair update (no filter)
    first = np.full(len(enc_tca), -1.0)
    np.maximum.at(first, ee, L)
    d = dangerous & (first > 0)
    res["no_filter"] = {"caught": float(w[d].sum() / max(nd, 1e-9)),
                        "caught_24h": float(w[d & (first >= 24)].sum() / max(nd, 1e-9))}
    return res


def main():
    files = sorted(glob.glob(str(ROOT / "research" / "cache" / "hindcast_w4*_*.npz")))
    runs = {}
    for f in files:
        name = Path(f).name.split("_")[1] + ("_" + Path(f).name.split("_")[2] if Path(f).name.split("_")[2].startswith("t") else "")
        runs[name] = analyse(f)
        print(name, "dangerous member-pair encounters", runs[name]["dangerous"], "over", runs[name]["test_days"], "days")
    out = {"windows": runs, "throughput_per_h": THROUGHPUT_PER_H, "summary": {}}
    print(f"{'R km':>5} {'caught':>14} {'>=24h':>14} {'FA/day':>8} {'median warn':>12} {'secure evals/day':>17}  hours/day at: " +
          ", ".join(THROUGHPUT_PER_H))
    for R in RADII:
        c = [r["radii"][R]["caught"] for r in runs.values()]
        c24 = [r["radii"][R]["caught_24h"] for r in runs.values()]
        mw = [r["radii"][R]["median_warning_h"] for r in runs.values()]
        ev = [r["radii"][R]["secure_evals_per_day"] for r in runs.values()]
        hours = {k: float(np.mean(ev) / v) for k, v in THROUGHPUT_PER_H.items()}
        fa = [r["radii"][R]["false_alarms_per_day"] for r in runs.values()]
        out["summary"][R] = {"caught": [float(np.mean(c)), float(np.std(c))], "caught_24h": [float(np.mean(c24)), float(np.std(c24))],
                             "false_alarms_per_day": float(np.mean(fa)),
                             "median_warning_h": float(np.mean(mw)), "secure_evals_per_day": float(np.mean(ev)), "hours_per_day": hours}
        print(f"{R:>5} {np.mean(c)*100:7.1f}+-{np.std(c)*100:4.1f}% {np.mean(c24)*100:7.1f}+-{np.std(c24)*100:4.1f}% "
              f"{np.mean(fa):8.1f} {np.mean(mw):10.1f} h {np.mean(ev):17,.0f}  " + ", ".join(f"{h:.2f}" for h in hours.values()))
    nf = [r["no_filter"]["caught_24h"] for r in runs.values()]
    print(f"no filter: caught >=24h {np.mean(nf)*100:.1f}%")
    # the two-stage design with owners' models, both operators' alerts (section 14)
    man = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(ROOT / "research" / "results" / "hindcast_manoeuvre_w4*.json")))]
    if man:
        cmp = {}
        for g in ("one_manoeuvred", "none_manoeuvred", "both_manoeuvred"):
            cmp[g] = {k: float(np.mean([r["foster_owner"][g]["both_sides"][k] for r in man])) for k in ("caught", "caught_24h", "false_alarms_per_day")}
            cmp[g]["dangerous"] = float(np.mean([r["groups"][g]["dangerous"] for r in man]))
        tot = sum(cmp[g]["dangerous"] for g in cmp)
        out["two_stage_owner_models_both_sides"] = {
            "caught": sum(cmp[g]["caught"] * cmp[g]["dangerous"] for g in cmp) / tot,
            "caught_24h": sum(cmp[g]["caught_24h"] * cmp[g]["dangerous"] for g in cmp) / tot,
            "false_alarms_per_day": sum(cmp[g]["false_alarms_per_day"] for g in cmp), "groups": cmp}
        print("two-stage, owners' models, both sides:", {k: round(v, 3) for k, v in out["two_stage_owner_models_both_sides"].items() if k != "groups"})
    p = ROOT / "research" / "results" / "hindcast_secure_all.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
