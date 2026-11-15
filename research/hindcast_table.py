"""Summarise hindcast results at equal false-alarm budgets.

    python research/hindcast_table.py research/results/hindcast_<name>_s*.json
"""
import json, sys
import numpy as np

SCHEMES = ["baseline", "circular_meas", "foster", "foster_bias", "foster_local", "foster_fl", "foster_fl_sqrt", "foster_central",
           "foster_pub", "foster_pub_local", "foster_pub_fl", "foster_pub_fl_sqrt", "foster_pub_central", "foster_pub_fl_dp", "foster_pub_owner"]


def at_budget(sweep, budget, c="caught", f="fa_day"):
    """Caught fraction when the threshold is set so false alarms/day = budget
    (log-interpolated along the threshold sweep; NaN if out of range)."""
    fa = np.array([p[f] for p in sweep]); cv = np.array([p[c] for p in sweep])
    o = np.argsort(fa); fa, cv = fa[o], cv[o]
    if budget < fa[0] or budget > fa[-1]:
        return float("nan")
    return float(np.interp(np.log(budget), np.log(np.maximum(fa, 1e-9)), cv))


def summarise(runs, budgets=(30, 100, 300, 1000)):
    out = {}
    for s in SCHEMES:
        if s not in runs[0]["results"]:
            continue
        r = {k: float(np.mean([x["results"][s]["all"][k] for x in runs])) for k in
             ("caught_frac", "caught_24h_frac", "median_warning_h", "false_alarms_per_day", "warning_hours_per_dangerous")}
        mm = {k: float(np.mean([x["results"][s]["member_vs_member"][k] for x in runs])) for k in
              ("caught_frac", "caught_24h_frac", "false_alarms_per_day")}
        r["all_at_budget"] = {b: float(np.nanmean([at_budget(x["sweep"][s], b) for x in runs])) for b in budgets}
        r["all24_at_budget"] = {b: float(np.nanmean([at_budget(x["sweep"][s], b, "caught_24h") for x in runs])) for b in budgets}
        r["mm_at_budget"] = {b: float(np.nanmean([at_budget(x["sweep"][s], b / 10, "mm_caught", "mm_fa_day") for x in runs])) for b in budgets}
        r["mm24_at_budget"] = {b: float(np.nanmean([at_budget(x["sweep"][s], b / 10, "mm_caught_24h", "mm_fa_day") for x in runs])) for b in budgets}
        r["mm"] = mm
        out[s] = r
    return out


if __name__ == "__main__":
    runs = [json.load(open(p)) for p in sys.argv[1:]]
    S = summarise(runs)
    print(f"{len(runs)} run(s); dangerous/day {runs[0]['dangerous_per_day']:.0f}; test days {runs[0]['n_test_days']:.2f}")
    print(f"{'scheme':15s} {'caught':>7s} {'>=24h':>6s} {'warn h':>6s} {'FA/day':>7s} | caught at FA/day 30/100/300/1000 | >=24h at same | member-member at FA/day 3/10/30/100")
    for s, r in S.items():
        print(f"{s:15s} {r['caught_frac']:7.3f} {r['caught_24h_frac']:6.3f} {r['median_warning_h']:6.1f} {r['false_alarms_per_day']:7.0f} | "
              + " ".join(f"{v:.3f}" for v in r["all_at_budget"].values()) + " | "
              + " ".join(f"{v:.3f}" for v in r["all24_at_budget"].values()) + " | "
              + " ".join(f"{v:.3f}" for v in r["mm_at_budget"].values()))
