"""Rolling evaluation: the hindcast over several 4-day test windows, each
with its models trained only on the days before it (expanding window).

    python research/hindcast_rolling.py research/results/hindcast_w4_t*_s0.json research/results/hindcast_w4_s0.json

Prints, per window and as mean +- standard deviation over windows, the key
numbers of each step, and writes research/results/hindcast_rolling.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hindcast_table import at_budget  # noqa: E402

STEPS = [("baseline", "Baseline (circular)"), ("foster", "+ Foster, table"),
         ("foster_pub_owner", "+ Foster + owners' models"), ("foster_pub_owner_dp2", "  ... DP eps 2"),
         ("foster_pub", "Public model only"), ("foster_pub_fl_sqrt", "FedAvg"),
         ("foster_pub_local", "Local-only")]


def window_row(r, s):
    a = r["results"][s]["all"]
    return {"caught": a["caught_frac"], "fa_day": a["false_alarms_per_day"],
            "warn_h": a["median_warning_h"],
            "all_at_100": at_budget(r["sweep"][s], 100),
            "mm_at_10": at_budget(r["sweep"][s], 10, "mm_caught", "mm_fa_day")}


def main(paths):
    runs = sorted((json.loads(Path(p).read_text()) for p in paths), key=lambda r: r["test_days"][0])
    out = {"windows": [{"test_days": r["test_days"], "split_days": r["split_days"],
                        "dangerous_per_day": r["dangerous_per_day"]} for r in runs], "steps": {}}
    print("windows (test days):", ", ".join(f"{r['test_days'][0]:.1f}-{r['test_days'][1]:.1f}" for r in runs))
    print(f"{'step':28s} {'caught@1e-4':>18s} {'FA/day@1e-4':>16s} {'caught@100FA':>18s} {'mm caught@10FA':>18s}")
    for s, label in STEPS:
        if any(s not in r["results"] for r in runs):
            continue
        rows = [window_row(r, s) for r in runs]
        agg = {k: (float(np.nanmean([x[k] for x in rows])), float(np.nanstd([x[k] for x in rows])))
               for k in rows[0]}
        out["steps"][s] = {"per_window": rows, "mean_sd": agg}
        f = lambda k, pct=True: (f"{agg[k][0]*100:5.1f} +- {agg[k][1]*100:4.1f}%" if pct
                                 else f"{agg[k][0]:6.0f} +- {agg[k][1]:4.0f}")
        print(f"{label:28s} {f('caught'):>18s} {f('fa_day', False):>16s} {f('all_at_100'):>18s} {f('mm_at_10'):>18s}")
    # paired comparison: owners' models vs baseline and vs FedAvg, per window
    for a, b in (("foster_pub_owner", "baseline"), ("foster_pub_owner", "foster_pub_fl_sqrt"),
                 ("foster_pub_owner_dp2", "foster_pub_fl_sqrt"), ("foster_pub_owner", "foster_pub")):
        if a in out["steps"] and b in out["steps"]:
            d1 = [x["all_at_100"] - y["all_at_100"] for x, y in zip(out["steps"][a]["per_window"], out["steps"][b]["per_window"])]
            d2 = [x["mm_at_10"] - y["mm_at_10"] for x, y in zip(out["steps"][a]["per_window"], out["steps"][b]["per_window"])]
            print(f"{a} - {b}: all@100 {np.round(np.array(d1)*100,1)} pts, mm@10 {np.round(np.array(d2)*100,1)} pts")
            out.setdefault("paired", {})[f"{a}-{b}"] = {"all_at_100": d1, "mm_at_10": d2}
    p = Path(__file__).resolve().parent / "results" / "hindcast_rolling.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p.name)


if __name__ == "__main__":
    main(sys.argv[1:])
