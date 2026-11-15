"""Dilution-aware rule (Pc >= 1e-4 OR max-Pc >= T2) over the rolling windows.

    python research/hindcast_dilution.py research/results/hindcast_w4_t*_s0.json research/results/hindcast_w4_s0.json

For each window, T2 is set (log-interpolated along the T2 sweep) so the rule
raises a given number of false alarms a day: the circular baseline's, or the
plain rule's own plus 10% / 25%. Reports caught, caught >= 24 h, median
warning and warning hours per dangerous encounter, mean +- sd over windows.
"""
import json
import sys
from pathlib import Path

import numpy as np

KEYS = ("caught_frac", "caught_24h_frac", "median_warning_h", "warning_hours_per_dangerous")


def at_fa(rows, fa):
    x = np.log([max(r["false_alarms_per_day"], 1e-9) for r in rows])
    o = np.argsort(x)
    if not (x[o][0] <= np.log(fa) <= x[o][-1]):
        return {k: float("nan") for k in KEYS}
    return {k: float(np.interp(np.log(fa), x[o], np.array([r[k] for r in rows])[o])) for k in KEYS}


def main(paths):
    runs = sorted((json.loads(Path(p).read_text()) for p in paths), key=lambda r: r["test_days"][0])
    table = {}

    def add(label, vals):
        table.setdefault(label, []).append(vals)

    for r in runs:
        b = r["results"]["baseline"]["all"]
        add("Baseline (circular), Pc >= 1e-4", {**{k: b[k] for k in KEYS}, "fa": b["false_alarms_per_day"]})
        for s, lab in (("foster", "+ Foster, table"), ("foster_pub_owner", "+ Foster + owners' models"),
                       ("foster_pub_owner_dp2", "+ Foster + owners' models, DP eps 2")):
            p = r["results"][s]["all"]
            add(f"{lab}, Pc >= 1e-4", {**{k: p[k] for k in KEYS}, "fa": p["false_alarms_per_day"]})
            for mult in (1.10, 1.25):
                v = at_fa(r["hybrid"][s], p["false_alarms_per_day"] * mult)
                add(f"{lab}, + max-Pc, FA +{int(round((mult - 1) * 100))}%", {**v, "fa": p["false_alarms_per_day"] * mult})
            v = at_fa(r["hybrid"][s], b["false_alarms_per_day"])
            add(f"{lab}, + max-Pc, FA = baseline's", {**v, "fa": b["false_alarms_per_day"]})
    print(f"{len(runs)} windows. mean +- sd")
    print(f"{'rule':58s} {'FA/day':>11s} {'caught':>13s} {'>=24h':>13s} {'median warn':>13s} {'warn h/dangerous':>16s}")
    out = {}
    for lab, rows in table.items():
        m = {k: (float(np.nanmean([x[k] for x in rows])), float(np.nanstd([x[k] for x in rows]))) for k in rows[0]}
        out[lab] = m
        print(f"{lab:58s} {m['fa'][0]:6.0f}+-{m['fa'][1]:3.0f} {m['caught_frac'][0]*100:6.1f}+-{m['caught_frac'][1]*100:4.1f}% "
              f"{m['caught_24h_frac'][0]*100:6.1f}+-{m['caught_24h_frac'][1]*100:4.1f}% {m['median_warning_h'][0]:7.1f}+-{m['median_warning_h'][1]:3.1f}h "
              f"{m['warning_hours_per_dangerous'][0]:9.1f}+-{m['warning_hours_per_dangerous'][1]:3.1f}")
    p = Path(__file__).resolve().parent / "results" / "hindcast_dilution.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p.name)


if __name__ == "__main__":
    main(sys.argv[1:])
