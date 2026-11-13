"""Measured TLE error vs age from a Space-Track gp_history file.

    python research/tle_error_fit.py data/historical/gp_history_week.tle

Every element set is propagated to the epochs of the same object's later
sets (up to 72 h later) and compared there with the later set, in the later
set's RTN frame. Robust sigma = 1.4826 x MAD per age bin, Starlink and all
others separately (satcollision.tle_history.fit_tle_errors). Also counts
element sets that jump more than 5 robust sigma from the previous one
(likely manoeuvres). Writes research/results/tle_error_<file>.json.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satcollision.tle_history import TLEHistory, detect_jumps, fit_tle_errors, format_table, group_of  # noqa: E402


def main(path, max_objects=1500):
    t0 = time.time()
    hist = TLEHistory.load(path)
    print(f"{hist.n_sets} element sets, {hist.n_objects} objects, loaded in {time.time() - t0:.1f} s")
    t0 = time.time()
    model = fit_tle_errors(hist, max_objects_per_group=max_objects)
    print(f"fit in {time.time() - t0:.1f} s ({max_objects} objects per group)\n")
    print(format_table(model))
    t0 = time.time()
    jumps = detect_jumps(hist, model, 5.0)
    by_group = {}
    for j in jumps:
        g = group_of(j["name"])
        by_group[g] = by_group.get(g, 0) + 1
    print(f"\njumps > 5 sigma: {len(jumps)} ({by_group}) in {time.time() - t0:.1f} s")
    out = ROOT / "research" / "results" / f"tle_error_{Path(path).stem}.json"
    out.write_text(json.dumps({"file": Path(path).name, "n_sets": hist.n_sets, "n_objects": hist.n_objects,
                               "max_objects_per_group": max_objects, "model": model.to_dict(),
                               "jumps_over_5_sigma": by_group}, indent=1), encoding="utf-8")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
