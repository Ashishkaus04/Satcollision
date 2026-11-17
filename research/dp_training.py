"""Better DP training for owner models (RESULTS.md section 21).

    PYTHONPATH=src python research/dp_training.py <gp_history files...>

Per member fleet, models trained on errors before the split (the last window of
section 10c) and scored on the fleet's errors after it (held-out NLL, along-track
1-sigma coverage):
  no_dp             the owner model without DP (section 10b)
  sample_eps2_old   DP-SGD per training sample, epsilon 2, 5 epochs, clip 1 (sections 10b to 18)
  sample_eps2_tuned the same guarantee with the training tuned on a validation split
                    of the fleet's own satellites before the split (epochs x clip grid)
  sat_eps8, sat_eps2  satellite-level DP: one satellite's whole data is the protected
                    unit (error_learning.train_owner_dp_satellite)
Writes research/results/dp_training.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satcollision import error_learning as el  # noqa: E402
from satcollision import hindcast as hc  # noqa: E402

GRID = [(5, 1.0), (10, 0.5), (20, 0.5), (20, 2.0), (40, 1.0)]   # (epochs, clip) for sample-level eps 2


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ev(m, d):
    r = el.evaluate(m, d)
    return {"nll": r["nll"], "cover1_t": r["cover1"][1]}


def main(files):
    cfg = json.loads((ROOT / "worlds" / "real_leo_history.json").read_text())["fleet"]["operators"]
    hist = hc.load_histories(files, cache_dir=str(ROOT / "cache"))
    owner, ops = hc.owners_of(hist.names, cfg)
    first, last = hist.first_epoch, hist.last_epoch
    split = max(first + 1.5, last - 8.0)
    lo = first + 0.5
    out = {"fleets": {}}
    for k, op in enumerate(ops):
        rows = np.nonzero(owner == k)[0]
        data = el.build_samples(hist, rows, lo, split, seed=0, max_samples=300_000)
        test = el.build_samples(hist, rows, split, last - 1.0, seed=1, max_samples=50_000)
        t0 = time.time()
        res = {"n_train": len(data), "n_satellites": int(len(np.unique(data.row)))}
        res["no_dp"] = ev(el.train_local(data, epochs=30, seed=0), test)
        res["sample_eps2_old"] = ev(el.train_owner_dp(data, target_epsilon=2.0, epochs=5, seed=0), test)
        # tune epochs x clip on a validation split of the fleet's satellites (before the split)
        tr, va = el.split_by_object(data, 0.2, seed=5)
        best = None
        for epochs, clip in GRID:
            m = el.train_owner_dp(tr, target_epsilon=2.0, epochs=epochs, max_grad_norm=clip, seed=0)
            v = el.val_nll(m, va)
            if best is None or v < best[0]:
                best = (v, epochs, clip)
        _, ep, cl = best
        res["sample_eps2_tuned"] = {**ev(el.train_owner_dp(data, target_epsilon=2.0, epochs=ep, max_grad_norm=cl, seed=0), test),
                                    "epochs": ep, "clip": cl}
        for eps in (8.0, 2.0):
            m = el.train_owner_dp_satellite(data, target_epsilon=eps, epochs=20, sats_per_step=64, per_sat=32,
                                            clip=1.0, seed=0)
            res[f"sat_eps{eps:g}"] = {**ev(m, test), "noise_multiplier": m.dp["noise_multiplier"],
                                      "sample_rate": m.dp["sample_rate"]}
        out["fleets"][op] = res
        log(f"{op:9s} ({res['n_satellites']} sats, {time.time() - t0:.0f} s): " + " | ".join(
            f"{n} {v['nll']:.2f}/{v['cover1_t']:.2f}" for n, v in res.items() if isinstance(v, dict)))
    p = ROOT / "research" / "results" / "dp_training.json"
    p.write_text(json.dumps(out, indent=1))
    log("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
