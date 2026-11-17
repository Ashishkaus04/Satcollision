"""Debris and non-member objects: what one operator can do alone (RESULTS.md 19).

    PYTHONPATH=src python research/hindcast_debris.py <gp_history files...>

Most dangerous encounters of member satellites are with objects nobody in
the federation flies: debris, rocket bodies and other operators' or dead
payloads. There is no one to share with, so each operator screens them
alone: its own satellite from its precise orbit, the other object from the
public catalogue. What matters is how well it models the public error of
those objects. Four rolling windows (section 10c caches); for each window
the public table, a public model (TLE labels, every object) and its
conformal calibration (section 17) are fitted on data before the split.

Schemes (Pc threshold 1e-4, and at equal false-alarm budgets):
  baseline           circular Pc, 0.5 km
  foster_table       Foster Pc, public error table
  foster_model       Foster Pc, public model (bias + spread)
  foster_model_cal   the same, conformally calibrated per age bin
Split by the other object's kind: debris (name contains DEB), rocket bodies
(R/B), other payloads (everything else not flown by a member).
Writes research/results/hindcast_debris.json.
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
from hindcast_fl import PRECISE, THRESHOLD, table_arrays  # noqa: E402
from hindcast_table import at_budget  # noqa: E402

WINDOWS = {"w4": None, "w4_t7.4": 7.4, "w4_t11.4": 11.4, "w4_t15.4": 15.4}
BUDGETS = [30, 100, 300]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def kind_of(name):
    n = (name or "").upper()
    if " DEB" in n or n.endswith("DEB") or "DEB " in n:
        return "debris"
    if "R/B" in n or "ROCKET" in n:
        return "rocket_body"
    return "payload"


def main(files):
    cfg = json.loads((ROOT / "worlds" / "real_leo_history.json").read_text())["fleet"]["operators"]
    hist = hc.load_histories(files, cache_dir=str(ROOT / "cache"))
    owner, ops = hc.owners_of(hist.names, cfg)
    first, last = hist.first_epoch, hist.last_epoch
    is_sl = np.array([group_of(n) == "starlink" for n in hist.names])
    kinds = np.array([kind_of(n) for n in hist.names])
    out = {"windows": {}}
    for name, td in WINDOWS.items():
        split = max(first + 1.5, last - 8.0) if td is None else first + 0.5 + td
        z = np.load(sorted(glob.glob(str(ROOT / "research" / "cache" / f"hindcast_{name}_*_{split:.3f}_*.npz")))[0])
        enc = {k: z[k] for k in z.files if not k.startswith(("asm_", "err_"))}
        asm = {k[4:]: z[k] for k in z.files if k.startswith("asm_")}
        err = {k[4:]: z[k] for k in z.files if k.startswith("err_")}
        n_days = float(min(enc["tca_jd"].max() - enc["tca_jd"].min(), 4.0))
        dangerous = hc.true_pc(enc, PRECISE) >= THRESHOLD
        other = asm["other_row"]
        t0 = time.time()
        lo, cal_lo = first + 0.5, split - 3.0
        table = fit_tle_errors(hc.subset_history(hist, split), max_objects_per_group=1500, seed=0)
        allr = np.arange(hist.n_objects)
        pub = el.train_local(el.build_samples(hist, allr, lo, cal_lo, seed=2, ages_per_set=2, max_samples=300_000), epochs=30)
        cal = el.build_samples(hist, np.nonzero(owner < 0)[0], cal_lo, split, seed=3, max_samples=100_000)
        bc, sc = el.predict(pub, cal.X, cal.age_h)
        scales = el.conformal_scales(bc, sc, cal.y, cal.age_h)
        log(f"{name}: models {time.time() - t0:.0f} s")
        Xo = el.features(hist, err["set"], err["age_h"])
        b, s = el.predict(pub, Xo, err["age_h"])
        sp, md = table_arrays(table, is_sl[other], err["age_h"])
        schemes = {
            "baseline": hc.assess_pc(enc, asm, err, None, circular_sigma_km=0.5),
            "foster_table": hc.assess_pc(enc, asm, err, np.sqrt(sp ** 2 + md ** 2), PRECISE),
            "foster_model": hc.assess_pc(enc, asm, err, s, PRECISE, bias_rtn=b),
            "foster_model_cal": hc.assess_pc(enc, asm, err, el.apply_scales(s, err["age_h"], scales), PRECISE, bias_rtn=b),
        }
        subsets = {"all_non_member": owner[other] < 0}
        for k in ("debris", "rocket_body", "payload"):
            subsets[k] = (owner[other] < 0) & (kinds[other] == k)
        res = {}
        for sname, pc in schemes.items():
            res[sname] = {}
            for sub, m in subsets.items():
                r = hc.score(asm, pc, dangerous, n_days, THRESHOLD, mask=m, enc_weight=enc["weight"])
                sweep = [{"fa_day": (x := hc.score(asm, pc, dangerous, n_days, float(T), mask=m, enc_weight=enc["weight"]))["false_alarms_per_day"],
                          "caught": x["caught_frac"], "caught_24h": x["caught_24h_frac"]} for T in np.logspace(-8, -2, 19)]
                r["at_budget"] = {B: at_budget(sweep, B) for B in BUDGETS}
                r["at_budget_24h"] = {B: at_budget(sweep, B, "caught_24h") for B in BUDGETS}
                res[sname][sub] = r
            a = res[sname]["all_non_member"]
            log(f"  {sname:17s} non-member: dangerous/day {a['dangerous'] / n_days:.0f} caught {a['caught_frac']:.3f} "
                f"(24h {a['caught_24h_frac']:.3f}) FA/day {a['false_alarms_per_day']:.0f} | at 100 FA/day "
                f"{a['at_budget'][100]:.3f} (24h {a['at_budget_24h'][100]:.3f})")
        # share of all dangerous (member-side) encounters that are with non-members
        mm = np.zeros(len(enc["tca_jd"]), bool)
        e_nm = np.unique(asm["enc"][owner[other] < 0])
        e_all = np.unique(asm["enc"])
        w = enc["weight"]
        res["share_dangerous_non_member"] = float((w[e_nm] * dangerous[e_nm]).sum() / max((w[e_all] * dangerous[e_all]).sum(), 1e-9))
        res["dangerous_per_day_by_kind"] = {k: float((w[np.unique(asm["enc"][subsets[k]])] * dangerous[np.unique(asm["enc"][subsets[k]])]).sum() / n_days)
                                            for k in ("debris", "rocket_body", "payload")}
        out["windows"][name] = res
        log(f"  non-member share of dangerous encounters {res['share_dangerous_non_member']:.2f}; by kind/day "
            f"{ {k: round(v, 1) for k, v in res['dangerous_per_day_by_kind'].items()} }")
    out["summary"] = {}
    for sname in ("baseline", "foster_table", "foster_model", "foster_model_cal"):
        out["summary"][sname] = {}
        for sub in ("all_non_member", "debris", "rocket_body", "payload"):
            rows = [out["windows"][w][sname][sub] for w in out["windows"]]
            out["summary"][sname][sub] = {
                "caught": float(np.mean([r["caught_frac"] for r in rows])),
                "caught_24h": float(np.mean([r["caught_24h_frac"] for r in rows])),
                "fa_day": float(np.mean([r["false_alarms_per_day"] for r in rows])),
                "at_100": float(np.nanmean([r["at_budget"][100] for r in rows])),
                "at_100_24h": float(np.nanmean([r["at_budget_24h"][100] for r in rows])),
                "dangerous_per_day": float(np.mean([r["dangerous"] for r in rows])) / 4.0}
    out["share_dangerous_non_member"] = float(np.mean([out["windows"][w]["share_dangerous_non_member"] for w in out["windows"]]))
    print(json.dumps(out["summary"], indent=1))
    print("share of dangerous encounters that are with non-members:", round(out["share_dangerous_non_member"], 3))
    p = ROOT / "research" / "results" / "hindcast_debris.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    log("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
