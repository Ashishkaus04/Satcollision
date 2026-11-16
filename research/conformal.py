"""Conformal calibration of the error models (RESULTS.md section 17).

    PYTHONPATH=src python research/conformal.py <gp_history files...>

A. TLE labels, four weeks (last window's split): models trained on days up to
   split - 3, calibrated (per age bin and axis, error_learning.conformal_scales)
   on [split - 3, split), tested on errors after the split. Along-track 1- and
   2-sigma coverage and Gaussian NLL, before and after, for the public table,
   the public model and each member's owner model.
B. Precise labels (Sentinel-1, Swarm, COSMIC-2): the public model recalibrated
   with only the owner's calibration numbers (computed on its precise orbits
   before the split) against a full owner model trained on precise labels.
C. Screening, last window: Foster Pc with the public model for every object,
   before and after conformal scaling, at the fixed threshold and at equal
   false-alarm budgets.
Writes research/results/conformal.json.
"""
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from satcollision import error_learning as el  # noqa: E402
from satcollision import hindcast as hc  # noqa: E402
from satcollision.tle_history import fit_tle_errors, group_of, to_rtn  # noqa: E402
from hindcast_fl import PRECISE, THRESHOLD, table_arrays  # noqa: E402
from hindcast_table import at_budget  # noqa: E402
import precise_vs_tle as pvt  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def metrics(bias, sigma, d):
    ref = el.reference_scale(d.age_h)
    return {"cover1_t": el.coverage(bias, sigma, d.y)[1], "cover2_t": el.coverage(bias, sigma, d.y, 2.0)[1],
            "cover1": el.coverage(bias, sigma, d.y), "nll": el.gaussian_nll(bias, sigma, d.y, ref), "n": len(d)}


def before_after(pred_fn, cal, test):
    bc, sc = pred_fn(cal)
    scales = el.conformal_scales(bc, sc, cal.y, cal.age_h)
    bt, st = pred_fn(test)
    return {"before": metrics(bt, st, test), "after": metrics(bt, el.apply_scales(st, test.age_h, scales), test),
            "scales_t": [float(x) for x in scales[:, 1]]}, scales


def main(files):
    cfg = json.loads((ROOT / "worlds" / "real_leo_history.json").read_text())["fleet"]["operators"]
    hist = hc.load_histories(files, cache_dir=str(ROOT / "cache"))
    owner, ops = hc.owners_of(hist.names, cfg)
    first, last = hist.first_epoch, hist.last_epoch
    split = max(first + 1.5, last - 8.0)
    lo, cal_lo, test_hi = first + 0.5, split - 3.0, last - 1.0
    out = {"split_day": split - first, "A": {}, "B": {}, "C": {}}
    is_sl = np.array([group_of(n) == "starlink" for n in hist.names])

    # ---------------- A ----------------
    table = fit_tle_errors(hc.subset_history(hist, cal_lo), max_objects_per_group=1500, seed=0)
    allr = np.arange(hist.n_objects)
    pub = el.train_local(el.build_samples(hist, allr, lo, cal_lo, seed=2, ages_per_set=2, max_samples=300_000), epochs=30)
    pub_cal = el.build_samples(hist, allr, cal_lo, split, seed=3, ages_per_set=2, max_samples=200_000)

    def table_pred(d):
        sp, md = table_arrays(table, is_sl[d.row], d.age_h)
        return np.zeros_like(sp), np.sqrt(sp ** 2 + md ** 2)

    def model_pred(m):
        return lambda d: el.predict(m, d.X, d.age_h)

    pub_scales = el.conformal_scales(*model_pred(pub)(pub_cal), pub_cal.y, pub_cal.age_h)
    tbl_scales = el.conformal_scales(*table_pred(pub_cal), pub_cal.y, pub_cal.age_h)
    groups = {"unowned": np.nonzero(owner < 0)[0]}
    groups.update({op: np.nonzero(owner == k)[0] for k, op in enumerate(ops)})
    for g, rows in groups.items():
        test = el.build_samples(hist, rows, split, test_hi, seed=4, max_samples=50_000)
        r = {}
        for name, fn, sc in (("public_table", table_pred, tbl_scales), ("public_model", model_pred(pub), pub_scales)):
            b, s = fn(test)
            r[name] = {"before": metrics(b, s, test), "after": metrics(b, el.apply_scales(s, test.age_h, sc), test)}
        if g != "unowned":
            k = ops.index(g)
            own = el.train_local(el.build_samples(hist, rows, lo, cal_lo, seed=0, max_samples=300_000), epochs=30)
            cal = el.build_samples(hist, rows, cal_lo, split, seed=5, max_samples=100_000)
            r["owner_model"], _ = before_after(model_pred(own), cal, test)
        out["A"][g] = r
        log(f"A {g:9s} " + " | ".join(f"{n}: cover1_t {v['before']['cover1_t']:.2f}->{v['after']['cover1_t']:.2f} "
                                      f"nll {v['before']['nll']:.2f}->{v['after']['nll']:.2f}" for n, v in r.items()))

    # ---------------- B ----------------
    precise = sorted(glob.glob(str(ROOT / "data" / "precise" / "*.npz")))
    P, mission_of = pvt.load_precise(precise)
    pub_b = el.train_local(el.build_samples(hist, allr, lo, cal_lo, seed=2, ages_per_set=2, max_samples=300_000),
                           epochs=30, seed=1)
    per_mission = {}
    for norad in np.unique(P["norad"]):
        m = P["norad"] == norad
        jd_k, r_k, v_k = P["jd"][m], P["r_teme_km"][m], P["v_teme_km_s"][m]
        row = hist.row_of[int(norad)]
        sets = np.arange(hist.start[row], hist.start[row + 1])
        rng = np.random.default_rng(int(norad))
        s = np.repeat(sets, 12)
        age = np.exp(rng.uniform(math.log(0.1), math.log(72.0), len(s)))
        t = hist.epochs[s] + age / 24.0
        r_pred, _, ok1 = hc.propagate_pairs(hist, s, t)
        r_true, v_true, ok3 = pvt.hermite(jd_k, r_k, v_k, t)
        ok = ok1 & ok3 & (t < hist.last_epoch - 0.5)
        y = to_rtn(r_pred[ok] - r_true[ok], r_true[ok], v_true[ok])
        d = el.ErrorSamples(el.features(hist, s[ok], age[ok]), y, age[ok], np.full(int(ok.sum()), row))
        per_mission.setdefault(mission_of[int(norad)], []).append((d, t[ok]))
    for mission, parts in per_mission.items():
        def pick(f):
            return el.ErrorSamples.concat([d.subset(f(t)) for d, t in parts])
        tr, cal, test = pick(lambda t: t < cal_lo), pick(lambda t: (t >= cal_lo) & (t < split)), pick(lambda t: t >= split)
        own = el.train_local(tr, epochs=60)
        r = {"n_train": len(tr), "n_cal": len(cal), "n_test": len(test)}
        r["public_model"], _ = before_after(model_pred(pub_b), cal, test)
        r["owner_model_precise"], _ = before_after(model_pred(own), cal, test)
        out["B"][mission] = r
        log(f"B {mission:9s} n_cal {len(cal)} | " + " | ".join(
            f"{n}: cover1_t {v['before']['cover1_t']:.2f}->{v['after']['cover1_t']:.2f} nll {v['before']['nll']:.2f}->{v['after']['nll']:.2f}"
            for n, v in r.items() if isinstance(v, dict)))

    # ---------------- C ----------------
    f = sorted(glob.glob(str(ROOT / "research" / "cache" / f"hindcast_w4_*_{split:.3f}_*.npz")))
    if f:
        z = np.load(f[0])
        enc = {k: z[k] for k in z.files if not k.startswith(("asm_", "err_"))}
        asm = {k[4:]: z[k] for k in z.files if k.startswith("asm_")}
        err = {k[4:]: z[k] for k in z.files if k.startswith("err_")}
        n_days = float(min(enc["tca_jd"].max() - enc["tca_jd"].min(), 4.0))
        dangerous = hc.true_pc(enc, PRECISE) >= THRESHOLD
        Xo = el.features(hist, err["set"], err["age_h"])
        b, s = el.predict(pub, Xo, err["age_h"])
        s_c = el.apply_scales(s, err["age_h"], pub_scales)
        sp, md = table_arrays(table, is_sl[asm["other_row"]], err["age_h"])
        st = np.sqrt(sp ** 2 + md ** 2)
        st_c = el.apply_scales(st, err["age_h"], tbl_scales)
        for name, sig, bias in (("public_model", s, b), ("public_model_conformal", s_c, b),
                                ("public_table", st, None), ("public_table_conformal", st_c, None)):
            pc = hc.assess_pc(enc, asm, err, sig, PRECISE, bias_rtn=bias)
            r = hc.score(asm, pc, dangerous, n_days, THRESHOLD, enc_weight=enc["weight"])
            sweep = [{"fa_day": hc.score(asm, pc, dangerous, n_days, float(T), enc_weight=enc["weight"])["false_alarms_per_day"],
                      "caught": hc.score(asm, pc, dangerous, n_days, float(T), enc_weight=enc["weight"])["caught_frac"]}
                     for T in np.logspace(-8, -2, 19)]
            out["C"][name] = {"at_1e-4": r, "caught_at_100fa": at_budget(sweep, 100), "caught_at_300fa": at_budget(sweep, 300)}
            log(f"C {name:24s} at 1e-4: caught {r['caught_frac']:.3f} (24h {r['caught_24h_frac']:.3f}) FA/day "
                f"{r['false_alarms_per_day']:.0f} | caught at 100 FA/day {out['C'][name]['caught_at_100fa']:.3f}")
    p = ROOT / "research" / "results" / "conformal.json"
    p.write_text(json.dumps(out, indent=1))
    log("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
