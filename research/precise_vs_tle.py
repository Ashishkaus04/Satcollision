"""How wrong is the TLE-derived truth? Sentinel-1C/1D precise orbits as ground truth.

    PYTHONPATH=src python research/precise_vs_tle.py data/precise/sentinel1_2026-10-12_2026-11-13.npz \
        data/historical/gp_history_oct16a.tle ... data/historical/gp_history_week.tle

Every hindcast number (RESULTS.md section 10) scores a public prediction
against the *next published TLE* (hindcast.truth_weights), and the error
models are trained on that same TLE-vs-later-TLE difference. Here, for two
satellites with independent centimetre-level orbits (ESA's Sentinel-1 POD),
we measure:

1. the real error of the public prediction (prediction - precise orbit) by
   age, against the TLE-derived error (prediction - next TLE) at the same
   instants: does TLE consistency understate the error, and by how much?
2. the error of the TLE-derived truth itself (next TLE - precise orbit);
3. what an owner with precise labels gains: a model trained on its own
   precise-orbit errors against the public model (TLE labels, every object)
   and against the same owner trained on TLE labels, all scored on held-out
   precise-label errors after the split day.

Writes research/results/precise_vs_tle.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satcollision import error_learning as el  # noqa: E402
from satcollision import hindcast as hc  # noqa: E402
from satcollision.tle_history import to_rtn  # noqa: E402

BINS = [(0, 3), (3, 6), (6, 12), (12, 24), (24, 36), (36, 48), (48, 72)]


def hermite(jd_k, r_k, v_k, jd):
    """Cubic Hermite interpolation of position (and velocity) on a 60 s grid;
    NaN where the grid has a gap around jd."""
    idx = np.searchsorted(jd_k, jd) - 1
    ok = (idx >= 0) & (idx < len(jd_k) - 1)
    i = np.clip(idx, 0, len(jd_k) - 2)
    h = (jd_k[i + 1] - jd_k[i]) * 86400.0
    ok &= h < 90.0
    s = ((jd - jd_k[i]) * 86400.0 / np.where(h > 0, h, 1))[:, None]
    h = h[:, None]
    p0, p1, m0, m1 = r_k[i], r_k[i + 1], v_k[i] * h, v_k[i + 1] * h
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    r = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
    dv = ((6 * s**2 - 6 * s) * p0 + (3 * s**2 - 4 * s + 1) * m0 + (-6 * s**2 + 6 * s) * p1 + (3 * s**2 - 2 * s) * m1) / h
    r[~ok] = np.nan
    dv[~ok] = np.nan
    return r, dv, ok


def robust(x):
    return float(1.4826 * np.median(np.abs(x - np.median(x)))), float(np.median(x))


def load_precise(npz_list):
    """Several precise-orbit files as one; mission name from each file name."""
    parts = {k: [] for k in ("norad", "jd", "r_teme_km", "v_teme_km_s")}
    mission = {}
    for f in npz_list:
        z = np.load(f)
        for k in parts:
            parts[k].append(z[k])
        for n in np.unique(z["norad"]):
            mission[int(n)] = Path(f).stem.split("_")[0]
    return {k: np.concatenate(v) for k, v in parts.items()}, mission


def main(npz, files, split_day=19.9):
    npz_list = npz.split(",")
    P, mission_of = load_precise(npz_list)
    hist = hc.load_histories(files, cache_dir=str(ROOT / "cache"))
    first = hist.first_epoch
    split = first + split_day
    out = {"precise_files": [Path(f).name for f in npz_list], "satellites": {}, "split_day": split_day}
    all_samples = {}
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
        r_tle, v_tle, ok2 = hc.truth_pairs(hist, np.full(len(s), row), t)
        r_true, v_true, ok3 = hermite(jd_k, r_k, v_k, t)
        ta, tb, _ = hc.truth_weights(hist, np.full(len(s), row), t)
        ok = ok1 & ok2 & ok3 & (ta != s) & (tb != s) & (t < hist.last_epoch - 0.5)
        s, age, t = s[ok], age[ok], t[ok]
        e_precise = to_rtn(r_pred[ok] - r_true[ok], r_true[ok], v_true[ok])
        e_tle = to_rtn(r_pred[ok] - r_tle[ok], r_tle[ok], v_tle[ok])
        e_truth = to_rtn(r_tle[ok] - r_true[ok], r_true[ok], v_true[ok])
        # at the set's own epoch: the TLE fit error itself
        r0, _, okk = hc.propagate_pairs(hist, sets, hist.epochs[sets])
        p0, pv0, okp = hermite(jd_k, r_k, v_k, hist.epochs[sets])
        g = okk & okp
        e_fit = to_rtn(r0[g] - p0[g], p0[g], pv0[g])
        sat = {"name": hist.names[row], "mission": mission_of[int(norad)], "n_sets": int(len(sets)), "n_samples": int(ok.sum()),
               "fit_error_at_epoch": {"sigma_rtc_km": [robust(e_fit[:, k])[0] for k in range(3)],
                                      "median_rtc_km": [robust(e_fit[:, k])[1] for k in range(3)], "n": int(g.sum())},
               "by_age": []}
        print(f"\n{hist.names[row]} ({norad}): {len(sets)} sets, {ok.sum()} samples; "
              f"fit error at epoch, robust sigma R/T/C {np.round(sat['fit_error_at_epoch']['sigma_rtc_km'], 3)} km, "
              f"median T {sat['fit_error_at_epoch']['median_rtc_km'][1]:.3f} km")
        print("  age (h)    n   sigma T: precise  TLE-truth  ratio | median T: precise TLE-truth | truth's own sigma T")
        for lo, hi in BINS:
            b = (age >= lo) & (age < hi)
            if b.sum() < 10:
                continue
            sp, mp = robust(e_precise[b, 1]); st, mt = robust(e_tle[b, 1]); sx, _ = robust(e_truth[b, 1])
            spr, _ = robust(e_precise[b, 0]); str_, _ = robust(e_tle[b, 0])
            sat["by_age"].append({"age_h": [lo, hi], "n": int(b.sum()), "sigma_t_precise": sp, "sigma_t_tle": st,
                                  "median_t_precise": mp, "median_t_tle": mt, "sigma_t_truth_err": sx,
                                  "sigma_r_precise": spr, "sigma_r_tle": str_})
            print(f"  {lo:>2}-{hi:<3} {b.sum():5d}   {sp:8.3f}  {st:9.3f}  {sp / max(st, 1e-9):5.2f} | "
                  f"{mp:8.3f} {mt:9.3f} | {sx:8.3f}")
        out["satellites"][str(int(norad))] = sat
        all_samples[int(norad)] = (s, age, t, e_precise, e_tle)

    # ---- an owner with precise labels, per mission: does it beat the public model? ----
    def samples(err_key, sel_fn, sats):
        parts = []
        for norad in sats:
            s, age, t, e_p, e_t = all_samples[norad]
            sel = sel_fn(t)
            y = e_p if err_key == "precise" else e_t
            parts.append(el.ErrorSamples(el.features(hist, s[sel], age[sel]), y[sel], age[sel],
                                         np.full(int(sel.sum()), hist.row_of[norad])))
        return el.ErrorSamples.concat(parts)

    pub_train = el.build_samples(hist, np.arange(hist.n_objects), first + 0.5, split, seed=2, ages_per_set=2,
                                 max_samples=300_000)
    publics = [el.train_local(pub_train, epochs=30, seed=seed) for seed in range(3)]
    out["owner_test"] = {}
    for mission in sorted(set(mission_of[n] for n in all_samples)):
        sats = [n for n in all_samples if mission_of[n] == mission]
        tr_p = samples("precise", lambda t: t < split, sats)
        tr_t = samples("tle", lambda t: t < split, sats)
        te_p = samples("precise", lambda t: t >= split, sats)
        print(f"\n{mission}: precise-label owner of {len(sats)} satellites: train {len(tr_p)}, test {len(te_p)} samples")
        res = {}
        for seed in range(3):
            own_precise = el.train_local(tr_p, epochs=60, seed=seed)
            own_tle = el.train_local(tr_t, epochs=60, seed=seed)
            dp = el.train_owner_dp(tr_p, target_epsilon=2.0, epochs=20, batch=64, seed=seed)
            for k, mdl in (("public_model_tle_labels", publics[seed]), ("owner_model_precise_labels", own_precise),
                           ("owner_model_tle_labels", own_tle), ("owner_model_precise_dp_eps2", dp)):
                res.setdefault(k, []).append(el.evaluate(mdl, te_p))
        summary = {k: {"nll": float(np.mean([x["nll"] for x in v])), "nll_sd": float(np.std([x["nll"] for x in v])),
                       "cover1_t": float(np.mean([x["cover1"][1] for x in v])),
                       "mae_t_corrected_km": float(np.mean([x["mae_t_corrected_km"] for x in v]))}
                   for k, v in res.items()}
        for k, v in summary.items():
            print(f"  {k:30s} NLL {v['nll']:.3f} +- {v['nll_sd']:.3f}  along-track 1-sigma coverage {v['cover1_t']:.2f}  "
                  f"median |error| after bias correction {v['mae_t_corrected_km']:.3f} km")
        out["owner_test"][mission] = {"satellites": sats, "train_samples": len(tr_p), "test_samples": len(te_p),
                                      "models": summary, "raw_median_abs_t_km": float(np.median(np.abs(te_p.y[:, 1])))}
    tag = "_".join(sorted(set(mission_of.values())))
    p = ROOT / "research" / "results" / ("precise_vs_tle.json" if tag == "sentinel1" else f"precise_vs_tle_{tag}.json")
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
