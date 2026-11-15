"""Hindcast on Space-Track history: baseline -> +Foster -> +Foster+FL.

    PYTHONPATH=src python research/hindcast_fl.py data/historical/gp_history_week.tle [more files] \
        --train-days 1.5 --seed 0

Every number here comes from real element sets: a twin at update time u
predicts the other object from the newest public set (epoch <= u), knows
its own satellite exactly, and is scored against the truth (the first set
published after TCA, propagated back; hindcast.truth_weights). Nothing is
drawn from an assumed error distribution.

Schemes (Pc threshold 1e-4, updates every 8 h, 72 h horizon):
  baseline        circular Pc, combined sigma 0.5 km (the original twin)
  circular_meas   circular Pc, combined sigma from the measured error table
  foster          Foster Pc, measured RTN table (Starlink / other, by age)
  foster_bias     foster + the table's median bias subtracted (public)
  foster_fl       Foster with the FedAvg error model (bias + sigma) for
                  members' satellites; the table for everything else
  foster_local    same with the assessor's local-only model
  foster_central  same with one model on all members' data pooled

The error table and all models see only data from before the split; test
encounters have every update after it. Results go to
research/results/hindcast_<name>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satcollision import error_learning as el  # noqa: E402
from satcollision import hindcast as hc  # noqa: E402
from satcollision.tle_history import fit_tle_errors, group_of  # noqa: E402

THRESHOLD = 1e-4
PRECISE = 0.354
STRATA = ((10.0, 1.0), (30.0, 0.1), (100.0, 0.01))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def table_arrays(model, is_starlink: np.ndarray, age_h: np.ndarray):
    """Vectorised tle_history.ErrorModel.sigma_rtn: (spread, median) per
    axis, interpolated in age; beyond the last bin the last value grows
    linearly with age, as the world does."""
    spread = np.zeros((len(age_h), 3))
    median = np.zeros((len(age_h), 3))
    for grp, sel in (("starlink", is_starlink), ("other", ~is_starlink)):
        rows = [r for r in model.table.get(grp, []) if r["n"] >= 20]
        if not rows or not sel.any():
            continue
        ages = np.array([r["age_mid_h"] for r in rows])
        a = age_h[sel]
        grow = np.maximum(1.0, a / ages[-1])
        for k, c in enumerate("rtc"):
            sp = np.array([r[f"sigma_{c}_km"] for r in rows])
            md = np.array([r[f"median_{c}_km"] for r in rows])
            spread[sel, k] = np.interp(a, ages, sp) * np.where(a > ages[-1], grow, 1.0)
            median[sel, k] = np.interp(a, ages, md) * np.where(a > ages[-1], grow, 1.0)
    return np.maximum(spread, 0.05), median


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--name", default=None)
    ap.add_argument("--train-days", type=float, default=None,
                    help="training window length from the start (default: all but the last 4 days + 72 h)")
    ap.add_argument("--warmup-days", type=float, default=0.5)
    ap.add_argument("--test-days", type=float, default=None)
    ap.add_argument("--update-h", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--max-samples", type=int, default=300_000)
    ap.add_argument("--no-secure-check", action="store_true")
    ap.add_argument("--dp-clip", type=float, default=0.5)
    ap.add_argument("--dp-noise", type=float, default=1.0)
    a = ap.parse_args()

    cfg = json.loads((ROOT / "worlds" / "real_leo_history.json").read_text())["fleet"]["operators"]
    hist = hc.load_histories(a.files, cache_dir=str(ROOT / "cache"))
    owner, ops = hc.owners_of(hist.names, cfg)
    first, last = hist.first_epoch, hist.last_epoch
    span = last - first
    train_lo = first + a.warmup_days
    if a.train_days is None:
        split = max(train_lo + 1.0, last - 1.0 - 3.0 - 4.0)
    else:
        split = train_lo + a.train_days
    test_lo = split + 3.0
    test_hi = last - 1.0 if a.test_days is None else min(last - 1.0, test_lo + a.test_days)
    n_days = test_hi - test_lo
    name = a.name or Path(a.files[0]).stem + (f"_x{len(a.files)}" if len(a.files) > 1 else "")
    log(f"{hist.n_sets} sets, {hist.n_objects} objects, {span:.1f} days; train {train_lo - first:.1f}-{split - first:.1f} d, "
        f"test TCAs {test_lo - first:.1f}-{test_hi - first:.1f} d ({n_days:.2f} d)")
    log("members:", {op: int((owner == k).sum()) for k, op in enumerate(ops)})

    # ---- hindcast geometry (cached) ----
    cache = ROOT / "research" / "cache" / f"hindcast_{name}_{hist.fingerprint()}_{split:.3f}_{test_hi:.3f}.npz"
    if cache.exists():
        z = np.load(cache)
        enc = {k: z[k] for k in z.files if not k.startswith("asm_") and not k.startswith("err_")}
        asm = {k[4:]: z[k] for k in z.files if k.startswith("asm_")}
        err = {k[4:]: z[k] for k in z.files if k.startswith("err_")}
        log(f"loaded {cache.name}: {len(enc['tca_jd'])} approaches, {len(asm['enc'])} assessments")
    else:
        t0 = time.time()
        sc = hc.screen_truth(hist, owner, test_lo, test_hi, strata=STRATA, dense_op=ops.index("Starlink"),
                             progress=lambda i, n: (i % 1200 == 0) and log(f"  screen {i}/{n}"))
        log(f"screen: {len(sc['tca_jd'])} approaches kept ({sc['weight'].sum():.0f} weighted) in {time.time() - t0:.0f} s")
        enc = hc.approach_geometry(hist, sc)
        upd = hc.update_times(split, test_hi, a.update_h)
        asm = hc.assessments(enc, owner, upd)
        err = hc.public_errors(hist, enc, asm)
        ok = err["ok"] & (asm["update_jd"] >= split)
        asm = {k: v[ok] for k, v in asm.items()}
        err = {k: v[ok] for k, v in err.items()}
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **enc, **{f"asm_{k}": v for k, v in asm.items()},
                            **{f"err_{k}": v for k, v in err.items()})
        log(f"geometry + errors: {len(asm['enc'])} assessments, {time.time() - t0:.0f} s total")

    dangerous = hc.true_pc(enc, PRECISE) >= THRESHOLD
    w_enc = enc["weight"]
    log(f"dangerous approaches: {int(dangerous.sum())} ({dangerous.sum() / n_days:.0f} / day)")

    other = asm["other_row"]
    other_owner = owner[other]
    is_sl = np.array([group_of(hist.names[r]) == "starlink" for r in range(hist.n_objects)])[other]
    age = err["age_h"]

    # ---- public error table from the training window ----
    t0 = time.time()
    table = fit_tle_errors(hc.subset_history(hist, split), max_objects_per_group=1500, seed=a.seed)
    spread, median = table_arrays(table, is_sl, age)
    sig_table = np.sqrt(spread ** 2 + median ** 2)
    log(f"error table fitted in {time.time() - t0:.0f} s")

    # ---- training data per operator (own fleet, before the split) ----
    t0 = time.time()
    train, test = {}, {}
    for k, op in enumerate(ops):
        rows = np.nonzero(owner == k)[0]
        train[op] = el.build_samples(hist, rows, train_lo, split, seed=a.seed, max_samples=a.max_samples)
        test[op] = el.build_samples(hist, rows, test_lo - 3.0, test_hi, seed=a.seed + 1, max_samples=50_000)
    log("samples:", {op: len(train[op]) for op in ops}, f"{time.time() - t0:.0f} s")
    # Public data only: the same TLE-vs-later-TLE labels for every object in
    # the catalogue, which anyone can compute without any operator's help.
    all_rows = np.arange(hist.n_objects)
    pub_train = el.build_samples(hist, all_rows, train_lo, split, seed=a.seed + 2, ages_per_set=2,
                                 max_samples=a.max_samples)
    unowned_rows = np.nonzero(owner < 0)[0]
    test_unowned = el.build_samples(hist, unowned_rows, test_lo - 3.0, test_hi, seed=a.seed + 3, max_samples=50_000)

    # ---- models ----
    t0 = time.time()
    epochs = a.rounds
    local = {op: el.train_local(train[op], epochs=epochs, seed=a.seed) for op in ops}
    log(f"local models {time.time() - t0:.0f} s")
    t0 = time.time()
    central = el.train_local(el.ErrorSamples.concat([train[op] for op in ops]), epochs=epochs, seed=a.seed)
    log(f"central model {time.time() - t0:.0f} s")
    public = el.train_local(pub_train, epochs=epochs, seed=a.seed)
    tsl_u = np.array([group_of(hist.names[r]) == "starlink" for r in test_unowned.row])
    sp_u, md_u = table_arrays(table, tsl_u, test_unowned.age_h)
    quality_unowned = {"table": el.table_nll(np.sqrt(sp_u ** 2 + md_u ** 2), np.zeros_like(md_u), test_unowned),
                       "public_model": el.evaluate(public, test_unowned)}
    log(f"public model: unowned objects NLL table {quality_unowned['table']['nll']:.3f} "
        f"public model {quality_unowned['public_model']['nll']:.3f}; "
        f"along-track 1-sigma coverage {quality_unowned['table']['cover1'][1]:.2f} vs {quality_unowned['public_model']['cover1'][1]:.2f}")
    t0 = time.time()
    fed, fed_log = el.fedavg(train, rounds=a.rounds, local_epochs=1, seed=a.seed)
    log(f"FedAvg {time.time() - t0:.0f} s, {fed_log.bytes_per_update} bytes per update, best round {fed_log.best_round} of {len(fed_log.history)}")
    secure_check = None
    if not a.no_secure_check:
        t0 = time.time()
        small = {op: d.subset(np.arange(min(len(d), 5000))) for op, d in train.items()}
        f1, _ = el.fedavg(small, rounds=3, seed=a.seed)
        f2, _ = el.fedavg(small, rounds=3, seed=a.seed, secure=True)
        secure_check = float(np.abs(el.flat(f1) - el.flat(f2)).max())
        log(f"secure aggregation check: max weight difference {secure_check:.2e} ({time.time() - t0:.0f} s)")
    fed_sqrt, fsq_log = el.fedavg(train, rounds=a.rounds, local_epochs=1, seed=a.seed, weighting="sqrt")
    # DP-FedAvg: each operator's update clipped and noised before it leaves
    # (operator-level DP; epsilon by RDP composition over the rounds run).
    fed_dp, dp_log = el.fedavg(train, rounds=a.rounds, local_epochs=1, seed=a.seed, weighting="sqrt",
                               clip=a.dp_clip, noise_multiplier=a.dp_noise)
    dp_eps = el.gaussian_rdp_epsilon(a.dp_noise, len(dp_log.history))
    log(f"DP-FedAvg: clip {a.dp_clip}, noise x{a.dp_noise}, {len(dp_log.history)} rounds -> epsilon {dp_eps:.1f} (delta 1e-5)")

    # ---- error-model quality on held-out test-period samples ----
    quality = {}
    test_all = el.ErrorSamples.concat([test[op] for op in ops])
    tsl = np.array([group_of(hist.names[r]) == "starlink" for r in test_all.row])
    sp_t, md_t = table_arrays(table, tsl, test_all.age_h)
    owner_t = owner[test_all.row]
    for k, op in enumerate(ops):
        sel = owner_t == k
        d = test_all.subset(sel)
        q = {"table": el.table_nll(np.sqrt(sp_t[sel] ** 2 + md_t[sel] ** 2), np.zeros_like(md_t[sel]), d),
             "table_bias": el.table_nll(sp_t[sel], md_t[sel], d),
             "fedavg": el.evaluate(fed, d), "fedavg_sqrt": el.evaluate(fed_sqrt, d),
             "central": el.evaluate(central, d), "own_local": el.evaluate(local[op], d),
             "public_model": el.evaluate(public, d), "fedavg_dp": el.evaluate(fed_dp, d)}
        # local models of *other* operators predicting this fleet: what a
        # local-only twin uses when it screens this operator's satellites
        others = [el.evaluate(local[o], d)["nll"] for o in ops if o != op and len(train[o])]
        q["others_local_mean_nll"] = float(np.mean(others)) if others else None
        quality[op] = q
        log(f"  {op:9s} n={len(d):6d} NLL table {q['table']['nll']:.3f} table+bias {q['table_bias']['nll']:.3f} "
            f"fed {q['fedavg']['nll']:.3f} central {q['central']['nll']:.3f} own-local {q['own_local']['nll']:.3f} "
            f"others-local {q['others_local_mean_nll']}")

    # ---- per-assessment bias and sigma under each model ----
    member_other = other_owner >= 0
    Xo = el.features(hist, err["set"], age)

    allmask = np.ones(len(age), bool)

    def apply(member_models, unowned_model=None):
        """bias, sigma per assessment: ``member_models`` [(assessor mask,
        model)] for members' satellites, ``unowned_model`` (or the table)
        for everything else."""
        bias = np.zeros_like(median)
        sig = sig_table.copy()
        for mask, mdl in member_models:
            sel = member_other & mask
            if sel.any():
                bias[sel], sig[sel] = el.predict(mdl, Xo[sel], age[sel])
        if unowned_model is not None:
            sel = ~member_other
            bias[sel], sig[sel] = el.predict(unowned_model, Xo[sel], age[sel])
        return bias, sig

    loc_models = [(asm["operator"] == k, local[op]) for k, op in enumerate(ops)]
    b_fed, s_fed = apply([(allmask, fed)])
    b_fsq, s_fsq = apply([(allmask, fed_sqrt)])
    b_cen, s_cen = apply([(allmask, central)])
    b_loc, s_loc = apply(loc_models)
    b_pub, s_pub = apply([(allmask, public)], public)
    b_fedp, s_fedp = apply([(allmask, fed)], public)
    b_fsqp, s_fsqp = apply([(allmask, fed_sqrt)], public)
    b_cenp, s_cenp = apply([(allmask, central)], public)
    b_locp, s_locp = apply(loc_models, public)
    b_dpp, s_dpp = apply([(allmask, fed_dp)], public)
    # Suggested alternative: every member shares (to the federation) one
    # model of its *own* fleet's error, and a twin uses the owner's model
    # for that owner's satellites.
    b_own = np.zeros_like(median); s_own = sig_table.copy()
    sel = ~member_other
    b_own[sel], s_own[sel] = el.predict(public, Xo[sel], age[sel])
    for k, op in enumerate(ops):
        sel = other_owner == k
        if sel.any():
            b_own[sel], s_own[sel] = el.predict(local[op], Xo[sel], age[sel])

    circ_meas = np.sqrt(PRECISE ** 2 + np.mean(sig_table ** 2, axis=1))
    schemes = {
        "baseline": hc.assess_pc(enc, asm, err, None, circular_sigma_km=0.5),
        "circular_meas": hc.assess_pc(enc, asm, err, None, circular_sigma_km=circ_meas),
        "foster": hc.assess_pc(enc, asm, err, sig_table, PRECISE),
        "foster_bias": hc.assess_pc(enc, asm, err, spread, PRECISE, bias_rtn=median),
        "foster_local": hc.assess_pc(enc, asm, err, s_loc, PRECISE, bias_rtn=b_loc),
        "foster_fl": hc.assess_pc(enc, asm, err, s_fed, PRECISE, bias_rtn=b_fed),
        "foster_fl_sqrt": hc.assess_pc(enc, asm, err, s_fsq, PRECISE, bias_rtn=b_fsq),
        "foster_central": hc.assess_pc(enc, asm, err, s_cen, PRECISE, bias_rtn=b_cen),
        "foster_pub": hc.assess_pc(enc, asm, err, s_pub, PRECISE, bias_rtn=b_pub),
        "foster_pub_local": hc.assess_pc(enc, asm, err, s_locp, PRECISE, bias_rtn=b_locp),
        "foster_pub_fl": hc.assess_pc(enc, asm, err, s_fedp, PRECISE, bias_rtn=b_fedp),
        "foster_pub_fl_sqrt": hc.assess_pc(enc, asm, err, s_fsqp, PRECISE, bias_rtn=b_fsqp),
        "foster_pub_central": hc.assess_pc(enc, asm, err, s_cenp, PRECISE, bias_rtn=b_cenp),
        "foster_pub_fl_dp": hc.assess_pc(enc, asm, err, s_dpp, PRECISE, bias_rtn=b_dpp),
        "foster_pub_owner": hc.assess_pc(enc, asm, err, s_own, PRECISE, bias_rtn=b_own),
    }

    subsets = {"all": None, "member_vs_member": member_other}
    for k, op in enumerate(ops):
        subsets[f"by_{op}"] = asm["operator"] == k
    results = {}
    for sname, pc in schemes.items():
        results[sname] = {sub: hc.score(asm, pc, dangerous, n_days, THRESHOLD, mask=m, enc_weight=w_enc)
                          for sub, m in subsets.items()}
        r = results[sname]["all"]
        rm = results[sname]["member_vs_member"]
        log(f"{sname:15s} caught {r['caught_frac']:.3f} (>=24h {r['caught_24h_frac']:.3f}) median warn {r['median_warning_h']:.1f} h "
            f"FA/day {r['false_alarms_per_day']:.1f} | member-member caught {rm['caught_frac']:.3f} FA/day {rm['false_alarms_per_day']:.1f}")

    # ---- threshold sweep: caught vs false alarms ----
    sweep = {}
    for sname, pc in schemes.items():
        sweep[sname] = []
        for T in np.logspace(-8, -2, 25):
            s_all = hc.score(asm, pc, dangerous, n_days, float(T), enc_weight=w_enc)
            s_mm = hc.score(asm, pc, dangerous, n_days, float(T), mask=member_other, enc_weight=w_enc)
            sweep[sname].append({"T": float(T), "caught": s_all["caught_frac"], "caught_24h": s_all["caught_24h_frac"],
                                 "fa_day": s_all["false_alarms_per_day"], "mm_caught": s_mm["caught_frac"],
                                 "mm_caught_24h": s_mm["caught_24h_frac"], "mm_fa_day": s_mm["false_alarms_per_day"],
                                 "warn_h_per_dangerous": s_all["warning_hours_per_dangerous"]})

    out = ROOT / "research" / "results" / f"hindcast_{name}_s{a.seed}.json"
    out.write_text(json.dumps({
        "files": [Path(f).name for f in a.files], "seed": a.seed, "threshold": THRESHOLD, "update_h": a.update_h,
        "split_days": split - first, "test_days": [test_lo - first, test_hi - first], "n_test_days": n_days,
        "strata": STRATA, "members": {op: int((owner == k).sum()) for k, op in enumerate(ops)},
        "n_approaches": int(len(enc["tca_jd"])), "n_assessments": int(len(asm["enc"])),
        "dangerous_per_day": float(dangerous.sum() / n_days),
        "dangerous_member_member": int(dangerous[np.unique(asm["enc"][member_other])].sum()),
        "train_samples": {op: len(train[op]) for op in ops}, "fedavg_bytes_per_update": fed_log.bytes_per_update,
        "secure_aggregation_max_abs_diff": secure_check, "table": table.to_dict(),
        "model_quality": quality, "model_quality_unowned": quality_unowned,
        "fedavg_history": fed_log.history, "fedavg_best_round": fed_log.best_round,
        "fedavg_sqrt_best_round": fsq_log.best_round,
        "dp": {"clip": a.dp_clip, "noise_multiplier": a.dp_noise, "rounds": len(dp_log.history), "epsilon": dp_eps, "delta": 1e-5}, "results": results, "sweep": sweep,
    }, indent=1), encoding="utf-8")
    log("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
