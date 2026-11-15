"""Manoeuvres as private knowledge: what does each operator's knowledge of its
own manoeuvres add to screening?

    PYTHONPATH=src python research/hindcast_manoeuvre.py <gp_history files...> [--name w4]

Uses the cached four-week hindcast (research/hindcast_fl.py, same split).
A manoeuvre is a jump between consecutive element sets of a member satellite
(tle_history.detect_jumps: more than 10 robust sigma and more than 1 km, the
world's rule), dated halfway between the two sets. An operator knows its own
manoeuvres (its twin flies the real trajectory: in the hindcast its own
satellite is the truth); the other operator sees them only when a new TLE
appears.

A shared error model cannot use a manoeuvre plan without revealing it, so the
question is asked where the knowledge naturally lives: for a member-vs-member
encounter, compare
  * one side alone: the operator that did NOT manoeuvre assesses the one that
    did, from public data (what any single twin can do), with
  * both sides: either operator's alert counts, as in the two-stage scheme,
    where the manoeuvring operator assesses with its own (manoeuvred) orbit.
Split by whether a satellite manoeuvred in the 72 h before TCA.
Writes research/results/hindcast_manoeuvre_<name>.json.
"""
import argparse
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
from satcollision.tle_history import detect_jumps, fit_tle_errors, group_of  # noqa: E402
from hindcast_fl import PRECISE, THRESHOLD, table_arrays  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def first_alert_lead(asm, pc, n_enc, T):
    """(n_enc, 2) lead (h) of the first alert of each side, -1 if none."""
    out = np.full((n_enc, 2), -1.0)
    hit = pc >= T
    np.maximum.at(out, (asm["enc"], asm["side"]), np.where(hit, asm["lead_h"], -1.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--name", default="w4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-days", type=float, default=None)
    a = ap.parse_args()
    cfg = json.loads((ROOT / "worlds" / "real_leo_history.json").read_text())["fleet"]["operators"]
    hist = hc.load_histories(a.files, cache_dir=str(ROOT / "cache"))
    owner, ops = hc.owners_of(hist.names, cfg)
    first, last = hist.first_epoch, hist.last_epoch
    split = (max(first + 0.5 + 1.0, last - 1.0 - 3.0 - 4.0) if a.train_days is None
             else first + 0.5 + a.train_days)
    cache = sorted(glob.glob(str(ROOT / "research" / "cache" / f"hindcast_{a.name}_*_{split:.3f}_*.npz")))
    z = np.load(cache[0])
    enc = {k: z[k] for k in z.files if not k.startswith(("asm_", "err_"))}
    asm = {k[4:]: z[k] for k in z.files if k.startswith("asm_")}
    err = {k[4:]: z[k] for k in z.files if k.startswith("err_")}
    n_enc = len(enc["tca_jd"])
    test_lo = split + 3.0
    n_days = float(min(enc["tca_jd"].max(), test_lo + 4.0) - test_lo)
    dangerous = hc.true_pc(enc, PRECISE) >= THRESHOLD
    w = enc["weight"]
    log(f"{n_enc} approaches, {len(asm['enc'])} assessments, {n_days:.2f} test days")

    # ---- manoeuvres of member satellites ----
    t0 = time.time()
    table = fit_tle_errors(hc.subset_history(hist, split), max_objects_per_group=1500, seed=a.seed)
    member_rows = np.nonzero(owner >= 0)[0]
    jumps = detect_jumps(hist, table, n_sigma=10.0, rows=member_rows)
    jumps = [j for j in jumps if max(abs(x) for x in j["rtn_km"]) >= 1.0]
    man_t = {}
    for j in jumps:
        man_t.setdefault(hist.row_of[j["norad_id"]], []).append(j["epoch_jd"] - j["gap_h"] / 48.0)
    log(f"{len(jumps)} manoeuvres on {len(man_t)} member satellites in {(last - first):.1f} days "
        f"({time.time() - t0:.0f} s)")

    def manoeuvred(rows, tca):
        out = np.zeros(len(rows), bool)
        for i, (r, t) in enumerate(zip(rows, tca)):
            ts = man_t.get(int(r))
            if ts:
                ts = np.asarray(ts)
                out[i] = np.any((ts > t - 3.0) & (ts < t))
        return out

    mm = (owner[enc["row_a"]] >= 0) & (owner[enc["row_b"]] >= 0)
    man_a = np.zeros(n_enc, bool); man_b = np.zeros(n_enc, bool)
    idx = np.nonzero(mm | (owner[enc["row_a"]] >= 0))[0]
    man_a[idx] = manoeuvred(enc["row_a"][idx], enc["tca_jd"][idx])
    idx = np.nonzero(owner[enc["row_b"]] >= 0)[0]
    man_b[idx] = manoeuvred(enc["row_b"][idx], enc["tca_jd"][idx])

    # ---- schemes ----
    other = asm["other_row"]
    is_sl = np.array([group_of(hist.names[r]) == "starlink" for r in range(hist.n_objects)])[other]
    sp, md = table_arrays(table, is_sl, err["age_h"])
    sig_table = np.sqrt(sp ** 2 + md ** 2)
    train_lo = first + 0.5
    t0 = time.time()
    local = {}
    for k, op in enumerate(ops):
        d = el.build_samples(hist, np.nonzero(owner == k)[0], train_lo, split, seed=a.seed, max_samples=300_000)
        local[k] = el.train_local(d, epochs=30, seed=a.seed)
    pub = el.train_local(el.build_samples(hist, np.arange(hist.n_objects), train_lo, split, seed=a.seed + 2,
                                          ages_per_set=2, max_samples=300_000), epochs=30, seed=a.seed)
    Xo = el.features(hist, err["set"], err["age_h"])
    b_own, s_own = el.OwnerModels(local, pub).predict(Xo, err["age_h"], owner[other])
    log(f"models {time.time() - t0:.0f} s")
    schemes = {"baseline": hc.assess_pc(enc, asm, err, None, circular_sigma_km=0.5),
               "foster": hc.assess_pc(enc, asm, err, sig_table, PRECISE),
               "foster_owner": hc.assess_pc(enc, asm, err, s_own, PRECISE, bias_rtn=b_own)}

    # ---- member-vs-member encounters: one side vs both ----
    groups = {"one_manoeuvred": mm & (man_a ^ man_b), "both_manoeuvred": mm & man_a & man_b,
              "none_manoeuvred": mm & ~man_a & ~man_b}
    res = {"n_jumps": len(jumps), "n_manoeuvring_sats": len(man_t), "test_days": n_days, "groups": {}}
    for g, sel in groups.items():
        dsel = sel & dangerous
        res["groups"][g] = {"encounters": float(w[sel].sum()), "dangerous": float(w[dsel].sum())}
    for name, pc in schemes.items():
        lead = first_alert_lead(asm, pc, n_enc, THRESHOLD)
        res[name] = {}
        for g, sel in groups.items():
            both = np.maximum(lead[:, 0], lead[:, 1])
            if g == "one_manoeuvred":
                # side 0 assesses object b; if b manoeuvred, side 0 sees it only from public data
                blind = np.where(man_b, lead[:, 0], lead[:, 1])      # the operator that did not manoeuvre
                knows = np.where(man_b, lead[:, 1], lead[:, 0])      # the operator that manoeuvred
                views = (("not_manoeuvring_side_alone", blind), ("manoeuvring_side_alone", knows), ("both_sides", both))
            else:
                views = (("side_a_alone", lead[:, 0]), ("side_b_alone", lead[:, 1]), ("both_sides", both))
            r = {}
            for key, L in views:
                caught = sel & dangerous & (L > 0)
                fa = sel & ~dangerous & (L > 0)
                nd = max(float(w[sel & dangerous].sum()), 1e-9)
                r[key] = {"caught": float(w[caught].sum() / nd), "caught_24h": float(w[caught & (L >= 24)].sum() / nd),
                          "median_warning_h": float(np.median(L[caught])) if caught.any() else 0.0,
                          "false_alarms_per_day": float(w[fa].sum() / n_days)}
            log(f"{name:13s} {g:16s} dangerous {res['groups'][g]['dangerous']:4.0f} | " + " | ".join(
                f"{k}: {v['caught']:.2f} (>=24h {v['caught_24h']:.2f}) FA {v['false_alarms_per_day']:.0f}/d" for k, v in r.items()))
            # fair check: each single side with its threshold lowered until it raises as many false
            # alarms as both sides together at 1e-4
            target = r["both_sides"]["false_alarms_per_day"]
            for key, col in (("side0_at_equal_fa", 0), ("side1_at_equal_fa", 1)):
                best = None
                for T in np.logspace(-4, -9, 26):
                    L = first_alert_lead(asm, pc, n_enc, T)[:, col]
                    fa = float(w[sel & ~dangerous & (L > 0)].sum() / n_days)
                    if fa >= target or T == 1e-9:
                        nd = max(float(w[sel & dangerous].sum()), 1e-9)
                        best = {"T": float(T), "false_alarms_per_day": fa,
                                "caught": float(w[sel & dangerous & (L > 0)].sum() / nd),
                                "caught_24h": float(w[sel & dangerous & (L >= 24)].sum() / nd)}
                        break
                r[key] = best
            log(f"   equal-FA single sides: " + ", ".join(f"{k} caught {v['caught']:.2f} at FA {v['false_alarms_per_day']:.0f}/d"
                                                       for k, v in r.items() if k.endswith("equal_fa")))
            res[name][g] = r
    # how much of the public error of manoeuvring satellites is the manoeuvre
    oth_man = np.where(asm["side"] == 0, man_b[asm["enc"]], man_a[asm["enc"]]) & (owner[other] >= 0)
    for lab, m in (("other manoeuvred", oth_man), ("other did not", (owner[other] >= 0) & ~oth_man)):
        e = np.abs(err["err_rtn"][m, 1])
        res.setdefault("public_error_along_track_km", {})[lab] = {
            "n": int(m.sum()), "median": float(np.median(e)), "p90": float(np.percentile(e, 90))}
        log(f"public along-track error, member satellites, {lab}: median {np.median(e):.2f} km, 90th pct "
            f"{np.percentile(e, 90):.2f} km (n={m.sum()})")
    p = ROOT / "research" / "results" / (f"hindcast_manoeuvre_{a.name}.json" if a.train_days is None
                                         else f"hindcast_manoeuvre_{a.name}.json")
    p.write_text(json.dumps(res, indent=1))
    log("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
