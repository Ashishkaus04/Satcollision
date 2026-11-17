"""Safety vs privacy: one table and one figure for the report (RESULTS.md section 20).

    python research/privacy_safety.py

Rows are designs from "share nothing" to "share everything"; columns are
what is shared, the data an operator sends a day, what a rival learns about a
target satellite, and safety for member-vs-member encounters. Everything is
read from saved results:
  * safety: hindcast_manoeuvre_w4*.json (each twin alone, both operators'
    alerts), hindcast_hybrid.json (filters + secure Pc), avoidance_outcome.json
    (risk removed when operators act);
  * leakage: the section 7 attacker on the 10 Sep catalogue
    (starlink_..._leakage_k8_far3_win24.0_s1.json: one day of observations;
    ..._winall_s1.json: one week), median over targets of the error of the
    rival's estimate of the target's along-track offset, as a share of the
    public uncertainty (0.67 = learned nothing, 0 = knows it exactly), and
    owner_model_privacy.json for what a shared owner model adds;
  * data volumes: computed below from the measured counts and sizes (stated
    with each row; orders of magnitude, per operator per day, averaged over
    the six members).
"""
import glob
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parent / "figures"
NO_INFO = 0.674          # median |N(0, 1)|: the error ratio of a rival that learned nothing


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def leak(channel, window="win24.0"):
    d = load(f"starlink_20260910T0758_168h_10km__leakage_k8_far3_{window}_s1.json")
    if not d or channel not in d["results"]:
        return float("nan")
    return float(np.median([r["err_ratio"][1] for r in d["results"][channel]]))


def union_catch(scheme):
    man = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(R / "hindcast_manoeuvre_w4*.json")))]
    tot = c = c24 = fa = 0.0
    for g in ("one_manoeuvred", "none_manoeuvred", "both_manoeuvred"):
        d = np.mean([r["groups"][g]["dangerous"] for r in man])
        v = [r[scheme][g]["both_sides"] for r in man]
        tot += d
        c += d * np.mean([x["caught"] for x in v])
        c24 += d * np.mean([x["caught_24h"] for x in v])
        fa += np.mean([x["false_alarms_per_day"] for x in v])
    return c / tot, c24 / tot, fa


def main():
    av = load("avoidance_outcome.json")["summary"]
    hy = load("hindcast_hybrid.json")["summary"]
    priv = load("owner_model_privacy.json")["operators"]
    model_leak = {op: 1 - v["position"]["after_owner_model_bias_dp_eps2"] / v["position"]["median_along_track_km_public_prediction"]
                  for op, v in priv.items()}
    b = union_catch("baseline")
    o = union_catch("foster_owner")
    evals_per_op = 180_000 * 2 / 6          # secure evaluations a day an average member takes part in
    rows = [
        {"design": "Each twin alone, circular Pc", "shared": "nothing",
         "data_per_day": "0", "leak_day": NO_INFO, "leak_week": NO_INFO, "model_leak": None,
         "caught": b[0], "caught_24h": b[1], "fa_day": b[2], "risk_removed": av["baseline"]["risk_removed_frac"][0]},
        {"design": "Each twin alone + shared owner models (DP eps 2)", "shared": "one 6.3 kB model per operator per update",
         "data_per_day": "about 1 kB sent, 5 kB received (weekly models)", "leak_day": NO_INFO, "leak_week": NO_INFO,
         "model_leak": max(model_leak.values()),
         "caught": o[0], "caught_24h": o[1], "fa_day": o[2], "risk_removed": av["two_stage"]["risk_removed_frac"][0]},
        {"design": "Two-stage: abstracted alerts + secure Pc on alerted pairs (original design)",
         "shared": "alert, severity tier and TCA per alerted encounter; secure-step bit",
         "data_per_day": "about 10-20 kB of signed alerts, plus secure-step traffic for alerted pairs",
         "leak_day": leak("SECBIT"), "leak_week": leak("SECBIT", "winall"), "model_leak": None,
         "caught": o[0], "caught_24h": o[1], "fa_day": o[2], "risk_removed": None},
        {"design": "Two-stage, hardened (only 'act now' revealed)", "shared": "act/no-act per approach",
         "data_per_day": "as above", "leak_day": leak("MINIMAL"), "leak_week": leak("MINIMAL", "winall"),
         "model_leak": None, "caught": o[0], "caught_24h": o[1], "fa_day": o[2], "risk_removed": None},
        {"design": "Fixed 20 km public filter + secure Pc", "shared": "secret shares; act/no-act per filtered pair",
         "data_per_day": f"about {evals_per_op * 35e3 / 1e9:.1f} GB of MPC traffic (35 kB per evaluation; reveals nothing) and {evals_per_op / 8 / 1e3:.0f} kB of decisions",
         "leak_day": leak("FILTERSEC"), "leak_week": leak("FILTERSEC", "winall"), "model_leak": None,
         "caught": hy["fixed_radius"]["180000"]["caught"][0], "caught_24h": hy["fixed_radius"]["180000"]["caught_24h"][0],
         "fa_day": hy["fixed_radius"]["180000"]["false_alarms_per_day"][0], "risk_removed": av["filter20"]["risk_removed_frac"][0]},
        {"design": "Adopted: model-set filter (owner models, DP eps 2) + secure Pc",
         "shared": "secret shares; act/no-act per filtered pair; owner models (DP)",
         "data_per_day": f"as above, plus the owner models",
         "leak_day": leak("FILTERSEC"), "leak_week": leak("FILTERSEC", "winall"), "model_leak": max(model_leak.values()),
         "caught": hy["owner_models_dp2"]["180000"]["caught"][0], "caught_24h": hy["owner_models_dp2"]["180000"]["caught_24h"][0],
         "fa_day": hy["owner_models_dp2"]["180000"]["false_alarms_per_day"][0], "risk_removed": av["adaptive"]["risk_removed_frac"][0]},
        {"design": "Full sharing of precise orbits (no privacy)", "shared": "every precise ephemeris",
         "data_per_day": "about 1 GB of ephemerides (average fleet: ~2,000 satellites x 3 days at 60 s)",
         "leak_day": 0.0, "leak_week": 0.0, "model_leak": None,
         "caught": 1.0, "caught_24h": 1.0, "fa_day": None, "risk_removed": None},
    ]
    (R / "privacy_safety.json").write_text(json.dumps({"rows": rows, "no_info": NO_INFO}, indent=1))

    print(f"{'design':72s} {'leak 1 d':>8s} {'leak 1 wk':>9s} {'caught':>7s} {'>=24h':>6s} {'FA/day':>7s} {'risk rem.':>9s}")
    for r in rows:
        f = lambda v, pct=True: "  n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v * 100:.0f}%" if pct else f"{v:.0f}")
        print(f"{r['design'][:72]:72s} {f(r['leak_day']):>8s} {f(r['leak_week']):>9s} {f(r['caught']):>7s} {f(r['caught_24h']):>6s} "
              f"{f(r['fa_day'], False):>7s} {f(r['risk_removed']):>9s}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    cols = ["#7f7f7f", "#ff7f0e", "#1f77b4", "#17becf", "#2ca02c", "#9467bd", "#d62728"]
    short = ["Alone, circular", "Alone + owner models", "Two-stage alerts", "Two-stage, hardened",
             "Filter 20 km + secure", "Adopted design", "Full sharing"]
    for r, c, s in zip(rows, cols, short):
        x = r["leak_day"] / NO_INFO * 100          # 100 = rival learned nothing
        ax.scatter(x, r["caught_24h"] * 100, s=90, color=c, zorder=3)
        off = (6, -13) if s in ("Two-stage, hardened", "Alone + owner models") else (6, 5)
        ax.annotate(s, (x, r["caught_24h"] * 100), textcoords="offset points", xytext=off, fontsize=8)
    ax.set_xlabel("Privacy kept: rival's remaining error on a target, after one day (% of no-information)")
    ax.set_ylabel("Dangerous member-pair encounters caught >= 24 h ahead (%)")
    ax.set_xlim(-5, 110); ax.set_ylim(40, 105); ax.grid(alpha=0.3)
    ax.set_title("Safety vs privacy, member-vs-member encounters")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_privacy_safety.{ext}", dpi=200)
    print("wrote fig_privacy_safety.png")


if __name__ == "__main__":
    main()
