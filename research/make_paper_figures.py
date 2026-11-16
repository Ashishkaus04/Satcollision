"""Figures for RESULTS.md sections 10 to 15 (and the report), from saved results.

    python research/make_paper_figures.py

Reads research/results/*.json only (no data, no recomputation) and writes
research/figures/fig_<name>.png and .pdf. One function per figure; a figure
whose results file is missing is skipped with a note.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parent / "figures"
C = {"baseline": "#7f7f7f", "foster": "#1f77b4", "owner": "#9467bd", "owner_dp": "#c5b0d5",
     "public": "#2ca02c", "fedavg": "#d62728", "local": "#ff7f0e"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_{name}.{ext}", dpi=200)
    plt.close(fig)
    print("wrote", f"fig_{name}.png")


def fig_rolling():
    """Section 10c: the three steps and the comparisons, mean +- sd over four windows."""
    d = load("hindcast_rolling.json")
    if not d:
        return print("skip rolling")
    steps = [("baseline", "Baseline\n(circular)", C["baseline"]), ("foster", "+ Foster\n(table)", C["foster"]),
             ("foster_pub_owner", "+ Foster +\nowners' models", C["owner"]),
             ("foster_pub_owner_dp2", "... with DP\n(eps 2)", C["owner_dp"]),
             ("foster_pub", "Public model\n(comparison)", C["public"]), ("foster_pub_fl_sqrt", "FedAvg\n(comparison)", C["fedavg"]),
             ("foster_pub_local", "Local-only\n(comparison)", C["local"])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    for ax, key, title in ((axes[0], "all_at_100", "All encounters, 100 false alarms a day"),
                           (axes[1], "mm_at_10", "Member vs member, 10 false alarms a day")):
        m = [d["steps"][s]["mean_sd"][key][0] * 100 for s, _, _ in steps]
        sd = [d["steps"][s]["mean_sd"][key][1] * 100 for s, _, _ in steps]
        ax.bar(range(len(steps)), m, yerr=sd, color=[c for *_, c in steps], capsize=3)
        ax.axvline(3.5, color="#bbbbbb", lw=0.8, ls=":")
        ax.set_xticks(range(len(steps)), [l for _, l, _ in steps], fontsize=7.5)
        ax.set_ylabel("Dangerous encounters caught (%)")
        ax.set_title(title)
    fig.suptitle("Four rolling test windows (mean and standard deviation)", fontsize=10)
    save(fig, "rolling_steps")


def fig_precise():
    """Section 11: real error (vs precise orbit) and TLE-derived error by age, every precise satellite."""
    files = sorted(R.glob("precise_vs_tle*.json"))
    if not files:
        return print("skip precise")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    cols = {"sentinel1": "#1f77b4", "cosmic2": "#ff7f0e", "swarm": "#2ca02c"}
    seen = set()
    for f in files:
        d = json.loads(f.read_text())
        for nid, sat in d["satellites"].items():
            mis = sat.get("mission", "sentinel1")
            mid = [(b["age_h"][0] + b["age_h"][1]) / 2 for b in sat["by_age"]]
            real = [b["sigma_t_precise"] for b in sat["by_age"]]
            tle = [b["sigma_t_tle"] for b in sat["by_age"]]
            lab = None if mis in seen else mis
            seen.add(mis)
            axes[0].plot(mid, real, "-o", color=cols.get(mis, "k"), ms=3, label=lab and f"{lab}: vs precise orbit")
            axes[0].plot(mid, tle, "--", color=cols.get(mis, "k"), lw=1, alpha=0.8, label=lab and f"{lab}: vs next TLE")
            axes[1].plot(mid, np.array(real) / np.maximum(tle, 1e-9), "-o", color=cols.get(mis, "k"), ms=3, label=lab)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Age of the public element set (h)"); axes[0].set_ylabel("Along-track robust sigma (km)")
    axes[0].set_title("Error of the public prediction"); axes[0].legend(fontsize=7)
    axes[1].set_xscale("log"); axes[1].axhline(1, color="#999999", lw=0.8)
    axes[1].set_xlabel("Age of the public element set (h)"); axes[1].set_ylabel("Real / TLE-derived sigma")
    axes[1].set_title("How much TLE-derived truth understates the error"); axes[1].legend(fontsize=8)
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    for ax in axes:
        ax.xaxis.set_major_locator(FixedLocator([1.5, 3, 6, 12, 24, 48]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    save(fig, "precise_truth")


def fig_owner_precise():
    """Section 11: held-out NLL of owner models with precise labels vs public model, per mission."""
    rows = []
    for f in sorted(R.glob("precise_vs_tle*.json")):
        d = json.loads(f.read_text())
        ot = d.get("owner_test", {})
        if "models" in ot:
            ot = {"sentinel1": ot}
        rows += [(m, v["models"]) for m, v in ot.items()]
    if not rows:
        return print("skip owner precise")
    keys = [("public_model_tle_labels", "Public model (TLE labels)", C["public"]),
            ("owner_model_tle_labels", "Owner, TLE labels", C["local"]),
            ("owner_model_precise_labels", "Owner, precise labels", C["owner"]),
            ("owner_model_precise_dp_eps2", "Owner, precise labels, DP eps 2", C["owner_dp"])]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(rows))
    for i, (k, lab, col) in enumerate(keys):
        ax.bar(x + (i - 1.5) * 0.2, [m[k]["nll"] for _, m in rows], 0.2, yerr=[m[k]["nll_sd"] for _, m in rows],
               color=col, label=lab, capsize=2)
    ax.set_xticks(x, [r for r, _ in rows]); ax.axhline(0, color="#999999", lw=0.6)
    ax.set_ylabel("Held-out NLL against the precise orbit (lower is better)")
    ax.set_title("Owner models trained on precise orbits"); ax.legend(fontsize=8)
    save(fig, "owner_precise")


def fig_dilution():
    """Section 12: the maximum-Pc rule, caught vs false alarms, last test window."""
    d = load("hindcast_w4_s0.json")
    if not d or "hybrid" not in d:
        return print("skip dilution")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    b = d["results"]["baseline"]["all"]
    ax.plot(b["false_alarms_per_day"], b["caught_frac"] * 100, "s", color=C["baseline"], ms=8, label="Baseline at 1e-4")
    for name, lab, col in (("foster", "+ Foster (table) + max-Pc", C["foster"]),
                           ("foster_pub_owner", "+ Foster + owners' models + max-Pc", C["owner"])):
        rows = d["hybrid"][name]
        ax.plot([r["false_alarms_per_day"] for r in rows], [r["caught_frac"] * 100 for r in rows], "-o", ms=3, color=col, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("False alarms per day"); ax.set_ylabel("Dangerous encounters caught (%)")
    ax.set_title("Alert if Pc >= 1e-4 or max-Pc >= T2 (sweep of T2)"); ax.legend(fontsize=8)
    save(fig, "dilution")


def fig_mpc():
    """Section 13: throughput of the secure step."""
    d = load("mpc_timing.json")
    if not d:
        return print("skip mpc")
    rows = [("Literature figure\n(used before)", 18_000, "#bbbbbb"),
            ("Semi-honest,\nWAN (modelled)", 450_000, C["foster"]), ("Semi-honest,\nLAN (modelled)", 4.4e6, C["foster"]),
            ("Semi-honest,\nlaptop (measured)", 3600 / 0.00044, C["foster"]),
            ("Malicious (sy-rep-ring),\nlaptop (measured)", 3600 / 0.0139, C["fedavg"])]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(rows)), [r[1] for r in rows], color=[r[2] for r in rows])
    for i, r in enumerate(rows):
        ax.text(i, r[1] * 1.15, f"{r[1]:,.0f}", ha="center", fontsize=8)
    ax.set_yscale("log"); ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.set_ylabel("Secure Pc decisions per hour (3-party cluster)")
    ax.set_title("Measured cost of the secure step (MP-SPDZ)")
    save(fig, "mpc_throughput")


def fig_manoeuvre():
    """Section 14: who assesses, for encounters where one satellite manoeuvred."""
    runs = [json.loads(f.read_text()) for f in sorted(R.glob("hindcast_manoeuvre_w4*.json"))]
    if not runs:
        return print("skip manoeuvre")
    g = "one_manoeuvred"
    views = [("not_manoeuvring_side_alone", "Operator that did not\nmanoeuvre, alone", C["local"]),
             ("manoeuvring_side_alone", "Operator that\nmanoeuvred, alone", C["owner"]),
             ("both_sides", "Both operators\n(either alert)", C["public"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, lab in ((axes[0], "caught", "Caught (%)"), (axes[1], "caught_24h", "Caught at least 24 h ahead (%)")):
        m = [np.mean([r["foster_owner"][g][v][key] for r in runs]) * 100 for v, _, _ in views]
        sd = [np.std([r["foster_owner"][g][v][key] for r in runs]) * 100 for v, _, _ in views]
        ax.bar(range(3), m, yerr=sd, color=[c for *_, c in views], capsize=3)
        ax.set_xticks(range(3), [l for _, l, _ in views], fontsize=8); ax.set_ylabel(lab)
    fig.suptitle("Encounters where one satellite manoeuvred in the 72 h before TCA (four windows)", fontsize=10)
    save(fig, "manoeuvre")


def fig_privacy():
    """Section 15: what a shared owner model reveals."""
    d = load("owner_model_privacy.json")
    if not d:
        return print("skip privacy")
    ops = list(d["operators"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(ops))
    for i, (k, lab, col) in enumerate((("after_public_model_bias", "Public model", C["public"]),
                                       ("after_owner_model_bias_no_dp", "Owner model", C["owner"]),
                                       ("after_owner_model_bias_dp_eps2", "Owner model, DP eps 2", C["owner_dp"]))):
        red = [100 * (1 - d["operators"][o]["position"][k] / d["operators"][o]["position"]["median_along_track_km_public_prediction"]) for o in ops]
        axes[0].bar(x + (i - 1) * 0.27, red, 0.27, color=col, label=lab)
    axes[0].axhline(0, color="#999999", lw=0.6)
    axes[0].set_xticks(x, ops, fontsize=8); axes[0].set_ylabel("Rival's along-track error reduction (%)")
    axes[0].set_title("Position leak"); axes[0].legend(fontsize=8)
    for i, (k, lab, col) in enumerate((("no_dp", "No DP", C["owner"]), ("dp_eps2", "DP eps 2", C["owner_dp"]))):
        axes[1].bar(x + (i - 0.5) * 0.35, [d["operators"][o]["membership_auc"][k]["per_satellite"] for o in ops], 0.35,
                    color=col, label=lab)
    axes[1].axhline(0.5, color="#999999", lw=0.8, ls="--")
    axes[1].set_ylim(0.35, 0.65); axes[1].set_xticks(x, ops, fontsize=8)
    axes[1].set_ylabel("Membership attack AUC (0.5 = guessing)"); axes[1].set_title("Membership inference, per satellite")
    axes[1].legend(fontsize=8)
    save(fig, "owner_privacy")


def fig_secure_all():
    """Section 16: public filter + secure Pc for every close member pair."""
    d = load("hindcast_secure_all.json")
    if not d:
        return print("skip secure_all")
    R = sorted(int(r) for r in d["summary"])
    S = d["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.errorbar(R, [S[str(r)]["caught"][0] * 100 for r in R], yerr=[S[str(r)]["caught"][1] * 100 for r in R],
                fmt="-o", color=C["owner"], capsize=3, label="Caught")
    ax.errorbar(R, [S[str(r)]["caught_24h"][0] * 100 for r in R], yerr=[S[str(r)]["caught_24h"][1] * 100 for r in R],
                fmt="--o", color=C["owner"], capsize=3, label="Caught >= 24 h ahead")
    ts = d.get("two_stage_owner_models_both_sides")
    if ts:
        ax.axhline(ts["caught"] * 100, color=C["fedavg"], lw=1, label="Two-stage, owners' models: caught")
        ax.axhline(ts["caught_24h"] * 100, color=C["fedavg"], lw=1, ls="--", label="Two-stage: caught >= 24 h")
    ax.set_xscale("log"); ax.set_xticks(R, [str(r) for r in R])
    ax.set_xlabel("Public filter radius (km)"); ax.set_ylabel("Dangerous member-pair encounters (%)")
    ax.set_title("Member vs member, four windows"); ax.legend(fontsize=7)
    ax = axes[1]
    for k, col in zip(d["throughput_per_h"], (C["foster"], C["public"], C["local"], C["fedavg"])):
        ax.plot(R, [S[str(r)]["hours_per_day"][k] for r in R], "-o", ms=3, color=col, label=k)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(R, [str(r) for r in R])
    ax.axhline(24, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("Public filter radius (km)"); ax.set_ylabel("Secure computation, cluster-hours per day")
    ax.set_title("Cost at the measured MPC speeds"); ax.legend(fontsize=7)
    save(fig, "secure_all")


def fig_hybrid():
    """Section 16b: fixed vs model-adaptive filter, caught vs secure evaluations a day (mean of four windows)."""
    d = load("hindcast_hybrid.json")
    if not d:
        return print("skip hybrid")

    def at_b(curve, b, key):
        x = np.log([max(p["secure_evals_per_day"], 1) for p in curve]); o = np.argsort(x)
        if not (x[o][0] <= np.log(b) <= x[o][-1]):
            return np.nan
        return float(np.interp(np.log(b), x[o], np.array([p[key] for p in curve])[o]))

    grid = np.logspace(3.5, 6.2, 40)
    sch = [("fixed_radius", "Fixed radius", C["baseline"], "--"), ("table", "Adaptive: public table", C["foster"], "-"),
           ("public_model", "Adaptive: public model", C["public"], "-"),
           ("owner_models", "Adaptive: owners' models", C["owner"], "-"),
           ("owner_models_dp2", "Adaptive: owners' models, DP eps 2", C["owner_dp"], ":")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, lab in ((axes[0], "caught", "Caught (%)"), (axes[1], "caught_24h", "Caught at least 24 h ahead (%)")):
        for s_, l_, col, ls in sch:
            y = [np.nanmean([at_b(d["windows"][w][s_], b, key) for w in d["windows"]]) * 100 for b in grid]
            ax.plot(grid, y, ls, color=col, label=l_)
        ax.set_xscale("log"); ax.set_xlabel("Secure evaluations per day"); ax.set_ylabel(lab)
        ax.axvline(180_000, color="#bbbbbb", lw=0.8, ls=":")
    axes[0].legend(fontsize=7)
    fig.suptitle("Member-vs-member encounters: which pairs get a secure check (four windows)", fontsize=10)
    save(fig, "hybrid_filter")


if __name__ == "__main__":
    for f in (fig_rolling, fig_precise, fig_owner_precise, fig_dilution, fig_mpc, fig_manoeuvre, fig_privacy, fig_secure_all, fig_hybrid):
        f()
