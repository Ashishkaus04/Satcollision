#!/usr/bin/env python3
r"""
Render the project's evaluation figures into ``docs/figures/``.

Every figure here is generated from the same code paths the demo and the
backtests use — none of the numbers are typed in by hand, so a figure in
the report can always be regenerated and cannot drift away from what the
system actually does. Each figure is written three ways: a 300-dpi PNG (for
slides), a vector PDF (for the report, so it stays sharp at any size), and
a CSV of the exact series behind it (so a number in the text can be checked
against the plot, and so the data survives even if the figure is redrawn).

Usage:

    PYTHONPATH=src python3 scripts/make_figures.py                # everything but the backtest figure
    PYTHONPATH=src python3 scripts/make_figures.py --skip-fl      # skip the slow PyTorch/Opacus figure
    PYTHONPATH=src python3 scripts/make_figures.py \
        --backtest data/starlink.tle Starlink data/kuiper.tle Kuiper --hours 24

The backtest figure needs real TLE files (see
``scripts/run_cross_operator_backtest.py`` for how to fetch them). Its
screening result is cached as JSON under ``docs/figures/`` so a re-render
does not re-run a 30-40s propagation, and so the run's numbers are pinned
to the TLE epoch they came from rather than silently changing tomorrow.

Design notes, since a panel may ask why the charts look the way they do:

* The four series colours are a fixed, validated set (blue / orange / aqua
  / violet). They were checked for colour-vision-deficiency separation
  rather than picked by eye, and every series also carries its own line
  style or a direct value label, so no figure depends on colour alone —
  which matters for a thesis that will be printed in greyscale.
* Nothing here is a dual-axis chart. Where two measures of different scale
  need comparing (detection lead time in minutes vs. data exposure in per
  cent), they get two panels rather than two y-scales on one plot, because
  a reader cannot tell which curve belongs to which axis without decoding a
  legend first.
* Grid lines, axes and annotations are deliberately recessive; the data is
  the darkest thing on the page.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display in a headless run; must precede pyplot
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = REPO_ROOT / "docs" / "figures"

# Validated categorical palette (light surface). Checked with the data-viz
# validator on the all-pairs list: worst CVD deltaE 9.2, worst normal-vision
# deltaE 16.3, both above their floors.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
SURFACE = "#fcfcfb"
INK, INK_SOFT, INK_FAINT = "#0b0b0b", "#52514e", "#a3a29c"


def _apply_style() -> None:
    """One house style for every figure, so the report reads as one document."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": INK_FAINT,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 0.8,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.7,
        "lines.markersize": 5,
    })


def _save(fig, name: str, rows: list[list], header: list[str]) -> None:
    """Write PNG + PDF + the CSV of the series behind the figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png, pdf, csv_path = (FIGURE_DIR / f"{name}.png", FIGURE_DIR / f"{name}.pdf",
                          FIGURE_DIR / f"{name}.csv")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    # ``CreationDate: None`` drops the timestamp matplotlib would otherwise
    # embed in the PDF. Without it every re-render produces different bytes
    # for an unchanged figure, so all six PDFs show up as modified in git on
    # every run and a real change is impossible to spot among the noise.
    fig.savefig(pdf, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  wrote {png.relative_to(REPO_ROOT)}, {pdf.name}, {csv_path.name}")


# --------------------------------------------------------------------------
# 1. Reputation trajectories
# --------------------------------------------------------------------------

def figure_reputation_trajectories(n_rounds: int = 20, defect_round: int = 12, seed: int = 7):
    """Reputation over time, one line per operator, grouped by strategy.

    The point of the figure is the ``on_off`` line: it is indistinguishable
    from an honest operator until it defects, and then falls below its own
    starting score within a few rounds. That is the asymmetric rise/fall
    rate doing its job, and it is hard to show in a table.
    """
    from satcollision.reputation import ReputationParams, simulate_reputation_rounds

    result = simulate_reputation_rounds(n_rounds=n_rounds, defect_from_round=defect_round, seed=seed)
    params = ReputationParams()
    style = {
        "honest": (BLUE, "-", "o"),
        "free_rider": (ORANGE, "--", "s"),
        "spoofer": (VIOLET, "-.", "^"),
        "on_off": (AQUA, ":", "D"),
    }
    rounds = np.arange(1, n_rounds + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    # Tier thresholds first, so they sit behind the data as reference lines.
    for value, label in ((params.quarantine_threshold, "quarantine"),
                         (params.degraded_threshold, "degraded"),
                         (params.standard_threshold, "full access")):
        ax.axhline(value, color=INK_FAINT, lw=0.8, ls=(0, (2, 3)), zorder=1)
        # Labelled inside the plot at the left, where no series is running,
        # so the right margin stays free for the per-operator direct labels.
        ax.text(0.8, value + 0.012, label, color=INK_FAINT, fontsize=7.5,
                va="bottom", ha="left", zorder=4,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))

    ax.axvline(defect_round, color=INK_FAINT, lw=0.8, zorder=1)
    ax.text(defect_round - 0.25, 0.985, "Zeta defects here", color=INK_SOFT, fontsize=8,
            ha="right", va="top")

    seen_strategies: set[str] = set()
    rows = []
    for operator, behaviour in result["behaviours"].items():
        colour, linestyle, marker = style[behaviour]
        trajectory = result["state"].trajectory(operator)
        rows.append([operator, behaviour] + [round(v, 4) for v in trajectory])
        # One legend entry per strategy, not per operator: three honest
        # operators are three samples of one behaviour, not three categories.
        label = behaviour.replace("_", "-") if behaviour not in seen_strategies else None
        seen_strategies.add(behaviour)
        ax.plot(rounds, trajectory, color=colour, ls=linestyle, marker=marker,
                markevery=4, markeredgecolor=SURFACE, markeredgewidth=1.0,
                label=label, zorder=3)
        # Direct label in ink, not in the series colour — the marker beside
        # it already carries the identity.
        ax.annotate(operator, (rounds[-1], trajectory[-1]), textcoords="offset points",
                    xytext=(6, -2), fontsize=8, color=INK)

    ax.set_xlabel("Reporting period")
    ax.set_ylabel("Reputation score")
    ax.set_title("Reputation separates honest operators from three kinds of misbehaviour")
    ax.set_xlim(0.5, n_rounds + 2.6)
    ax.set_ylim(0.1, 1.0)
    ax.set_xticks(range(2, n_rounds + 1, 2))
    ax.legend(loc="lower left", ncol=4, bbox_to_anchor=(0, -0.22))

    _save(fig, "fig_reputation_trajectories", rows,
          ["operator", "strategy"] + [f"round_{i}" for i in rounds])
    return result


# --------------------------------------------------------------------------
# 2. Robustness sweep
# --------------------------------------------------------------------------

def figure_robustness_sweep(n_trials: int = 400):
    """Aggregation error against the fraction of dishonest reporters.

    Worth knowing before anyone asks in a viva: the plain mean's curve is
    not monotone, and that is structural rather than noise (it is stable
    to within 0.1 between 200 and 800 trials). ``robustness_sweep`` cycles
    adversaries through three attack types, so raising the dishonest share
    changes the *mix* as well as the count — at 30% a sign-flip attack is
    present and drags the mean hard; at 40% an added spoof-high pushes the
    other way and partially cancels it. The figure says so in a footnote
    rather than quietly smoothing the dip away.
    """
    from satcollision.evaluate import robustness_sweep

    sweep = robustness_sweep(n_trials=n_trials)
    fractions = np.array(sweep["adversary_fractions"], dtype=float)
    series = (
        ("Plain mean", sweep["mean_error"], BLUE, "-", "o"),
        ("Trimmed mean", sweep["trimmed_mean_error"], ORANGE, "--", "s"),
        ("Krum", sweep["krum_error"], AQUA, "-.", "^"),
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, values, colour, linestyle, marker in series:
        ax.plot(fractions * 100, values, color=colour, ls=linestyle, marker=marker,
                markeredgecolor=SURFACE, markeredgewidth=1.0, label=label, zorder=3)
        ax.annotate(f"{values[-1]:.2f}", (fractions[-1] * 100, values[-1]),
                    textcoords="offset points", xytext=(6, 0), fontsize=8, color=INK)

    ax.set_xlabel("Share of operators reporting dishonestly (%)")
    ax.set_ylabel("Aggregation error vs. honest truth")
    ax.set_title("A single liar moves the plain mean; the robust rules barely notice")
    ax.set_xlim(-2, fractions[-1] * 100 + 8)
    ax.set_ylim(bottom=0)
    ax.set_xticks([f * 100 for f in fractions])
    ax.legend(loc="upper left")
    ax.text(0.5, -0.24,
            "Adversaries cycle through spoof-high, suppress and sign-flip, so the attack mix changes with the share, not just\n"
            "the count — hence the plain mean's peak at 30% (a sign-flip dominates) and its partial cancellation at 40%.",
            transform=ax.transAxes, ha="center", fontsize=8, color=INK_SOFT)

    rows = [[f"{f:.2f}", f"{m:.4f}", f"{t:.4f}", f"{k:.4f}"] for f, m, t, k in zip(
        fractions, sweep["mean_error"], sweep["trimmed_mean_error"], sweep["krum_error"])]
    _save(fig, "fig_robustness_sweep", rows,
          ["adversary_fraction", "mean_error", "trimmed_mean_error", "krum_error"])


# --------------------------------------------------------------------------
# 3. Reputation-weighted aggregation across rounds
# --------------------------------------------------------------------------

def figure_weighted_aggregation_error(simulation: dict):
    """Why a memoryless robust rule is not the end of the story.

    Trimmed mean re-decides who the outlier is every round from scratch.
    Reputation-weighted aggregation carries the history, so its error keeps
    falling as the federation learns who to discount — which is visible as
    two curves separating, and invisible in any single-round comparison.
    """
    history = simulation["error_history"]
    rounds = [row["round"] for row in history]
    series = (
        ("Plain mean", [r["mean"] for r in history], BLUE, "-", "o"),
        ("Trimmed mean", [r["trimmed_mean"] for r in history], ORANGE, "--", "s"),
        ("Reputation-weighted", [r["reputation_weighted"] for r in history], AQUA, "-.", "D"),
    )

    def trailing_mean(values: list[float], window: int = 5) -> list[float]:
        """Per-round error is noisy by construction — each round draws a fresh
        set of encounters — and the claim being made is about the level the
        error settles at, not any single round. So the readable line is the
        trailing mean, with the raw rounds kept visible behind it rather than
        hidden, so nothing is smoothed away silently."""
        return [float(np.mean(values[max(0, i - window + 1): i + 1])) for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, values, colour, linestyle, marker in series:
        ax.plot(rounds, values, color=colour, lw=0.9, alpha=0.3, zorder=2)
        smoothed = trailing_mean(values)
        ax.plot(rounds, smoothed, color=colour, ls=linestyle, marker=marker, markevery=4,
                markeredgecolor=SURFACE, markeredgewidth=1.0, label=label, zorder=3)
        ax.annotate(f"{smoothed[-1]:.2f}", (rounds[-1], smoothed[-1]), textcoords="offset points",
                    xytext=(8, -2), fontsize=8.5, color=INK)

    ax.set_xlabel("Reporting period")
    ax.set_ylabel("Aggregation error vs. honest truth")
    ax.set_title("Carrying reputation across rounds beats deciding fresh each round")
    ax.set_xlim(0.5, rounds[-1] + 2.4)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(2, rounds[-1] + 1, 2))
    ax.legend(loc="upper left")
    ax.text(0.5, -0.20,
            "Bold lines are a five-period trailing mean; the faint lines behind them are the raw per-period error.",
            transform=ax.transAxes, ha="center", fontsize=8, color=INK_SOFT)

    rows = [[r["round"], f"{r['mean']:.4f}", f"{r['trimmed_mean']:.4f}",
             f"{r['reputation_weighted']:.4f}"] for r in history]
    _save(fig, "fig_weighted_aggregation_error", rows,
          ["round", "mean_error", "trimmed_mean_error", "reputation_weighted_error"])


# --------------------------------------------------------------------------
# 4. Three-scenario comparison
# --------------------------------------------------------------------------

def figure_three_scenarios(seed: int = 42, tca_offset_min: float = 30.0):
    """The project's central claim, in two panels.

    Detection lead time (minutes) and raw data exposure (per cent) are
    measures of different kinds on different scales. Plotting them on one
    pair of axes would need a second y-axis, which is the single most
    misread thing in a technical report — so they get one panel each, with
    a shared category order so the eye can travel between them.
    """
    from satcollision.evaluate import run_three_scenario_comparison

    comparison = run_three_scenario_comparison(seed=seed, tca_offset_min=tca_offset_min)
    scenarios = list(comparison["scenarios"].values())
    names = [s.name.replace(" (Proposed)", "\n(Proposed)") for s in scenarios]
    lead_minutes = [s.detection_lead_time_s / 60.0 for s in scenarios]
    exposure = [s.raw_data_exposed_pct for s in scenarios]
    colours = [BLUE, ORANGE, AQUA]

    fig, (ax_lead, ax_exposure) = plt.subplots(1, 2, figsize=(8.6, 4.0))
    for ax, values, title, unit in (
        (ax_lead, lead_minutes, "Detection lead time", "minutes"),
        (ax_exposure, exposure, "Raw trajectory data exposed", "% of own data"),
    ):
        bars = ax.bar(names, values, color=colours, width=0.6, zorder=3)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, value),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=9, color=INK)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(values) * 1.18 if max(values) else 1)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=8.5)

    fig.suptitle("The federated design matches full data sharing on warning time, at zero exposure",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()

    rows = [[s.name, f"{s.detection_lead_time_s/60:.2f}", f"{s.raw_data_exposed_pct:.1f}",
             s.cross_operator_signal_shared] for s in scenarios]
    _save(fig, "fig_three_scenarios", rows,
          ["scenario", "lead_time_min", "raw_data_exposed_pct", "cross_operator_signal"])


# --------------------------------------------------------------------------
# 5. Federated learning comparison
# --------------------------------------------------------------------------

def figure_federated_learning(n_samples: int = 400, n_rounds: int = 8, seed: int = 42):
    """FedAvg with and without DP-SGD, against centralized and local-only.

    Drawn as a dot plot rather than bars on purpose. All four scores sit
    between 0.96 and 0.99, so bars would either start at zero (and show
    four indistinguishable full-height columns) or start at 0.95 — a
    truncated bar axis, which exaggerates differences because a bar's
    *length* is what encodes its value. A dot encodes by position, so a
    zoomed axis is honest for it, and the accuracy-to-F1 connector makes
    the within-scenario spread readable at the same time.
    """
    from satcollision.federated_learning import compare_scenarios

    result = compare_scenarios(["Alpha", "Beta", "Gamma"], n_samples_per_operator=n_samples,
                                n_rounds=n_rounds, local_epochs=2, seed=seed)
    labels = ["Centralized\n(pools raw data)", "Local only\n(no sharing)",
              "Federated\n(FedAvg)", "Federated\n+ DP-SGD"]
    keys = ["centralized", "local_only_mean", "federated_no_dp", "federated_dp"]
    accuracy = [result[k]["accuracy"] for k in keys]
    f1 = [result[k]["f1"] for k in keys]

    y = np.arange(len(labels))[::-1]  # first scenario at the top
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    for yi, acc, f1_value in zip(y, accuracy, f1):
        ax.plot([acc, f1_value], [yi, yi], color=INK_FAINT, lw=1.2, zorder=2)
    ax.scatter(accuracy, y, s=70, color=BLUE, marker="o", zorder=3,
               edgecolor=SURFACE, linewidth=1.2, label="Accuracy")
    ax.scatter(f1, y, s=70, color=ORANGE, marker="s", zorder=3,
               edgecolor=SURFACE, linewidth=1.2, label="F1")
    for yi, acc, f1_value in zip(y, accuracy, f1):
        lower, upper = (acc, f1_value) if acc <= f1_value else (f1_value, acc)
        ax.annotate(f"{lower:.3f}", (lower, yi), textcoords="offset points", xytext=(-9, -3),
                    ha="right", fontsize=8, color=INK)
        ax.annotate(f"{upper:.3f}", (upper, yi), textcoords="offset points", xytext=(9, -3),
                    ha="left", fontsize=8, color=INK)

    ax.set_yticks(y, labels)
    ax.set_xlabel("Score on the shared held-out set")
    ax.set_xlim(min(min(accuracy), min(f1)) - 0.02, max(max(accuracy), max(f1)) + 0.016)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_title("Federated training lands close to pooling the data; DP-SGD buys privacy for a little accuracy")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, -0.30))

    epsilons = result.get("federated_dp_final_round_epsilons", {})
    if epsilons:
        epsilon = float(list(epsilons.values())[0])
        ax.text(1.0, -0.30, f"DP-SGD run at final-round epsilon = {epsilon:.2f} (delta = 1e-5) per operator.",
                transform=ax.transAxes, ha="right", fontsize=8, color=INK_SOFT)

    rows = [[label.replace("\n", " "), f"{a:.4f}", f"{f:.4f}"] for label, a, f in zip(labels, accuracy, f1)]
    _save(fig, "fig_federated_learning", rows, ["scenario", "accuracy", "f1"])


# --------------------------------------------------------------------------
# 6. Real backtest miss-distance distribution
# --------------------------------------------------------------------------

def _backtest_miss_distances(path_a: str, name_a: str, path_b: str, name_b: str,
                              hours: float, cache: Path) -> dict:
    """Screen two real catalogs and cache the resulting miss distances.

    Cached deliberately: TLE files are re-published daily, so a figure
    regenerated next week from fresh files would quietly disagree with the
    numbers quoted in the report text. The cache pins both the distances
    and the epoch they were computed from.
    """
    if cache.exists():
        print(f"  using cached screening result {cache.name} (delete it to re-run)")
        return json.loads(cache.read_text(encoding="utf-8"))

    import datetime
    from satcollision.fleets import load_tle_file, _epoch_from_calendar
    from satcollision.twin import propagate_window, screen_conjunctions, DEFAULT_PC_THRESHOLD

    objects = (load_tle_file(path_a, operator_name=name_a)
               + load_tle_file(path_b, operator_name=name_b))
    now = datetime.datetime.now(datetime.timezone.utc)
    _, jd0, fr0 = _epoch_from_calendar(now.year, now.month, now.day, now.hour + now.minute / 60.0)
    print(f"  propagating {len(objects):,} real objects across {hours:.0f}h (this takes a moment)...")
    prop = propagate_window(objects, jd0, fr0, duration_s=hours * 3600.0, step_s=30.0)
    conjunctions = screen_conjunctions(objects, prop, screening_distance_km=25.0)
    cross = [c for c in conjunctions if c.object_a.operator != c.object_b.operator]

    payload = {
        "operator_a": name_a,
        "operator_b": name_b,
        "n_objects": len(objects),
        "window_hours": hours,
        "computed_at_utc": now.isoformat(timespec="seconds"),
        "pc_threshold": DEFAULT_PC_THRESHOLD,
        "cross_operator_miss_km": sorted(c.miss_distance_km for c in cross),
        "same_operator_count": len(conjunctions) - len(cross),
        "max_cross_operator_pc": max((c.pc for c in cross), default=0.0),
        "n_above_threshold": sum(1 for c in cross if c.pc >= DEFAULT_PC_THRESHOLD),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  cached screening result to {cache.name}")
    return payload


def figure_backtest_distribution(path_a: str, name_a: str, path_b: str, name_b: str, hours: float):
    """Where real cross-operator close approaches actually fall.

    The honest headline of this figure is a negative: on current catalogs
    nothing between two operators crosses the operational risk threshold.
    Showing the whole distribution rather than a single "no events" line is
    what makes that a result instead of an absence — it says how close the
    two constellations come, and how often.
    """
    cache = FIGURE_DIR / f"backtest_{name_a.lower()}_{name_b.lower()}_{int(hours)}h.json"
    data = _backtest_miss_distances(path_a, name_a, path_b, name_b, hours, cache)
    distances = np.array(data["cross_operator_miss_km"], dtype=float)
    if distances.size == 0:
        print(f"  no cross-operator conjunctions within 25 km — nothing to plot for "
              f"{name_a} vs {name_b}; the honest result is reported in the README instead.")
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bins = np.linspace(0, 25, 26)
    ax.hist(distances, bins=bins, color=BLUE, rwidth=0.9, zorder=3)
    closest = float(distances.min())
    ax.axvline(closest, color=ORANGE, lw=1.6, ls="--", zorder=4)
    ax.annotate(f"closest approach\n{closest:.2f} km", (closest, ax.get_ylim()[1] * 0.82),
                textcoords="offset points", xytext=(10, 0), fontsize=8.5, color=INK)

    ax.set_xlabel("Miss distance at closest approach (km)")
    ax.set_ylabel("Number of close approaches")
    # State what the run actually found rather than assuming the negative:
    # the honest headline changes if a catalog update ever does produce a
    # threshold crossing, and the figure should say so without an edit.
    n_above = int(data.get("n_above_threshold", 0))
    verdict = ("none above the risk threshold" if n_above == 0
               else f"{n_above:,} above the risk threshold")
    ax.set_title(f"{name_a} vs {name_b}: {len(distances):,} close approaches over "
                 f"{int(hours)}h, {verdict}")
    ax.grid(axis="x", visible=False)
    ax.text(0.5, -0.30,
            f"{data['n_objects']:,} real catalogued objects, screened {data['computed_at_utc']}. "
            f"Highest cross-operator Pc: {data['max_cross_operator_pc']:.1e} "
            f"(threshold {data['pc_threshold']:.0e}).\n"
            "Counts rise with distance because the volume of space being screened grows with it, and the series "
            "stops at 25 km because that is the\nscreening cut-off — neither is a property of the orbits. The "
            "left-hand tail is the part that matters operationally.",
            transform=ax.transAxes, ha="center", fontsize=8, color=INK_SOFT)

    counts, edges = np.histogram(distances, bins=bins)
    rows = [[f"{edges[i]:.1f}", f"{edges[i+1]:.1f}", int(counts[i])] for i in range(len(counts))]
    _save(fig, f"fig_backtest_{name_a.lower()}_{name_b.lower()}", rows,
          ["bin_start_km", "bin_end_km", "n_close_approaches"])


# --------------------------------------------------------------------------
# 7. Real-data three-scenario comparison
# --------------------------------------------------------------------------

def figure_real_evaluation(json_path: Path | None = None):
    """Detection lead time across every real threshold-crossing encounter.

    Drawn as two empirical cumulative distributions rather than a pair of
    bars, because the interesting thing is not "the average went up" but
    *how the whole distribution moves*: how many encounters get an hour of
    warning, how many get minutes, and how many no-cooperation misses
    entirely. A bar chart of two medians would hide all three.

    Reads the JSON written by ``scripts/run_real_evaluation.py``.
    """
    path = json_path or (FIGURE_DIR / "real_evaluation.json")
    if not path.exists():
        print(f"  skipped: {path.name} not found — run scripts/run_real_evaluation.py first.")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    head = data["headline"]
    rows = head.get("encounters", [])
    if not rows:
        print("  skipped: the run found no threshold-crossing encounters to plot.")
        return

    summary = head.get("summary", {})
    horizon_h = float(head.get("screening_horizon_h", 72.0))
    nocoop_h = np.sort(np.array([r["lead_time_nocoop_min"] for r in rows], dtype=float) / 60.0)
    fractions = np.arange(1, len(nocoop_h) + 1) / len(nocoop_h) * 100.0
    median_nocoop = float(np.median(nocoop_h))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    # One measured distribution, one reference line. The cooperative curve is
    # deliberately not drawn as a series: it is a constant at the screening
    # horizon (cooperation flags an encounter as soon as it is screened), and
    # drawing a flat step at the axis limit would imply a measurement where
    # there is an architectural property.
    ax.step(nocoop_h, fractions, where="post", color=ORANGE, ls="--",
            label="No cooperation", zorder=3)
    ax.axvline(horizon_h, color=BLUE, lw=2.0, zorder=3)
    ax.annotate(f"Federated / full sharing\nflagged on entry at {horizon_h:.0f} h",
                (horizon_h, 50), textcoords="offset points", xytext=(-10, 0),
                ha="right", va="center", fontsize=8.5, color=INK)

    ax.axvline(median_nocoop, color=INK_FAINT, lw=0.9, ls=(0, (2, 3)), zorder=2)
    ax.annotate(f"median {median_nocoop:.1f} h", (median_nocoop, 2), textcoords="offset points",
                xytext=(5, 0), fontsize=8, color=INK_SOFT, ha="left", va="bottom")

    ax.set_xlabel("Warning time before closest approach (hours)")
    ax.set_ylabel("% of encounters with this much warning or less")
    ax.set_ylim(0, 102)
    ax.set_xlim(0, horizon_h * 1.04)
    ax.set_title(f"How late no-cooperation is, across {len(rows)} real threshold-crossing encounters")
    ax.legend(loc="upper left")

    ax.text(0.5, -0.30,
            f"{head['n_objects']:,} real {head['catalog']} objects over {head['window_hours']:.0f}h, "
            f"screened {head['computed_at_utc']}. Orbits, geometry and Pc unmodified; ownership assigned "
            "by NORAD parity.\nCooperation buys a median "
            f"{summary.get('median_lead_time_gain_h', 0):.1f} extra hours of warning — federated at 0% "
            "raw-data exposure, full sharing at 100%.",
            transform=ax.transAxes, ha="center", fontsize=8, color=INK_SOFT)

    csv_rows = [[r["encounter_id"], r["miss_distance_m"], r["pc"],
                 r["lead_time_nocoop_min"], r["lead_time_cooperative_min"], r["lead_time_gain_min"]]
                for r in sorted(rows, key=lambda r: -r["pc"])]
    _save(fig, "fig_real_evaluation", csv_rows,
          ["encounter_id", "miss_distance_m", "pc", "lead_time_nocoop_min",
           "lead_time_cooperative_min", "lead_time_gain_min"])


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-fl", action="store_true",
                        help="skip the federated-learning figure (needs torch + opacus, slowest)")
    parser.add_argument("--backtest", nargs=4, metavar=("TLE_A", "NAME_A", "TLE_B", "NAME_B"),
                        help="also render the real cross-operator backtest distribution")
    parser.add_argument("--hours", type=float, default=24.0,
                        help="backtest window in hours (default 24)")
    args = parser.parse_args()

    _apply_style()
    print(f"Rendering figures into {FIGURE_DIR.relative_to(REPO_ROOT)}/ ...")

    print("[1] reputation trajectories")
    simulation = figure_reputation_trajectories()

    print("[2] robustness sweep")
    figure_robustness_sweep()

    print("[3] reputation-weighted aggregation error")
    figure_weighted_aggregation_error(simulation)

    print("[4] three-scenario comparison")
    figure_three_scenarios()

    if args.skip_fl:
        print("[5] federated learning — skipped (--skip-fl)")
    else:
        print("[5] federated learning (slow: trains real models)")
        try:
            figure_federated_learning()
        except ImportError as exc:
            print(f"  skipped: {exc}. Install torch + opacus, or pass --skip-fl.")

    if args.backtest:
        print("[6] real backtest distribution")
        figure_backtest_distribution(*args.backtest, hours=args.hours)
    else:
        print("[6] real backtest distribution — skipped (pass --backtest to render it)")

    print("[7] real-data three-scenario comparison")
    figure_real_evaluation()

    print("\nDone. Each figure is written as .png (slides), .pdf (report) and .csv (the data).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
