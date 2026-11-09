#!/usr/bin/env python3
r"""
Run the real evaluation under several probability-of-collision models and
compare them, on one propagation and one encounter population.

The point is not that a refined Pc model gives different numbers — of course
it does. The point is to say *why* it differs, in a way a report can defend.
A Foster-vs-circular run moves for two independent reasons:

1. **The encounter-plane projection.** Foster scores the component of the
   separation perpendicular to the relative velocity; ``twin.compute_pc`` is
   handed the raw 3D separation at the closest *sampled* timestep and has no
   direction information left to project with. With a 30-second screening
   grid and several km/s of relative speed, a sample can sit a long way along
   the track from the true minimum, and all of that error lands in the
   along-velocity direction. On synthetic fleets the in-plane miss runs about
   0.78-0.84x the raw miss. Through ``exp(-d^2 / 2 sigma^2)`` that is not a
   rounding difference.
2. **The anisotropic covariance**, which is the thing the module was built for.

Reporting a single Foster-vs-circular delta would attribute all of it to (2).
So this script runs four columns, and the middle two are what separate them:

    circular              today's numbers, unchanged — the baseline
    circular-in-plane     same estimator, fed the projected miss  -> isolates (1)
    foster(k=1)           isotropic covariance                    -> cross-checks (1)
    foster(k=8)           anisotropic                             -> adds (2)

``circular-in-plane`` and ``foster(k=1)`` should land on top of each other. If
they do, the decomposition holds and the step between ``foster(k=1)`` and
``foster(k=8)`` is the anisotropy result, cleanly. If they do not, something
is wrong and the comparison should not be reported.

**One propagation, one population.** The library's ``run_split_catalog_evaluation``
stamps its own epoch from ``datetime.now()``, so calling it four times would
screen four slightly different catalogues and the columns would not be paired.
This script therefore does its own screening from the public building blocks,
once, and re-scores that single set of conjunctions under each model. Every
row in the output describes the same physical encounter in all four columns.

Population selection is reported both ways, because they answer different
questions. The paired table scores the encounters the **circular** model puts
above threshold, so the columns are comparable row by row. The counts table
reports how many encounters **each** model admits above threshold on its own,
because a model that flags more conjunctions is itself a result.

Usage:

    PYTHONPATH=src python3 scripts/run_pc_model_comparison.py \
        --catalog data/starlink.tle Starlink \
        --hours 168

    # quick pass while iterating
    PYTHONPATH=src python3 scripts/run_pc_model_comparison.py \
        --catalog data/kuiper.tle Kuiper --hours 24 --limit 20

Writes ``docs/pc_model_comparison.md`` and
``docs/figures/pc_model_comparison.json``.

Cost: the two circular columns are effectively free. The Foster columns
evaluate a quadrature at every step of every lead-time scan, which measured at
roughly 0.5-1 s per encounter on a 10,713-object Starlink run — so a few
minutes for a full 237-encounter column. Expect the upper end: ``--limit``
selects the highest-Pc encounters, and those cross the threshold early and
scan fastest, so a limited pass flatters the per-encounter cost. The
propagation and screening pass, shared by every column, is its own ~2 minutes.

``--limit`` caps the encounter count for a quick pass and ``--models`` selects
a subset of columns.
"""
from __future__ import annotations

import argparse
import datetime
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "docs" / "pc_model_comparison.md"
JSON_PATH = REPO_ROOT / "docs" / "figures" / "pc_model_comparison.json"

# Column order is the argument: baseline, then one effect at a time.
COLUMNS = ("circular", "circular-in-plane", "foster(k=1)", "foster(k=8)")


def _check(path: str) -> str:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        print(f"ERROR: {path} does not exist or is empty.")
        print("See scripts/run_cross_operator_backtest.py's docstring for how to fetch a TLE file.")
        sys.exit(1)
    return path


def _build_models(names):
    """Map column name -> (pc_model, use_in_plane_miss).

    ``circular-in-plane`` is not a different estimator. It is the circular one
    fed the projected miss distance, which is why it isolates the projection
    effect: estimator held constant, input changed.
    """
    from satcollision.pc import CircularPcModel, FosterPcModel

    catalogue = {
        "circular": (None, False),
        "circular-in-plane": (CircularPcModel(), True),
        "foster(k=1)": (FosterPcModel(along_track_ratio=1.0), False),
        "foster(k=8)": (FosterPcModel(along_track_ratio=8.0), False),
    }
    return {name: catalogue[name] for name in names}


def _screen_once(tle_path, catalog_name, hours, step_s, screening_km, epoch="catalogue"):
    """Propagate and screen the catalogue exactly once, under the circular model.

    Uses the public building blocks rather than ``run_split_catalog_evaluation``
    so that the epoch, the propagation and the candidate set are shared by every
    column. The Pc stamped here is the circular one; each column re-scores it.
    """
    from satcollision.fleets import _epoch_from_calendar, load_tle_file
    from satcollision.real_evaluation import assign_two_operators, resolve_epoch
    from satcollision.twin import propagate_window, screen_conjunctions

    objects = assign_two_operators(load_tle_file(tle_path, operator_name=catalog_name))
    now = resolve_epoch(objects, epoch)
    _warn_if_stale(objects, now)
    _, jd0, fr0 = _epoch_from_calendar(now.year, now.month, now.day, now.hour + now.minute / 60.0)
    propagation = propagate_window(objects, jd0, fr0, duration_s=hours * 3600.0, step_s=step_s)
    conjunctions = screen_conjunctions(objects, propagation, screening_distance_km=screening_km)
    return objects, conjunctions, now


def _parse_epoch(value: str):
    import datetime as _dt
    if value in ("catalogue", "now"):
        return value
    return _dt.datetime.fromisoformat(value)


def _warn_if_stale(objects, start) -> None:
    from satcollision.real_evaluation import STALE_TLE_WARNING_DAYS, median_tle_age_days
    age = median_tle_age_days(objects, start)
    print(f"      window starts {start.isoformat(timespec='minutes')}; "
          f"median TLE age there {age:.1f} days")
    if age > STALE_TLE_WARNING_DAYS:
        print(f"      WARNING: element sets are {age:.0f} days old at the window start. SGP4 "
              "error grows quickly with age; results will not be reliable. Use "
              "--epoch catalogue or download a fresh catalogue.")


def _rescore(conjunction, pc_model, use_in_plane, sigma_km):
    """Return a copy of the conjunction as this column sees it.

    The Pc on a Conjunction is what the threshold filter and the output row
    read, so a column that changes the estimator has to change that too, or
    the run would report one model's Pc beside another model's lead times.
    """
    from satcollision.pc import EncounterGeometry, resolve_pc_model
    from satcollision.twin import compute_pc

    geometry = EncounterGeometry.from_conjunction(conjunction)
    if use_in_plane:
        miss_km = geometry.in_plane_miss_km
        return replace(conjunction, miss_distance_km=miss_km,
                       pc=compute_pc(miss_km, sigma_km))
    if pc_model is None:
        return conjunction
    return replace(conjunction, pc=resolve_pc_model(pc_model).pc(geometry, sigma_km))


def _score_column(name, pc_model, use_in_plane, paired, threshold, horizon_s, sigma_km, verbose):
    """Evaluate one column across the shared encounter set."""
    from satcollision.pc import EncounterGeometry
    from satcollision.real_evaluation import evaluate_encounter

    started = time.perf_counter()
    rows = []
    for index, conjunction in enumerate(paired, start=1):
        scored = _rescore(conjunction, pc_model, use_in_plane, sigma_km)
        evaluation = evaluate_encounter(
            scored, threshold=threshold, horizon_s=horizon_s,
            pc_model=None if use_in_plane else pc_model,
        )
        row = evaluation.as_row()
        row["column"] = name
        # Both miss distances on every row, always. ``miss_distance_m`` is
        # whatever that column fed its estimator, which differs between
        # columns by design; these two do not, so a row stays traceable.
        row["raw_miss_m"] = conjunction.miss_distance_km * 1000.0
        row["in_plane_miss_m"] = (
            EncounterGeometry.from_conjunction(conjunction).in_plane_miss_km * 1000.0
        )
        rows.append(row)
        if verbose and index % 25 == 0:
            print(f"      {name}: {index}/{len(paired)} "
                  f"({time.perf_counter() - started:.0f}s elapsed)")
    return rows, time.perf_counter() - started


def _admitted_count(conjunctions, pc_model, use_in_plane, threshold, horizon_s, sigma_km):
    """How many encounters this column would put above threshold on its own.

    Separate from the paired table on purpose: a model that admits more
    conjunctions is a result in itself, but a population that changes between
    columns cannot be compared row by row.
    """
    above = [c for c in conjunctions
             if _rescore(c, pc_model, use_in_plane, sigma_km).pc >= threshold]
    scorable = [c for c in above if c.tca_offset_s >= horizon_s]
    return len(above), len(scorable)


def _median(values):
    return statistics.median(values) if values else None


def _column_summary(rows):
    """Aggregate one column.

    ``median_miss_m`` is the miss distance this column actually *scored*, which
    is the raw one for every column except ``circular-in-plane``. The raw and
    in-plane values are both on every row, so the difference is never implicit.
    """
    if not rows:
        return {"n_encounters": 0}
    gains = [r["lead_time_gain_min"] for r in rows]
    return {
        "n_encounters": len(rows),
        "median_pc": _median([r["pc"] for r in rows]),
        "max_pc": max(r["pc"] for r in rows),
        "median_miss_m": _median([r["miss_distance_m"] for r in rows]),
        "median_raw_miss_m": _median([r["raw_miss_m"] for r in rows]),
        "median_in_plane_miss_m": _median([r["in_plane_miss_m"] for r in rows]),
        "median_lead_time_nocoop_h": _median([r["lead_time_nocoop_min"] for r in rows]) / 60.0,
        "median_lead_time_cooperative_h": _median(
            [r["lead_time_cooperative_min"] for r in rows]) / 60.0,
        "median_lead_time_gain_h": _median(gains) / 60.0,
        "n_undetected_by_nocoop": sum(1 for r in rows if r["lead_time_nocoop_min"] <= 0.0),
    }


def _outcome_breakdown(rows):
    """Split one column's encounters by which regime flags them at all.

    A single median gain hides the case that matters most under a non-circular
    model: an encounter that only the *poorly tracked* regime flags. With a
    large miss distance, inflating sigma raises Pc (probability dilution run
    in reverse), so degraded public tracking can cross the threshold on an
    encounter that precise tracking correctly dismisses. That is not "no
    cooperation detected it earlier" — it is a false alarm, i.e. a manoeuvre
    the operator would have paid fuel for and cooperation would have avoided.
    A lead-time gain is only meaningful where both regimes flag the encounter.
    """
    both = [r for r in rows if r["lead_time_cooperative_min"] > 0 and r["lead_time_nocoop_min"] > 0]
    coop_only = [r for r in rows if r["lead_time_cooperative_min"] > 0 >= r["lead_time_nocoop_min"]]
    nocoop_only = [r for r in rows if r["lead_time_nocoop_min"] > 0 >= r["lead_time_cooperative_min"]]
    neither = [r for r in rows if r["lead_time_cooperative_min"] <= 0 and r["lead_time_nocoop_min"] <= 0]
    gains_both = [r["lead_time_gain_min"] / 60.0 for r in both]
    return {
        "n": len(rows),
        "both_flag": len(both),
        "only_cooperative_flags": len(coop_only),
        "only_no_cooperation_flags": len(nocoop_only),
        "neither_flags": len(neither),
        "median_gain_where_both_flag_h": _median(gains_both),
    }


def _breakdown_table(breakdowns) -> str:
    lines = [
        "| Column | Both regimes flag | Only cooperative flags (no-coop misses it) | "
        "Only no-cooperation flags (false alarm cooperation avoids) | Neither flags | "
        "Median gain where both flag |",
        "|---|---|---|---|---|---|",
    ]
    for name, b in breakdowns.items():
        gain = b["median_gain_where_both_flag_h"]
        lines.append(
            f"| `{name}` | {b['both_flag']} | {b['only_cooperative_flags']} | "
            f"{b['only_no_cooperation_flags']} | {b['neither_flags']} | "
            + (f"{gain:.2f} h" if gain is not None else "—") + " |")
    return "\n".join(lines) + "\n"


def _projection_stats(conjunctions):
    """How much of the raw miss distance the encounter-plane projection removes."""
    from satcollision.pc import EncounterGeometry

    ratios = []
    for conjunction in conjunctions:
        geometry = EncounterGeometry.from_conjunction(conjunction)
        if conjunction.miss_distance_km > 0.0:
            ratios.append(geometry.in_plane_miss_km / conjunction.miss_distance_km)
    if not ratios:
        return {}
    ratios.sort()
    return {
        "n": len(ratios),
        "median": _median(ratios),
        "p10": ratios[int(0.10 * (len(ratios) - 1))],
        "p90": ratios[int(0.90 * (len(ratios) - 1))],
        "min": ratios[0],
        "max": ratios[-1],
    }


def _summary_table(summaries) -> str:
    names = list(summaries)
    lines = [
        "| Column | Encounters | Median Pc | Median miss scored (m) | Median warning, no cooperation | Median gain |",
        "|---|---|---|---|---|---|",
    ]
    for name in names:
        s = summaries[name]
        if not s.get("n_encounters"):
            lines.append(f"| `{name}` | 0 | — | — | — | — |")
            continue
        lines.append(
            f"| `{name}` | {s['n_encounters']} | {s['median_pc']:.2e} | "
            f"{s['median_miss_m']:.0f} | {s['median_lead_time_nocoop_h']:.2f} h | "
            f"{s['median_lead_time_gain_h']:.2f} h |"
        )
    return "\n".join(lines) + "\n"


def _counts_table(counts) -> str:
    lines = [
        "| Column | Above threshold | Scorable after burn-in |",
        "|---|---|---|",
    ]
    for name, (above, scorable) in counts.items():
        lines.append(f"| `{name}` | {above} | {scorable} |")
    return "\n".join(lines) + "\n"


def _consistency_note(summaries) -> str:
    """State whether the decomposition actually held, rather than assuming it."""
    a, b = summaries.get("circular-in-plane"), summaries.get("foster(k=1)")
    if not (a and b and a.get("n_encounters") and b.get("n_encounters")):
        return ("_Both `circular-in-plane` and `foster(k=1)` are needed to check the "
                "decomposition; this run did not produce both._\n")
    pc_a, pc_b = a["median_pc"], b["median_pc"]
    rel = abs(pc_a - pc_b) / max(pc_a, pc_b)
    gap_h = abs(a["median_lead_time_gain_h"] - b["median_lead_time_gain_h"])
    verdict = "holds" if rel < 0.05 and gap_h < 1.0 else "**does not hold**"
    return (
        f"`circular-in-plane` and `foster(k=1)` are the same calculation reached two ways — "
        f"the circular formula at the projected miss, and the Foster integral with an isotropic "
        f"covariance. Their median Pc differs by {rel * 100:.2f}% and their median gain by "
        f"{gap_h:.2f} h, so the decomposition {verdict}.\n\n"
        + ("" if verdict == "holds" else
           "> Because it does not hold, the split between the projection effect and the "
           "anisotropy effect below is not trustworthy and should not be reported until the "
           "discrepancy is understood.\n")
    )


def _render_markdown(payload) -> str:
    run, summaries = payload["run"], payload["summaries"]
    projection = payload["projection"]
    baseline = summaries.get("circular", {})
    anisotropic = summaries.get("foster(k=8)", {})

    parts = [
        "# Probability-of-collision model comparison",
        "",
        f"_Generated by `scripts/run_pc_model_comparison.py`. Propagation window starts "
        f"{run.get('window_start_utc', run['computed_at_utc'])}._",
        "",
        "## What is being compared, and what is held fixed",
        "",
        f"One {run['catalog']} catalogue ({run['n_objects']:,} objects), propagated once with SGP4 "
        f"across a {run['window_hours']:.0f}-hour window at a {run['step_s']:.0f}-second step and "
        f"screened once at {run['screening_km']:.0f} km. Every column below re-scores **that single "
        "set of conjunctions** — same epoch, same orbits, same candidate pairs, same regime sigmas, "
        "same threshold. Only the probability-of-collision estimator changes.",
        "",
        "The paired table scores the encounters the **circular** model puts above threshold, so the "
        "columns describe the same physical encounters row by row. How many encounters each model "
        "admits on its own is reported separately, further down.",
        "",
        "## Why four columns and not two",
        "",
        "A Foster-vs-circular comparison moves for two independent reasons, and only one of them is "
        "the anisotropic covariance the model was built for. The other is that Foster scores the "
        "component of the separation perpendicular to the relative velocity, while the circular "
        "formula is handed the raw 3D separation at the closest *sampled* timestep and has no "
        "direction left to project with. The screening grid is "
        f"{run['step_s']:.0f} seconds wide, so a sample can sit well along the track from the true "
        "minimum, and all of that error is in the along-velocity direction.",
        "",
        "| Column | What it is | What it isolates |",
        "|---|---|---|",
        "| `circular` | today's estimator, unchanged | the baseline every existing result uses |",
        "| `circular-in-plane` | circular formula, projected miss | the projection / TCA-resolution effect |",
        "| `foster(k=1)` | Foster, isotropic covariance | cross-checks the line above |",
        "| `foster(k=8)` | Foster, along-track anisotropy | adds the anisotropy effect |",
        "",
    ]

    if projection:
        parts += [
            "### How much the projection removes",
            "",
            f"Across {projection['n']} screened conjunctions the in-plane miss is a median "
            f"**{projection['median']:.3f}x** the raw miss "
            f"(p10 {projection['p10']:.3f}, p90 {projection['p90']:.3f}, "
            f"range {projection['min']:.3f}-{projection['max']:.3f}). Through "
            "`exp(-d^2 / 2 sigma^2)` a shift of that size is not a rounding difference, which is "
            "why it has to be separated out rather than folded into an anisotropy claim.",
            "",
        ]

    parts += [
        "## Results",
        "",
        _summary_table(summaries),
        "",
        _consistency_note(summaries),
        "",
        "### Encounters admitted above threshold, per model",
        "",
        "A model that flags more conjunctions is a result in its own right, but a population that "
        "changes between columns cannot be compared row by row — which is why the table above holds "
        "the population fixed and this one does not.",
        "",
        _counts_table(payload["counts"]),
        "",
    ]

    breakdowns = payload.get("breakdowns")
    if breakdowns is None:  # older JSON: derive it from the stored rows
        breakdowns = {name: _outcome_breakdown([r for r in payload["encounters"]
                                                if r["column"] == name])
                      for name in summaries}
    parts += [
        "## Which regime flags each encounter",
        "",
        "The median gain above treats every encounter alike. Under a non-circular model that is "
        "misleading, because some encounters are flagged by only one regime, and the two cases "
        "mean opposite things:",
        "",
        "- **Only cooperative flags:** precise tracking sees a real risk that public tracking "
        "never raises. Cooperation is the only way the operator hears about it.",
        "- **Only no-cooperation flags:** the miss distance is large compared with precise "
        "tracking's uncertainty, so precise tracking correctly calls it safe, while degraded "
        "public tracking inflates the uncertainty enough to push Pc over the threshold. That is a "
        "false alarm: an unnecessary, fuel-costly manoeuvre that cooperation avoids. It shows up "
        "as a *negative* gain in the median above, which is why the median alone should not be "
        "quoted.",
        "",
        _breakdown_table(breakdowns),
    ]
    if baseline.get("n_encounters") and anisotropic.get("n_encounters"):
        b8 = breakdowns.get("foster(k=8)", {})
        parts += [
            "## Reading it",
            "",
            f"Under the circular model the median warning-time gain is "
            f"**{baseline['median_lead_time_gain_h']:.2f} h**. Under the anisotropic model "
            f"(`foster(k=8)`) the median over the same encounters is "
            f"**{anisotropic['median_lead_time_gain_h']:.2f} h**, but that number mixes three "
            "different outcomes. Read the breakdown instead: "
            + (f"where both regimes flag an encounter ({b8.get('both_flag')} of {b8.get('n')}), "
               f"cooperation still gives a median **{b8['median_gain_where_both_flag_h']:.2f} h** "
               "more warning; "
               if b8.get("median_gain_where_both_flag_h") is not None else "")
            + (f"in {b8.get('only_no_cooperation_flags')} more, cooperation's benefit is avoiding a "
               "false alarm rather than an earlier warning. "
               if b8.get("only_no_cooperation_flags") else "")
            + "The intermediate columns say how much of the move from the circular model is the "
            "projection and how much is the shape of the error ellipsoid; the report should quote "
            "that split and this breakdown, not the endpoints alone.",
            "",
        ]

    parts += [
        "## Method notes",
        "",
        f"- Threshold {run['pc_threshold']:.0e}, screening horizon {run['horizon_h']:.0f} h, "
        f"combined sigma {run['combined_sigma_km']} km.",
        "- Anisotropy is introduced at **constant trace**: `foster(k=8)` redistributes the same "
        "total uncertainty into the along-track axis rather than adding any, so the comparison "
        "isolates the shape of the error ellipsoid from its size.",
        "- Every encounter row in the JSON carries `pc_model`, `raw_miss_m` and `miss_distance_m`, "
        "so a row can always be traced back to which column produced it.",
        "",
    ]
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", nargs=2, metavar=("TLE", "NAME"), required=True,
                        help="real catalogue to split into two operators")
    parser.add_argument("--hours", type=float, default=168.0, help="propagation window (default 168)")
    parser.add_argument("--step", type=float, default=30.0, help="propagation step in seconds")
    parser.add_argument("--models", nargs="+", default=list(COLUMNS), choices=list(COLUMNS),
                        help="which columns to run (default: all four)")
    parser.add_argument("--rerender", action="store_true",
                        help="rebuild the markdown from the existing JSON without re-running")
    parser.add_argument("--limit", type=int, default=None,
                        help="score at most this many encounters, highest Pc first (quick pass)")
    parser.add_argument("--epoch", default="catalogue",
                        help="window start: 'catalogue' (default; the newest TLE epoch in the "
                             "file, so the run stays inside the data's validity and is "
                             "reproducible), 'now', or an ISO time such as 2026-09-10T12:00")
    args = parser.parse_args()

    if args.rerender:
        if not JSON_PATH.exists():
            print(f"ERROR: {JSON_PATH} not found; run without --rerender first.")
            return 1
        payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        DOC_PATH.write_text(_render_markdown(payload), encoding="utf-8")
        print(f"Re-rendered {DOC_PATH.relative_to(REPO_ROOT)} from the cached run.")
        return 0

    from satcollision.real_evaluation import (
        DEFAULT_EVALUATION_SCREEN_KM, SCREENING_HORIZON_S,
    )
    from satcollision.twin import DEFAULT_COMBINED_SIGMA_KM, DEFAULT_PC_THRESHOLD

    tle_path, catalog_name = _check(args.catalog[0]), args.catalog[1]
    threshold = DEFAULT_PC_THRESHOLD
    horizon_s = SCREENING_HORIZON_S
    sigma_km = DEFAULT_COMBINED_SIGMA_KM
    screening_km = DEFAULT_EVALUATION_SCREEN_KM

    print(f"Screening {catalog_name} once, {args.hours:.0f}h window at {args.step:.0f}s ...")
    started = time.perf_counter()
    objects, conjunctions, computed_at = _screen_once(
        tle_path, catalog_name, args.hours, args.step, screening_km,
        epoch=_parse_epoch(args.epoch))
    cross_operator = [c for c in conjunctions
                      if c.object_a.operator != c.object_b.operator]
    print(f"      {len(objects):,} objects, {len(cross_operator):,} cross-operator close approaches "
          f"({time.perf_counter() - started:.0f}s)")

    # The shared population: what the circular model puts above threshold and
    # the burn-in rule keeps. Every column scores exactly these encounters.
    above = [c for c in cross_operator if c.pc >= threshold]
    paired = [c for c in above if c.tca_offset_s >= horizon_s]
    paired.sort(key=lambda c: -c.pc)
    if args.limit is not None:
        paired = paired[: args.limit]
    print(f"      {len(above)} above threshold, {len(paired)} scored "
          f"({len(above) - len(paired)} inside the burn-in window or dropped by --limit)")

    if not paired:
        print("Nothing above threshold to compare. Try a longer --hours or a denser catalogue.")
        return 1

    models = _build_models(args.models)
    all_rows, summaries, counts, timings = [], {}, {}, {}
    for position, (name, (pc_model, use_in_plane)) in enumerate(models.items(), start=1):
        print(f"[{position}/{len(models)}] {name} ...")
        rows, elapsed = _score_column(name, pc_model, use_in_plane, paired,
                                      threshold, horizon_s, sigma_km, verbose=True)
        all_rows += rows
        summaries[name] = _column_summary(rows)
        counts[name] = _admitted_count(cross_operator, pc_model, use_in_plane,
                                       threshold, horizon_s, sigma_km)
        timings[name] = elapsed
        s = summaries[name]
        print(f"      median Pc {s['median_pc']:.3e}, median gain "
              f"{s['median_lead_time_gain_h']:.2f} h, {elapsed:.0f}s")

    payload = {
        "run": {
            "catalog": catalog_name,
            "n_objects": len(objects),
            "window_hours": args.hours,
            "step_s": args.step,
            "screening_km": screening_km,
            "pc_threshold": threshold,
            "horizon_h": horizon_s / 3600.0,
            "combined_sigma_km": sigma_km,
            "computed_at_utc": computed_at.isoformat(timespec="seconds"),
            "window_start_utc": computed_at.isoformat(timespec="seconds"),
            "epoch_mode": args.epoch,
            "n_cross_operator": len(cross_operator),
            "n_above_threshold": len(above),
            "n_paired": len(paired),
            "columns": list(models),
            "seconds_per_column": timings,
        },
        "projection": _projection_stats(cross_operator),
        "summaries": summaries,
        "breakdowns": {name: _outcome_breakdown([r for r in all_rows if r["column"] == name])
                       for name in summaries},
        "counts": counts,
        "encounters": all_rows,
    }

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"\nWrote {DOC_PATH.relative_to(REPO_ROOT)} and {JSON_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
