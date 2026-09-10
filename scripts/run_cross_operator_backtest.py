#!/usr/bin/env python3
r"""
Cross-operator historical backtest: the real version of the project's core
claim, run on genuine current orbital data from two independent operators.

run_backtest.py (Starlink-vs-itself) validates that the propagation /
screening / Pc pipeline behaves correctly against real orbital dynamics —
useful, but it doesn't touch the actual novel contribution, because a
single operator already has full internal visibility into its own fleet
and needs no federation to see it. This script screens for real
conjunctions BETWEEN two different real operators, and for any that cross
the Pc threshold, runs the full pipeline this project is actually about:
signal abstraction (Section 3), then a negotiated deconfliction plan
(Section 4, Enhancement 2) — all on two real, independently-downloaded TLE
catalogs, with no synthetic data involved on either side.

Usage:
    PYTHONPATH=src python3 scripts/run_cross_operator_backtest.py \
        data/starlink.tle Starlink \
        data/oneweb.tle OneWeb \
        [duration_hours]

duration_hours is optional (default 24). Widening it (e.g. 168 for a full
week) increases the chance of catching a rarer real close approach, at a
roughly proportional runtime cost (~30-40s per 24h against a combined
catalog of ~10,000-12,000 real objects, based on prior benchmarking).

Get a second operator's TLE file the same way as the first — in a normal
browser or a plain PowerShell window (not through any automation/bridge
tool, which may sit behind a restrictive proxy):

    Invoke-WebRequest -Uri "https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle" -OutFile data\oneweb.tle
    Invoke-WebRequest -Uri "https://celestrak.org/NORAD/elements/gp.php?GROUP=kuiper&FORMAT=tle" -OutFile data\kuiper.tle

Then check it's actually non-empty before running this script:
    Get-Item data\oneweb.tle | Select-Object Length
"""
import sys
import datetime
from pathlib import Path

import numpy as np

from satcollision.fleets import load_tle_file, _epoch_from_calendar
from satcollision.twin import propagate_window, screen_conjunctions, compute_pc, DEFAULT_PC_THRESHOLD
from satcollision.signal import abstract_conjunction
from satcollision.deconfliction import negotiate_maneuver
from satcollision.federation import OperatorReport, mean_aggregate


def _load_operator(path: str, operator_name: str):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        print(f"ERROR: {path} does not exist or is empty.")
        print("See the module docstring at the top of this script for how to fetch a real TLE file.")
        sys.exit(1)
    objects = load_tle_file(path, operator_name=operator_name)
    print(f"  loaded {len(objects)} real {operator_name} objects from {path}")
    return objects


def main():
    if len(sys.argv) not in (5, 6):
        print(__doc__)
        sys.exit(1)
    path_a, name_a, path_b, name_b = sys.argv[1:5]
    duration_hours = float(sys.argv[5]) if len(sys.argv) == 6 else 24.0

    print(f"Loading two real operator catalogs: {name_a} and {name_b} ...")
    objects_a = _load_operator(path_a, name_a)
    objects_b = _load_operator(path_b, name_b)
    objects = objects_a + objects_b
    print(f"  combined catalog: {len(objects)} real tracked objects\n")

    now = datetime.datetime.now(datetime.timezone.utc)
    epoch_days, jd0, fr0 = _epoch_from_calendar(now.year, now.month, now.day, now.hour + now.minute / 60.0)

    duration_s = duration_hours * 60 * 60
    step_s = 30.0
    print(f"Propagating across a {duration_s/3600:.0f}-hour window (step={step_s}s)...")
    prop = propagate_window(objects, jd0, fr0, duration_s=duration_s, step_s=step_s)

    print("Screening for conjunctions (<=25km)...")
    all_conjunctions = screen_conjunctions(objects, prop, screening_distance_km=25.0)

    cross_operator = [c for c in all_conjunctions if c.object_a.operator != c.object_b.operator]
    same_operator = [c for c in all_conjunctions if c.object_a.operator == c.object_b.operator]
    print(f"  {len(all_conjunctions)} total conjunctions found "
          f"({len(cross_operator)} cross-operator, {len(same_operator)} within the same operator)\n")

    if not cross_operator:
        print(f"No {name_a}<->{name_b} pairs came within 25km of each other in this 24-hour window.")
        print("That's a real, honest result, not a bug -- different constellations often occupy")
        print("genuinely separated altitude/inclination shells. Report the closest approach found")
        print("anyway, for context:")
        # Even with a wider net, nothing above may have been close; re-screen at 200km just to
        # report *some* honest distance rather than silence.
        wide = screen_conjunctions(objects, prop, screening_distance_km=200.0)
        wide_cross = [c for c in wide if c.object_a.operator != c.object_b.operator]
        if wide_cross:
            closest = min(wide_cross, key=lambda c: c.miss_distance_km)
            print(f"  closest {name_a}<->{name_b} approach within 200km: {closest.pair_label} "
                  f"miss={closest.miss_distance_km:.2f}km (Pc negligible at this distance)")
        else:
            print(f"  no {name_a}<->{name_b} pair came within 200km either -- these two "
                  f"constellations occupy clearly separated orbital regimes right now.")
        return

    cross_operator.sort(key=lambda c: -c.pc)
    above_threshold = [c for c in cross_operator if c.pc >= DEFAULT_PC_THRESHOLD]
    print(f"{len(above_threshold)} cross-operator conjunction(s) cross the Pc threshold "
          f"({DEFAULT_PC_THRESHOLD:.0e}):\n")

    for c in above_threshold[:20]:
        print(f"  {c.pair_label:32s} TCA={c.tca_offset_s/60:7.1f}min  "
              f"miss={c.miss_distance_km*1000:9.1f}m  Pc={c.pc:.2e}  geometry={c.geometry_class}")
        sig_a, sig_b = abstract_conjunction(c)
        plan = negotiate_maneuver(sig_a, sig_b)
        print(f"      signal A: {sig_a}")
        print(f"      signal B: {sig_b}")
        print(f"      negotiated plan: {plan.primary_operator} maneuvers primary (+), "
              f"{plan.secondary_operator} holds secondary (-)")
        print()

    if not above_threshold and cross_operator:
        print("  none crossed the Pc threshold. Closest cross-operator approaches found "
              "(useful context even without a threshold crossing):")
        for c in cross_operator[:5]:
            print(f"    {c.pair_label:32s} miss={c.miss_distance_km*1000:9.1f}m  "
                  f"Pc={c.pc:.2e}  geometry={c.geometry_class}")

    # Real per-operator reporting-period counts, straight from what was actually found --
    # replaces the synthetic Poisson-sampled counts used in run_demo.py's federation
    # walkthrough with genuine numbers from this run.
    print("\nReal per-operator report vectors (threshold/elevated/critical), from this run:")
    all_signals = []
    for c in cross_operator:
        result = abstract_conjunction(c)
        if result:
            all_signals.extend(result)
    reports = []
    for op_name in (name_a, name_b):
        report = OperatorReport(
            operator=op_name,
            counts=np.array([
                sum(1 for s in all_signals if s.sender_operator == op_name and s.severity_tier == tier)
                for tier in ("threshold", "elevated", "critical")
            ], dtype=float),
        )
        reports.append(report)
        print(f"  {op_name}: {report.counts}")

    # Only 2 real operators here, so plain averaging is the honest tool --
    # Byzantine-robust aggregation (trimmed-mean/Krum) only means something
    # with enough reporters that a dishonest minority can be outvoted; that
    # is validated separately, with a realistic operator count, in
    # evaluate.robustness_sweep() using synthetic reports.
    federation_aggregate = mean_aggregate(reports)
    print(f"  federation-level aggregate this period: {federation_aggregate}")


if __name__ == "__main__":
    main()
