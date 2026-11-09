#!/usr/bin/env python3
r"""
End-to-end demonstration of the federated conjunction-screening pipeline.

Runs the core path in front of an audience, in order:

    1. Orbital propagation     real TLEs, real SGP4
    2. Digital twin            screening -> time of closest approach -> Pc
    3. Signal abstraction      the privacy boundary, and what crosses it
    4. Federated aggregation   correlated risk view from abstracted signals

Step 3 is the one worth pausing on. It prints the full private state of an
encounter — both objects' positions and velocities, the exact miss distance,
the exact collision probability — directly above the handful of coarse fields
that are permitted to leave the operator, and calls the structural check that
enforces the difference. The contrast is the argument.

The run has two halves. The live half propagates a real catalogue and screens
it while the audience watches, which is what makes the pipeline credible. The
second half reports the headline figures from the full 168-hour evaluation,
which takes minutes rather than seconds and therefore cannot be run live.
Reading those from the stored run rather than retyping them applies the same
discipline as the dashboard: the figures quoted cannot drift from the figures
measured.

Usage:

    PYTHONPATH=src python3 scripts/run_midsem_demo.py

    # smaller and faster, if the room's machine is slow
    PYTHONPATH=src python3 scripts/run_midsem_demo.py --catalog data/kuiper.tle Kuiper

Typical runtime is under twenty seconds. Check it on the machine you will
present from, not only on your own.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_JSON = REPO_ROOT / "docs" / "figures" / "real_evaluation.json"

RULE = "=" * 78
THIN = "-" * 78

def banner(title: str) -> None:
    print()
    print(RULE)
    print(f"  {title}")
    print(RULE)


def step(n: int, title: str) -> None:
    print()
    print(f"[{n}/4] {title}")
    print(THIN)


def title_card(catalog: str) -> None:
    banner("FEDERATED DIGITAL TWINS FOR SATELLITE COLLISION AVOIDANCE")
    print("  Privacy-preserving conjunction screening across operator boundaries")
    print()
    print(f"  Live pipeline on the {catalog} catalogue, followed by the measured")
    print("  result from the full 168-hour evaluation.")


def check(path: str) -> str:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        print(f"\nERROR: {path} is missing or empty.")
        print("See scripts/run_cross_operator_backtest.py for how to fetch a TLE file.")
        sys.exit(1)
    return path


def layer_1_and_2(tle_path: str, name: str, hours: float, step_s: float, screening_km: float):
    """Propagate a real catalogue and screen it, live."""
    from satcollision.fleets import _epoch_from_calendar, load_tle_file
    from satcollision.real_evaluation import assign_two_operators
    from satcollision.twin import DEFAULT_PC_THRESHOLD, propagate_window, screen_conjunctions

    step(1, "ORBITAL PROPAGATION — real catalogue, real SGP4")
    objects = assign_two_operators(load_tle_file(tle_path, operator_name=name))
    now = datetime.datetime.now(datetime.timezone.utc)
    _, jd0, fr0 = _epoch_from_calendar(now.year, now.month, now.day, now.hour + now.minute / 60.0)

    print(f"  Catalogue            {name}")
    print(f"  Objects              {len(objects):,}")
    print(f"  Window               {hours:.0f} h from {now:%Y-%m-%d %H:%M} UTC, {step_s:.0f} s step")
    print("  Ownership            assigned by NORAD parity into Operator-A / Operator-B")
    print("                       (the only assigned quantity; orbits and geometry are real)")

    started = time.perf_counter()
    propagation = propagate_window(objects, jd0, fr0, duration_s=hours * 3600.0, step_s=step_s)
    print(f"  Propagated           {len(objects) * propagation.positions.shape[1]:,} states "
          f"in {time.perf_counter() - started:.1f} s")

    step(2, "DIGITAL TWIN — screening, closest approach, collision probability")
    started = time.perf_counter()
    conjunctions = screen_conjunctions(objects, propagation, screening_distance_km=screening_km)
    cross = [c for c in conjunctions if c.object_a.operator != c.object_b.operator]
    above = [c for c in cross if c.pc >= DEFAULT_PC_THRESHOLD]

    print(f"  Screening radius     {screening_km:.0f} km (k-d tree broad phase)")
    print(f"  Close approaches     {len(conjunctions):,} found in {time.perf_counter() - started:.1f} s")
    print(f"  Cross-operator       {len(cross):,}")
    print(f"  Above Pc {DEFAULT_PC_THRESHOLD:.0e}       {len(above):,}")

    if not above:
        print("\n  No encounter crossed the threshold in this short window.")
        print("  Try a longer --hours; the stored full run below is the reported result.")
        return None

    worst = max(above, key=lambda c: c.pc)
    print()
    print("  Highest-risk cross-operator encounter in this window:")
    print(f"    Objects            {worst.object_a.label}  <->  {worst.object_b.label}")
    print(f"    Miss distance      {worst.miss_distance_km * 1000.0:,.0f} m")
    print(f"    Relative speed     {worst.relative_speed_km_s:.2f} km/s ({worst.geometry_class})")
    print(f"    Time to closest    {worst.tca_offset_s / 3600.0:.1f} h")
    print(f"    Collision prob.    {worst.pc:.3e}")
    return worst


def layer_3(conjunction) -> list:
    """The privacy boundary: show exactly what is allowed to cross it."""
    from satcollision.signal import abstract_conjunction, assert_no_raw_state

    step(3, "SIGNAL ABSTRACTION — the privacy boundary")

    print("  What the operator holds privately (never transmitted):")
    print(f"    position A         {conjunction.position_a_km.round(3).tolist()} km")
    print(f"    velocity A         {conjunction.velocity_a_km_s.round(4).tolist()} km/s")
    print(f"    position B         {conjunction.position_b_km.round(3).tolist()} km")
    print(f"    velocity B         {conjunction.velocity_b_km_s.round(4).tolist()} km/s")
    print(f"    miss distance      {conjunction.miss_distance_km:.6f} km")
    print(f"    exact Pc           {conjunction.pc:.6e}")

    signals = abstract_conjunction(conjunction)
    if not signals:
        print("\n  (Below threshold — no signal would be emitted.)")
        return []

    print()
    print("  What actually crosses the operator boundary:")
    for signal in signals:
        assert_no_raw_state(signal)
        print(f"    from {signal.sender_operator}:")
        for f in fields(signal):
            print(f"      {f.name:<20s} {getattr(signal, f.name)!r}")

    private_values = 6 + 3 + 3 + 3 + 3  # scalars + four 3-vectors
    shared_values = len(fields(signals[0]))
    print()
    print(f"  No position, no velocity, no orbital element, no miss distance, no exact Pc.")
    print(f"  assert_no_raw_state() enforces this structurally — a future field named")
    print(f"  'position' or 'miss_distance' makes the system fail loudly, not leak quietly.")
    print(f"  Roughly {private_values} private numbers reduce to {shared_values} coarse fields.")
    return signals


def layer_4(signals) -> None:
    """Aggregate abstracted signals into a correlated risk view."""
    from satcollision.federation import SEVERITY_TIERS, build_operator_report, mean_aggregate

    step(4, "FEDERATED AGGREGATION — correlated risk view")

    operators = sorted({s.sender_operator for s in signals})
    reports = [build_operator_report(op, signals) for op in operators]

    print(f"  Severity tiers       {list(SEVERITY_TIERS)}")
    for report in reports:
        print(f"  {report.operator:<20s} counts {report.counts.astype(int).tolist()}")

    aggregate = mean_aggregate(reports)
    print(f"  Aggregate            {aggregate.round(3).tolist()}")
    print()
    print("  Both operators independently observed the same physical encounter and")
    print("  produced the same encounter identifier, so the hub can correlate them")
    print("  without either operator revealing which satellite it was concerned about.")
    print()
    print("  This is the point the rest of the system builds on: robust aggregation,")
    print("  secure aggregation and signed contributions all wrap this same step.")


def stored_evaluation() -> None:
    """Report the headline numbers from the full run, which is too slow to run live."""
    banner("MEASURED RESULT — full 168-hour evaluation")

    if not EVAL_JSON.exists():
        print(f"  {EVAL_JSON.relative_to(REPO_ROOT)} not found.")
        print("  Run: PYTHONPATH=src python3 scripts/run_real_evaluation.py "
              "--catalog data/starlink.tle Starlink")
        return

    payload = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    head = payload["headline"]
    summary = head.get("summary", {})

    print(f"  Run computed         {head.get('computed_at_utc', 'unknown')}")
    print(f"  Catalogue            {head.get('catalog')} — {head.get('n_objects', 0):,} objects")
    print(f"  Window               {head.get('window_hours', 0):.0f} h at "
          f"{head.get('step_s', 0):.0f} s, screened at {head.get('screening_km', 0):.0f} km")
    print(f"  Above threshold      {head.get('n_above_threshold')}")
    print(f"  Excluded as burn-in  {head.get('n_excluded_burn_in')}")
    print(f"  Scored               {head.get('n_scored')}")

    if summary.get("n_encounters"):
        nocoop_h = summary["median_lead_time_nocoop_min"] / 60.0
        coop_h = summary["median_lead_time_cooperative_min"] / 60.0
        gain_h = summary["median_lead_time_gain_min"] / 60.0
        print()
        print(f"  Median warning, no cooperation      {nocoop_h:6.1f} h")
        print(f"  Median warning, federated           {coop_h:6.1f} h  "
              f"(capped by the {head.get('screening_horizon_h', 72):.0f} h decision horizon)")
        print(f"  Median gain from cooperation        {gain_h:6.1f} h")
        exposure = summary.get("raw_data_exposed_pct", {})
        if exposure:
            print()
            print(f"  Raw trajectory data exposed — full sharing  "
                  f"{exposure.get('full_sharing', 0):.0f}%")
            print(f"  Raw trajectory data exposed — federated     "
                  f"{exposure.get('federated', 0):.0f}%")

    negative = payload.get("negative")
    if negative and negative.get("encounters"):
        worst = max(negative["encounters"], key=lambda r: r["pc"])
        print()
        print("  Genuine cross-operator threshold crossing (no assigned ownership):")
        print(f"    {worst['encounter_id']}   {worst['miss_distance_m']:,.0f} m   "
              f"Pc {worst['pc']:.2e}   {worst['geometry_class']}")

    print()
    print("  Under the standard circular probability model; a refined non-circular")
    print("  model is in development.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", nargs=2, metavar=("TLE", "NAME"),
                        default=["data/starlink.tle", "Starlink"],
                        help="catalogue for the live half (default: Starlink)")
    parser.add_argument("--hours", type=float, default=12.0,
                        help="live propagation window (default 12; keep it short)")
    parser.add_argument("--step", type=float, default=30.0, help="propagation step in seconds")
    parser.add_argument("--screening-km", type=float, default=5.0, help="screening radius")
    args = parser.parse_args()

    tle_path, name = check(args.catalog[0]), args.catalog[1]
    title_card(args.catalog[1])
    started = time.perf_counter()

    worst = layer_1_and_2(tle_path, name, args.hours, args.step, args.screening_km)
    if worst is not None:
        signals = layer_3(worst)
        if signals:
            layer_4(signals)

    stored_evaluation()

    print()
    print(RULE)
    print(f"  Demonstration complete in {time.perf_counter() - started:.1f} s.")
    print(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
