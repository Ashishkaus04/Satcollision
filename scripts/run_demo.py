#!/usr/bin/env python3
"""
End-to-end walkthrough of every implemented layer, in one run:

    simulation -> digital twin -> signal abstraction -> federation
    -> deconfliction -> 3-scenario evaluation -> robustness sweep

Run with:  PYTHONPATH=src python3 scripts/run_demo.py
Writes a plain-text report to docs/sample_run.md as a side effect.
"""
import io
import sys
from contextlib import redirect_stdout

import numpy as np

from satcollision.fleets import build_synthetic_fleets, inject_actor_encounter, _epoch_from_calendar
from satcollision.twin import propagate_window, screen_conjunctions, DEFAULT_PC_THRESHOLD
from satcollision.signal import abstract_conjunction
from satcollision.federation import OperatorReport, mean_aggregate, trimmed_mean_aggregate, krum_aggregate, \
    inject_adversarial_report, aggregation_error
from satcollision.deconfliction import negotiate_maneuver, simulate_uncoordinated_vs_negotiated
from satcollision.evaluate import run_three_scenario_comparison, robustness_sweep


def main():
    print("=" * 78)
    print("FEDERATED DIGITAL TWINS FOR SATELLITE CONSTELLATION COLLISION AVOIDANCE")
    print("End-to-end demonstration run")
    print("=" * 78)

    # ---- 1. Simulation layer ----
    print("\n[1] Building synthetic fleets (3 operators + ambient debris)...")
    objects = build_synthetic_fleets(seed=42)
    by_op = {}
    for o in objects:
        by_op[o.operator] = by_op.get(o.operator, 0) + 1
    print("    population:", by_op, f"  (total {len(objects)} tracked objects)")

    print("    engineering one genuine close approach for the walkthrough...")
    actor_a, actor_b = inject_actor_encounter(objects, tca_offset_min=30.0)
    print(f"    actor A: {actor_a.label}   actor B: {actor_b.label}")

    # ---- 2. Digital twin: screening + Pc ----
    print("\n[2] Propagating 40-minute window and screening for conjunctions...")
    epoch_days, jd0, fr0 = _epoch_from_calendar(2026, 9, 9, 0.0)
    samples = propagate_window(objects, jd0, fr0, duration_s=40 * 60, step_s=10.0)
    conjunctions = screen_conjunctions(objects, samples, screening_distance_km=25.0)
    print(f"    {len(conjunctions)} conjunction(s) found within the 25km screening volume:")
    for c in conjunctions:
        flag = "  <== crosses Pc threshold" if c.pc >= DEFAULT_PC_THRESHOLD else ""
        print(f"      {c.pair_label:24s} TCA={c.tca_offset_s/60:5.1f}min  "
              f"miss={c.miss_distance_km*1000:8.1f}m  Pc={c.pc:.2e}  "
              f"geometry={c.geometry_class}{flag}")

    actor_conj = next(
        c for c in conjunctions
        if {c.object_a.norad_id, c.object_b.norad_id} == {actor_a.norad_id, actor_b.norad_id}
    )

    # ---- 3. Signal abstraction ----
    print("\n[3] Abstracting the flagged conjunction into shareable signals...")
    sig_a, sig_b = abstract_conjunction(actor_conj)
    print("    signal from A:", sig_a)
    print("    signal from B:", sig_b)
    print("    (no position/velocity/orbital-element field exists on this object —")
    print("     see signal.assert_no_raw_state, exercised in tests/test_twin_and_signal.py)")

    # ---- 4. Federation: aggregation under attack ----
    print("\n[4] Federation aggregation: 5 honest operators + 1 adversary reporting")
    print("    period counts (threshold/elevated/critical encounters)...")
    rng = np.random.default_rng(1)
    honest = [OperatorReport(operator=f"Op{i}", counts=rng.poisson(3, size=3).astype(float)) for i in range(5)]
    adversary = inject_adversarial_report(
        OperatorReport(operator="Adversary", counts=np.array([2.0, 1.0, 0.0])),
        attack_type="spoof_high", magnitude=50.0,
    )
    all_reports = honest + [adversary]
    mean_est = mean_aggregate(all_reports)
    trimmed_est = trimmed_mean_aggregate(all_reports, trim_fraction=0.2)
    krum_est = krum_aggregate(all_reports, n_byzantine=1)
    print(f"    plain mean   : {mean_est}   error vs honest truth = {aggregation_error(mean_est, honest):.3f}")
    print(f"    trimmed mean : {trimmed_est}   error = {aggregation_error(trimmed_est, honest):.3f}")
    print(f"    Krum         : {krum_est}   error = {aggregation_error(krum_est, honest):.3f}")

    # ---- 5. Deconfliction ----
    print("\n[5] Maneuver deconfliction negotiation...")
    plan = negotiate_maneuver(sig_a, sig_b)
    print("   ", plan)
    deconf_stats = simulate_uncoordinated_vs_negotiated(n_trials=20000)
    print(f"    uncoordinated independent guessing: safe {deconf_stats['uncoordinated_safe_fraction']:.1%} of the time")
    print(f"    negotiated protocol:                safe {deconf_stats['negotiated_safe_fraction']:.1%} of the time")

    # ---- 6. Three-scenario comparison ----
    print("\n[6] Three-scenario comparison on this same engineered encounter...")
    comparison = run_three_scenario_comparison(seed=42, tca_offset_min=30.0)
    for sc in comparison["scenarios"].values():
        print(f"    {sc.name:22s} lead_time={sc.detection_lead_time_s/60:6.2f} min   "
              f"raw_data_exposed={sc.raw_data_exposed_pct:5.1f}%   "
              f"cross_operator_signal={sc.cross_operator_signal_shared}")

    # ---- 7. Robustness sweep ----
    print("\n[7] Robustness sweep: aggregation error vs. fraction of dishonest operators...")
    sweep = robustness_sweep(n_trials=60)
    print("    adv_frac   mean_err   trimmed_mean_err   krum_err")
    for i, frac in enumerate(sweep["adversary_fractions"]):
        print(f"    {frac:>6.0%}     {sweep['mean_error'][i]:7.3f}      {sweep['trimmed_mean_error'][i]:7.3f}"
              f"            {sweep['krum_error'][i]:7.3f}")

    print("\n" + "=" * 78)
    print("Done. See README.md for what's built vs. future work, and")
    print("major_project_final_definition.md (project root) for the full writeup.")
    print("=" * 78)


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    text = buf.getvalue()
    sys.stdout.write(text)
    with open("docs/sample_run.md", "w", encoding="utf-8") as fh:
        fh.write("# Sample run output\n\n```\n" + text + "```\n")
