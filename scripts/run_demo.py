#!/usr/bin/env python3
"""
End-to-end walkthrough of every implemented layer, in one run:

    simulation -> digital twin (+ Kalman refinement) -> signal abstraction
    -> signing/tamper-evident log -> federation (plain + secure/MPC)
    -> deconfliction -> 3-scenario evaluation -> robustness sweep
    -> federated learning (FedAvg + DP-SGD) -> incentives/reputation
    -> plain-language incident summaries

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
from satcollision.identity import (
    build_operator_identities, public_key_registry, sign_signal, sign_report,
    verify_signed, SignedLog, OperatorIdentity,
)
from satcollision.secure_aggregation import (
    build_mpc_states, secure_sum_aggregate, secure_mean_aggregate, compute_masked_share,
)
from satcollision.kalman import refine_pc_for_conjunction
from satcollision.federated_learning import compare_scenarios
from satcollision.reputation import (
    ReputationState, simulate_reputation_rounds, reputation_weighted_aggregate,
    tailor_alert,
)
from satcollision.summaries import (
    build_incident_report, summarize_alert, summarize_conjunction,
    summarize_federation_round, summarize_maneuver_plan, summarize_reputation,
)


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

    # ---- 2b. Kalman-filter state estimation: a real evolving sigma instead of the fixed constant ----
    print("\n[2b] Kalman-filter refinement: re-deriving this encounter's combined sigma from an")
    print("     actual tracking history instead of the fixed DEFAULT_COMBINED_SIGMA_KM=0.5km...")
    kalman_result = refine_pc_for_conjunction(actor_conj, jd0, fr0, seed=0)
    print(f"    fixed-sigma Pc   (sigma=0.5km, constant)      : {kalman_result['fixed_sigma_pc']:.3e}")
    print(f"    Kalman-derived Pc (sigma={kalman_result['kalman_sigma_km']:.4f}km at TCA, evolved): "
          f"{kalman_result['kalman_sigma_pc']:.3e}")
    print(f"    ({kalman_result['seconds_since_last_measurement']:.0f}s since the simulated last "
          "tracking update at TCA -- this is exactly what the fixed constant can't represent)")

    # ---- 3. Signal abstraction ----
    print("\n[3] Abstracting the flagged conjunction into shareable signals...")
    sig_a, sig_b = abstract_conjunction(actor_conj)
    print("    signal from A:", sig_a)
    print("    signal from B:", sig_b)
    print("    (no position/velocity/orbital-element field exists on this object —")
    print("     see signal.assert_no_raw_state, exercised in tests/test_twin_and_signal.py)")

    # ---- 3b. Cryptographic identity, signing, and tamper-evident logging ----
    print("\n[3b] Cryptographic signing (real Ed25519 keys) + hash-chained log...")
    operator_names = [actor_a.operator, actor_b.operator, "Op0", "Op1", "Op2", "Op3", "Op4", "Adversary"]
    identities = build_operator_identities(operator_names)
    trusted_keys = public_key_registry(identities)
    fed_log = SignedLog()

    signed_a = sign_signal(identities[actor_a.operator], sig_a)
    signed_b = sign_signal(identities[actor_b.operator], sig_b)
    fed_log.append(signed_a)
    fed_log.append(signed_b)
    print(f"    {actor_a.operator} signs its signal; {actor_b.operator} verifies it: "
          f"{verify_signed(signed_a, trusted_keys)}")

    print("    tamper check: mutating the signed payload after signing...")
    import dataclasses as _dc
    tampered_signal = _dc.replace(signed_a.signal, geometry_class="head-on")
    tampered = _dc.replace(signed_a, signal=tampered_signal)
    print(f"      verify(tampered content, same signature) = {verify_signed(tampered, trusted_keys)}  "
          "(must be False)")

    print("    impersonation check: an attacker signs with their own key, claiming to be "
          f"'{actor_a.operator}'...")
    forger = OperatorIdentity.generate(actor_a.operator)  # attacker has NO real identity for this operator
    forged = sign_signal(forger, sig_a)
    print(f"      verify(forged signature, real trusted registry) = {verify_signed(forged, trusted_keys)}  "
          "(must be False -- registry only trusts the real keypair on file)")

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

    print("    every report above also gets signed and appended to the same tamper-evident log")
    print("    (note: the adversary's SPOOFED CONTENT still verifies -- signing proves who sent")
    print("     a message and that it wasn't altered in transit, not that its contents are honest;")
    print("     that is what Byzantine-robust aggregation above, and secure aggregation, are for)...")
    for report in all_reports:
        fed_log.append(sign_report(identities[report.operator], report))
    chain_ok, chain_reason = fed_log.verify_chain(trusted_keys)
    print(f"    hash-chained log: {len(fed_log.entries)} entries, tail_hash={fed_log.tail_hash()[:16]}..., "
          f"verify_chain() = {chain_ok}")

    print("    now simulating a cover-up: someone edits an already-logged entry after the fact...")
    entries = fed_log.entries
    edited_report = _dc.replace(entries[2].signed.report, counts=np.array([0.0, 0.0, 0.0]))
    edited_signed = _dc.replace(entries[2].signed, report=edited_report)
    object.__setattr__(fed_log._entries[2], "signed", edited_signed)
    chain_ok_after, chain_reason_after = fed_log.verify_chain(trusted_keys)
    print(f"    verify_chain() after the edit = {chain_ok_after}  reason: {chain_reason_after}")

    # ---- 4b. Secure aggregation: same 6 reports, but the "aggregator" never sees a plaintext one ----
    print("\n[4b] Secure aggregation on the same 6 reports (Bonawitz-style pairwise-masked")
    print("     additive secret sharing over a 127-bit field, real X25519 key agreement)...")
    reports_by_operator = {r.operator: r for r in all_reports}
    mpc_states = build_mpc_states(list(reports_by_operator.keys()))
    for operator in reports_by_operator:
        share = compute_masked_share(mpc_states, reports_by_operator, operator)
        print(f"      {operator:10s} plaintext={reports_by_operator[operator].counts}   "
              f"masked share (first coord) = {share[0]}")
    secure_mean = secure_mean_aggregate(mpc_states, reports_by_operator)
    print(f"    plain mean_aggregate()      : {mean_est}")
    print(f"    secure_mean_aggregate()     : {secure_mean}")
    print("    (identical result -- the aggregator computed the same mean, but its own")
    print("     arithmetic never touched a single operator's plaintext count vector; the")
    print("     'masked share' values above are exactly what it saw instead. Note this loses")
    print("     the Byzantine-robustness demonstrated above -- the adversary's spoofed value")
    print("     is still baked into this mean, since masked shares can't be trimmed/Krummed.")
    print("     See README 'Trust model' for why these two protections don't compose for free.)")

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

    # ---- 8. Real federated learning: FedAvg + DP-SGD training an actual model ----
    print("\n[8] Federated learning: FedAvg + DP-SGD (Opacus) training a real risk-classifier")
    print("    across Alpha/Beta/Gamma's own non-IID local encounter data, evaluated on a held-out")
    print("    set mixing every operator's regime (never any single operator's own distribution)...")
    fl_result = compare_scenarios(["Alpha", "Beta", "Gamma"], n_samples_per_operator=400, n_rounds=8,
                                   local_epochs=2, seed=42)
    print(f"    centralized (pools raw data, no privacy -- upper bound) : {fl_result['centralized']}")
    print(f"    local-only  (each operator alone, mean across 3)        : {fl_result['local_only_mean']}")
    print(f"    federated, no DP  (FedAvg only)                         : {fl_result['federated_no_dp']}")
    print(f"    federated + DP-SGD (FedAvg + Opacus)                    : {fl_result['federated_dp']}")
    print(f"    final-round (epsilon, delta=1e-5) per operator under DP-SGD: "
          f"{fl_result['federated_dp_final_round_epsilons']}")
    print("    (federated should land close to centralized without ever pooling raw data; DP-SGD")
    print("     costs a little more accuracy in exchange for the formal epsilon above on every")
    print("     operator's shared update -- see README 'Trust model' for what that epsilon means.)")

    # ---- 9. Incentives: reputation earned from observable behaviour ----
    print("\n[9] Incentive/reputation layer: 20 reporting periods, 6 operators, 4 strategies.")
    print("    Reputation is built only from things the federation can already see -- whether a")
    print("    counterparty attests the encounter you reported (or the one you stayed silent about),")
    print("    how far your counts sit from the robust consensus, and whether you contributed at all...")
    rep = simulate_reputation_rounds(n_rounds=20, defect_from_round=12, seed=7)
    print("    operator   strategy     score trajectory (every 4th round)            final   alert tier")
    for operator, behaviour in rep["behaviours"].items():
        traj = rep["state"].trajectory(operator)
        sparse = "  ".join(f"{traj[i]:.2f}" for i in range(3, len(traj), 4))
        print(f"    {operator:10s} {behaviour:11s} {sparse}      {rep['final_scores'][operator]:.2f}    "
              f"{rep['final_tiers'][operator]}")
    print("    (Zeta is the interesting one: it behaves exactly like an honest operator for 12 rounds,")
    print(f"     peaks at {max(rep['state'].trajectory('Zeta')[:12]):.2f}, then defects -- and because reputation falls ~3.5x faster")
    print("     than it rises, 8 rounds of lying cost more than 12 rounds of honesty bought. Banking")
    print("     good behaviour to spend on one big lie is a net loss, which is the point.)")

    print("\n    aggregation error vs. the honest operators' own truth, mean of the last 5 rounds:")
    print(f"      plain mean            : {rep['mean_error_last_5']:.3f}")
    print(f"      trimmed mean          : {rep['trimmed_error_last_5']:.3f}   (robust, but memoryless)")
    print(f"      reputation-weighted   : {rep['reputation_error_last_5']:.3f}   (carries history across rounds)")

    print("\n    the same 6 reports from step [4], but aggregated with reputation weights")
    print("    (the adversary has spent five rounds being caught, and no longer has influence)...")
    demo_state = ReputationState(scores={**{f"Op{i}": 0.9 for i in range(5)}, "Adversary": 0.12})
    weighted_est = reputation_weighted_aggregate(all_reports, demo_state)
    print(f"      plain mean            : {mean_est}   error = {aggregation_error(mean_est, honest):.3f}")
    print(f"      trimmed mean          : {trimmed_est}   error = {aggregation_error(trimmed_est, honest):.3f}")
    print(f"      reputation-weighted   : {weighted_est}   error = "
          f"{aggregation_error(weighted_est, honest):.3f}")

    print("\n    reciprocity -- what each tier actually receives when an alert is shared:")
    for tier in ("full", "standard", "degraded", "suspended"):
        print(f"      {tier:10s} {sorted(tailor_alert(sig_a, tier).keys())}")
    print("    ...and the hard safety floor: a CRITICAL alert is delivered in full to everyone,")
    print("    whatever their reputation --")
    critical_signal = _dc.replace(sig_a, severity_tier="critical")
    print(f"      suspended operator, critical alert: {sorted(tailor_alert(critical_signal, 'suspended').keys())}")
    print("    (reputation allocates influence and privilege; it must never be a mechanism for")
    print("     withholding a collision warning, because the debris harms third parties who had")
    print("     no part in the misbehaviour. See README 'Trust model'.)")

    # ---- 10. Plain-language incident summaries ----
    print("\n[10] Plain-language summaries of everything above -- deterministic templates,")
    print("     no language model anywhere: a safety summary has to be reproducible, auditable,")
    print("     and incapable of inventing a number that was never computed...")

    print("\n     the flagged encounter, as its OWN operator sees it (full fidelity):")
    print(f"       {summarize_conjunction(actor_conj)}")
    print("\n     the same encounter for a non-technical reader (audience='executive'):")
    print(f"       {summarize_conjunction(actor_conj, audience='executive')}")

    print("\n     the same encounter as the COUNTERPARTY sees it, at three reputation tiers --")
    print("     note the summary is rendered from the tailored alert dict, so a lower tier")
    print("     cannot leak through a template edit (summaries.assert_no_undisclosed_terms):")
    for tier in ("full", "standard", "degraded"):
        print(f"       [{tier}] {summarize_alert(tailor_alert(sig_b, tier))}")

    print("\n     the agreed maneuver, written as an instruction for one side:")
    print(f"       {summarize_maneuver_plan(plan, viewpoint_operator=plan.secondary_operator)}")

    print("\n     what step [4]'s federation round actually concluded:")
    print("       " + summarize_federation_round(
        n_operators=len(all_reports),
        plain_estimate=mean_est, robust_estimate=trimmed_est,
        plain_error=aggregation_error(mean_est, honest),
        robust_error=aggregation_error(trimmed_est, honest),
        quarantined=["Adversary"],
    ))

    print("\n     what the quarantined operator is told about its own standing:")
    print(f"       {summarize_reputation(demo_state.score('Adversary'), demo_state.tier('Adversary'), 'Adversary')}")

    print("\n     the whole thing as one incident report -- the detecting operator's own copy:")
    owner_report = build_incident_report(
        actor_conj,
        tailor_alert(sig_b, "full"),
        plan=plan,
        viewpoint_operator=plan.primary_operator,
        log_tail_hash=fed_log.tail_hash(),
    )
    for line in owner_report.as_text().splitlines():
        print(f"       {line}")

    print("\n     ...and the same incident as a DEGRADED counterparty receives it. The private")
    print("     view isn't hidden from this copy, it was never passed in (conjunction=None):")
    counterparty_report = build_incident_report(
        None,
        tailor_alert(sig_b, "degraded"),
        plan=plan,
        viewpoint_operator=plan.secondary_operator,
        log_tail_hash=fed_log.tail_hash(),
    )
    for line in counterparty_report.as_text().splitlines():
        print(f"       {line}")

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
