import numpy as np
import pytest

from satcollision.federation import OperatorReport, mean_aggregate
from satcollision.reputation import (
    ReputationParams, ReputationState, RoundEvidence,
    alert_tier, attest_participation, evaluate_round, gather_attestations,
    parse_encounter_participants, reputation_weighted_aggregate, reputation_weights,
    simulate_reputation_rounds, tailor_alert, update_scores,
)
from satcollision.signal import AbstractedSignal


def _signal(encounter_id: str, operator: str, severity: str = "threshold") -> AbstractedSignal:
    return AbstractedSignal(
        encounter_id=encounter_id,
        sender_operator=operator,
        local_object_label=f"{operator}-1",
        tca_offset_s=1200.0,
        geometry_class="crossing",
        severity_tier=severity,
    )


# --------------------------------------------------------------- attestation

def test_parse_encounter_participants_round_trips_and_rejects_junk():
    assert parse_encounter_participants("ENC-41000-42000-1800") == (41000, 42000)
    with pytest.raises(ValueError):
        parse_encounter_participants("not-an-encounter")


def test_attest_participation_is_true_only_for_the_owner():
    eid = "ENC-41000-42000-1800"
    assert attest_participation(eid, {41000, 41001}) is True
    assert attest_participation(eid, {42000}) is True
    assert attest_participation(eid, {43000}) is False


def test_gather_attestations_returns_one_set_per_operator():
    attested = gather_attestations(
        ["ENC-41000-42000-600", "ENC-41001-43000-900"],
        {"Alpha": {41000, 41001}, "Beta": {42000}, "Gamma": {43000}},
    )
    assert attested["Alpha"] == {"ENC-41000-42000-600", "ENC-41001-43000-900"}
    assert attested["Beta"] == {"ENC-41000-42000-600"}
    assert attested["Gamma"] == {"ENC-41001-43000-900"}


# ------------------------------------------------------------------ evidence

def _two_operator_round(alpha_signals, beta_signals):
    signals = alpha_signals + beta_signals
    reports = [
        OperatorReport(operator="Alpha", counts=np.array([float(len(alpha_signals)), 0.0, 0.0])),
        OperatorReport(operator="Beta", counts=np.array([float(len(beta_signals)), 0.0, 0.0])),
    ]
    ownership = {"Alpha": {41000}, "Beta": {42000}}
    return evaluate_round(["Alpha", "Beta"], signals, reports, ownership)


def test_honest_reporting_earns_full_corroboration():
    eid = "ENC-41000-42000-600"
    evidence = _two_operator_round([_signal(eid, "Alpha")], [_signal(eid, "Beta")])
    assert evidence["Alpha"].corroboration == 1.0
    assert evidence["Alpha"].n_matched == 1
    assert evidence["Alpha"].participation == 1.0


def test_withholding_a_signal_is_detected_without_raw_data():
    """Beta is a party to the encounter Alpha reported, and says nothing."""
    eid = "ENC-41000-42000-600"
    evidence = _two_operator_round([_signal(eid, "Alpha")], [])
    assert evidence["Beta"].n_involved == 1
    assert evidence["Beta"].n_missed == 1
    assert evidence["Beta"].corroboration == 0.0
    assert evidence["Beta"].participation == 0.0


def test_fabricated_encounter_is_unbacked_and_costs_corroboration():
    real = "ENC-41000-42000-600"
    phantom = "ENC-90000-90001-600"  # neither operator owns those catalogue numbers
    evidence = _two_operator_round(
        [_signal(real, "Alpha"), _signal(phantom, "Alpha")], [_signal(real, "Beta")]
    )
    assert evidence["Alpha"].n_unbacked == 1
    assert evidence["Alpha"].corroboration == pytest.approx(0.5)  # 1 matched / (1 involved + 1 unbacked)


def test_consensus_agreement_penalises_the_outlier_not_the_crowd():
    signals = [_signal("ENC-41000-42000-600", "Alpha"), _signal("ENC-41000-42000-600", "Beta")]
    reports = [
        OperatorReport(operator="Alpha", counts=np.array([2.0, 1.0, 0.0])),
        OperatorReport(operator="Beta", counts=np.array([2.0, 1.0, 0.0])),
        OperatorReport(operator="Gamma", counts=np.array([3.0, 1.0, 0.0])),
        OperatorReport(operator="Loud", counts=np.array([60.0, 40.0, 30.0])),
    ]
    ownership = {"Alpha": {41000}, "Beta": {42000}, "Gamma": {43000}, "Loud": {44000}}
    evidence = evaluate_round(list(ownership), signals, reports, ownership)
    assert evidence["Loud"].consensus_agreement < 0.1
    assert evidence["Alpha"].consensus_agreement > 0.9


def test_round_score_stays_in_unit_interval():
    best = RoundEvidence("Op", corroboration=1.0, consensus_agreement=1.0, participation=1.0)
    worst = RoundEvidence("Op", corroboration=0.0, consensus_agreement=0.0, participation=0.0)
    assert best.round_score() == pytest.approx(1.0)
    assert worst.round_score() == pytest.approx(0.0)


# -------------------------------------------------------------- score update

def _state_with(score: float, operator: str = "Op") -> ReputationState:
    return ReputationState(scores={operator: score})


def test_reputation_falls_faster_than_it_rises():
    """The defence against 'bank honest rounds, then spend them on one lie'."""
    params = ReputationParams()
    start = 0.5
    good = update_scores(_state_with(start), {"Op": RoundEvidence("Op", 1.0, 1.0, 1.0)}, params)
    bad = update_scores(_state_with(start), {"Op": RoundEvidence("Op", 0.0, 0.0, 0.0)}, params)
    gained = good.score("Op") - start
    lost = start - bad.score("Op")
    assert lost > gained


def test_one_bad_round_costs_more_than_several_good_rounds_gained():
    params = ReputationParams()
    state = ReputationState()
    for _ in range(3):
        state = update_scores(state, {"Op": RoundEvidence("Op", 1.0, 1.0, 1.0)}, params)
    peak = state.score("Op")
    state = update_scores(state, {"Op": RoundEvidence("Op", 0.0, 0.0, 0.0)}, params)
    assert state.score("Op") < params.initial_score < peak


def test_scores_are_bounded_and_history_is_recorded():
    params = ReputationParams()
    state = ReputationState()
    for _ in range(50):
        state = update_scores(state, {"Op": RoundEvidence("Op", 0.0, 0.0, 0.0)}, params)
    assert state.score("Op") >= params.floor
    assert state.rounds == 50
    assert len(state.history) == 50
    assert len(state.trajectory("Op")) == 50


def test_a_fresh_identity_starts_below_an_established_honest_one():
    """Whitewashing — dumping a damaged identity for a new one — must not pay."""
    state = ReputationState()
    for _ in range(10):
        state = update_scores(state, {"Established": RoundEvidence("Established", 1.0, 1.0, 1.0)})
    assert state.score("NeverSeenBefore") == state.params.initial_score
    assert state.score("NeverSeenBefore") < state.score("Established")


# ------------------------------------------------------- weighted aggregation

def test_quarantined_operator_gets_zero_weight():
    reports = [
        OperatorReport(operator="Good", counts=np.array([2.0, 1.0, 0.0])),
        OperatorReport(operator="Bad", counts=np.array([50.0, 50.0, 50.0])),
    ]
    state = ReputationState(scores={"Good": 0.9, "Bad": 0.05})
    weights = reputation_weights(reports, state)
    assert weights[1] == 0.0
    assert weights[0] == pytest.approx(1.0)


def test_weights_fall_back_to_uniform_if_everyone_is_quarantined():
    reports = [
        OperatorReport(operator="A", counts=np.array([1.0, 0.0, 0.0])),
        OperatorReport(operator="B", counts=np.array([2.0, 0.0, 0.0])),
    ]
    state = ReputationState(scores={"A": 0.01, "B": 0.02})
    weights = reputation_weights(reports, state)
    assert np.allclose(weights, np.array([0.5, 0.5]))


def test_reputation_weighted_aggregate_beats_plain_mean_against_a_known_liar():
    honest = [OperatorReport(operator=f"Op{i}", counts=np.array([2.0, 1.0, 0.0])) for i in range(4)]
    liar = OperatorReport(operator="Liar", counts=np.array([60.0, 40.0, 30.0]), is_adversarial=True)
    reports = honest + [liar]
    truth = np.mean([r.counts for r in honest], axis=0)
    state = ReputationState(scores={**{f"Op{i}": 0.9 for i in range(4)}, "Liar": 0.1})

    plain_err = np.linalg.norm(mean_aggregate(reports) - truth)
    weighted_err = np.linalg.norm(reputation_weighted_aggregate(reports, state) - truth)
    assert weighted_err < plain_err


# ------------------------------------------------------------- alert tiering

def test_alert_tier_boundaries():
    params = ReputationParams()
    assert alert_tier(0.05, params) == "suspended"
    assert alert_tier(0.30, params) == "degraded"
    assert alert_tier(0.50, params) == "standard"
    assert alert_tier(0.95, params) == "full"


def test_tailor_alert_discloses_progressively_more_with_reputation():
    signal = _signal("ENC-41000-42000-600", "Alpha", severity="elevated")
    suspended = tailor_alert(signal, "suspended")
    degraded = tailor_alert(signal, "degraded")
    standard = tailor_alert(signal, "standard")
    full = tailor_alert(signal, "full")

    assert set(suspended) < set(degraded) < set(standard) < set(full)
    assert "severity_tier" not in degraded
    assert "geometry_class" not in standard
    assert full["geometry_class"] == "crossing"
    assert full["deconfliction_eligible"] is True


def test_critical_alerts_are_never_degraded_whatever_the_reputation():
    """Safety override: reputation allocates influence, never collision warnings."""
    critical = _signal("ENC-41000-42000-600", "Alpha", severity="critical")
    alert = tailor_alert(critical, "suspended")
    assert alert["tier"] == "full"
    assert alert["safety_override"] is True
    assert alert["geometry_class"] == "crossing"


def test_safety_override_can_be_disabled_explicitly():
    critical = _signal("ENC-41000-42000-600", "Alpha", severity="critical")
    alert = tailor_alert(critical, "suspended", safety_override=False)
    assert alert["tier"] == "suspended"
    assert "geometry_class" not in alert


def test_tailor_alert_rejects_an_unknown_tier():
    with pytest.raises(ValueError):
        tailor_alert(_signal("ENC-41000-42000-600", "Alpha"), "platinum")


# ---------------------------------------------------------------- simulation

def test_simulation_separates_honest_from_misbehaving_operators():
    result = simulate_reputation_rounds(n_rounds=20, seed=7)
    scores = result["final_scores"]
    honest = [op for op, b in result["behaviours"].items() if b == "honest"]
    cheats = [op for op, b in result["behaviours"].items() if b in ("free_rider", "spoofer")]
    assert min(scores[op] for op in honest) > max(scores[op] for op in cheats)
    assert all(result["final_tiers"][op] == "full" for op in honest)


def test_on_off_attacker_ends_below_where_it_started():
    """Banking honest rounds then defecting must be a net loss, not a shield."""
    result = simulate_reputation_rounds(n_rounds=20, defect_from_round=12, seed=7)
    trajectory = result["state"].trajectory("Zeta")
    peak = max(trajectory[:12])
    assert trajectory[-1] < ReputationParams().initial_score < peak


def test_reputation_weighted_aggregation_beats_memoryless_defences_over_rounds():
    result = simulate_reputation_rounds(n_rounds=20, seed=7)
    assert result["reputation_error_last_5"] < result["trimmed_error_last_5"]
    assert result["trimmed_error_last_5"] < result["mean_error_last_5"]
