import numpy as np

from satcollision.federation import (
    OperatorReport, mean_aggregate, trimmed_mean_aggregate, krum_aggregate,
    inject_adversarial_report, aggregation_error, apply_local_dp_noise,
)
from satcollision.deconfliction import negotiate_maneuver, simulate_uncoordinated_vs_negotiated
from satcollision.signal import AbstractedSignal


def _honest_reports(n=6, seed=1):
    rng = np.random.default_rng(seed)
    return [OperatorReport(operator=f"Op{i}", counts=rng.poisson(3, size=3).astype(float)) for i in range(n)]


def test_trimmed_mean_and_krum_beat_plain_mean_under_attack():
    honest = _honest_reports()
    adversary = inject_adversarial_report(
        OperatorReport(operator="Adv", counts=np.array([2.0, 1.0, 0.0])),
        attack_type="spoof_high", magnitude=50.0,
    )
    reports = honest + [adversary]

    mean_err = aggregation_error(mean_aggregate(reports), honest)
    trimmed_err = aggregation_error(trimmed_mean_aggregate(reports, trim_fraction=0.2), honest)
    krum_err = aggregation_error(krum_aggregate(reports, n_byzantine=1), honest)

    assert trimmed_err < mean_err
    assert krum_err < mean_err


def test_local_dp_noise_preserves_nonnegativity_and_shape():
    report = OperatorReport(operator="Op0", counts=np.array([3.0, 1.0, 0.0]))
    noisy = apply_local_dp_noise(report, epsilon=1.0, seed=0)
    assert noisy.counts.shape == report.counts.shape
    assert (noisy.counts >= 0).all()


def test_negotiate_maneuver_is_deterministic_and_complementary():
    sig_a = AbstractedSignal(encounter_id="ENC-1-2-100", sender_operator="Alpha", local_object_label="Alpha-1",
                              tca_offset_s=100.0, geometry_class="crossing", severity_tier="elevated")
    sig_b = AbstractedSignal(encounter_id="ENC-1-2-100", sender_operator="Beta", local_object_label="Beta-2",
                              tca_offset_s=100.0, geometry_class="crossing", severity_tier="elevated")
    plan1 = negotiate_maneuver(sig_a, sig_b)
    plan2 = negotiate_maneuver(sig_a, sig_b)  # must be reproducible, no randomness
    assert plan1 == plan2
    assert plan1.primary_direction != plan1.secondary_direction
    assert {plan1.primary_operator, plan1.secondary_operator} == {"Alpha", "Beta"}


def test_negotiated_deconfliction_beats_uncoordinated():
    stats = simulate_uncoordinated_vs_negotiated(n_trials=5000, seed=1)
    assert stats["negotiated_safe_fraction"] == 1.0
    assert 0.4 < stats["uncoordinated_safe_fraction"] < 0.6  # ~50% by construction
