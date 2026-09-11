import numpy as np

from satcollision.federation import OperatorReport, mean_aggregate
from satcollision.secure_aggregation import (
    build_mpc_states,
    compute_masked_share,
    secure_sum_aggregate,
    secure_mean_aggregate,
    _pairwise_seed,
    FIELD_PRIME,
    SCALE,
)


def _reports(seed=0, n=5):
    rng = np.random.default_rng(seed)
    names = [f"Op{i}" for i in range(n)]
    return {name: OperatorReport(operator=name, counts=rng.poisson(3, size=3).astype(float)) for name in names}


def test_secure_sum_matches_plaintext_sum_exactly():
    reports = _reports()
    states = build_mpc_states(list(reports.keys()))

    secure_total = secure_sum_aggregate(states, reports)
    true_total = np.sum([r.counts for r in reports.values()], axis=0)

    assert np.allclose(secure_total, true_total, atol=1e-6)


def test_secure_sum_handles_fractional_dp_noised_counts():
    rng = np.random.default_rng(3)
    reports = {
        f"Op{i}": OperatorReport(operator=f"Op{i}", counts=rng.normal(loc=3.0, scale=1.5, size=3))
        for i in range(6)
    }
    states = build_mpc_states(list(reports.keys()))

    secure_total = secure_sum_aggregate(states, reports)
    true_total = np.sum([r.counts for r in reports.values()], axis=0)
    assert np.allclose(secure_total, true_total, atol=1e-6)


def test_secure_mean_matches_plain_mean_aggregate():
    reports = _reports(seed=7, n=8)
    states = build_mpc_states(list(reports.keys()))

    secure_mean = secure_mean_aggregate(states, reports)
    plain_mean = mean_aggregate(list(reports.values()))
    assert np.allclose(secure_mean, plain_mean, atol=1e-6)


def test_single_masked_share_does_not_reveal_plaintext():
    reports = _reports(seed=1, n=5)
    states = build_mpc_states(list(reports.keys()))

    share = compute_masked_share(states, reports, "Op0")
    plaintext_scaled = [int(round(v * SCALE)) for v in reports["Op0"].counts]

    # The mask should dominate: the masked share looks nothing like the
    # plaintext value scaled into the same units, and sits near the scale
    # of the whole field rather than near the scale of the plaintext.
    assert share != plaintext_scaled
    assert any(abs(s - p) > 10 ** 30 for s, p in zip(share, plaintext_scaled))


def test_masked_shares_are_deterministic_given_same_keys_and_reports():
    reports = _reports(seed=2, n=4)
    states = build_mpc_states(list(reports.keys()))

    share1 = compute_masked_share(states, reports, "Op1")
    share2 = compute_masked_share(states, reports, "Op1")
    assert share1 == share2


def test_share_difference_isolates_plaintext_difference_under_fixed_peer_masks():
    names = ["Op0", "Op1", "Op2", "Op3"]
    states = build_mpc_states(names)
    reports_a = {n: OperatorReport(operator=n, counts=np.array([1.0, 2.0, 3.0])) for n in names}
    reports_b = dict(reports_a)
    reports_b["Op1"] = OperatorReport(operator="Op1", counts=np.array([5.0, 2.0, 3.0]))

    share_a = compute_masked_share(states, reports_a, "Op1")
    share_b = compute_masked_share(states, reports_b, "Op1")

    diff = [(b - a) % FIELD_PRIME for a, b in zip(share_a, share_b)]
    expected = [int(round((5.0 - 1.0) * SCALE)) % FIELD_PRIME, 0, 0]
    assert diff == expected


def test_pairwise_seed_is_symmetric_regardless_of_argument_order():
    states = build_mpc_states(["Alpha", "Beta"])
    assert _pairwise_seed(states, "Alpha", "Beta") == _pairwise_seed(states, "Beta", "Alpha")


def test_secure_sum_requires_at_least_one_report():
    states = build_mpc_states([])
    try:
        secure_sum_aggregate(states, {})
        assert False, "expected ValueError"
    except ValueError:
        pass
