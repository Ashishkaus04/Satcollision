"""
Federation layer: cross-operator aggregation of abstracted signals.

Per Section 5 of the project definition, the federation combines abstracted
signals across the whole participating federation — not just pairwise — to
surface patterns a single operator's own data can't show (e.g. "risk is
concentrated in the 550-600km/53deg shell this reporting period"). Each
operator's contribution to that aggregate is a small fixed-length count
vector (encounters observed this period, broken down by severity tier),
never a raw signal list — this is itself a second, coarser abstraction
layer on top of ``signal.py``'s per-encounter one.

Three aggregation strategies are implemented so their robustness can be
compared head-to-head (this is exactly what Enhancement #1 — adversarial-
resistant aggregation — needs to demonstrate):

* :func:`mean_aggregate` — plain average. Fast, but a single dishonest
  operator can shift it arbitrarily (unbounded influence).
* :func:`trimmed_mean_aggregate` — drops the highest/lowest
  ``trim_fraction`` of values per dimension before averaging. Cheap and
  tolerates a bounded fraction of dishonest reporters.
* :func:`krum_aggregate` — selects the single most "central" report (the
  one with the smallest sum of squared distances to its closest
  neighbours), the classic Blanchard et al. Byzantine-robust rule; tolerant
  of up to roughly n/2 - 2 adversarial reporters.

:func:`apply_local_dp_noise` implements a simple Laplace-mechanism local
differential privacy step (each operator perturbs its own count vector
before it ever leaves their boundary), and :func:`inject_adversarial_report`
simulates the standard attack models named in the project definition's tech
stack (sign-flip, scaling/spoof, zeroing/suppression, free-riding).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

SEVERITY_TIERS = ("threshold", "elevated", "critical")


@dataclass
class OperatorReport:
    operator: str
    counts: np.ndarray  # length len(SEVERITY_TIERS), one count per tier
    is_adversarial: bool = False  # ground-truth label, for evaluation only —
    # never used by the aggregation functions themselves, which must work
    # without knowing in advance who is dishonest.


def build_operator_report(operator: str, abstracted_signals: list, severity_tiers=SEVERITY_TIERS) -> OperatorReport:
    """Summarize one operator's abstracted signals from one reporting period
    into the small fixed-length count vector that actually crosses into the
    federation aggregate."""
    counts = np.zeros(len(severity_tiers))
    for sig in abstracted_signals:
        if sig.sender_operator != operator:
            continue
        if sig.severity_tier in severity_tiers:
            counts[severity_tiers.index(sig.severity_tier)] += 1
    return OperatorReport(operator=operator, counts=counts)


def apply_local_dp_noise(report: OperatorReport, epsilon: float = 1.0, seed: int | None = None) -> OperatorReport:
    """Local differential privacy: add Laplace noise (scale = 1/epsilon,
    since one additional/removed encounter changes any single count by at
    most 1) to an operator's own report before sharing it. Smaller
    ``epsilon`` = more privacy, more noise.
    """
    rng = np.random.default_rng(seed)
    noisy = report.counts + rng.laplace(loc=0.0, scale=1.0 / epsilon, size=report.counts.shape)
    noisy = np.clip(noisy, a_min=0.0, a_max=None)  # counts can't be negative
    return OperatorReport(operator=report.operator, counts=noisy, is_adversarial=report.is_adversarial)


def inject_adversarial_report(report: OperatorReport, attack_type: str, magnitude: float = 5.0) -> OperatorReport:
    """Simulate one dishonest operator, using the attack models named in the
    project definition's evaluation plan.

    * ``"spoof_high"`` — inflates counts to force other operators into
      unnecessary maneuvers (a "cry wolf" attack).
    * ``"suppress"`` — zeroes counts to hide the reporter's own risky
      behaviour from the federation.
    * ``"sign_flip"`` — reports the negative of its true counts (clipped to
      zero downstream), a standard Byzantine attack model in the FL
      literature.
    * ``"free_ride"`` — reports all zeros regardless of true activity,
      benefiting from others' signals while contributing none of its own.
    """
    counts = report.counts.copy()
    if attack_type == "spoof_high":
        counts = counts + magnitude
    elif attack_type == "suppress":
        counts = np.zeros_like(counts)
    elif attack_type == "sign_flip":
        counts = -counts * magnitude
    elif attack_type == "free_ride":
        counts = np.zeros_like(counts)
    else:
        raise ValueError(f"unknown attack_type: {attack_type}")
    return OperatorReport(operator=report.operator, counts=counts, is_adversarial=True)


def mean_aggregate(reports: list[OperatorReport]) -> np.ndarray:
    return np.mean([r.counts for r in reports], axis=0)


def trimmed_mean_aggregate(reports: list[OperatorReport], trim_fraction: float = 0.2) -> np.ndarray:
    matrix = np.stack([r.counts for r in reports])  # shape (n_reports, n_dims)
    n = matrix.shape[0]
    k = int(np.floor(trim_fraction * n))
    result = np.zeros(matrix.shape[1])
    for dim in range(matrix.shape[1]):
        col = np.sort(matrix[:, dim])
        trimmed = col[k: n - k] if n - k > k else col
        result[dim] = trimmed.mean()
    return result


def krum_aggregate(reports: list[OperatorReport], n_byzantine: int = 1) -> np.ndarray:
    """Classic Krum (Blanchard et al., 2017): score each report by the sum
    of squared distances to its ``n - n_byzantine - 2`` closest neighbours,
    and return the single lowest-scoring (most "central") report.
    """
    matrix = np.stack([r.counts for r in reports])
    n = matrix.shape[0]
    n_keep = max(1, n - n_byzantine - 2)
    scores = []
    for i in range(n):
        dists = np.linalg.norm(matrix - matrix[i], axis=1)
        dists = np.sort(dists)[1:]  # drop the zero self-distance
        scores.append(np.sum(dists[:n_keep]))
    best_idx = int(np.argmin(scores))
    return matrix[best_idx]


def aggregation_error(estimate: np.ndarray, honest_reports: list[OperatorReport]) -> float:
    """Distance between an aggregate and the "ground truth" mean computed
    from honest reports only — the metric the evaluation harness uses to
    compare aggregation strategies under attack (Section 7 of the project
    definition).
    """
    true_mean = np.mean([r.counts for r in honest_reports], axis=0)
    return float(np.linalg.norm(estimate - true_mean))
