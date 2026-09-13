"""
Incentive / reputation layer (Enhancement #4 of the project definition).

Every layer built so far protects the federation from a *message*: signing
stops an outsider forging one, robust aggregation stops one round's lying
insider skewing one aggregate, secure aggregation stops the aggregator
reading one report. None of them give an operator any reason to keep
participating honestly over time — a Byzantine-robust aggregate quietly
discards an adversary's report and then greets it identically next round,
and a free-rider that never contributes a signal receives exactly the same
alerts as an operator carrying real reporting cost. That is the gap this
module closes: it is the part of the system that makes honest participation
the *individually* rational strategy rather than merely the collectively
desirable one.

Three pieces:

1. **Evidence** (:func:`evaluate_round`) — a per-round, per-operator
   observation of behaviour, built only from things the federation can see
   without any raw orbital data crossing a boundary:

   * *corroboration* — an encounter involves exactly two objects, and the
     encounter id (``ENC-<norad_a>-<norad_b>-<tca>``) names both. So every
     operator can check **locally** whether one of those objects is its own
     (:func:`attest_participation`) and answer with a single boolean. An
     operator that is a party to an encounter someone else reported, and
     never reported it itself, is withholding; a signal whose encounter no
     one — not even its sender — attests to owning is unbacked.
   * *consensus agreement* — how far this operator's period count vector
     sits from the Byzantine-robust (trimmed-mean) consensus of all
     reports, scaled by the spread the honest population actually shows
     that round rather than by a magic constant.
   * *participation* — did it report at all.

2. **Score update** (:func:`update_scores`) — an exponentially-weighted
   score in [0, 1] with **asymmetric** rates: reputation rises slowly and
   falls fast. This is the standard defence against the "on-off" or
   build-then-betray attack, where an adversary banks honest rounds and
   spends them on one big lie; here the lie costs more than the rounds that
   paid for it (:func:`simulate_reputation_rounds` demonstrates exactly
   this against an ``on_off`` operator). New joiners start at
   ``initial_score`` — deliberately below where an honest operator settles,
   so abandoning a damaged identity and rejoining fresh ("whitewashing")
   is a downgrade, not an escape.

3. **Consequences** — what makes it an incentive rather than a scoreboard:

   * :func:`reputation_weighted_aggregate` weights each operator's
     influence on the federated result by its score (quarantining anyone
     below the threshold entirely). Unlike trimmed-mean/Krum, this carries
     *memory* across rounds: an operator that lied last round has reduced
     influence this round even if this round's report looks plausible.
   * :func:`alert_tier` / :func:`tailor_alert` make the *detail* an
     operator receives back from the federation proportional to its score.
     A quarantined free-rider still learns that something is happening; it
     stops learning the geometry, timing, and deconfliction partner that
     make an alert operationally actionable, all of which exist only
     because other operators paid to produce them.

**Safety override.** ``tailor_alert`` never degrades a ``critical``-tier
alert, whatever the recipient's reputation. Reputation governs influence
and privilege; it must never become a mechanism for withholding a
collision warning from an operator whose satellite is about to be hit,
because the people harmed by that are third parties (everyone downstream
of the debris) who had no part in the misbehaviour. This is a deliberate
design limit, not an oversight, and it is the honest answer to "what if
reputation is wrong about someone".

**What this does not solve.** Sybil attacks — reputation is only as strong
as the cost of a new identity, and here that cost comes entirely from
``identity.py``'s registry being permissioned (a real federation would
anchor it in launch licensing, which is already a heavyweight regulatory
identity). Collusion — a coalition that mutually attests each other's
phantom encounters raises each other's corroboration; the bound is that
encounter ids name NORAD objects from a public catalogue, so a coalition
can only fabricate encounters between objects it genuinely owns, which
limits the lie to its own fleet. And reputation is a *behavioural*
signal, not a correctness proof: a well-meaning operator with a
mis-calibrated Pc pipeline looks, from the outside, somewhat like a liar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .federation import (
    OperatorReport, build_operator_report, mean_aggregate, trimmed_mean_aggregate,
)
from .signal import AbstractedSignal

# Ordered worst -> best. The tier an operator sits in decides how much of a
# shared alert it gets to see (see tailor_alert).
ALERT_TIERS = ("suspended", "degraded", "standard", "full")


@dataclass(frozen=True)
class ReputationParams:
    """Tuning constants, kept in one place so the demo and the tests agree.

    ``fall_rate > rise_rate`` is the important one: it is what makes an
    "act honest for N rounds, then lie once" strategy lose more than it
    banked.
    """

    initial_score: float = 0.50      # where a newly admitted operator starts
    rise_rate: float = 0.10          # EWMA step when a round looks better than the current score
    fall_rate: float = 0.35          # ... and when it looks worse. Deliberately ~3.5x faster.
    corroboration_weight: float = 0.40
    consensus_weight: float = 0.40
    participation_weight: float = 0.20
    floor: float = 0.02              # never exactly zero: an operator must be able to recover
    ceiling: float = 1.00
    quarantine_threshold: float = 0.25   # below this, zero weight in the aggregate
    degraded_threshold: float = 0.45     # below this, alerts are stripped to a bare flag
    standard_threshold: float = 0.70     # below this, alerts lose geometry/partner detail
    weight_exponent: float = 2.0     # >1 sharpens the influence gap between honest and suspect
    consensus_scale_floor: float = 0.5   # keeps agreement finite when every report is identical


@dataclass(frozen=True)
class RoundEvidence:
    """One operator's observable behaviour in one reporting period."""

    operator: str
    corroboration: float          # [0,1] — reported encounters it is genuinely party to
    consensus_agreement: float    # [0,1] — closeness of its counts to the robust consensus
    participation: float          # 1.0 if it submitted a report at all, else 0.0
    n_involved: int = 0           # encounters it attested being a party to
    n_matched: int = 0            # ... of which it also reported
    n_missed: int = 0             # ... of which it stayed silent about
    n_unbacked: int = 0           # signals it sent for encounters nobody owns

    def round_score(self, params: ReputationParams = ReputationParams()) -> float:
        """Collapse this round's evidence into a single [0,1] observation."""
        total_w = params.corroboration_weight + params.consensus_weight + params.participation_weight
        score = (
            params.corroboration_weight * self.corroboration
            + params.consensus_weight * self.consensus_agreement
            + params.participation_weight * self.participation
        ) / total_w
        return float(min(1.0, max(0.0, score)))


@dataclass
class ReputationState:
    """The federation's running view of every operator's trustworthiness."""

    scores: dict[str, float] = field(default_factory=dict)
    rounds: int = 0
    history: list[dict[str, float]] = field(default_factory=list)
    params: ReputationParams = field(default_factory=ReputationParams)

    def score(self, operator: str) -> float:
        """Current score, admitting an unseen operator at ``initial_score``."""
        return self.scores.get(operator, self.params.initial_score)

    def tier(self, operator: str) -> str:
        return alert_tier(self.score(operator), self.params)

    def trajectory(self, operator: str) -> list[float]:
        """Per-round score history for one operator (for plots/tables)."""
        return [snapshot.get(operator, float("nan")) for snapshot in self.history]


# --------------------------------------------------------------------------
# Evidence gathering
# --------------------------------------------------------------------------

def parse_encounter_participants(encounter_id: str) -> tuple[int, int]:
    """Pull the two NORAD ids out of an ``ENC-<a>-<b>-<tca>`` encounter id.

    Note what this is *not*: the encounter id carries catalogue numbers,
    which are public (every tracked object has one), not state vectors. The
    abstraction boundary in ``signal.py`` is untouched by anything here.
    """
    parts = encounter_id.split("-")
    if len(parts) < 4 or parts[0] != "ENC":
        raise ValueError(f"malformed encounter_id: {encounter_id!r}")
    return int(parts[1]), int(parts[2])


def attest_participation(encounter_id: str, owned_norad_ids: set[int]) -> bool:
    """Run **by an operator, on its own machine**: am I a party to this encounter?

    The federation broadcasts an encounter id and receives back one boolean
    per operator. That boolean reveals nothing the operator would not have
    revealed by reporting the encounter honestly in the first place, which
    is precisely why withholding is detectable without anyone handing over
    a fleet catalogue.
    """
    norad_a, norad_b = parse_encounter_participants(encounter_id)
    return norad_a in owned_norad_ids or norad_b in owned_norad_ids


def gather_attestations(
    encounter_ids: list[str],
    ownership: dict[str, set[int]],
) -> dict[str, set[str]]:
    """Collect every operator's attestations for this round's encounter ids.

    In a deployment this is a round trip over ``hub.py``; here it is a loop,
    but the data crossing the boundary is identical — one boolean per
    (operator, encounter) pair.
    """
    attested: dict[str, set[str]] = {operator: set() for operator in ownership}
    for operator, owned in ownership.items():
        for encounter_id in encounter_ids:
            if attest_participation(encounter_id, owned):
                attested[operator].add(encounter_id)
    return attested


def _consensus_agreement(
    report: OperatorReport | None,
    reports: list[OperatorReport],
    params: ReputationParams,
) -> float:
    """How close this report sits to the robust consensus, in [0,1].

    Two deliberate choices here. First, the reference point is the
    coordinate-wise *median* of all reports, not the plain mean and not
    ``trimmed_mean_aggregate``: a loud adversary drags a mean toward itself
    (which would make the honest operators look like the outliers), and
    trimming is no help in a small federation — ``trim_fraction=0.2`` with
    four reporters trims nothing at all, since ``floor(0.2 * 4) == 0``, so
    the "robust" reference silently degrades back into the plain mean
    exactly when few operators make each one's influence largest. The
    median stays robust at every federation size. Second, an operator is
    only penalised
    for the deviation it has in *excess* of the round's median deviation:
    reporting periods genuinely differ in how busy and how varied they are,
    and an operator that is no further from consensus than a typical member
    of the federation has not given any evidence of misbehaviour. Only
    standing out relative to that round's own spread costs score.
    """
    if report is None or not reports:
        return 0.0
    consensus = np.median(np.stack([r.counts for r in reports]), axis=0)
    deviations = np.array([float(np.linalg.norm(r.counts - consensus)) for r in reports])
    median_deviation = float(np.median(deviations))
    scale = max(median_deviation, params.consensus_scale_floor)
    own = float(np.linalg.norm(report.counts - consensus))
    excess = max(0.0, own - median_deviation)
    return float(math.exp(-excess / scale))


def evaluate_round(
    operators: list[str],
    signals: list[AbstractedSignal],
    reports: list[OperatorReport],
    ownership: dict[str, set[int]],
    params: ReputationParams = ReputationParams(),
) -> dict[str, RoundEvidence]:
    """Turn one reporting period into per-operator evidence.

    ``signals`` is every abstracted signal the federation saw this period
    (from all senders), ``reports`` the period count vectors, and
    ``ownership`` the attestation oracle — in a deployment, the operators
    themselves answering :func:`attest_participation`.
    """
    encounter_ids = sorted({sig.encounter_id for sig in signals})
    attested = gather_attestations(encounter_ids, ownership)
    reports_by_operator = {r.operator: r for r in reports}
    # An encounter nobody attests owning is a phantom: it was reported into
    # existence by whoever sent it.
    owned_encounters = set().union(*attested.values()) if attested else set()

    evidence: dict[str, RoundEvidence] = {}
    for operator in operators:
        involved = attested.get(operator, set())
        reported = {sig.encounter_id for sig in signals if sig.sender_operator == operator}

        matched = reported & involved
        missed = involved - reported
        unbacked = {eid for eid in reported - involved if eid not in owned_encounters}

        denominator = len(involved) + len(unbacked)
        corroboration = (len(matched) / denominator) if denominator else 1.0

        report = reports_by_operator.get(operator)
        # Participation means *contributing something another operator can
        # vouch for*, not merely being connected and not merely emitting
        # volume. A free-rider that files an empty report while attesting
        # real encounters contributed nothing; a fabricator that files a
        # loud report backed by no attested encounter contributed nothing
        # either, and should not earn participation credit for noise. An
        # operator with genuinely nothing to report in a quiet period is
        # not penalised — there was nothing to contribute.
        contributed = report is not None and (
            len(matched) > 0 or (not involved and not unbacked)
        )
        evidence[operator] = RoundEvidence(
            operator=operator,
            corroboration=float(min(1.0, max(0.0, corroboration))),
            consensus_agreement=_consensus_agreement(report, reports, params),
            participation=1.0 if contributed else 0.0,
            n_involved=len(involved),
            n_matched=len(matched),
            n_missed=len(missed),
            n_unbacked=len(unbacked),
        )
    return evidence


# --------------------------------------------------------------------------
# Score update
# --------------------------------------------------------------------------

def update_scores(
    state: ReputationState,
    evidence: dict[str, RoundEvidence],
    params: ReputationParams | None = None,
) -> ReputationState:
    """Fold one round's evidence into the running scores (asymmetric EWMA).

    Returns a **new** state; the input is left untouched so a caller can
    replay or branch a history (useful for the what-if comparison in
    :func:`simulate_reputation_rounds`).
    """
    params = params or state.params
    scores = dict(state.scores)
    for operator, ev in evidence.items():
        current = scores.get(operator, params.initial_score)
        observed = ev.round_score(params)
        rate = params.rise_rate if observed >= current else params.fall_rate
        updated = current + rate * (observed - current)
        scores[operator] = float(min(params.ceiling, max(params.floor, updated)))
    return ReputationState(
        scores=scores,
        rounds=state.rounds + 1,
        history=state.history + [dict(scores)],
        params=params,
    )


# --------------------------------------------------------------------------
# Consequence 1: reputation-weighted aggregation
# --------------------------------------------------------------------------

def reputation_weights(
    reports: list[OperatorReport],
    state: ReputationState,
    params: ReputationParams | None = None,
) -> np.ndarray:
    """Normalized aggregation weights, quarantining anyone below threshold.

    Falls back to uniform weights if *everyone* is quarantined — a
    federation that trusts nobody should degrade to the plain mean, not
    divide by zero.
    """
    params = params or state.params
    raw = []
    for report in reports:
        score = state.score(report.operator)
        raw.append(0.0 if score < params.quarantine_threshold else score ** params.weight_exponent)
    weights = np.array(raw, dtype=float)
    if weights.sum() <= 0:
        return np.full(len(reports), 1.0 / len(reports))
    return weights / weights.sum()


def reputation_weighted_aggregate(
    reports: list[OperatorReport],
    state: ReputationState,
    params: ReputationParams | None = None,
) -> np.ndarray:
    """Aggregate period counts, weighting each operator by its reputation.

    The difference from ``trimmed_mean_aggregate`` worth stating on a
    panel: trimmed mean is memoryless — it re-decides who the outlier is
    from scratch every round, so an adversary that alternates between
    plausible and outrageous reports pays nothing for the plausible ones.
    This carries the history, so influence is earned over rounds and lost
    in one.
    """
    params = params or state.params
    weights = reputation_weights(reports, state, params)
    matrix = np.stack([r.counts for r in reports])
    return np.asarray(weights @ matrix, dtype=float)


# --------------------------------------------------------------------------
# Consequence 2: reciprocity — how much alert detail you get back
# --------------------------------------------------------------------------

def alert_tier(score: float, params: ReputationParams = ReputationParams()) -> str:
    if score < params.quarantine_threshold:
        return "suspended"
    if score < params.degraded_threshold:
        return "degraded"
    if score < params.standard_threshold:
        return "standard"
    return "full"


def tailor_alert(
    signal: AbstractedSignal,
    tier: str,
    safety_override: bool = True,
) -> dict:
    """Strip a shared alert down to what the recipient's tier has earned.

    ``full`` sees everything the abstraction layer allows (severity,
    geometry class, TCA timing — enough to plan a deconfliction);
    ``standard`` loses geometry (so it knows how urgent and when, but must
    do its own geometry work); ``degraded`` gets a bare "something involves
    you this period" flag; ``suspended`` gets only the encounter id, which
    it could have derived from the public catalogue anyway.

    ``safety_override`` is the hard floor: a ``critical`` severity alert is
    always delivered in full. Reputation exists to allocate influence and
    privilege, not to decide who is allowed to avoid a collision — the
    debris from a withheld warning does not respect the reputation of whose
    satellite made it.
    """
    if tier not in ALERT_TIERS:
        raise ValueError(f"unknown alert tier: {tier!r}")

    overridden = safety_override and signal.severity_tier == "critical"
    effective = "full" if overridden else tier

    alert: dict = {"encounter_id": signal.encounter_id, "tier": effective}
    if overridden:
        alert["safety_override"] = True
    if effective == "suspended":
        return alert
    alert["involves_you"] = True
    if effective == "degraded":
        return alert
    alert["severity_tier"] = signal.severity_tier
    alert["tca_offset_s"] = signal.tca_offset_s
    if effective == "standard":
        return alert
    alert["geometry_class"] = signal.geometry_class
    alert["counterparty_operator"] = signal.sender_operator
    alert["deconfliction_eligible"] = True
    return alert


# --------------------------------------------------------------------------
# Multi-round simulation (the thing the evaluation chapter plots)
# --------------------------------------------------------------------------

BEHAVIOURS = ("honest", "free_rider", "spoofer", "on_off")


def _round_signals(
    operator: str,
    behaviour: str,
    involved: list[str],
    rng: np.random.Generator,
    phantom_pool: list[str],
    is_defecting: bool,
) -> list[AbstractedSignal]:
    """The signals one operator actually emits this round, given its strategy."""
    if behaviour == "free_rider":
        return []
    emitted: list[str] = []
    if behaviour == "honest" or (behaviour == "on_off" and not is_defecting):
        emitted = list(involved)
    elif behaviour == "spoofer" or (behaviour == "on_off" and is_defecting):
        # Reports a minority of what it genuinely sees and invents extra
        # encounters between objects it does not own (the "cry wolf" attack
        # from federation.inject_adversarial_report, expressed at signal level).
        emitted = [eid for eid in involved if rng.random() < 0.3]
        emitted += list(rng.choice(phantom_pool, size=min(3, len(phantom_pool)), replace=False))

    tiers = ("threshold", "elevated", "critical")
    signals = []
    for eid in emitted:
        signals.append(
            AbstractedSignal(
                encounter_id=str(eid),
                sender_operator=operator,
                local_object_label=f"{operator}-obj",
                tca_offset_s=float(rng.integers(300, 3600)),
                geometry_class="crossing",
                severity_tier=str(rng.choice(tiers, p=[0.6, 0.3, 0.1])),
            )
        )
    return signals


def simulate_reputation_rounds(
    behaviours: dict[str, str] | None = None,
    n_rounds: int = 20,
    encounters_per_round: int = 8,
    defect_from_round: int = 12,
    seed: int = 7,
) -> dict:
    """Run a multi-round federation and watch reputations separate.

    Four strategies, each an operator: ``honest`` reports everything it is
    party to; ``free_rider`` reports nothing but stays connected;
    ``spoofer`` under-reports its own encounters and fabricates others;
    ``on_off`` behaves exactly like ``honest`` until ``defect_from_round``
    and then switches to ``spoofer`` — the strategy that beats a memoryless
    defence like trimmed mean, and the one the asymmetric rise/fall rates
    exist to punish.

    Also tracks, each round, the aggregation error of plain mean, trimmed
    mean and reputation-weighted aggregation against the honest operators'
    own truth, which is the head-to-head number for the evaluation chapter.
    """
    behaviours = behaviours or {
        "Alpha": "honest",
        "Beta": "honest",
        "Gamma": "honest",
        "Delta": "free_rider",
        "Epsilon": "spoofer",
        "Zeta": "on_off",
    }
    operators = list(behaviours)
    rng = np.random.default_rng(seed)

    # Each operator owns a disjoint block of catalogue numbers.
    ownership = {op: set(range(40000 + 1000 * i, 40000 + 1000 * i + 200)) for i, op in enumerate(operators)}
    unowned = list(range(90000, 90200))  # objects in the public catalogue nobody here operates

    state = ReputationState()
    error_history: list[dict[str, float]] = []

    for round_idx in range(n_rounds):
        # --- generate this round's genuine encounters between real pairs ---
        genuine: dict[str, list[str]] = {op: [] for op in operators}
        for _ in range(encounters_per_round):
            op_a, op_b = rng.choice(operators, size=2, replace=False)
            norad_a = int(rng.choice(sorted(ownership[op_a])))
            norad_b = int(rng.choice(sorted(ownership[op_b])))
            lo, hi = min(norad_a, norad_b), max(norad_a, norad_b)
            eid = f"ENC-{lo}-{hi}-{int(rng.integers(300, 3600))}"
            genuine[op_a].append(eid)
            genuine[op_b].append(eid)

        phantom_pool = [
            f"ENC-{a}-{a + 1}-{int(rng.integers(300, 3600))}"
            for a in rng.choice(unowned, size=6, replace=False)
        ]

        signals: list[AbstractedSignal] = []
        for operator in operators:
            is_defecting = behaviours[operator] == "on_off" and round_idx >= defect_from_round
            signals += _round_signals(
                operator, behaviours[operator], genuine[operator], rng, phantom_pool, is_defecting
            )

        reports = [build_operator_report(op, signals) for op in operators]
        honest_reports = [
            r for r in reports
            if behaviours[r.operator] == "honest"
            or (behaviours[r.operator] == "on_off" and round_idx < defect_from_round)
        ]

        evidence = evaluate_round(operators, signals, reports, ownership, state.params)
        state = update_scores(state, evidence)

        truth = np.mean([r.counts for r in honest_reports], axis=0)
        error_history.append({
            "round": round_idx + 1,
            "mean": float(np.linalg.norm(mean_aggregate(reports) - truth)),
            "trimmed_mean": float(np.linalg.norm(trimmed_mean_aggregate(reports) - truth)),
            "reputation_weighted": float(
                np.linalg.norm(reputation_weighted_aggregate(reports, state) - truth)
            ),
        })

    return {
        "behaviours": behaviours,
        "final_scores": dict(state.scores),
        "final_tiers": {op: state.tier(op) for op in operators},
        "score_history": state.history,
        "error_history": error_history,
        "mean_error_last_5": float(np.mean([e["mean"] for e in error_history[-5:]])),
        "trimmed_error_last_5": float(np.mean([e["trimmed_mean"] for e in error_history[-5:]])),
        "reputation_error_last_5": float(np.mean([e["reputation_weighted"] for e in error_history[-5:]])),
        "state": state,
    }
