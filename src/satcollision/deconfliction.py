"""
Maneuver deconfliction layer (Enhancement #2, project definition Sec. 4).

The failure mode this defends against: once both operators in an encounter
have independently detected the same threshold-crossing conjunction (they
each screen the same public catalog, so both *can* detect it without any
private exchange), each might independently decide to maneuver away from
the predicted collision point. If they pick their escape directions
without coordinating, there is a real chance the two shifts cancel out
instead of adding up — leaving the satellites just as close together as
before, or closer. :func:`negotiate_maneuver` fixes this with a
deterministic, previously-agreed protocol rule that assigns the two sides
*complementary* escape directions — computed from nothing but the public
``encounter_id`` both sides already share, so it costs zero additional
private data exchange. :func:`simulate_uncoordinated_vs_negotiated`
quantifies exactly how much this is worth.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from .signal import AbstractedSignal


@dataclass(frozen=True)
class ManeuverPlan:
    encounter_id: str
    primary_operator: str
    primary_object: str
    primary_direction: str  # "+" (a shared, arbitrary but fixed escape-axis convention)
    secondary_operator: str
    secondary_object: str
    secondary_direction: str  # always the complement of primary_direction
    rationale: str


def negotiate_maneuver(signal_a: AbstractedSignal, signal_b: AbstractedSignal) -> ManeuverPlan:
    """Assign complementary escape directions to both sides of one encounter.

    Both inputs must be the two :class:`AbstractedSignal` instances
    produced by :func:`satcollision.signal.abstract_conjunction` for the
    same encounter. The assignment is a pure function of ``encounter_id``
    (itself derived only from public catalog IDs and time, per
    ``signal.py``) — so both operators, computing it independently, always
    agree on the same plan without a single extra bit of private
    information changing hands.
    """
    if signal_a.encounter_id != signal_b.encounter_id:
        raise ValueError("negotiate_maneuver requires both signals from the same encounter")

    digest = hashlib.sha256(signal_a.encounter_id.encode()).hexdigest()
    a_is_primary = int(digest[:8], 16) % 2 == 0
    primary, secondary = (signal_a, signal_b) if a_is_primary else (signal_b, signal_a)

    return ManeuverPlan(
        encounter_id=signal_a.encounter_id,
        primary_operator=primary.sender_operator,
        primary_object=primary.local_object_label,
        primary_direction="+",
        secondary_operator=secondary.sender_operator,
        secondary_object=secondary.local_object_label,
        secondary_direction="-",
        rationale=(
            "Directions are assigned by a rule fixed in advance and keyed only on the "
            "public encounter_id, so both operators derive the same complementary plan "
            "independently — guaranteeing the two maneuvers add up instead of risking "
            "cancellation, with no additional private data exchanged."
        ),
    )


def _new_separation_km(dir_a: int, dir_b: int, shift_km: float) -> float:
    """1-D escape-axis model: both satellites sit ~0 apart along this axis
    at TCA (that is what a threshold-crossing conjunction means); each
    moves by +-shift_km along it. Opposite signs add up to full separation;
    matching signs cancel and leave the conjunction essentially unresolved.
    """
    return abs(dir_a * shift_km - dir_b * shift_km)


def simulate_uncoordinated_vs_negotiated(
    n_trials: int = 10_000,
    shift_km: float = 1.0,
    safe_separation_km: float = 1.5,
    seed: int = 0,
) -> dict:
    """Quantify the value of the negotiated protocol vs. two operators each
    guessing an escape direction independently, with no coordination.

    Returns a dict with the fraction of trials that end up safely separated
    under each strategy. The negotiated strategy is always 1.0 by
    construction; the uncoordinated strategy is included so the improvement
    is measured rather than asserted.
    """
    rng = random.Random(seed)
    n_safe_uncoordinated = 0
    for _ in range(n_trials):
        dir_a = rng.choice((1, -1))
        dir_b = rng.choice((1, -1))
        if _new_separation_km(dir_a, dir_b, shift_km) >= safe_separation_km:
            n_safe_uncoordinated += 1

    negotiated_separation = _new_separation_km(1, -1, shift_km)
    return {
        "n_trials": n_trials,
        "shift_km": shift_km,
        "safe_separation_km": safe_separation_km,
        "uncoordinated_safe_fraction": n_safe_uncoordinated / n_trials,
        "negotiated_safe_fraction": 1.0 if negotiated_separation >= safe_separation_km else 0.0,
        "negotiated_separation_km": negotiated_separation,
    }
