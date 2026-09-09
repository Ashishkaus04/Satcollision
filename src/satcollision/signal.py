"""
Signal abstraction layer.

This is the privacy boundary described in Section 3 of the project
definition: a private, full-fidelity :class:`~satcollision.twin.Conjunction`
(which carries real positions/velocities via each object's SGP4 state)
never itself crosses an operator boundary. Only an :class:`AbstractedSignal`
does — and only when the conjunction's Pc has crossed the shared threshold
in the first place. An AbstractedSignal structurally cannot carry a state
vector: :func:`assert_no_raw_state` enforces that at the dataclass-field
level, not just by convention, so a future code change that accidentally
adds a position/velocity field to the signal fails loudly instead of
silently leaking data.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from .twin import Conjunction, DEFAULT_PC_THRESHOLD

# Fields an AbstractedSignal is allowed to carry. Anything resembling a
# state vector, orbital element, or raw coordinate is deliberately absent.
_ALLOWED_SIGNAL_FIELDS = {
    "encounter_id",
    "sender_operator",
    "local_object_label",
    "tca_offset_s",
    "geometry_class",
    "severity_tier",
    "threshold_crossed",
}

# Names that must never appear on an AbstractedSignal even if someone tries
# to add them later — used by assert_no_raw_state as a denylist safety net.
_FORBIDDEN_SUBSTRINGS = (
    "position", "velocity", "state_vector", "pos_", "vel_",
    "inclination", "raan", "eccentric", "semi_major", "mean_anomaly",
    "miss_distance", "pc",  # exact Pc value is also withheld — only a coarse tier crosses
)


@dataclass(frozen=True)
class AbstractedSignal:
    encounter_id: str
    sender_operator: str
    local_object_label: str  # the sending operator's own asset handle only
    tca_offset_s: float      # needed operationally for coordination timing
    geometry_class: str      # coarse: "head-on" / "crossing" / "overtaking"
    severity_tier: str       # coarse: "threshold" / "elevated" / "critical"
    threshold_crossed: bool = True


def assert_no_raw_state(signal: AbstractedSignal) -> None:
    """Fail loudly if an AbstractedSignal ever carries more than the allowed fields.

    This is the "explicit boundary check" called for in the project
    definition's signal-abstraction layer: a structural guarantee, checked
    at runtime (and exercised in tests/test_signal.py), rather than a
    comment trusting future contributors to remember the privacy rule.
    """
    field_names = {f.name for f in fields(signal)}
    unexpected = field_names - _ALLOWED_SIGNAL_FIELDS
    if unexpected:
        raise AssertionError(
            f"AbstractedSignal carries disallowed field(s) {unexpected} — "
            "these look like they could leak raw orbital/state data across "
            "the operator boundary."
        )
    for name in field_names:
        lowered = name.lower()
        if any(bad in lowered for bad in _FORBIDDEN_SUBSTRINGS) and name not in _ALLOWED_SIGNAL_FIELDS:
            raise AssertionError(f"Field name '{name}' looks like a state-vector leak.")


def severity_tier(pc: float, threshold: float = DEFAULT_PC_THRESHOLD) -> str:
    """Bucket a precise Pc into a coarse, shareable severity tier.

    Sharing the exact Pc value (even without the miss distance itself)
    weakly leaks information about miss distance/uncertainty; bucketing
    into tiers is the abstraction layer's way of sharing "how urgent is
    this" without sharing the number an outside party could try to invert.
    """
    if pc < threshold:
        return "below_threshold"
    if pc < 10 * threshold:
        return "threshold"
    if pc < 100 * threshold:
        return "elevated"
    return "critical"


def abstract_conjunction(
    conjunction: Conjunction,
    threshold: float = DEFAULT_PC_THRESHOLD,
) -> tuple[AbstractedSignal, AbstractedSignal] | None:
    """Convert a private Conjunction into the two abstracted signals shared
    across the federation boundary (one framed from each side's perspective),
    or ``None`` if the conjunction never crossed the sharing threshold at all
    — the majority of screened conjunctions, by design, never generate any
    shared signal.
    """
    if conjunction.pc < threshold:
        return None

    tier = severity_tier(conjunction.pc, threshold)
    encounter_id = (
        f"ENC-{min(conjunction.object_a.norad_id, conjunction.object_b.norad_id)}-"
        f"{max(conjunction.object_a.norad_id, conjunction.object_b.norad_id)}-"
        f"{int(conjunction.tca_offset_s)}"
    )

    signal_a = AbstractedSignal(
        encounter_id=encounter_id,
        sender_operator=conjunction.object_a.operator,
        local_object_label=conjunction.object_a.label,
        tca_offset_s=conjunction.tca_offset_s,
        geometry_class=conjunction.geometry_class,
        severity_tier=tier,
    )
    signal_b = AbstractedSignal(
        encounter_id=encounter_id,
        sender_operator=conjunction.object_b.operator,
        local_object_label=conjunction.object_b.label,
        tca_offset_s=conjunction.tca_offset_s,
        geometry_class=conjunction.geometry_class,
        severity_tier=tier,
    )
    assert_no_raw_state(signal_a)
    assert_no_raw_state(signal_b)
    return signal_a, signal_b
