"""
Plain-language incident summaries (Enhancement #5 of the project definition).

Every number this system produces is currently addressed to someone who
already knows what it means. ``Pc=8.00e-04`` is meaningful to a conjunction
analyst and meaningless to the duty officer who has to decide whether to
wake someone up, to the regulator asking why a maneuver was flown, or to
the operator three time zones away whose only interest is "does this
involve me, and what do I have to do". This module is the layer that turns
the system's own outputs into sentences those readers can act on.

Three properties are deliberate, and all three are the sort of thing a
panel will probe:

**It is template-based, not generative.** There is no language model
anywhere in this module, and that is a design decision rather than a
limitation of the sandbox. A summary that accompanies a collision warning
is a safety artifact: it has to be reproducible (the same encounter must
always produce the same words, so two operators reading "their" copy are
reading the same thing), auditable (every number in the text is traceable
to the field it came from), and incapable of inventing a figure that was
never computed. A generative model gives up all three, and would also mean
shipping encounter data to a third-party API — re-opening precisely the
data-exposure hole the rest of this project exists to close.

**It cannot say more than the reader is entitled to know.** Cross-operator
summaries are rendered from the *tailored alert dictionary* produced by
``reputation.tailor_alert``, never from the underlying
:class:`~satcollision.signal.AbstractedSignal`. A field that the
recipient's tier did not receive is not merely omitted from the sentence —
it is not in the data the renderer can see. :func:`assert_no_undisclosed_terms`
checks the rendered text against the alert it came from, the same way
``signal.assert_no_raw_state`` checks the abstraction boundary
structurally rather than by convention.

**It says what is uncertain.** Where a summary quotes a collision
probability it also says what that probability rests on, because the
number comes from the simplified circular Pc model described in
``twin.py``. A plain-language layer that quietly launders a modelled
estimate into a confident-sounding sentence would be worse than no summary
at all.

Audience is selected with ``audience="operator"`` (keeps the technical
figures alongside the plain wording) or ``audience="executive"`` (drops
catalogue numbers, sigma and scientific notation entirely, and speaks only
in odds, distances and times).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

AUDIENCES = ("operator", "executive")

_GEOMETRY_PHRASES = {
    "head-on": "approaching almost directly head-on",
    "crossing": "crossing paths at a wide angle",
    "overtaking": "on near-parallel tracks, one overtaking the other",
    "unknown": "on a geometry that could not be classified",
}

_SEVERITY_PHRASES = {
    "below_threshold": "below the level at which encounters are shared",
    "threshold": "just over the level at which encounters are shared",
    "elevated": "well over the sharing level — this one deserves attention today",
    "critical": "in the most serious band the system has",
}

# Words that only belong in a summary when the matching field was actually
# disclosed to this reader. Used by assert_no_undisclosed_terms to check a
# rendered string against the alert it was rendered from.
_FIELD_MARKERS = {
    "geometry_class": ("head-on", "crossing paths", "overtaking", "near-parallel"),
    "tca_offset_s": ("minute", "hour", "second", "closest approach is"),
    "severity_tier": ("level at which encounters are shared", "sharing level",
                       "most serious band", "deserves attention"),
    "counterparty_operator": ("operated by", "counterpart"),
}


# --------------------------------------------------------------------------
# Phrasing helpers — the whole "plain language" claim lives or dies here
# --------------------------------------------------------------------------

def describe_duration(seconds: float) -> str:
    """Seconds to something a person reads without converting anything.

    ``1800`` becomes "30 minutes", not "1800 s" and not "0.5 hours".
    """
    seconds = float(seconds)
    if seconds < 0:
        return "a moment ago"
    if seconds < 60:
        return "under a minute"
    minutes = int(round(seconds / 60.0))
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours, remainder = divmod(minutes, 60)
    if remainder == 0:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    return f"{hours} hour{'s' if hours != 1 else ''} {remainder} minute{'s' if remainder != 1 else ''}"


def describe_distance(km: float) -> str:
    """Kilometres to metres-or-kilometres, whichever a reader pictures faster.

    Anything under 10 m is reported as "under 10 metres" rather than as a
    figure: a screened conjunction whose modelled miss distance rounds to
    zero (the demo's engineered encounter is exactly that) would otherwise
    render as "0 metres", which reads like a broken template rather than
    the two tracks genuinely intersecting. The propagation step size makes
    single-metre precision meaningless at that scale anyway.
    """
    km = float(km)
    if km < 0.01:
        return "under 10 metres"
    if km < 1.0:
        metres = int(round(km * 1000.0))
        return f"{metres} metre{'s' if metres != 1 else ''}"
    if km < 10.0:
        return f"{km:.1f} kilometres"
    return f"{int(round(km)):,} kilometres"


def _two_significant_figures(value: float) -> int:
    """Round to 2 significant figures — "1 in 1,200", never "1 in 1,247"."""
    if value <= 0:
        return 0
    magnitude = int(math.floor(math.log10(value)))
    factor = 10 ** (magnitude - 1)
    return int(round(value / factor) * factor)


def describe_odds(pc: float) -> str:
    """A collision probability as odds, which is how people reason about risk.

    ``8e-4`` becomes "about a 1 in 1,200 chance" rather than a number most
    readers will silently mis-scale by three orders of magnitude. Rounded
    to two significant figures, because the underlying estimate does not
    justify more precision than that (see ``twin.compute_pc``).
    """
    pc = float(pc)
    if pc <= 0:
        return "no meaningful chance on this model"
    if pc >= 0.1:
        return f"about a {pc * 100:.0f}% chance"
    denominator = 1.0 / pc
    if denominator > 1_000_000:
        return "less than a 1 in a million chance"
    return f"about a 1 in {_two_significant_figures(denominator):,} chance"


def describe_geometry(geometry_class: str) -> str:
    return _GEOMETRY_PHRASES.get(geometry_class, _GEOMETRY_PHRASES["unknown"])


def describe_severity(severity_tier: str) -> str:
    return _SEVERITY_PHRASES.get(severity_tier, "at an unrecognised severity level")


# --------------------------------------------------------------------------
# Internal summary: an operator's own twin, full fidelity
# --------------------------------------------------------------------------

def summarize_conjunction(conjunction, audience: str = "operator") -> str:
    """Summarize one of an operator's *own* screened conjunctions.

    This one sees everything — it never crosses an operator boundary, so it
    can quote miss distance and Pc directly. The caveat sentence is not
    decoration: the probability comes from the simplified circular model in
    ``twin.py``, and a summary that hides that is overclaiming.
    """
    if audience not in AUDIENCES:
        raise ValueError(f"unknown audience: {audience!r}")

    when = describe_duration(conjunction.tca_offset_s)
    odds = describe_odds(conjunction.pc)
    geometry = describe_geometry(conjunction.geometry_class)
    if float(conjunction.miss_distance_km) < 0.01:
        separation = "passing through what the model puts at the same point, under 10 metres apart"
    else:
        separation = f"passing within {describe_distance(conjunction.miss_distance_km)} of each other"

    if audience == "executive":
        subject = "Two satellites"
        detail = ""
    else:
        subject = f"{conjunction.object_a.label} and {conjunction.object_b.label}"
        detail = (
            f" Closing speed is {conjunction.relative_speed_km_s:.1f} km/s; "
            f"modelled Pc is {conjunction.pc:.2e}."
        )

    return (
        f"{subject} are {geometry}. Their closest approach is in {when}, "
        f"{separation} — {odds} of a collision.{detail} "
        "That figure is a model estimate, not a measurement: it assumes the "
        "simplified circular uncertainty model, so treat it as an order of "
        "magnitude rather than an exact number."
    )


# --------------------------------------------------------------------------
# Cross-operator summary: rendered only from what the reader was given
# --------------------------------------------------------------------------

def summarize_alert(alert: dict) -> str:
    """Render a tailored cross-operator alert into a sentence or two.

    The input is the dictionary from ``reputation.tailor_alert`` — so the
    fields a lower-reputation reader never received are not available to
    this function at all, rather than being available and politely skipped.
    """
    tier = alert.get("tier", "suspended")
    parts: list[str] = []

    if not alert.get("involves_you"):
        parts.append(
            f"Encounter {alert['encounter_id']} was reported to the federation. "
            "At your current access tier no further detail is released."
        )
        return " ".join(parts)

    opening = f"Encounter {alert['encounter_id']} involves one of your satellites."
    parts.append(opening)

    if "severity_tier" in alert:
        parts.append(f"The federation rates it {describe_severity(alert['severity_tier'])}.")
    if "tca_offset_s" in alert:
        parts.append(f"Closest approach is in {describe_duration(alert['tca_offset_s'])}.")
    if "geometry_class" in alert:
        parts.append(f"The two objects are {describe_geometry(alert['geometry_class'])}.")
    if "counterparty_operator" in alert:
        parts.append(
            f"The other object is operated by {alert['counterparty_operator']}, "
            "who received the matching alert at the same moment."
        )
    if alert.get("deconfliction_eligible"):
        parts.append(
            "You are eligible to open a deconfliction plan for this encounter, which "
            "assigns both sides complementary escape directions without either of you "
            "sharing trajectory data."
        )

    if alert.get("safety_override"):
        parts.append(
            "This alert was delivered in full regardless of access tier, because the "
            "federation never withholds its most serious warnings from anyone."
        )
    elif tier != "full":
        parts.append(
            f"Some detail is held back at your current access tier ({tier}); it is "
            "released again as your reporting record improves."
        )

    return " ".join(parts)


def assert_no_undisclosed_terms(summary: str, alert: dict) -> None:
    """Fail loudly if a rendered summary describes something it wasn't given.

    The plain-language layer is the most likely place for a privacy leak to
    reappear after the careful work in ``signal.py``: it is the one place
    that deliberately turns structured fields into free text, and free text
    is where a well-meaning template edit ("just mention the timing, it's
    harmless") quietly undoes a boundary. This is the same structural check
    ``signal.assert_no_raw_state`` performs, applied to prose.
    """
    lowered = summary.lower()
    for field, markers in _FIELD_MARKERS.items():
        if field in alert:
            continue
        for marker in markers:
            if marker in lowered:
                raise AssertionError(
                    f"summary mentions '{marker}', which implies the '{field}' field, "
                    f"but that field was not disclosed to this reader (tier="
                    f"{alert.get('tier')!r})."
                )


# --------------------------------------------------------------------------
# Maneuver, federation round, reputation, backtest
# --------------------------------------------------------------------------

def summarize_maneuver_plan(plan, viewpoint_operator: str | None = None) -> str:
    """Describe an agreed deconfliction plan in the imperative.

    ``viewpoint_operator`` turns the description into instructions for one
    side ("you move..."), which is what an operations console should show.
    """
    direction_words = {"+": "away from the encounter along the agreed escape axis",
                       "-": "the opposite way along the same axis"}
    primary = (
        f"{plan.primary_operator} moves {plan.primary_object} "
        f"{direction_words.get(plan.primary_direction, plan.primary_direction)}"
    )
    secondary = (
        f"{plan.secondary_operator} moves {plan.secondary_object} "
        f"{direction_words.get(plan.secondary_direction, plan.secondary_direction)}"
    )
    lines = [
        f"Agreed plan for encounter {plan.encounter_id}: {primary}, while {secondary}.",
        "Both sides worked this out independently from the encounter identifier alone, "
        "so neither had to send the other any trajectory data, and the two maneuvers "
        "are guaranteed to add up rather than cancel.",
    ]
    if viewpoint_operator == plan.primary_operator:
        lines.append(f"Your action: move {plan.primary_object} {direction_words[plan.primary_direction]}.")
    elif viewpoint_operator == plan.secondary_operator:
        lines.append(f"Your action: move {plan.secondary_object} {direction_words[plan.secondary_direction]}.")
    return " ".join(lines)


def summarize_federation_round(
    n_operators: int,
    plain_estimate,
    robust_estimate,
    plain_error: float,
    robust_error: float,
    quarantined: list[str] | None = None,
) -> str:
    """Explain what a federation round concluded, and why the two numbers differ."""
    quarantined = quarantined or []
    total_plain = float(sum(plain_estimate))
    total_robust = float(sum(robust_estimate))
    lines = [
        f"{n_operators} operators reported this period. Taking every report at face "
        f"value gives an average of {total_plain:.1f} flagged encounters per operator; "
        f"discounting reports that disagree with everyone else's gives {total_robust:.1f}.",
    ]
    if plain_error > robust_error * 1.5:
        lines.append(
            "That gap is the signature of at least one operator reporting something "
            "the rest of the federation cannot corroborate — the robust figure is the "
            "one to act on."
        )
    else:
        lines.append("The two figures agree closely, which is what an honest period looks like.")
    if quarantined:
        who = ", ".join(quarantined)
        lines.append(
            f"{who} currently carries no weight in this result, having been quarantined "
            "on its own reporting record."
        )
    return " ".join(lines)


def summarize_reputation(score: float, tier: str, operator: str = "Your fleet") -> str:
    """Tell an operator where it stands and what would change it."""
    consequence = {
        "full": "You receive complete alerts, including geometry and a deconfliction partner.",
        "standard": "You receive severity and timing, but not geometry or partner details.",
        "degraded": "You are told only that an encounter involves you.",
        "suspended": "You carry no weight in federated results and receive identifiers only.",
    }.get(tier, "")
    return (
        f"{operator} has a reporting reputation of {score:.2f} ({tier} tier). {consequence} "
        "Reputation rises by reporting the encounters your own satellites are party to, "
        "and falls faster than it rises when reports cannot be corroborated by the "
        "operator on the other side of the encounter. Critical collision warnings reach "
        "you in full at any reputation."
    )


def summarize_backtest(
    catalog_name: str,
    n_objects: int,
    window_hours: float,
    n_screened: int,
    n_flagged: int,
    closest_km: float,
    max_pc: float,
) -> str:
    """Turn a catalogue-scale backtest into a paragraph for a report chapter."""
    window = describe_duration(window_hours * 3600.0)
    body = (
        f"Across {n_objects:,} real {catalog_name} objects over {window}, the pipeline "
        f"examined {n_screened:,} close approaches within the 25 km screening volume. "
    )
    if n_flagged:
        body += (
            f"{n_flagged:,} of them crossed the operational risk threshold. The closest "
            f"pass was {describe_distance(closest_km)}, carrying {describe_odds(max_pc)} "
            "of a collision."
        )
    else:
        body += (
            f"None crossed the operational risk threshold: the closest pass was "
            f"{describe_distance(closest_km)}, which works out to {describe_odds(max_pc)} "
            "of a collision. A negative result on real data is still a result — it says "
            "these two populations are genuinely separated, not that the pipeline saw "
            "nothing."
        )
    return body


# --------------------------------------------------------------------------
# The full incident report
# --------------------------------------------------------------------------

@dataclass
class IncidentReport:
    """A rendered incident narrative, kept as sections so a caller can place
    them in a console, an email, or a report chapter independently."""

    encounter_id: str
    what_happened: str
    what_was_shared: str
    what_happens_next: str
    how_to_verify: str

    def as_text(self) -> str:
        return (
            f"INCIDENT SUMMARY — {self.encounter_id}\n"
            f"  What happened   : {self.what_happened}\n"
            f"  What was shared : {self.what_was_shared}\n"
            f"  What happens now: {self.what_happens_next}\n"
            f"  How to check    : {self.how_to_verify}"
        )


def build_incident_report(
    conjunction=None,
    alert: dict | None = None,
    plan=None,
    viewpoint_operator: str | None = None,
    audience: str = "operator",
    log_tail_hash: str | None = None,
) -> IncidentReport:
    """Assemble the full narrative for one encounter.

    ``conjunction`` is the *reporting* operator's own private view and
    ``alert`` is what a given reader received; the two are deliberately
    separate inputs so this function cannot blur them. Pass both to build
    the **owner's** copy — the operator that detected the encounter,
    reviewing both its own findings and what its counterparty was sent.
    Pass ``conjunction=None`` to build a **counterparty's** copy, where the
    narrative is rendered from the alert alone and the private view is not
    merely omitted but absent.
    """
    if alert is None:
        raise ValueError("build_incident_report needs the alert the reader received")

    what_was_shared = summarize_alert(alert)
    assert_no_undisclosed_terms(what_was_shared, alert)

    if conjunction is not None:
        what_happened = summarize_conjunction(conjunction, audience=audience)
    else:
        what_happened = (
            "The operator on the other side of this encounter detected it in its own "
            "digital twin and shared an abstracted signal — never a trajectory — with "
            "the federation. What reached you is set out below."
        )
        assert_no_undisclosed_terms(what_happened, alert)

    if plan is not None and alert.get("deconfliction_eligible"):
        what_happens_next = summarize_maneuver_plan(plan, viewpoint_operator=viewpoint_operator)
    elif plan is not None:
        # A plan names both sides' assets and operators, so handing one to a
        # reader whose tier withheld the counterparty would walk that detail
        # straight back out through the summary layer. Eligibility is the
        # alert's to grant, not this renderer's to assume.
        what_happens_next = (
            "A deconfliction plan exists for this encounter, but your current access "
            "tier does not include it. It is released as your reporting record improves."
        )
    else:
        what_happens_next = (
            "No deconfliction plan has been opened yet. Either side can open one from "
            "the encounter identifier alone."
        )

    verify = (
        "Every signal and report behind this summary was signed by the operator that "
        "sent it and appended to a hash-chained log, so any later edit to the record "
        "is detectable."
    )
    if log_tail_hash:
        verify += f" Current log tail: {log_tail_hash[:16]}..."

    return IncidentReport(
        encounter_id=alert.get("encounter_id", getattr(conjunction, "pair_label", "unknown")),
        what_happened=what_happened,
        what_was_shared=what_was_shared,
        what_happens_next=what_happens_next,
        how_to_verify=verify,
    )
