r"""
The three-scenario comparison, run on real orbital data.

``evaluate.run_three_scenario_comparison`` answers the project's central
question on *one engineered* conjunction: it builds a close approach on
purpose, so the geometry is whatever the scenario generator chose. That is
fine for a walkthrough and weak as evidence. This module runs the same
comparison across every threshold-crossing conjunction found in a real
CelesTrak catalogue, so the headline number becomes a distribution measured
on genuine orbits rather than a single constructed case.

**The honest problem, stated up front.** Genuine cross-operator threshold
crossings are rare. Two week-long Starlink-Kuiper runs a day apart measured
a closest approach of 3.45 km (Pc ~ 1e-14, nowhere near mattering) and then
828 m (Pc 2.0e-4, above the operational threshold) — one crossing, out of
32 cross-operator close approaches in a week. Megaconstellations are
deliberately placed in separated altitude shells, so whether a given week
contains such an event is close to a coin flip on the catalogue of the day.

A single event is an existence proof: it establishes that the scenario this
project addresses is real rather than hypothetical, and it cannot support a
median, a spread, or any claim about what the system does *in general*. So
this module offers two runs, and the report should carry both:

* :func:`run_split_catalog_evaluation` — **the headline.** Take one real
  catalogue and assign each satellite to Operator A or Operator B by a
  fixed rule, then keep only the conjunctions whose two objects landed on
  opposite sides. Every orbit, miss distance, time-to-closest-approach and
  Pc is real and unmodified; the *ownership boundary* is the one synthetic
  element, and it is exactly the element a federation is about. Splitting
  on NORAD parity is deliberate: catalogue numbers are assigned by launch
  order, so parity is uncorrelated with orbital shell, and the split does
  not quietly select for (or against) the encounter geometries being
  measured. Splitting by altitude band, by contrast, would have produced a
  boundary that the orbits themselves respect, which is precisely the bias
  that makes the real cross-operator case empty.
* :func:`run_cross_constellation_evaluation` — **the existence check.** The
  same pipeline across two genuinely different operators, nothing assigned,
  reported with whatever it finds intact — zero crossings on most
  catalogues, occasionally one. Either outcome is reported rather than
  buried: a zero is what justifies the split above, and a crossing is the
  motivating scenario caught in the real sky.

**What the comparison can and cannot show.** Federated and full-sharing
produce the *same* detection lead time by construction: in both, each
operator tracks its own asset at full precision and learns about the other
side in time to act, so they see the same Pc curve. That equality is a
design property, not a measurement, and claiming otherwise would be
overclaiming. What the real data does measure is the quantity that is
genuinely unknown in advance: **how much warning no-cooperation actually
loses on real encounter geometries** — which depends on each encounter's
miss distance, and so varies from encounter to encounter in a way no single
engineered case can represent.

**One modelling fix, made here rather than inherited.**
``evaluate.sigma_no_cooperation`` parameterises degraded public-tracking
precision as a *fraction* of the encounter's own time-to-TCA: it starts at
3 km "far out" and improves linearly to 0.5 km at closest approach, where
"far out" means the start of the scenario. That is harmless for the demo's
single 30-minute encounter and badly wrong across a week-long catalogue
run, because it makes tracking precision depend on how far ahead the
screening window happened to open. An encounter six days out would appear
to be detected days in advance, and the "lead time" measured would really
be measuring the window length.

**A second correction the real data forced.** An encounter whose closest
approach falls inside the first :data:`SCREENING_HORIZON_S` of the
propagation window cannot be given a full horizon of warning by *any*
scenario — the window simply did not open early enough. On the first real
run, 194 of 432 threshold crossings fell in that burn-in period, and every
one of them scored an identical lead time under both regimes (gain exactly
zero), because both were clipped by the window rather than by tracking
quality. Including them does not make the result conservative, it makes it
meaningless: it mixes "no-cooperation was late" with "we started watching
late". Both run functions therefore exclude encounters inside the burn-in
and report how many were dropped, so the exclusion is visible rather than
silent.

So the real-data path uses :func:`sigma_public_tracking` instead, which is
a function of *absolute* time before TCA: tracking is poor while the
encounter is more than :data:`SCREENING_HORIZON_S` away and improves
linearly to the same near-TCA precision. Lead times are also capped at that
horizon, because an operator screening a rolling 3-day window cannot act on
something it has not yet screened. The consequence worth stating in the
report: the cooperative lead time saturates at the horizon (with
cooperation you are warned as soon as the encounter enters screening), so
the figure to draw is the *distribution of how late no-cooperation is*,
against that horizon as a reference line.
"""

from __future__ import annotations

import datetime
import statistics
from dataclasses import dataclass, replace

from .evaluate import (
    SIGMA_GOOD_KM, SIGMA_NOCOOP_FAR_KM, SIGMA_NOCOOP_NEAR_KM, detection_lead_time,
)
from .fleets import TrackedObject, load_tle_file, _epoch_from_calendar
from .signal import abstract_conjunction
from .twin import DEFAULT_PC_THRESHOLD, propagate_window, screen_conjunctions

# A threshold crossing needs a sub-kilometre miss distance: with the
# simplified circular Pc model (HBR 20 m, sigma 0.5 km) the crossing point of
# Pc = 1e-4 sits at d = sqrt(2 * sigma^2 * ln(HBR^2 / (2 * sigma^2 * Pc))) ~= 1.02 km,
# and a *larger* sigma only lowers Pc further. Screening at 5 km therefore
# cannot miss a threshold-crossing encounter, while keeping a week-long run
# over ~11,000 objects to a manageable number of candidate pairs. (The
# backtest scripts keep the full 25 km volume, because there the point is the
# distribution of close approaches rather than the threshold crossings.)
DEFAULT_EVALUATION_SCREEN_KM = 5.0

# Operators screen a rolling window rather than the whole future: conjunction
# data messages are typically issued from about three days before closest
# approach, and a maneuver decision is made inside that window. Lead time is
# therefore measured against this horizon, not against however far ahead the
# propagation happened to start.
SCREENING_HORIZON_S = 72.0 * 3600.0


def sigma_public_tracking(lead_time_s: float, horizon_s: float = SCREENING_HORIZON_S) -> float:
    """Degraded public-tracking uncertainty as a function of *absolute* time to TCA.

    Unlike ``evaluate.sigma_no_cooperation`` (a fraction of each encounter's
    own time-to-TCA, fine for a single engineered case) this is anchored to
    real clock time, so an encounter six days out and one six hours out are
    scored on the same scale — see this module's docstring for why that
    matters once the comparison runs across a whole catalogue.
    """
    if horizon_s <= 0:
        return SIGMA_NOCOOP_NEAR_KM
    frac = max(0.0, min(1.0, lead_time_s / horizon_s))
    return SIGMA_NOCOOP_NEAR_KM + frac * (SIGMA_NOCOOP_FAR_KM - SIGMA_NOCOOP_NEAR_KM)


SPLIT_RULE_DESCRIPTION = (
    "NORAD catalogue number parity (even -> Operator-A, odd -> Operator-B); "
    "catalogue numbers follow launch order, so parity is uncorrelated with "
    "orbital shell and does not bias which encounters cross the boundary"
)


# --------------------------------------------------------------------------
# Ownership assignment
# --------------------------------------------------------------------------

def assign_two_operators(
    objects: list[TrackedObject],
    name_a: str = "Operator-A",
    name_b: str = "Operator-B",
) -> list[TrackedObject]:
    """Relabel a real catalogue as two operators, by NORAD parity.

    Returns new :class:`TrackedObject` instances; the input list and the
    underlying SGP4 records are untouched, so the same catalogue can be
    re-split differently without reloading it.
    """
    return [
        replace(obj, operator=(name_a if obj.norad_id % 2 == 0 else name_b))
        for obj in objects
    ]


# --------------------------------------------------------------------------
# Per-encounter evaluation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EncounterEvaluation:
    """One real conjunction, scored under all three scenarios."""

    encounter_id: str
    operator_a: str
    operator_b: str
    object_a: str
    object_b: str
    tca_offset_s: float
    miss_distance_km: float
    pc: float
    geometry_class: str
    lead_time_nocoop_s: float
    lead_time_cooperative_s: float  # identical for full-sharing and federated

    @property
    def lead_time_gain_s(self) -> float:
        return self.lead_time_cooperative_s - self.lead_time_nocoop_s

    @property
    def detected_by_nocoop(self) -> bool:
        return self.lead_time_nocoop_s > 0.0

    def as_row(self) -> dict:
        return {
            "encounter_id": self.encounter_id,
            "operator_a": self.operator_a,
            "operator_b": self.operator_b,
            "object_a": self.object_a,
            "object_b": self.object_b,
            "tca_offset_min": round(self.tca_offset_s / 60.0, 2),
            "miss_distance_m": round(self.miss_distance_km * 1000.0, 1),
            "pc": self.pc,
            "geometry_class": self.geometry_class,
            "lead_time_nocoop_min": round(self.lead_time_nocoop_s / 60.0, 2),
            "lead_time_cooperative_min": round(self.lead_time_cooperative_s / 60.0, 2),
            "lead_time_gain_min": round(self.lead_time_gain_s / 60.0, 2),
        }


def evaluate_encounter(
    conjunction,
    threshold: float = DEFAULT_PC_THRESHOLD,
    horizon_s: float = SCREENING_HORIZON_S,
) -> EncounterEvaluation:
    """Score one real conjunction under no-cooperation and cooperative tracking.

    The cooperative lead time covers both full-sharing and federated: both
    give every operator full-precision knowledge of its own asset plus
    timely notice of the encounter, so both see the same Pc curve. They
    differ in what crosses the boundary to achieve it, which is the
    exposure axis rather than the timing axis.

    Both are measured inside the same rolling screening horizon, so an
    encounter that happens to sit far out in the propagation window is not
    credited with days of imaginary warning.
    """
    tca_s = conjunction.tca_offset_s
    miss_km = conjunction.miss_distance_km
    # Scan the last `window` seconds before TCA. detection_lead_time counts
    # lead time from the value it is given, so passing the window (rather
    # than the raw TCA offset) is what anchors both regimes to the horizon.
    window = min(tca_s, horizon_s)

    # 60-second scan resolution: a three-day window at the 10-second default
    # would be ~26,000 Pc evaluations per encounter per regime for a precision
    # nobody reports, since lead times are quoted in minutes.
    lead_cooperative, _ = detection_lead_time(
        miss_km, window, sigma_fn=lambda lead: SIGMA_GOOD_KM, threshold=threshold, step_s=60.0
    )
    lead_nocoop, _ = detection_lead_time(
        miss_km, window, sigma_fn=lambda lead: sigma_public_tracking(lead, horizon_s),
        threshold=threshold, step_s=60.0,
    )

    signals = abstract_conjunction(conjunction, threshold=threshold)
    encounter_id = signals[0].encounter_id if signals else conjunction.pair_label

    return EncounterEvaluation(
        encounter_id=encounter_id,
        operator_a=conjunction.object_a.operator,
        operator_b=conjunction.object_b.operator,
        object_a=conjunction.object_a.label,
        object_b=conjunction.object_b.label,
        tca_offset_s=tca_s,
        miss_distance_km=miss_km,
        pc=conjunction.pc,
        geometry_class=conjunction.geometry_class,
        lead_time_nocoop_s=lead_nocoop,
        lead_time_cooperative_s=lead_cooperative,
    )


def summarize_evaluations(evaluations: list[EncounterEvaluation]) -> dict:
    """Aggregate per-encounter results into the numbers a report quotes.

    Medians rather than means lead the summary: lead time is bounded above
    by each encounter's own time-to-TCA, so the distribution is skewed and
    a mean would be pulled around by how far ahead the screening window
    happened to start.
    """
    if not evaluations:
        return {"n_encounters": 0}

    nocoop = [e.lead_time_nocoop_s / 60.0 for e in evaluations]
    coop = [e.lead_time_cooperative_s / 60.0 for e in evaluations]
    gain = [e.lead_time_gain_s / 60.0 for e in evaluations]
    missed = [e for e in evaluations if not e.detected_by_nocoop]

    # Reported in hours as well as minutes: against a three-day screening
    # horizon these quantities run to thousands of minutes, which no reader
    # converts in their head.
    return {
        "n_encounters": len(evaluations),
        "median_lead_time_nocoop_min": round(statistics.median(nocoop), 2),
        "median_lead_time_cooperative_min": round(statistics.median(coop), 2),
        "median_lead_time_gain_min": round(statistics.median(gain), 2),
        "mean_lead_time_gain_min": round(statistics.fmean(gain), 2),
        "max_lead_time_gain_min": round(max(gain), 2),
        "median_lead_time_nocoop_h": round(statistics.median(nocoop) / 60.0, 2),
        "median_lead_time_cooperative_h": round(statistics.median(coop) / 60.0, 2),
        "median_lead_time_gain_h": round(statistics.median(gain) / 60.0, 2),
        "mean_lead_time_gain_h": round(statistics.fmean(gain) / 60.0, 2),
        "min_lead_time_nocoop_h": round(min(nocoop) / 60.0, 2),
        "max_lead_time_nocoop_h": round(max(nocoop) / 60.0, 2),
        "n_missed_entirely_by_nocoop": len(missed),
        "pct_missed_entirely_by_nocoop": round(100.0 * len(missed) / len(evaluations), 1),
        "raw_data_exposed_pct": {"no_cooperation": 0.0, "full_sharing": 100.0, "federated": 0.0},
        "median_miss_distance_m": round(
            statistics.median([e.miss_distance_km * 1000.0 for e in evaluations]), 1),
    }


# --------------------------------------------------------------------------
# Whole-catalogue runs
# --------------------------------------------------------------------------



def _split_out_burn_in(conjunctions: list, horizon_s: float) -> tuple[list, list]:
    """Separate encounters that can be scored from those the window clipped.

    An encounter whose TCA falls within ``horizon_s`` of the start of the
    propagation window has its warning time limited by when the window
    opened, identically in every scenario — so it carries no information
    about how much warning cooperation buys, and averaging it in only
    dilutes the ones that do. Returned separately rather than dropped
    silently, so the run can report how many it set aside.
    """
    scorable = [c for c in conjunctions if c.tca_offset_s >= horizon_s]
    burn_in = [c for c in conjunctions if c.tca_offset_s < horizon_s]
    return scorable, burn_in


def _screen_real_catalogue(objects, hours: float, step_s: float, screening_km: float):
    """Propagate and screen a real catalogue from now, returning conjunctions."""
    now = datetime.datetime.now(datetime.timezone.utc)
    _, jd0, fr0 = _epoch_from_calendar(now.year, now.month, now.day, now.hour + now.minute / 60.0)
    propagation = propagate_window(objects, jd0, fr0, duration_s=hours * 3600.0, step_s=step_s)
    conjunctions = screen_conjunctions(objects, propagation, screening_distance_km=screening_km)
    return conjunctions, now


def run_split_catalog_evaluation(
    tle_path: str,
    catalog_name: str = "Starlink",
    hours: float = 168.0,
    step_s: float = 30.0,
    screening_km: float = DEFAULT_EVALUATION_SCREEN_KM,
    threshold: float = DEFAULT_PC_THRESHOLD,
    horizon_s: float = SCREENING_HORIZON_S,
) -> dict:
    """Headline run: one real catalogue, split into two operators by parity.

    Everything measured here — orbits, geometry, miss distance, Pc, timing —
    comes from the real catalogue. The only assigned quantity is which of
    two operators owns each satellite, and the result section should say so
    in exactly those words.
    """
    objects = assign_two_operators(load_tle_file(tle_path, operator_name=catalog_name))
    conjunctions, computed_at = _screen_real_catalogue(objects, hours, step_s, screening_km)

    cross_operator = [c for c in conjunctions if c.object_a.operator != c.object_b.operator]
    above_threshold = [c for c in cross_operator if c.pc >= threshold]
    scorable, burn_in = _split_out_burn_in(above_threshold, horizon_s)
    evaluations = [evaluate_encounter(c, threshold=threshold, horizon_s=horizon_s)
                    for c in scorable]

    return {
        "mode": "split_catalog",
        "catalog": catalog_name,
        "ownership": SPLIT_RULE_DESCRIPTION,
        "n_objects": len(objects),
        "window_hours": hours,
        "step_s": step_s,
        "screening_km": screening_km,
        "pc_threshold": threshold,
        "screening_horizon_h": horizon_s / 3600.0,
        "computed_at_utc": computed_at.isoformat(timespec="seconds"),
        "n_conjunctions_screened": len(conjunctions),
        "n_cross_operator": len(cross_operator),
        "n_above_threshold": len(above_threshold),
        "n_scored": len(scorable),
        "n_excluded_burn_in": len(burn_in),
        "closest_cross_operator_km": min((c.miss_distance_km for c in cross_operator), default=None),
        "max_cross_operator_pc": max((c.pc for c in cross_operator), default=0.0),
        "summary": summarize_evaluations(evaluations),
        "encounters": [e.as_row() for e in evaluations],
    }


def run_cross_constellation_evaluation(
    tle_path_a: str,
    name_a: str,
    tle_path_b: str,
    name_b: str,
    hours: float = 168.0,
    step_s: float = 30.0,
    screening_km: float = DEFAULT_EVALUATION_SCREEN_KM,
    threshold: float = DEFAULT_PC_THRESHOLD,
    horizon_s: float = SCREENING_HORIZON_S,
) -> dict:
    """The honest negative: two genuinely different operators, unmodified.

    Expected to find zero threshold crossings on current catalogues. The
    run still reports how close the two constellations come and the highest
    Pc reached, because "nothing crossed the threshold" is only a result if
    it comes with the distance at which nothing crossed it.
    """
    objects = (load_tle_file(tle_path_a, operator_name=name_a)
               + load_tle_file(tle_path_b, operator_name=name_b))
    conjunctions, computed_at = _screen_real_catalogue(objects, hours, step_s, screening_km)

    cross_operator = [c for c in conjunctions if c.object_a.operator != c.object_b.operator]
    above_threshold = [c for c in cross_operator if c.pc >= threshold]
    scorable, burn_in = _split_out_burn_in(above_threshold, horizon_s)
    evaluations = [evaluate_encounter(c, threshold=threshold, horizon_s=horizon_s)
                    for c in scorable]

    return {
        "mode": "cross_constellation",
        "catalog": f"{name_a} x {name_b}",
        "ownership": "genuine — each object keeps its real operator",
        "n_objects": len(objects),
        "window_hours": hours,
        "step_s": step_s,
        "screening_km": screening_km,
        "pc_threshold": threshold,
        "screening_horizon_h": horizon_s / 3600.0,
        "computed_at_utc": computed_at.isoformat(timespec="seconds"),
        "n_conjunctions_screened": len(conjunctions),
        "n_cross_operator": len(cross_operator),
        "n_above_threshold": len(above_threshold),
        "n_scored": len(scorable),
        "n_excluded_burn_in": len(burn_in),
        "closest_cross_operator_km": min((c.miss_distance_km for c in cross_operator), default=None),
        "max_cross_operator_pc": max((c.pc for c in cross_operator), default=0.0),
        "summary": summarize_evaluations(evaluations),
        "encounters": [e.as_row() for e in evaluations],
    }
