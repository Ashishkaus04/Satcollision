import pytest

from satcollision.deconfliction import negotiate_maneuver
from satcollision.reputation import tailor_alert
from satcollision.signal import AbstractedSignal
from satcollision.summaries import (
    assert_no_undisclosed_terms, build_incident_report, describe_distance,
    describe_duration, describe_odds, summarize_alert, summarize_backtest,
    summarize_conjunction, summarize_federation_round, summarize_maneuver_plan,
    summarize_reputation,
)


class _Obj:
    """Minimal stand-in for a TrackedObject — the summary layer only ever
    reads labels, never orbital state."""

    def __init__(self, label):
        self.label = label


class _Conjunction:
    def __init__(self, pc=8e-4, miss_km=0.12, tca_s=1800.0, geometry="crossing"):
        self.object_a = _Obj("Alpha-90000")
        self.object_b = _Obj("Beta-99000")
        self.tca_offset_s = tca_s
        self.miss_distance_km = miss_km
        self.pc = pc
        self.geometry_class = geometry
        self.relative_speed_km_s = 10.4

    @property
    def pair_label(self):
        return f"{self.object_a.label} <-> {self.object_b.label}"


def _signal(severity="elevated"):
    return AbstractedSignal(
        encounter_id="ENC-90000-99000-1800",
        sender_operator="Alpha",
        local_object_label="Alpha-90000",
        tca_offset_s=1800.0,
        geometry_class="crossing",
        severity_tier=severity,
    )


# ----------------------------------------------------------- phrasing helpers

@pytest.mark.parametrize("seconds,expected", [
    (30, "under a minute"),
    (60, "1 minute"),
    (1800, "30 minutes"),
    (3600, "1 hour"),
    (7500, "2 hours 5 minutes"),
])
def test_describe_duration_reads_like_a_person_wrote_it(seconds, expected):
    assert describe_duration(seconds) == expected


@pytest.mark.parametrize("km,expected", [
    (0.12, "120 metres"),
    (1.57, "1.6 kilometres"),
    (25.0, "25 kilometres"),
])
def test_describe_distance_picks_a_unit_a_reader_can_picture(km, expected):
    assert describe_distance(km) == expected


def test_describe_odds_converts_probability_into_odds():
    assert describe_odds(8e-4) == "about a 1 in 1,200 chance"
    assert describe_odds(1e-9) == "less than a 1 in a million chance"
    assert describe_odds(0.25) == "about a 25% chance"


def test_describe_odds_rounds_to_two_significant_figures():
    """The underlying Pc model does not justify more precision than this."""
    assert describe_odds(1 / 1247.0) == "about a 1 in 1,200 chance"


# -------------------------------------------------------- internal summaries

def test_conjunction_summary_states_the_model_caveat():
    text = summarize_conjunction(_Conjunction())
    assert "30 minutes" in text
    assert "120 metres" in text
    assert "1 in 1,200" in text
    assert "model estimate, not a measurement" in text


def test_executive_summary_drops_catalogue_labels_and_notation():
    text = summarize_conjunction(_Conjunction(), audience="executive")
    assert "Alpha-90000" not in text
    assert "Pc" not in text
    assert "e-0" not in text  # no scientific notation survives
    assert "1 in 1,200" in text


def test_unknown_audience_is_rejected():
    with pytest.raises(ValueError):
        summarize_conjunction(_Conjunction(), audience="haiku")


# ------------------------------------------------- tier-aware disclosure

def test_full_tier_summary_carries_every_disclosed_detail():
    alert = tailor_alert(_signal(), "full")
    text = summarize_alert(alert)
    assert "30 minutes" in text
    assert "crossing paths" in text
    assert "operated by Alpha" in text
    assert_no_undisclosed_terms(text, alert)


def test_standard_tier_summary_keeps_timing_but_drops_geometry():
    alert = tailor_alert(_signal(), "standard")
    text = summarize_alert(alert)
    assert "30 minutes" in text
    assert "crossing" not in text
    assert_no_undisclosed_terms(text, alert)


def test_degraded_tier_summary_says_only_that_it_involves_you():
    alert = tailor_alert(_signal(), "degraded")
    text = summarize_alert(alert)
    assert "involves one of your satellites" in text
    assert "minute" not in text
    assert_no_undisclosed_terms(text, alert)


def test_suspended_tier_summary_releases_nothing_beyond_the_identifier():
    alert = tailor_alert(_signal(), "suspended")
    text = summarize_alert(alert)
    assert "ENC-90000-99000-1800" in text
    assert "no further detail" in text
    assert_no_undisclosed_terms(text, alert)


def test_lower_tiers_never_produce_longer_summaries_than_higher_ones():
    lengths = [len(summarize_alert(tailor_alert(_signal(), tier)))
               for tier in ("suspended", "degraded", "standard", "full")]
    assert lengths == sorted(lengths)


def test_critical_alert_summary_explains_the_safety_override():
    alert = tailor_alert(_signal(severity="critical"), "suspended")
    text = summarize_alert(alert)
    assert "most serious band" in text
    assert "regardless of access tier" in text
    assert_no_undisclosed_terms(text, alert)


def test_the_disclosure_guard_actually_catches_a_leak():
    """A template edit that mentions timing to a degraded reader must fail."""
    alert = tailor_alert(_signal(), "degraded")
    leaky = summarize_alert(alert) + " Closest approach is in 30 minutes."
    with pytest.raises(AssertionError):
        assert_no_undisclosed_terms(leaky, alert)


def test_summaries_are_deterministic():
    """Same encounter, same words — two operators must read the same text."""
    alert = tailor_alert(_signal(), "full")
    assert summarize_alert(alert) == summarize_alert(alert)
    assert summarize_conjunction(_Conjunction()) == summarize_conjunction(_Conjunction())


# --------------------------------------------------- maneuver and federation

def test_maneuver_summary_gives_the_reader_their_own_instruction():
    plan = negotiate_maneuver(_signal(), AbstractedSignal(
        encounter_id="ENC-90000-99000-1800", sender_operator="Beta",
        local_object_label="Beta-99000", tca_offset_s=1800.0,
        geometry_class="crossing", severity_tier="elevated",
    ))
    text = summarize_maneuver_plan(plan, viewpoint_operator=plan.secondary_operator)
    assert "Your action" in text
    assert plan.secondary_object in text
    assert "neither had to send the other any trajectory data" in text


def test_federation_round_summary_flags_a_disagreeing_reporter():
    text = summarize_federation_round(
        n_operators=6, plain_estimate=[11.7, 10.8, 11.7], robust_estimate=[4.0, 3.3, 4.5],
        plain_error=13.7, robust_error=0.8, quarantined=["Adversary"],
    )
    assert "cannot corroborate" in text
    assert "Adversary" in text


def test_federation_round_summary_is_calm_when_reports_agree():
    text = summarize_federation_round(
        n_operators=5, plain_estimate=[4.0, 3.0, 4.0], robust_estimate=[3.9, 3.0, 4.1],
        plain_error=0.2, robust_error=0.19,
    )
    assert "what an honest period looks like" in text


def test_reputation_summary_names_the_consequence_and_the_remedy():
    text = summarize_reputation(0.36, "degraded")
    assert "0.36" in text
    assert "only that an encounter involves you" in text
    assert "falls faster than it rises" in text
    assert "Critical collision warnings reach you in full" in text


def test_backtest_summary_states_a_negative_result_honestly():
    text = summarize_backtest("Starlink vs Kuiper", 12_000, 168.0, 41_002, 0, 1.57, 5.7e-6)
    assert "None crossed" in text
    assert "1.6 kilometres" in text
    assert "A negative result on real data is still a result" in text


# ----------------------------------------------------------- incident report

def test_counterparty_copy_has_no_access_to_the_private_view():
    """conjunction=None: the owner's fidelity is absent, not merely skipped."""
    alert = tailor_alert(_signal(), "degraded")
    report = build_incident_report(None, alert, log_tail_hash="abc123")
    assert "120 metres" not in report.what_happened
    assert "never a trajectory" in report.what_happened
    assert "minute" not in report.what_happened


def test_a_plan_is_withheld_from_a_reader_whose_tier_withheld_the_counterparty():
    plan = negotiate_maneuver(_signal(), AbstractedSignal(
        encounter_id="ENC-90000-99000-1800", sender_operator="Beta",
        local_object_label="Beta-99000", tca_offset_s=1800.0,
        geometry_class="crossing", severity_tier="elevated",
    ))
    degraded = build_incident_report(None, tailor_alert(_signal(), "degraded"), plan=plan)
    full = build_incident_report(None, tailor_alert(_signal(), "full"), plan=plan)
    assert "does not include it" in degraded.what_happens_next
    assert "Beta-99000" not in degraded.what_happens_next
    assert "Beta-99000" in full.what_happens_next


def test_incident_report_requires_the_alert_the_reader_received():
    with pytest.raises(ValueError):
        build_incident_report(_Conjunction())


def test_incident_report_separates_the_private_view_from_the_shared_one():
    conjunction = _Conjunction()
    alert = tailor_alert(_signal(), "degraded")
    report = build_incident_report(conjunction, alert, log_tail_hash="f1753d2fc4aa6667abc")

    assert "120 metres" in report.what_happened      # the owner's own view is specific
    assert "minute" not in report.what_was_shared     # the reader's view is not
    assert "hash-chained log" in report.how_to_verify
    assert "No deconfliction plan" in report.what_happens_next
    assert report.encounter_id == "ENC-90000-99000-1800"
    assert "INCIDENT SUMMARY" in report.as_text()
