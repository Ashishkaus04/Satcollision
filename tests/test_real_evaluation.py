import pytest

from satcollision.evaluate import SIGMA_GOOD_KM
from satcollision.fleets import TrackedObject
from satcollision.real_evaluation import (
    DEFAULT_EVALUATION_SCREEN_KM, SCREENING_HORIZON_S, EncounterEvaluation,
    _split_out_burn_in, assign_two_operators, evaluate_encounter, sigma_public_tracking,
    summarize_evaluations,
)
from satcollision.twin import DEFAULT_PC_THRESHOLD, compute_pc


def _objects(n=10):
    """Catalogue stand-ins: the ownership split only reads norad_id, never the orbit."""
    return [TrackedObject(norad_id=40000 + i, operator="REAL", satrec=None) for i in range(n)]


class _Conj:
    """A Conjunction-shaped stand-in — the evaluation reads only these fields."""

    def __init__(self, miss_km, tca_s, operator_a="Operator-A", operator_b="Operator-B",
                 norad_a=41000, norad_b=42001):
        self.object_a = TrackedObject(norad_id=norad_a, operator=operator_a, satrec=None)
        self.object_b = TrackedObject(norad_id=norad_b, operator=operator_b, satrec=None)
        self.tca_offset_s = tca_s
        self.miss_distance_km = miss_km
        self.pc = compute_pc(miss_km, combined_sigma_km=SIGMA_GOOD_KM)
        self.geometry_class = "crossing"
        self.relative_speed_km_s = 11.2

    @property
    def pair_label(self):
        return f"{self.object_a.label} <-> {self.object_b.label}"


# ------------------------------------------------------------ ownership split

def test_parity_split_is_deterministic_and_balanced():
    objects = _objects(10)
    first = assign_two_operators(objects)
    second = assign_two_operators(objects)
    assert [o.operator for o in first] == [o.operator for o in second]
    assert sum(o.operator == "Operator-A" for o in first) == 5


def test_split_does_not_mutate_the_loaded_catalogue():
    objects = _objects(4)
    assign_two_operators(objects)
    assert all(o.operator == "REAL" for o in objects)


def test_split_assigns_by_norad_parity():
    split = assign_two_operators(_objects(4))
    for obj in split:
        expected = "Operator-A" if obj.norad_id % 2 == 0 else "Operator-B"
        assert obj.operator == expected


def test_split_accepts_custom_operator_names():
    split = assign_two_operators(_objects(2), name_a="Alpha", name_b="Beta")
    assert {o.operator for o in split} == {"Alpha", "Beta"}


# --------------------------------------------------------- encounter scoring

def test_cooperative_tracking_never_detects_later_than_no_cooperation():
    """No-cooperation has strictly worse (larger) sigma far from TCA, so it
    can only ever detect at the same time or later — never earlier."""
    for miss_km, tca_s in ((0.1, 1800.0), (0.4, 3600.0), (0.8, 900.0)):
        result = evaluate_encounter(_Conj(miss_km, tca_s))
        assert result.lead_time_cooperative_s >= result.lead_time_nocoop_s
        assert result.lead_time_gain_s >= 0.0


def test_a_close_encounter_is_detected_by_cooperative_tracking():
    result = evaluate_encounter(_Conj(miss_km=0.1, tca_s=1800.0))
    assert result.lead_time_cooperative_s > 0.0
    assert result.pc >= DEFAULT_PC_THRESHOLD


def test_evaluation_row_reports_minutes_and_metres():
    row = evaluate_encounter(_Conj(miss_km=0.12, tca_s=1800.0)).as_row()
    assert row["miss_distance_m"] == pytest.approx(120.0)
    assert row["tca_offset_min"] == pytest.approx(30.0)
    assert row["lead_time_gain_min"] >= 0.0
    assert row["operator_a"] == "Operator-A"


def test_encounter_id_comes_from_the_abstraction_layer():
    """The evaluation identifies encounters the same way the federation does."""
    result = evaluate_encounter(_Conj(miss_km=0.1, tca_s=1800.0, norad_a=41000, norad_b=42001))
    assert result.encounter_id == "ENC-41000-42001-1800"


# ------------------------------------------------------------------ summary

def _evaluation(nocoop_s, coop_s):
    return EncounterEvaluation(
        encounter_id="ENC-1-2-100", operator_a="Operator-A", operator_b="Operator-B",
        object_a="A-1", object_b="B-2", tca_offset_s=1800.0, miss_distance_km=0.2,
        pc=5e-4, geometry_class="crossing",
        lead_time_nocoop_s=nocoop_s, lead_time_cooperative_s=coop_s,
    )


def test_summary_reports_medians_and_the_gain():
    summary = summarize_evaluations([
        _evaluation(600.0, 1200.0), _evaluation(300.0, 1500.0), _evaluation(900.0, 1800.0),
    ])
    assert summary["n_encounters"] == 3
    assert summary["median_lead_time_nocoop_min"] == pytest.approx(10.0)
    assert summary["median_lead_time_cooperative_min"] == pytest.approx(25.0)
    assert summary["median_lead_time_gain_min"] == pytest.approx(15.0)
    assert summary["raw_data_exposed_pct"]["federated"] == 0.0
    assert summary["raw_data_exposed_pct"]["full_sharing"] == 100.0


def test_summary_counts_encounters_no_cooperation_never_sees():
    summary = summarize_evaluations([_evaluation(0.0, 1200.0), _evaluation(600.0, 1200.0)])
    assert summary["n_missed_entirely_by_nocoop"] == 1
    assert summary["pct_missed_entirely_by_nocoop"] == pytest.approx(50.0)


def test_empty_summary_is_reported_honestly_rather_than_as_zeros():
    """A run with no threshold crossings must not look like a run with bad results."""
    assert summarize_evaluations([]) == {"n_encounters": 0}


def test_screening_distance_covers_every_possible_threshold_crossing():
    """The 5 km screen must not be able to miss a crossing: at sigma=0.5 km the
    Pc curve falls below threshold by ~1.02 km, and larger sigma only lowers it."""
    assert compute_pc(1.1, combined_sigma_km=SIGMA_GOOD_KM) < DEFAULT_PC_THRESHOLD
    assert compute_pc(DEFAULT_EVALUATION_SCREEN_KM, combined_sigma_km=SIGMA_GOOD_KM) < DEFAULT_PC_THRESHOLD
    assert compute_pc(1.0, combined_sigma_km=SIGMA_GOOD_KM) >= DEFAULT_PC_THRESHOLD


# --------------------------------------------- absolute-time tracking model

def test_public_tracking_precision_improves_as_tca_approaches():
    far = sigma_public_tracking(SCREENING_HORIZON_S)
    mid = sigma_public_tracking(SCREENING_HORIZON_S / 2)
    near = sigma_public_tracking(0.0)
    assert far > mid > near


def test_public_tracking_sigma_is_anchored_to_clock_time_not_to_tca():
    """The bug this model exists to avoid: an encounter six days out must not
    be scored as though tracking were fresh just because the window is long."""
    assert sigma_public_tracking(6 * 24 * 3600.0) == sigma_public_tracking(SCREENING_HORIZON_S)


def test_lead_time_is_capped_at_the_screening_horizon():
    """A conjunction a week away cannot be acted on before it is screened."""
    week_out = evaluate_encounter(_Conj(miss_km=0.1, tca_s=7 * 24 * 3600.0))
    assert week_out.lead_time_cooperative_s <= SCREENING_HORIZON_S
    inside = evaluate_encounter(_Conj(miss_km=0.1, tca_s=3600.0))
    assert inside.lead_time_cooperative_s <= 3600.0


def test_encounters_inside_the_burn_in_window_are_set_aside():
    """An encounter clipped by the window start scores identically in every
    scenario, so it must not be averaged in with the ones that aren't."""
    inside = _Conj(miss_km=0.1, tca_s=SCREENING_HORIZON_S / 2)
    outside = _Conj(miss_km=0.1, tca_s=SCREENING_HORIZON_S * 1.5)
    scorable, burn_in = _split_out_burn_in([inside, outside], SCREENING_HORIZON_S)
    assert scorable == [outside]
    assert burn_in == [inside]


def test_a_burn_in_encounter_would_have_shown_zero_gain():
    """Why the exclusion exists, asserted rather than asserted in prose."""
    clipped = evaluate_encounter(_Conj(miss_km=0.1, tca_s=SCREENING_HORIZON_S / 4))
    assert clipped.lead_time_gain_s == 0.0
    assert clipped.lead_time_nocoop_s == clipped.lead_time_cooperative_s
