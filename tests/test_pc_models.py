"""
Validation for the selectable Pc model: the wiring, not the integral.

``test_pc.py`` establishes that ``foster_pc`` computes the right number.
This file is about the other half of the risk — that threading a model choice
through screening, lead-time scanning and the evaluation entry points quietly
changes something it was not supposed to.

Two properties carry most of the weight here:

* **The default is not merely close to the old behaviour, it is the old
  behaviour.** Every assertion about the circular path uses ``==`` on floats,
  not ``approx``. The point of keeping circular as the default is that the
  124 committed tests, every figure, the dashboard and the report numbers
  stay valid without re-deriving them; "almost identical" would not support
  that claim, and a tolerance would hide exactly the drift worth catching.
* **A model swap changes the estimator and nothing else.** Same encounters,
  same screening, same regime sigmas, same threshold. If a Foster run screened
  a different set of encounters than the circular run, the two result files
  would not be comparable and the step-4 comparison would be meaningless.

The third recurring theme is that missing geometry must be loud. A model that
needs state vectors and silently falls back to the circular formula when they
are absent would produce a comparison run that is partly a comparison of the
circular model with itself, with nothing in the output saying which rows.
"""

import math

import numpy as np
import pytest

from satcollision.evaluate import SIGMA_GOOD_KM, detection_lead_time
from satcollision.fleets import (
    _epoch_from_calendar,
    build_synthetic_fleets,
    inject_actor_encounter,
)
from satcollision.pc import (
    CIRCULAR_PC_MODEL,
    CircularPcModel,
    EncounterGeometry,
    FosterPcModel,
    encounter_plane_miss_km,
    pc_model_name,
    resolve_pc_model,
)
from satcollision.real_evaluation import evaluate_encounter
from satcollision.twin import (
    DEFAULT_COMBINED_SIGMA_KM,
    DEFAULT_PC_THRESHOLD,
    compute_pc,
    propagate_window,
    screen_conjunctions,
)

TCA_OFFSET_MIN = 30.0


def _screened(pc_model=None):
    """One engineered encounter, screened end to end, under a given model."""
    objects = build_synthetic_fleets()
    actor_a, actor_b = inject_actor_encounter(objects, tca_offset_min=TCA_OFFSET_MIN)
    _, jd0, fr0 = _epoch_from_calendar(2026, 9, 9, 0.0)
    samples = propagate_window(
        objects, jd0, fr0, duration_s=int(TCA_OFFSET_MIN * 60) + 600, step_s=10.0
    )
    conjunctions = screen_conjunctions(objects, samples, pc_model=pc_model)
    actor = next(
        c for c in conjunctions
        if {c.object_a.norad_id, c.object_b.norad_id} == {actor_a.norad_id, actor_b.norad_id}
    )
    return conjunctions, actor


# --------------------------------------------------------------------------
# The screening pass records the geometry it used to throw away
# --------------------------------------------------------------------------

def test_screening_records_tca_state_vectors():
    conjunctions, actor = _screened()
    assert all(c.has_state_vectors for c in conjunctions)

    for vector in (actor.position_a_km, actor.position_b_km,
                   actor.velocity_a_km_s, actor.velocity_b_km_s):
        assert vector.shape == (3,)
        assert np.all(np.isfinite(vector))


def test_recorded_state_reproduces_the_scalars_derived_from_it():
    """The vectors and the scalars must describe the same encounter.

    Miss distance is |r_rel| and relative speed is |v_rel|. If the recorded
    state came from a different timestep than the one the minimum was found
    at, this is where it shows up — and a Pc model fed that state would be
    scoring a geometry the rest of the pipeline never saw.
    """
    _, actor = _screened()
    assert float(np.linalg.norm(actor.r_rel_km)) == pytest.approx(
        actor.miss_distance_km, rel=1.0e-12
    )
    assert float(np.linalg.norm(actor.v_rel_km_s)) == pytest.approx(
        actor.relative_speed_km_s, rel=1.0e-12
    )


def test_state_vectors_are_copies_not_views_into_the_propagation_arrays():
    """Mutating a returned vector must not corrupt a sibling conjunction."""
    conjunctions, actor = _screened()
    before = [c.miss_distance_km for c in conjunctions]
    actor.position_a_km[0] += 1000.0
    assert [c.miss_distance_km for c in conjunctions] == before


# --------------------------------------------------------------------------
# The default path is the old path
# --------------------------------------------------------------------------

def test_default_screening_is_bit_identical_to_the_explicit_circular_model():
    _, default_actor = _screened(pc_model=None)
    _, circular_actor = _screened(pc_model=CircularPcModel())
    assert default_actor.pc == circular_actor.pc  # exact, not approx
    assert default_actor.miss_distance_km == circular_actor.miss_distance_km


def test_circular_model_matches_compute_pc_exactly():
    geometry = EncounterGeometry(miss_distance_km=0.828)
    for sigma in (0.1, 0.5, 3.0):
        assert CIRCULAR_PC_MODEL.pc(geometry, sigma) == compute_pc(0.828, sigma)


def test_default_lead_time_is_bit_identical_to_the_explicit_circular_model():
    _, actor = _screened()
    geometry = EncounterGeometry.from_conjunction(actor)
    window = actor.tca_offset_s

    baseline = detection_lead_time(
        actor.miss_distance_km, window, sigma_fn=lambda lead: SIGMA_GOOD_KM, step_s=60.0
    )
    explicit = detection_lead_time(
        actor.miss_distance_km, window, sigma_fn=lambda lead: SIGMA_GOOD_KM, step_s=60.0,
        pc_model=CircularPcModel(), geometry=geometry,
    )
    assert baseline == explicit


def test_default_encounter_evaluation_is_bit_identical():
    _, actor = _screened()
    baseline = evaluate_encounter(actor)
    explicit = evaluate_encounter(actor, pc_model=CircularPcModel())

    assert baseline.lead_time_nocoop_s == explicit.lead_time_nocoop_s
    assert baseline.lead_time_cooperative_s == explicit.lead_time_cooperative_s
    assert baseline.pc_model == "circular"


# --------------------------------------------------------------------------
# Foster at ratio 1.0 is the continuity check
# --------------------------------------------------------------------------

def _perpendicular_geometry(miss_km=0.828):
    """A true closest approach: separation exactly normal to relative velocity.

    Built by hand rather than screened, because a grid search lands *near* TCA
    and the residual along-track offset is the very thing the next few tests
    are trying to hold constant.
    """
    position_a = np.array([7000.0, 0.0, 0.0])
    velocity_a = np.array([0.0, 7.5, 0.0])
    position_b = position_a + np.array([0.0, 0.0, miss_km])
    velocity_b = np.array([0.0, -7.5, 0.0])
    return EncounterGeometry(miss_km, position_a, position_b, velocity_a, velocity_b)


def _off_tca_geometry(miss_km=0.828):
    """The same separation, but pointing along the relative velocity.

    What a grid search produces when the sampled minimum sits away from the
    true one. The perpendicular component — the part that decides whether the
    objects hit — is a hundredth of the raw separation here.
    """
    position_a = np.array([7000.0, 0.0, 0.0])
    velocity_a = np.array([0.0, 7.5, 0.0])
    position_b = position_a + np.array([0.0, miss_km, 0.0])
    velocity_b = np.array([0.0, -7.5, 0.15])
    return EncounterGeometry(miss_km, position_a, position_b, velocity_a, velocity_b)


# --------------------------------------------------------------------------
# The encounter-plane projection is a second, separable effect
# --------------------------------------------------------------------------

def test_in_plane_miss_equals_the_raw_miss_at_a_true_closest_approach():
    geometry = _perpendicular_geometry()
    assert geometry.in_plane_miss_km == pytest.approx(geometry.miss_distance_km, rel=1.0e-12)


def test_in_plane_miss_strips_the_along_velocity_component():
    geometry = _off_tca_geometry()
    assert geometry.in_plane_miss_km < 0.02 * geometry.miss_distance_km

    # The projection is the perpendicular leg of a right triangle.
    r_rel, v_rel = geometry.r_rel_km, geometry.v_rel_km_s
    along = abs(float(np.dot(r_rel, v_rel)) / float(np.linalg.norm(v_rel)))
    assert math.hypot(geometry.in_plane_miss_km, along) == pytest.approx(
        float(np.linalg.norm(r_rel)), rel=1.0e-12
    )


def test_foster_scores_the_in_plane_miss_not_the_raw_one():
    """The finding that makes a naive model comparison misleading.

    At an off-TCA sample the refined model does not disagree with the circular
    one because the covariance is anisotropic — this runs at ratio 1.0, where
    it is not. It disagrees because the two are scoring different miss
    distances. Asserting it here keeps the step-4 comparison from being read as
    an anisotropy result when part of it is a TCA-resolution result.
    """
    geometry = _off_tca_geometry()
    sigma = DEFAULT_COMBINED_SIGMA_KM
    refined = FosterPcModel(along_track_ratio=1.0).pc(geometry, sigma)

    assert refined == pytest.approx(compute_pc(geometry.in_plane_miss_km, sigma), rel=1.0e-3)
    assert refined > 3.0 * compute_pc(geometry.miss_distance_km, sigma)


def test_encounter_plane_miss_rejects_a_stationary_encounter():
    with pytest.raises(ValueError, match="encounter plane is undefined"):
        encounter_plane_miss_km(np.array([0.0, 0.828, 0.0]), np.zeros(3))


def test_foster_at_unit_ratio_matches_circular_at_a_true_closest_approach():
    """Continuity, with the projection effect held at zero.

    The residual is the small-hard-body approximation in ``compute_pc`` and
    nothing else, and it shrinks as sigma grows — which is the signature of
    that approximation rather than of a wiring error.
    """
    geometry = _perpendicular_geometry()
    isotropic = FosterPcModel(along_track_ratio=1.0)

    errors = []
    for sigma in (0.5, 1.0, 3.0):
        refined = isotropic.pc(geometry, sigma)
        circular = compute_pc(geometry.miss_distance_km, sigma)
        assert refined == pytest.approx(circular, rel=1.0e-3)
        errors.append(abs(refined - circular) / circular)
    assert errors == sorted(errors, reverse=True), errors


def test_foster_at_unit_ratio_reproduces_the_circular_result():
    """An isotropic covariance must give back the isotropic answer.

    This is the check that has to pass before any comparison run means
    anything: it separates "the refined model disagrees because the error
    ellipsoid is elongated" from "the refined model disagrees because it is
    wired up wrong". The residual is the small-hard-body approximation in
    ``compute_pc``, not a difference of substance.
    """
    _, actor = _screened()
    geometry = EncounterGeometry.from_conjunction(actor)
    isotropic = FosterPcModel(along_track_ratio=1.0)

    for sigma in (0.5, 1.0, 3.0):
        assert isotropic.pc(geometry, sigma) == pytest.approx(
            compute_pc(actor.miss_distance_km, sigma), rel=1.0e-3
        )


def test_foster_at_unit_ratio_gives_the_same_lead_times():
    _, actor = _screened()
    geometry = EncounterGeometry.from_conjunction(actor)
    window = actor.tca_offset_s

    baseline, _ = detection_lead_time(
        actor.miss_distance_km, window,
        sigma_fn=lambda lead: SIGMA_GOOD_KM, step_s=60.0,
    )
    refined, _ = detection_lead_time(
        actor.miss_distance_km, window,
        sigma_fn=lambda lead: SIGMA_GOOD_KM, step_s=60.0,
        pc_model=FosterPcModel(along_track_ratio=1.0), geometry=geometry,
    )
    # Lead time is quantised to the 60 s scan step, so agreement is exact
    # unless the threshold crossing lands inside one step of a boundary.
    assert abs(baseline - refined) <= 60.0


def test_anisotropy_changes_the_answer_and_the_ratio_is_the_dial():
    _, actor = _screened()
    geometry = EncounterGeometry.from_conjunction(actor)

    values = [
        FosterPcModel(along_track_ratio=k).pc(geometry, DEFAULT_COMBINED_SIGMA_KM)
        for k in (1.0, 2.0, 4.0, 8.0)
    ]
    assert len(set(values)) == len(values), "ratio had no effect on Pc"
    assert all(0.0 <= v <= 1.0 for v in values)


# --------------------------------------------------------------------------
# Model swap changes the estimator and nothing else
# --------------------------------------------------------------------------

def test_a_model_swap_screens_the_same_encounters():
    """Different Pc, same population — or the two runs are not comparable."""
    circular, _ = _screened(pc_model=None)
    foster, _ = _screened(pc_model=FosterPcModel())

    def key(c):
        return (c.object_a.norad_id, c.object_b.norad_id)

    assert {key(c) for c in circular} == {key(c) for c in foster}
    by_key = {key(c): c for c in foster}
    for c in circular:
        other = by_key[key(c)]
        assert c.miss_distance_km == other.miss_distance_km
        assert c.tca_offset_s == other.tca_offset_s
        assert c.geometry_class == other.geometry_class


def test_evaluation_rows_record_which_model_produced_them():
    _, actor = _screened(pc_model=FosterPcModel())
    evaluation = evaluate_encounter(actor, pc_model=FosterPcModel())
    assert evaluation.pc_model == "foster(k=8)"
    assert evaluation.as_row()["pc_model"] == "foster(k=8)"


def test_model_names_are_distinct_and_resolve_handles_none():
    assert pc_model_name(None) == "circular"
    assert pc_model_name(CircularPcModel()) == "circular"
    assert pc_model_name(FosterPcModel(along_track_ratio=1.0)) == "foster(k=1)"
    assert pc_model_name(FosterPcModel(along_track_ratio=8.0)) == "foster(k=8)"
    assert resolve_pc_model(None) is CIRCULAR_PC_MODEL
    model = FosterPcModel()
    assert resolve_pc_model(model) is model


# --------------------------------------------------------------------------
# The per-object hook (asymmetry is not wired to the regimes yet)
# --------------------------------------------------------------------------

def test_symmetric_split_is_what_the_scalar_sigma_means():
    """``sigma_km`` is the combined sigma, so each object carries it over sqrt(2)."""
    _, actor = _screened()
    geometry = EncounterGeometry.from_conjunction(actor)
    model = FosterPcModel()
    sigma = DEFAULT_COMBINED_SIGMA_KM

    implicit = model.pc(geometry, sigma)
    explicit = model.pc(geometry, sigma,
                        sigma_a_km=sigma / math.sqrt(2.0),
                        sigma_b_km=sigma / math.sqrt(2.0))
    assert implicit == explicit


def test_per_object_sigmas_take_effect():
    """The hook the asymmetric regime will use — one object tracked better.

    Not yet wired to ``sigma_fn``, which supplies a single combined sigma, but
    asserted here so the interface cannot rot before it is used.
    """
    _, actor = _screened()
    geometry = EncounterGeometry.from_conjunction(actor)
    model = FosterPcModel()

    balanced = model.pc(geometry, 1.0, sigma_a_km=0.5, sigma_b_km=0.5)
    lopsided = model.pc(geometry, 1.0, sigma_a_km=0.05, sigma_b_km=0.5)
    assert balanced != lopsided


def test_circular_model_combines_per_object_sigmas_in_quadrature():
    geometry = EncounterGeometry(miss_distance_km=0.828)
    combined = math.hypot(0.3, 0.4)
    assert CIRCULAR_PC_MODEL.pc(geometry, 999.0, sigma_a_km=0.3, sigma_b_km=0.4) == (
        compute_pc(0.828, combined)
    )


@pytest.mark.parametrize("model", [CircularPcModel(), FosterPcModel()])
def test_half_specified_per_object_sigmas_are_rejected(model):
    _, actor = _screened()
    geometry = EncounterGeometry.from_conjunction(actor)
    with pytest.raises(ValueError, match="both"):
        model.pc(geometry, 0.5, sigma_a_km=0.3)
    with pytest.raises(ValueError, match="both"):
        model.pc(geometry, 0.5, sigma_b_km=0.3)


# --------------------------------------------------------------------------
# Missing or inconsistent geometry must be loud
# --------------------------------------------------------------------------

def test_foster_refuses_a_geometry_with_no_state_vectors():
    """No silent fallback to circular — that would corrupt a comparison run."""
    bare = EncounterGeometry(miss_distance_km=0.828)
    assert not bare.has_state_vectors
    with pytest.raises(ValueError, match="state vectors"):
        FosterPcModel().pc(bare, DEFAULT_COMBINED_SIGMA_KM)


def test_lead_time_with_foster_and_no_geometry_refuses():
    with pytest.raises(ValueError, match="state vectors"):
        detection_lead_time(
            0.828, 1800.0, sigma_fn=lambda lead: SIGMA_GOOD_KM,
            step_s=60.0, pc_model=FosterPcModel(),
        )


def test_lead_time_rejects_geometry_that_disagrees_with_the_miss_distance():
    _, actor = _screened()
    geometry = EncounterGeometry.from_conjunction(actor)
    with pytest.raises(ValueError, match="disagrees"):
        detection_lead_time(
            actor.miss_distance_km + 1.0, actor.tca_offset_s,
            sigma_fn=lambda lead: SIGMA_GOOD_KM, step_s=60.0,
            pc_model=CircularPcModel(), geometry=geometry,
        )


def test_hand_built_conjunctions_still_work_under_the_default_model():
    """Nothing in the pipeline may now *require* state vectors.

    ``summaries`` and several tests build Conjunction-like objects from
    scalars. The new fields are optional and the default model ignores them.
    """
    geometry = EncounterGeometry(miss_distance_km=0.5)
    lead, _ = detection_lead_time(
        0.5, 3600.0, sigma_fn=lambda lead: SIGMA_GOOD_KM,
        threshold=DEFAULT_PC_THRESHOLD, step_s=60.0,
    )
    assert lead > 0.0
    assert CIRCULAR_PC_MODEL.pc(geometry, SIGMA_GOOD_KM) > 0.0
