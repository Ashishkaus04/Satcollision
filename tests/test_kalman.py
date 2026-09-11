import numpy as np

from satcollision.fleets import build_synthetic_fleets, inject_actor_encounter, _epoch_from_calendar
from satcollision.twin import propagate_window, screen_conjunctions
from satcollision.kalman import (
    initial_state, predict, update, track_object, isotropic_sigma_km,
    combined_sigma_km, refine_pc_for_conjunction,
)


def test_predict_only_grows_uncertainty():
    state = initial_state(position_km=np.array([7000.0, 0.0, 0.0]), velocity_km_s=np.array([0.0, 7.5, 0.0]))
    sigma_0 = isotropic_sigma_km(state)
    for _ in range(5):
        state = predict(state, dt=30.0, accel_noise_km_s2=1e-6)
    sigma_after = isotropic_sigma_km(state)
    assert sigma_after > sigma_0


def test_update_shrinks_uncertainty_relative_to_predict_only():
    state = initial_state(position_km=np.array([7000.0, 0.0, 0.0]), velocity_km_s=np.array([0.0, 7.5, 0.0]))
    predicted = predict(state, dt=60.0, accel_noise_km_s2=1e-6)
    sigma_predict_only = isotropic_sigma_km(predicted)

    updated = update(predicted, measured_position_km=predicted.x[:3], position_noise_km=0.05)
    sigma_after_update = isotropic_sigma_km(updated)

    assert sigma_after_update < sigma_predict_only


def test_track_object_sigma_oscillates_between_measurements():
    rng = np.random.default_rng(0)
    n = 40
    t_offsets_s = np.arange(n) * 10.0
    # Simple straight-line "ground truth" -- the filter under test doesn't
    # need real orbital dynamics to exercise growth/shrink behaviour.
    true_positions = np.stack([7000.0 + 7.5 * t_offsets_s, np.zeros(n), np.zeros(n)], axis=1)
    true_velocities = np.tile(np.array([7.5, 0.0, 0.0]), (n, 1))

    states = track_object(
        true_positions_km=true_positions,
        true_velocities_km_s=true_velocities,
        t_offsets_s=t_offsets_s,
        measurement_interval_s=100.0,
        position_noise_km=0.05,
        accel_noise_km_s2=1e-6,
        rng=rng,
    )
    sigmas = [isotropic_sigma_km(s) for s in states]

    # Sigma should be smaller right after a measurement update than right
    # before the next one -- i.e. it is not monotonic, unlike a fixed
    # constant which never changes at all.
    idx_last_before_update = 9  # t=90s, just before the update at t=100s
    idx_right_after_update = 10  # t=100s, right after the update fires
    assert sigmas[idx_right_after_update] < sigmas[idx_last_before_update]


def test_combined_sigma_is_at_least_either_individual_component():
    state_a = initial_state(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), position_sigma_km=0.1)
    state_b = initial_state(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), position_sigma_km=0.2)
    combined = combined_sigma_km(state_a, state_b)
    assert combined >= isotropic_sigma_km(state_a)
    assert combined >= isotropic_sigma_km(state_b)


def test_refine_pc_for_conjunction_runs_on_a_real_engineered_encounter():
    objects = build_synthetic_fleets(seed=42)
    actor_a, actor_b = inject_actor_encounter(objects, tca_offset_min=30.0)
    epoch_days, jd0, fr0 = _epoch_from_calendar(2026, 9, 9, 0.0)
    prop = propagate_window(objects, jd0, fr0, duration_s=40 * 60, step_s=10.0)
    conjunctions = screen_conjunctions(objects, prop, screening_distance_km=25.0)
    actor_conj = next(
        c for c in conjunctions
        if {c.object_a.norad_id, c.object_b.norad_id} == {actor_a.norad_id, actor_b.norad_id}
    )

    result = refine_pc_for_conjunction(actor_conj, jd0, fr0, seed=0)

    assert result["pair_label"] == actor_conj.pair_label
    assert result["fixed_sigma_pc"] == actor_conj.pc
    assert result["kalman_sigma_km"] > 0.0
    assert 0.0 <= result["kalman_sigma_pc"] <= 1.0
