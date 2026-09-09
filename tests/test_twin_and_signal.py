from satcollision.fleets import build_synthetic_fleets, inject_actor_encounter, _epoch_from_calendar
from satcollision.twin import propagate_window, screen_conjunctions, compute_pc, DEFAULT_PC_THRESHOLD
from satcollision.signal import abstract_conjunction, assert_no_raw_state, AbstractedSignal


def _engineered_scenario(tca_offset_min=30.0):
    objs = build_synthetic_fleets()
    actor_a, actor_b = inject_actor_encounter(objs, tca_offset_min=tca_offset_min)
    epoch_days, jd0, fr0 = _epoch_from_calendar(2026, 9, 9, 0.0)
    samples = propagate_window(objs, jd0, fr0, duration_s=int(tca_offset_min * 60) + 600, step_s=10.0)
    conjunctions = screen_conjunctions(objs, samples)
    actor_conj = next(
        c for c in conjunctions
        if {c.object_a.norad_id, c.object_b.norad_id} == {actor_a.norad_id, actor_b.norad_id}
    )
    return actor_conj


def test_engineered_conjunction_crosses_threshold():
    conj = _engineered_scenario()
    assert conj.pc >= DEFAULT_PC_THRESHOLD
    assert conj.miss_distance_km < 0.1


def test_compute_pc_monotonic_in_miss_distance():
    near = compute_pc(0.01, combined_sigma_km=0.5)
    far = compute_pc(5.0, combined_sigma_km=0.5)
    assert near > far
    assert 0.0 <= far <= near <= 1.0


def test_abstract_conjunction_returns_none_below_threshold():
    conj = _engineered_scenario()
    assert abstract_conjunction(conj, threshold=1.1) is None  # impossibly high threshold, Pc<=1


def test_abstracted_signal_has_no_raw_state_fields():
    conj = _engineered_scenario()
    sig_a, sig_b = abstract_conjunction(conj)
    assert_no_raw_state(sig_a)
    assert_no_raw_state(sig_b)
    field_names = set(AbstractedSignal.__dataclass_fields__.keys())
    for forbidden in ("position", "velocity", "inclination", "raan", "miss_distance"):
        assert not any(forbidden in name for name in field_names)


def test_abstracted_signals_share_encounter_id_and_geometry():
    conj = _engineered_scenario()
    sig_a, sig_b = abstract_conjunction(conj)
    assert sig_a.encounter_id == sig_b.encounter_id
    assert sig_a.geometry_class == sig_b.geometry_class == conj.geometry_class
    assert sig_a.sender_operator != sig_b.sender_operator
