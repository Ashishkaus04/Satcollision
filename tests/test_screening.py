"""Refined screening: finds approaches between samples and matches direct SGP4."""

import numpy as np
import pytest

from satcollision.fleets import build_synthetic_fleets, inject_actor_encounter, _epoch_from_calendar
from satcollision.screening import screen_conjunctions_refined, _refine
from satcollision.twin import propagate_window, screen_conjunctions


@pytest.fixture(scope="module")
def scene():
    objects = build_synthetic_fleets()
    a, b = inject_actor_encounter(objects, tca_offset_min=30.25, crossing_angle_deg=55.0)
    _, jd0, fr0 = _epoch_from_calendar(2026, 9, 9, 0.0)
    return objects, a, b, jd0, fr0


def _actor_hits(conjs, a, b):
    return [c for c in conjs if {c.object_a.norad_id, c.object_b.norad_id} == {a.norad_id, b.norad_id}]


def test_refined_screen_finds_crossing_encounter_the_grid_can_miss(scene):
    objects, a, b, jd0, fr0 = scene
    refined = _actor_hits(screen_conjunctions_refined(objects, jd0, fr0, 3600.0, step_s=60.0,
                                                      screening_distance_km=5.0), a, b)
    assert len(refined) == 1
    c = refined[0]
    assert c.miss_distance_km < 1.0
    assert abs(c.tca_offset_s - 1815.0) < 60.0
    assert c.has_state_vectors
    # The miss distance is the true SGP4 minimum, not a sample.
    t, d, *_ = _refine(a, b, jd0, fr0, c.tca_offset_s, 5.0)
    assert d == pytest.approx(c.miss_distance_km, abs=1e-3)
    # TCA sits 15 s off the 60 s grid; at a crossing speed of several km/s the
    # grid screen never samples the pair inside 5 km, so it misses it entirely.
    assert c.relative_speed_km_s > 3.0
    grid = _actor_hits(screen_conjunctions(objects, propagate_window(objects, jd0, fr0, 3600.0, 60.0),
                                           screening_distance_km=5.0), a, b)
    assert grid == []


def test_refined_screen_is_step_independent(scene):
    objects, a, b, jd0, fr0 = scene
    ds = []
    for step in (10.0, 30.0, 60.0):
        hit = _actor_hits(screen_conjunctions_refined(objects, jd0, fr0, 3600.0, step_s=step,
                                                      screening_distance_km=5.0), a, b)
        ds.append(hit[0].miss_distance_km)
    assert np.ptp(ds) < 1e-3


def test_chunking_does_not_change_results(scene):
    objects, a, b, jd0, fr0 = scene
    one = screen_conjunctions_refined(objects, jd0, fr0, 3600.0, step_s=30.0, chunk_s=1e9)
    many = screen_conjunctions_refined(objects, jd0, fr0, 3600.0, step_s=30.0, chunk_s=600.0)
    key = lambda cs: sorted((c.object_a.norad_id, c.object_b.norad_id, round(c.tca_offset_s, 1)) for c in cs)
    assert key(one) == key(many)


def test_pair_filter_and_separate_approaches(scene):
    objects, a, b, jd0, fr0 = scene
    only_actor = screen_conjunctions_refined(
        objects, jd0, fr0, 3600.0, step_s=30.0,
        pair_filter=lambda x, y: {x.norad_id, y.norad_id} == {a.norad_id, b.norad_id})
    assert {(c.object_a.norad_id, c.object_b.norad_id) for c in only_actor} <= {
        (a.norad_id, b.norad_id), (b.norad_id, a.norad_id)}
