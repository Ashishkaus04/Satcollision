"""Historical replay logic, checked on the engineered encounter (whose truth is known)."""

import datetime

import pytest

from satcollision.fleets import build_synthetic_fleets, inject_actor_encounter, tle_epoch_utc
from satcollision.historical_replay import (
    EVENTS, describe, find_norad_by_name, latest_before, replay_event,
)

UTC = datetime.timezone.utc
EPOCH = datetime.datetime(2026, 9, 9, tzinfo=UTC)


@pytest.fixture(scope="module")
def actors():
    objects = build_synthetic_fleets()
    a, b = inject_actor_encounter(objects, tca_offset_min=30.0)
    return a, b


def test_replay_flags_the_engineered_encounter(actors):
    a, b = actors
    tca = EPOCH + datetime.timedelta(minutes=30)
    result = replay_event([a, b], a.norad_id, b.norad_id, tca, as_of_hours=(0.4, 0.1))
    assert result.flagged_at_all
    first = result.earliest_flag
    assert first.hours_before == pytest.approx(0.4)
    assert first.miss_distance_km < 1.0
    assert abs(first.tca_error_min) < 1.0
    assert "Flagged 0 h before" in describe(result)


def test_replay_never_uses_element_sets_from_the_future(actors):
    a, b = actors
    tca = EPOCH + datetime.timedelta(minutes=30)
    # 2 h before TCA is before both element sets' epoch: nothing may be used.
    result = replay_event([a, b], a.norad_id, b.norad_id, tca, as_of_hours=(2.0,))
    row = result.rows[0]
    assert row.epoch_a_utc is None and not row.flagged
    assert "no element set" in row.note
    assert not result.flagged_at_all


def test_latest_before_picks_newest_eligible(actors):
    a, _ = actors
    assert latest_before([a], tle_epoch_utc(a)) is a
    assert latest_before([a], tle_epoch_utc(a) - datetime.timedelta(seconds=1)) is None


def test_missing_object_is_an_error(actors):
    a, b = actors
    with pytest.raises(ValueError):
        replay_event([a], a.norad_id, b.norad_id, EPOCH)


def test_find_norad_by_name(actors):
    a, _ = actors
    a.meta["name"] = "STARLINK-1095 "
    assert find_norad_by_name([a], "starlink-1095") == a.norad_id
    assert find_norad_by_name([a], "STARLINK-2305") is None


def test_registry_events_cite_a_source():
    for event in EVENTS.values():
        assert event.source and event.tca_utc.tzinfo is not None
