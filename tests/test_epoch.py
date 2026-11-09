"""Run-start epoch selection for real-catalogue runs."""

import datetime

import pytest

from satcollision.fleets import catalogue_epoch, load_tle_file, tle_epoch_utc
from satcollision.real_evaluation import median_tle_age_days, resolve_epoch

from tests.test_environment import TWO_TLES

UTC = datetime.timezone.utc


@pytest.fixture
def objects(tmp_path):
    path = tmp_path / "two.tle"
    path.write_text(TWO_TLES, encoding="utf-8")
    return load_tle_file(str(path), operator_name="Kuiper")


def test_tle_epoch_utc_decodes_day_of_year(objects):
    # Day 252.52899245 of 2026: day 252 -> 9 Sep, .52899245 of a day -> 12:41:44.95 UTC.
    got = tle_epoch_utc(objects[1])
    want = datetime.datetime(2026, 9, 9, 12, 41, 44, 950000, tzinfo=UTC)
    assert abs((got - want).total_seconds()) < 0.5


def test_catalogue_epoch_is_newest_element_set(objects):
    assert catalogue_epoch(objects) == datetime.datetime(2026, 9, 10, 0, 35, tzinfo=UTC)


def test_resolve_epoch_modes(objects):
    assert resolve_epoch(objects, "catalogue") == catalogue_epoch(objects)
    fixed = datetime.datetime(2026, 9, 1, 0, 0)
    assert resolve_epoch(objects, fixed) == fixed.replace(tzinfo=UTC)
    now = resolve_epoch(objects, None)
    assert abs((now - datetime.datetime.now(UTC)).total_seconds()) < 120
    with pytest.raises(ValueError):
        resolve_epoch(objects, "yesterday")


def test_median_tle_age(objects):
    start = catalogue_epoch(objects)
    # Ages at the start are ~0 and ~0.496 days -> median ~0.248.
    assert median_tle_age_days(objects, start) == pytest.approx(0.248, abs=0.01)
