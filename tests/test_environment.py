"""Tests for the step-controlled simulation environment (environment.py)."""

import datetime

import numpy as np
import pytest

from satcollision.environment import SYNTHETIC_EPOCH, SyntheticEnvironment

# Two real Kuiper element sets from data/kuiper.tle (epochs: day 253.025 and 252.529 of 2026).
TWO_TLES = """\
KUIPER-00008
1 63724U 25088A   26253.02492513 -.00000124  00000+0  12673-4 0  9999
2 63724  51.9032 352.8782 0000312  93.1920 266.9078 14.79929457 75343
KUIPER-00009
1 63725U 25088B   26252.52899245  .00006570  00000+0  44064-3 0  9995
2 63725  51.9033   4.5726 0000768 108.7585 251.3473 15.07950617 75137
"""


@pytest.fixture(scope="module")
def env():
    return SyntheticEnvironment.from_synthetic_fleets(inject_encounter=True, step_size_s=60.0)


def test_synthetic_clock_starts_at_the_fleet_epoch(env):
    env.reset()
    y, m, d, h = SYNTHETIC_EPOCH
    expected = datetime.datetime(y, m, d, tzinfo=datetime.timezone.utc) + datetime.timedelta(hours=h)
    assert abs((env.sim_time - expected).total_seconds()) < 1.0


def test_engineered_encounter_is_visible_around_tca(env):
    """The injected pair is built to meet 30 minutes after the fleet epoch.
    Starting the clock anywhere else (e.g. wall-clock now) would never show it."""
    env.seek(30 * 60)
    snap = env.snapshot()
    actors = [c for c in snap.active_conjunctions if c.object_a.is_actor and c.object_b.is_actor]
    assert len(actors) == 1
    assert actors[0].miss_distance_km < 1.0


def test_step_advances_clock_and_reset_rewinds(env):
    env.reset()
    t0 = env.sim_time
    snap = env.step(n=3)
    assert snap.step_index == 3
    assert (env.sim_time - t0).total_seconds() == pytest.approx(180.0)
    env.reset()
    assert env.sim_time == t0


def test_snapshot_does_not_advance_time(env):
    env.reset()
    t0 = env.sim_time
    env.snapshot()
    env.snapshot()
    assert env.sim_time == t0


def test_arrays_line_up_with_snapshot(env):
    env.reset()
    snap = env.snapshot()
    positions, ids = env.positions_array()
    velocities, ids_v = env.velocities_array()
    assert ids == ids_v == [s.norad_id for s in snap.satellites]
    assert positions.shape == velocities.shape == (len(ids), 3)
    radii = np.linalg.norm(positions, axis=1)
    assert np.all(radii > 6378.0) and np.all(radii < 6378.0 + 2000.0)


def test_operator_filter_and_lookup(env):
    snap = env.snapshot()
    assert set(snap.operator_names) == {"Alpha", "Beta", "Gamma"}
    alpha = snap.get_operator_satellites("Alpha")
    assert alpha and all(s.operator == "Alpha" for s in alpha)
    assert snap.get_satellite(alpha[0].norad_id) is alpha[0]
    assert snap.get_satellite(-1) is None


def test_pc_threshold_parameter_is_honoured():
    strict = SyntheticEnvironment.from_synthetic_fleets(pc_threshold=1.0)
    strict.seek(30 * 60)
    snap = strict.snapshot()
    assert snap.conjunctions  # the encounter is still screened ...
    assert snap.active_conjunctions == []  # ... but nothing reaches Pc 1.0


def test_tle_environment_starts_at_catalogue_epoch(tmp_path):
    path = tmp_path / "two.tle"
    path.write_text(TWO_TLES, encoding="utf-8")
    env = SyntheticEnvironment.from_tle_files({"Kuiper": str(path)}, inject_encounter=True)
    # Newest epoch in the file is day 253.025 of 2026 = 10 Sep 2026 00:35 UTC.
    expected = datetime.datetime(2026, 9, 10, 0, 35, tzinfo=datetime.timezone.utc)
    assert abs((env.sim_time - expected).total_seconds()) < 60.0
    # The injected pair is built at that same epoch, so it meets 30 min in.
    env.seek(30 * 60)
    actors = [c for c in env.snapshot().active_conjunctions if c.object_a.is_actor]
    assert len(actors) == 1
