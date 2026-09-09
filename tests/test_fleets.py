import math

from satcollision.fleets import (
    build_synthetic_fleets, inject_actor_encounter, propagate, _epoch_from_calendar, R_EARTH_KM,
    OPERATOR_SHELLS,
)


def test_build_synthetic_fleets_shapes():
    objs = build_synthetic_fleets()
    operators = {o.operator for o in objs if not o.is_debris}
    assert operators == set(OPERATOR_SHELLS.keys())
    assert any(o.is_debris for o in objs)


def test_propagated_altitude_matches_shell_spec():
    objs = build_synthetic_fleets()
    epoch_days, jd0, fr0 = _epoch_from_calendar(2026, 9, 9, 0.0)
    for operator, shell in OPERATOR_SHELLS.items():
        sample = next(o for o in objs if o.operator == operator)
        pos, _ = propagate(sample, jd0, fr0)
        r = math.sqrt(sum(c * c for c in pos))
        altitude = r - R_EARTH_KM
        assert abs(altitude - shell["altitude_km"]) < 5.0  # small per-satellite jitter only


def test_inject_actor_encounter_produces_genuine_close_approach():
    objs = build_synthetic_fleets()
    actor_a, actor_b = inject_actor_encounter(objs, tca_offset_min=30.0)
    epoch_days, jd0, fr0 = _epoch_from_calendar(2026, 9, 9, 0.0)

    def dist_at(dt_min):
        fr = fr0 + (30.0 + dt_min) / (24 * 60)
        jd = jd0 + math.floor(fr)
        fr = fr - math.floor(fr)
        pa, _ = propagate(actor_a, jd, fr)
        pb, _ = propagate(actor_b, jd, fr)
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(pa, pb)))

    d_before = dist_at(-2.0)
    d_at = dist_at(0.0)
    d_after = dist_at(2.0)

    assert d_at < 0.5  # sub-500m at the engineered TCA
    assert d_at < d_before and d_at < d_after  # genuine local minimum, not a fluke
