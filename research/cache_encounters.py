"""Screen a real catalogue once (refined screen, NORAD-parity split) and cache
every cross-operator approach with its TCA state vectors, so the analyses can
be rerun with different Pc models, thresholds and sigmas without re-screening.

Usage: PYTHONPATH=src python research/cache_encounters.py --tle data/starlink.tle --hours 168 --screen-km 10
"""
import argparse, json, pathlib, time
import numpy as np
from satcollision.fleets import load_tle_file, catalogue_epoch, _epoch_from_calendar, tle_epoch_utc
from satcollision.real_evaluation import assign_two_operators, median_tle_age_days
from satcollision.screening import screen_conjunctions_refined

ap = argparse.ArgumentParser()
ap.add_argument("--tle", default="data/starlink.tle")
ap.add_argument("--name", default="Starlink")
ap.add_argument("--hours", type=float, default=168.0)
ap.add_argument("--step", type=float, default=30.0)
ap.add_argument("--screen-km", type=float, default=10.0)
ap.add_argument("--out", default=None)
a = ap.parse_args()

objs = assign_two_operators(load_tle_file(a.tle, operator_name=a.name))
start = catalogue_epoch(objs)
_, jd0, fr0 = _epoch_from_calendar(start.year, start.month, start.day, start.hour + start.minute / 60.0)
t0 = time.time()
conjs = screen_conjunctions_refined(objs, jd0, fr0, a.hours * 3600.0, step_s=a.step,
                                    screening_distance_km=a.screen_km,
                                    pair_filter=lambda x, y: x.operator != y.operator)
rt = time.time() - t0
arr = lambda f: np.array([f(c) for c in conjs], dtype=float)
out = pathlib.Path(a.out or f"research/cache/{a.name.lower()}_{start:%Y%m%dT%H%M}_{a.hours:g}h_{a.screen_km:g}km.npz")
out.parent.mkdir(parents=True, exist_ok=True)
meta = dict(tle=a.tle, catalog=a.name, n_objects=len(objs), start_utc=start.isoformat(),
            hours=a.hours, step_s=a.step, screen_km=a.screen_km, runtime_s=round(rt, 1),
            median_tle_age_days=round(median_tle_age_days(objs, start), 2),
            ownership="NORAD parity split", screening="refined")
np.savez_compressed(out,
    norad_a=arr(lambda c: c.object_a.norad_id), norad_b=arr(lambda c: c.object_b.norad_id),
    tca_s=arr(lambda c: c.tca_offset_s), miss_km=arr(lambda c: c.miss_distance_km),
    rel_speed=arr(lambda c: c.relative_speed_km_s),
    ra=np.array([c.position_a_km for c in conjs]), rb=np.array([c.position_b_km for c in conjs]),
    va=np.array([c.velocity_a_km_s for c in conjs]), vb=np.array([c.velocity_b_km_s for c in conjs]),
    meta=json.dumps(meta))
print(out, len(conjs), "approaches", f"{rt:.0f}s")
