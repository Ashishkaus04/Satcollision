"""Replay the two 2021 China Space Station / Starlink events from public element sets.

For as-of times every 6 h from 7 days before the event day, take for each
object the newest element set published *before* the as-of time, propagate
over the event window, and find every approach under 50 km with the refined
screen. Reports the closest predicted approach in the window per as-of time,
with circular Pc (sigma 0.5 km) and Foster x8 Pc, plus the public-tracking
(no cooperation) Pc at that lead time.

Usage: PYTHONPATH=src python research/replay_2021.py data/historical/css_2021.tle
"""
import sys, json, math, datetime as dt, pathlib
import numpy as np
from satcollision.fleets import load_tle_file, tle_epoch_utc, _epoch_from_calendar
from satcollision.screening import screen_conjunctions_refined
from satcollision.twin import compute_pc
from satcollision.pc import FosterPcModel, EncounterGeometry

UTC = dt.timezone.utc
EVENTS = {
    "css-starlink1095-2021": (48274, 44971, dt.datetime(2021, 7, 1, 0, 0, tzinfo=UTC), dt.datetime(2021, 7, 2, 0, 0, tzinfo=UTC)),
    "css-starlink2305-2021": (48274, 47989, dt.datetime(2021, 10, 21, 0, 0, tzinfo=UTC), dt.datetime(2021, 10, 22, 0, 0, tzinfo=UTC)),
}
H = 72.0


def sig_pub(lead_h, far=3.0, near=0.5):
    return near + min(max(lead_h / H, 0), 1) * (far - near)


objs = load_tle_file(sys.argv[1], operator_name="hist")
hist = {}
for o in objs:
    hist.setdefault(o.norad_id, []).append(o)
for v in hist.values():
    v.sort(key=tle_epoch_utc)
foster = FosterPcModel(along_track_ratio=8.0)
report = {}
for key, (na, nb, w0, w1) in EVENTS.items():
    rows = []
    as_of = w0 - dt.timedelta(days=7)
    while as_of <= w1:
        a = [o for o in hist[na] if tle_epoch_utc(o) <= as_of]
        b = [o for o in hist[nb] if tle_epoch_utc(o) <= as_of]
        if a and b:
            oa, ob = a[-1], b[-1]
            start = max(as_of, w0 - dt.timedelta(hours=12))
            _, jd0, fr0 = _epoch_from_calendar(start.year, start.month, start.day, start.hour + start.minute / 60)
            dur = (w1 + dt.timedelta(hours=12) - start).total_seconds()
            cs = screen_conjunctions_refined([oa, ob], jd0, fr0, dur, step_s=30.0, screening_distance_km=50.0)
            if cs:
                c = min(cs, key=lambda c: c.miss_distance_km)
                tca = start + dt.timedelta(seconds=c.tca_offset_s)
                lead_h = (tca - as_of).total_seconds() / 3600
                g = EncounterGeometry.from_conjunction(c)
                rows.append({"as_of": as_of.isoformat(), "lead_h": round(lead_h, 1),
                             "age_a_h": round((as_of - tle_epoch_utc(oa)).total_seconds() / 3600, 1),
                             "age_b_h": round((as_of - tle_epoch_utc(ob)).total_seconds() / 3600, 1),
                             "pred_tca": tca.isoformat(timespec="seconds"), "miss_km": round(c.miss_distance_km, 3),
                             "rel_speed": round(c.relative_speed_km_s, 2), "n_approaches_50km": len(cs),
                             "pc_circ_full": compute_pc(c.miss_distance_km, 0.5),
                             "pc_circ_nocoop": compute_pc(c.miss_distance_km, sig_pub(lead_h)),
                             "pc_foster8_full": foster.pc(g, 0.5),
                             "pc_foster8_nocoop": foster.pc(g, sig_pub(lead_h))})
            else:
                rows.append({"as_of": as_of.isoformat(), "note": "no approach under 50 km in window"})
        as_of += dt.timedelta(hours=6)
    report[key] = rows
    print(key)
    for r in rows:
        if "note" in r:
            print("  ", r["as_of"][:16], r["note"])
        else:
            print("  ", r["as_of"][:16], f"lead {r['lead_h']:6.1f}h  TLE age {r['age_a_h']:5.1f}/{r['age_b_h']:5.1f}h  pred {r['pred_tca'][:19]}  miss {r['miss_km']:7.3f} km  "
                  f"Pc circ {r['pc_circ_full']:.1e}/{r['pc_circ_nocoop']:.1e}  F8 {r['pc_foster8_full']:.1e}/{r['pc_foster8_nocoop']:.1e}")
out = pathlib.Path("research/results/replay_2021.json")
out.write_text(json.dumps(report, indent=1, default=float))
print(out)
