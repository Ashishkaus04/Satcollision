"""A small synthetic gp_history file for the history-world tests.

Four days of element sets, one every six hours per object: three Alpha and
two Beta satellites, one non-member object, and one Beta satellite built to
cross Alpha's first satellite an hour after the world starts. Each set is
the object's base orbit moved to the set's epoch with SGP4's own secular
rates (bstar 0, so the sets agree to round-off), plus a deterministic
along-track error of about 100 m. ALPHA-3's sets jump 30 km along-track
from day 2.5 on: a "manoeuvre" for the jump detector.
"""

import math
import random
from datetime import datetime, timedelta, timezone

from sgp4.api import WGS72, Satrec
from sgp4.exporter import export_tle

from satcollision.fleets import TrackedObject, _make_satrec, inject_actor_encounter

BASE = datetime(2026, 11, 7, 0, 0, tzinfo=timezone.utc)     # the world starts here
FIRST = BASE - timedelta(days=1)                              # first element set
SET_EVERY_H = 6.0
N_SETS = 17                                                   # four days
TCA_AFTER_S = 3600.0
FAST_TCA_AFTER_S = 7215.0      # halfway between two 30 s samples
JD_1950 = 2433281.5


def _epoch_days(when: datetime) -> float:
    return (when - datetime(1949, 12, 31, tzinfo=timezone.utc)).total_seconds() / 86400.0


def _re_epoch(base: Satrec, satnum: int, when: datetime, d_mean_anomaly: float = 0.0) -> Satrec:
    dt_min = (_epoch_days(when) - (base.jdsatepoch + base.jdsatepochF - JD_1950)) * 1440.0
    s = Satrec()
    s.sgp4init(WGS72, "i", satnum, _epoch_days(when), 0.0, 0.0, 0.0, base.ecco,
               (base.argpo + base.argpdot * dt_min) % (2 * math.pi), base.inclo,
               (base.mo + base.mdot * dt_min + d_mean_anomaly) % (2 * math.pi), base.no_kozai,
               (base.nodeo + base.nodedot * dt_min) % (2 * math.pi))
    return s


def base_objects():
    ep = _epoch_days(BASE)
    sats = [
        ("ALPHA-1", 90001, 550.0, 53.0, 0.0, 10.0),
        ("ALPHA-2", 90002, 550.0, 53.0, 120.0, 40.0),
        ("ALPHA-3", 90003, 560.0, 53.0, 240.0, 80.0),
        ("BETA-2", 90012, 600.0, 97.5, 60.0, 200.0),
        ("DEBRIS 1", 90020, 700.0, 82.0, 300.0, 150.0),
    ]
    out = []
    for name, sid, alt, inc, raan, ma in sats:
        s = _make_satrec(sid, alt, inc, raan, ma, 0.0001, ep, bstar=0.0)
        out.append((name, TrackedObject(norad_id=sid, operator=name.split("-")[0].split(" ")[0], satrec=s)))
    alpha1 = out[0][1]
    probe = TrackedObject(norad_id=alpha1.norad_id, operator="ALPHA", satrec=alpha1.satrec)
    _, b = inject_actor_encounter([probe], epoch=(BASE.year, BASE.month, BASE.day, 0.0),
                                  watch_operator_a="ALPHA", new_operator_b="BETA",
                                  tca_offset_min=TCA_AFTER_S / 60.0, crossing_angle_deg=40.0)
    b_sat = _re_epoch(b.satrec, 90011, BASE)   # same orbit, its own number, no drag
    out.append(("BETA-1", TrackedObject(norad_id=90011, operator="BETA", satrec=b_sat)))
    # A fast crossing of ALPHA-2 between screening samples: at
    # 8.5 km/s the pair is ~127 km apart at both neighbouring samples.
    alpha2 = out[1][1]
    probe = TrackedObject(norad_id=alpha2.norad_id, operator="ALPHA", satrec=alpha2.satrec)
    _, c = inject_actor_encounter([probe], epoch=(BASE.year, BASE.month, BASE.day, 0.0),
                                  watch_operator_a="ALPHA", new_operator_b="BETA",
                                  tca_offset_min=FAST_TCA_AFTER_S / 60.0, crossing_angle_deg=170.0)
    out.append(("BETA-3", TrackedObject(norad_id=90013, operator="BETA", satrec=_re_epoch(c.satrec, 90013, BASE))))
    return out


def write_history(path, duplicate_last=True, along_track_noise_km=0.1, seed=1) -> str:
    rng = random.Random(seed)
    lines = []
    for name, obj in base_objects():
        a_km = (398600.4418 / (obj.satrec.no_kozai / 60.0) ** 2) ** (1 / 3)
        for k in range(N_SETS):
            when = FIRST + timedelta(hours=SET_EVERY_H * k)
            dm = rng.gauss(0.0, along_track_noise_km) / a_km
            if name == "ALPHA-3" and k >= 14:
                dm += 30.0 / a_km
            l1, l2 = export_tle(_re_epoch(obj.satrec, obj.norad_id, when, dm))
            lines += [f"0 {name}", l1, l2]
    if duplicate_last:
        lines += lines[-3:]          # Space-Track sometimes re-issues a set
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return str(path)


def config(path, **over):
    cfg = {
        "name": "hist_fixture",
        "fleet": {"type": "history", "file": str(path), "warmup_h": 24.0,
                  "operators": {"Alpha": ["ALPHA"], "Beta": ["BETA"]},
                  "fit_objects_per_group": 50},
        "speed": 60, "tick_sim_s": 120, "screening_step_s": 30, "lookahead_s": 1800,
        "pc_model": "foster", "federation_period_s": 3600, "seed": 3,
    }
    for k, v in over.items():
        if k == "fleet":
            cfg["fleet"] = {**cfg["fleet"], **v}
        else:
            cfg[k] = v
    return cfg
