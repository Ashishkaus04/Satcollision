"""More precise-orbit ground truth: ESA Swarm A/B/C and FORMOSAT-7/COSMIC-2.

    python research/precise_orbits_more.py --start 2026-10-12 --end 2026-11-13

Both are public without a login:
* Swarm (ESA): Level 1b MODx_SC daily SP3 orbits (GPS-based, 1 s), from the
  Swarm dissemination server. Altitudes about 460 km (A, C) and 510 km (B):
  much lower and more drag-affected than Sentinel-1 (693 km).
* COSMIC-2 (UCAR CDAAC): near-real-time LEO orbit arcs (leoOrb, SP3, 60 s,
  about 2 h each) for the four satellites still operating (PRN L76, L77,
  L80, L81 = FM1, FM2, FM5, FM6), about 550 km, 24 deg inclination.

SP3 epochs are GPS time; GPS - UTC = 18 s. Positions are Earth-fixed (IGS08
/ ITRF) and rotated to TEME exactly as precise_orbits.py does. Kept at 60 s.
Writes data/precise/<mission>_<start>_<end>.npz (same layout as the
Sentinel-1 file: norad, jd, r_teme_km, v_teme_km_s, source).
"""
import argparse
import gzip
import io
import re
import sys
import tarfile
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precise_orbits import itrf_to_teme  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "satcollision-research/1.0"}
GPS_UTC_S = 18.0
SWARM = {"A": 39452, "B": 39451, "C": 39453}
COSMIC2 = {"L76": 44349, "L77": 44351, "L80": 44358, "L81": 44353}   # FM1, FM2, FM5, FM6
SWARM_LIST = "https://swarm-diss.eo.esa.int/?do=list&maxfiles=100000&pos=0&file=swarm%2FLevel1b%2FLatest_baselines%2FMODx_SC%2FSat_{}"
SWARM_GET = "https://swarm-diss.eo.esa.int/?do=download&file=swarm%2FLevel1b%2FLatest_baselines%2FMODx_SC%2FSat_{}%2F{}"
C2_GET = "https://data.cosmic.ucar.edu/gnss-ro/cosmic2/nrt/level1b/{y}/{doy:03d}/leoOrb_nrt_{y}_{doy:03d}.tar.gz"


def get(url, timeout=300):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def parse_sp3(text: str, step_s: int = 60):
    """{prn: [(unix_utc, r_km(3), v_km_s(3))]} at whole multiples of step_s (GPS time)."""
    out = {}
    t = None
    keep = False
    pending = {}
    for line in text.splitlines():
        if line.startswith("*"):
            f = line[1:].split()
            sec = float(f[5])
            tg = datetime(int(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4]), tzinfo=timezone.utc) + timedelta(seconds=sec)
            t = tg.timestamp()
            keep = abs((t + 1e-6) % step_s) < 1e-3 or abs(t % step_s - step_s) < 1e-3
        elif keep and line.startswith("P"):
            prn = line[1:4].strip()
            pending[prn] = [float(line[4:18]), float(line[18:32]), float(line[32:46])]
        elif keep and line.startswith("V"):
            prn = line[1:4].strip()
            if prn in pending:
                v = [float(line[4:18]) / 1e4, float(line[18:32]) / 1e4, float(line[32:46]) / 1e4]  # dm/s -> km/s
                out.setdefault(prn, []).append((t - GPS_UTC_S, pending.pop(prn), v))
    return out


def save(name, rows, start, end, tag):
    norad, jd, R, V = [], [], [], []
    for nid, lst in rows.items():
        seen = {}
        for t, r, v in lst:
            seen.setdefault(round(t), (t, r, v))
        for k in sorted(seen):
            t, r, v = seen[k]
            norad.append(nid); jd.append(t / 86400.0 + 2440587.5); R.append(r); V.append(v)
    jd = np.array(jd); R = np.array(R); V = np.array(V)
    Rt, Vt = itrf_to_teme(jd, R, V)
    out = ROOT / "data" / "precise" / f"{name}_{start:%Y-%m-%d}_{end:%Y-%m-%d}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, norad=np.array(norad), jd=jd, r_teme_km=Rt, v_teme_km_s=Vt,
                        source=np.full(len(jd), tag))
    for nid in rows:
        n = sum(1 for x in norad if x == nid)
        print(f"  {name} {nid}: {n} states ({n / 1440:.1f} days)")
    print("wrote", out.relative_to(ROOT))


def swarm(start, end):
    cache = ROOT / "cache" / "precise_swarm"
    cache.mkdir(parents=True, exist_ok=True)
    rows = {}
    for sat, nid in SWARM.items():
        names = sorted(set(re.findall(rf"SW_OPER_MOD{sat}_SC_1B_(\d{{8}})T\d{{6}}_\d{{8}}T\d{{6}}_\d{{4}}\.ZIP",
                                      get(SWARM_LIST.format(sat)).decode())))
        listing = get(SWARM_LIST.format(sat)).decode()
        for day in names:
            d = datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc)
            if not (start - timedelta(days=1) <= d < end):
                continue
            fn = sorted(re.findall(rf"SW_OPER_MOD{sat}_SC_1B_{day}T\d{{6}}_\d{{8}}T\d{{6}}_\d{{4}}\.ZIP", listing))[-1]
            p = cache / fn
            if not p.exists():
                p.write_bytes(get(SWARM_GET.format(sat, fn)))
                time.sleep(0.5)
            with zipfile.ZipFile(p) as z:
                sp3 = [n for n in z.namelist() if n.endswith(".sp3")][0]
                parsed = parse_sp3(z.read(sp3).decode("ascii", "replace"))
            for prn, lst in parsed.items():
                rows.setdefault(nid, []).extend(x for x in lst if start.timestamp() <= x[0] < end.timestamp())
        print(f"Swarm {sat}: {len(rows.get(nid, []))} states", flush=True)
    save("swarm", rows, start, end, 2)


def cosmic2(start, end):
    cache = ROOT / "cache" / "precise_cosmic2"
    cache.mkdir(parents=True, exist_ok=True)
    rows = {}
    d = start - timedelta(days=1)
    while d < end:
        y, doy = d.year, d.timetuple().tm_yday
        p = cache / f"leoOrb_nrt_{y}_{doy:03d}.tar.gz"
        if not p.exists():
            try:
                p.write_bytes(get(C2_GET.format(y=y, doy=doy)))
            except Exception as e:
                print("  missing", p.name, e)
                d += timedelta(days=1)
                continue
            time.sleep(0.5)
        with tarfile.open(p) as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                parsed = parse_sp3(tf.extractfile(m).read().decode("ascii", "replace"))
                for prn, lst in parsed.items():
                    if prn in COSMIC2:
                        rows.setdefault(COSMIC2[prn], []).extend(
                            x for x in lst if start.timestamp() <= x[0] < end.timestamp())
        d += timedelta(days=1)
    save("cosmic2", rows, start, end, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-10-12")
    ap.add_argument("--end", default="2026-11-13")
    ap.add_argument("--only", default="swarm,cosmic2")
    a = ap.parse_args()
    s = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc)
    if "cosmic2" in a.only:
        cosmic2(s, e)
    if "swarm" in a.only:
        swarm(s, e)


if __name__ == "__main__":
    main()
