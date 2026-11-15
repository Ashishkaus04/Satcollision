"""Download and convert ESA's precise orbits of Sentinel-1C and -1D.

    python research/precise_orbits.py --start 2026-10-12 --end 2026-11-13

Sentinel-1 orbit files are public on ESA's SNAP auxiliary-data server
(step.esa.int, no login; the ASF mirror lists them but needs an Earthdata login
to download): precise POEORB
(a few cm, published about 20 days late, each file 26 h) and restituted
RESORB (about 10 cm, available within hours, each file ~3.5 h). We take
POEORB where it exists and RESORB after that. Each file lists state vectors
every 10 s in the Earth-fixed ITRF frame; we keep one a minute and rotate
them into TEME (the frame SGP4 works in) with the IAU-82 sidereal time,
ignoring polar motion and the equation of the equinoxes (tens of metres,
negligible against TLE errors of hundreds of metres to kilometres).

Writes data/precise/sentinel1_<start>_<end>.npz (git-ignored, like all of
data/): norad, jd (UTC Julian date), r_teme_km, v_teme_km_s, source.
"""
import argparse
import math
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://step.esa.int/auxdata/orbits/Sentinel-1/"
SATS = {"S1C": 62261, "S1D": 66315}       # NORAD ids (checked against the TLE history)
UA = {"User-Agent": "satcollision-research/1.0"}


def get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def listing(kind: str, months) -> list:
    pat = rf"(S1[A-D])_OPER_AUX_{kind.upper()}_OPOD_(\d{{8}}T\d{{6}})_V(\d{{8}}T\d{{6}})_(\d{{8}}T\d{{6}})\.EOF\.zip"
    out = []
    f = lambda s: datetime.strptime(s, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    for sat in SATS:
        for (y, mth) in months:
            url = BASE + f"{kind.upper()}/{sat}/{y}/{mth:02d}/"
            try:
                html = get(url).decode("ascii", "replace")
            except Exception:
                continue
            for m in set(re.finditer(pat, html)):
                out.append({"sat": m.group(1), "made": f(m.group(2)), "v0": f(m.group(3)), "v1": f(m.group(4)),
                            "name": m.group(0), "kind": kind, "url": url + m.group(0)})
    return out


def choose(files: list, start: datetime, end: datetime) -> list:
    """For each satellite, cover [start, end) with POEORB first, then RESORB
    for whatever POEORB does not cover; among files with the same validity
    start keep the newest production."""
    chosen = []
    for sat in SATS:
        covered_until = start
        for kind in ("poeorb", "resorb"):
            cand = {}
            for f in files:
                if f["sat"] == sat and f["kind"] == kind and f["v1"] > covered_until and f["v0"] < end:
                    k = f["v0"]
                    if k not in cand or f["made"] > cand[k]["made"]:
                        cand[k] = f
            for f in sorted(cand.values(), key=lambda f: f["v0"]):
                if f["v1"] <= covered_until:
                    continue
                chosen.append(f)
            if kind == "poeorb" and chosen:
                covered_until = max([covered_until] + [c["v1"] for c in chosen if c["sat"] == sat])
    return chosen


def parse_eof(data: bytes):
    t, r, v = [], [], []
    txt = data.decode("ascii", "replace")
    for block in re.findall(r"<OSV>(.*?)</OSV>", txt, re.S):
        utc = re.search(r"<UTC>UTC=([^<]+)</UTC>", block).group(1)
        vals = [float(re.search(rf"<{k}[^>]*>([^<]+)</{k}>", block).group(1)) for k in ("X", "Y", "Z", "VX", "VY", "VZ")]
        t.append(datetime.strptime(utc[:26], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc))
        r.append(vals[:3]); v.append(vals[3:])
    return t, np.array(r) / 1000.0, np.array(v) / 1000.0


def jd_utc(t: datetime) -> float:
    return t.timestamp() / 86400.0 + 2440587.5


def gmst82(jd_ut1: np.ndarray) -> np.ndarray:
    """IAU-82 Greenwich mean sidereal time (rad), the angle SGP4's TEME uses."""
    tut1 = (jd_ut1 - 2451545.0) / 36525.0
    s = (-6.2e-6 * tut1 ** 3 + 0.093104 * tut1 ** 2 + (876600.0 * 3600 + 8640184.812866) * tut1 + 67310.54841)
    return np.mod(s * (2 * math.pi / 86400.0), 2 * math.pi)


def itrf_to_teme(jd: np.ndarray, r: np.ndarray, v: np.ndarray):
    th = gmst82(jd)
    c, s = np.cos(th), np.sin(th)
    rt = np.stack([c * r[:, 0] - s * r[:, 1], s * r[:, 0] + c * r[:, 1], r[:, 2]], 1)
    omega = 7.292115146706979e-5
    vf = v + np.cross(np.array([0.0, 0.0, omega]), r)      # inertial velocity in the Earth-fixed axes
    vt = np.stack([c * vf[:, 0] - s * vf[:, 1], s * vf[:, 0] + c * vf[:, 1], vf[:, 2]], 1)
    return rt, vt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-10-12")
    ap.add_argument("--end", default="2026-11-13")
    ap.add_argument("--step-s", type=float, default=60.0)
    a = ap.parse_args()
    start = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc)
    months = sorted({(d.year, d.month) for d in (start, end)} | {(start.year, start.month + 1) if start.month < 12 else (start.year + 1, 1)})
    files = listing("poeorb", months) + listing("resorb", months)
    chosen = choose(files, start, end)
    print(f"{len(chosen)} files: " + ", ".join(f"{s} {sum(1 for f in chosen if f['sat'] == s)}" for s in SATS))
    cache = ROOT / "cache" / "precise_eof"
    cache.mkdir(parents=True, exist_ok=True)
    rows = {s: {} for s in SATS}
    for i, f in enumerate(chosen):
        p = cache / f["name"]
        if not p.exists():
            p.write_bytes(get(f["url"]))
            time.sleep(0.3)
        import io, zipfile
        with zipfile.ZipFile(p) as z:
            t, r, v = parse_eof(z.read(z.namelist()[0]))
        for tt, rr, vv in zip(t, r, v):
            if not (start <= tt < end):
                continue
            sec = round(tt.timestamp())
            if sec % int(a.step_s) >= 10:      # vectors every 10 s: keep the first in each step
                continue
            bucket = sec - sec % int(a.step_s)
            # POEORB wins over RESORB at the same instant (chosen first)
            rows[f["sat"]].setdefault(bucket, (tt.timestamp(), rr, vv, f["kind"]))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(chosen)}", flush=True)
    norad, jd, R, V, src = [], [], [], [], []
    for s, d in rows.items():
        for sec in sorted(d):
            ts, rr, vv, kind = d[sec]          # the vector's own time, not the bucket
            norad.append(SATS[s]); jd.append(ts / 86400.0 + 2440587.5); R.append(rr); V.append(vv)
            src.append(0 if kind == "poeorb" else 1)
    jd = np.array(jd); R = np.array(R); V = np.array(V)
    Rt, Vt = itrf_to_teme(jd, R, V)
    out = ROOT / "data" / "precise" / f"sentinel1_{a.start}_{a.end}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, norad=np.array(norad), jd=jd, r_teme_km=Rt, v_teme_km_s=Vt, source=np.array(src))
    for s in SATS:
        n = sum(1 for x in norad if x == SATS[s])
        print(f"{s}: {n} states ({n * a.step_s / 86400:.1f} days)")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    sys.exit(main())
