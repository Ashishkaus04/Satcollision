#!/usr/bin/env python3
r"""
Historical-backtest runner: same screening/abstraction/federation pipeline
as scripts/run_demo.py, but against a REAL TLE file instead of the
synthetic fleets — satisfying the historical-backtesting requirement in
Section 7 of the project definition.

Usage:
    PYTHONPATH=src python3 scripts/run_backtest.py data/starlink.tle [duration_hours]

duration_hours is optional (default 24).

The TLE file must be a real download from CelesTrak, e.g.:
    https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle
Open that URL in your normal browser (not through any sandboxed/proxied
tool) and use "Save As" -> plain text, or run in a normal PowerShell
window (not one driven by an automation bridge, which may sit behind a
restrictive proxy):

    Invoke-WebRequest -Uri "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle" -OutFile data\starlink.tle

Then check it actually has content before running this script:
    Get-Item data\starlink.tle | Select-Object Length   # should be >0, typically several hundred KB
"""
import sys
import datetime
from pathlib import Path

from satcollision.fleets import load_tle_file, _epoch_from_calendar
from satcollision.twin import propagate_window, screen_conjunctions, DEFAULT_PC_THRESHOLD
from satcollision.signal import abstract_conjunction


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    tle_path = sys.argv[1]
    duration_hours = float(sys.argv[2]) if len(sys.argv) == 3 else 24.0

    p = Path(tle_path)
    if not p.exists() or p.stat().st_size == 0:
        print(f"ERROR: {tle_path} does not exist or is empty.")
        print("See the module docstring at the top of this script for how to fetch a real TLE file.")
        sys.exit(1)

    print(f"Loading real TLE data from {tle_path} ...")
    objects = load_tle_file(tle_path, operator_name="Starlink")
    print(f"  loaded {len(objects)} real tracked objects")

    # Use "now" (UTC) as the propagation epoch — TLE epochs are all close to
    # their download time, so this keeps the propagation window valid.
    now = datetime.datetime.now(datetime.timezone.utc)
    epoch_days, jd0, fr0 = _epoch_from_calendar(now.year, now.month, now.day, now.hour + now.minute / 60.0)

    duration_s = duration_hours * 60 * 60
    step_s = 30.0
    print(f"Propagating {len(objects)} real objects across a {duration_hours:.0f}-hour window "
          f"(step={step_s}s)...")
    prop = propagate_window(objects, jd0, fr0, duration_s=duration_s, step_s=step_s)

    print("Screening for real conjunctions (<=25km)...")
    conjunctions = screen_conjunctions(objects, prop, screening_distance_km=25.0)
    print(f"Found {len(conjunctions)} real conjunction(s) within the screening volume.\n")

    above_threshold = [c for c in conjunctions if c.pc >= DEFAULT_PC_THRESHOLD]
    print(f"{len(above_threshold)} of those cross the Pc threshold ({DEFAULT_PC_THRESHOLD:.0e}):\n")
    for c in above_threshold[:20]:
        print(f"  {c.pair_label:28s} TCA={c.tca_offset_s/60:7.1f}min  "
              f"miss={c.miss_distance_km*1000:9.1f}m  Pc={c.pc:.2e}  geometry={c.geometry_class}")
        sig_a, sig_b = abstract_conjunction(c)
        print(f"      -> abstracted signal: {sig_a}")

    if not above_threshold and conjunctions:
        print("  (none crossed the Pc threshold in this window -- try a longer duration_hours,")
        print("   or a different/larger TLE group.)")


if __name__ == "__main__":
    main()
