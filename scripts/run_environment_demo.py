"""
Demo / usage script for the SyntheticEnvironment.

Run from the project root with:
    PYTHONPATH=src python scripts/run_environment_demo.py

This script exercises every public API of SyntheticEnvironment without
needing Jupyter: it creates an environment from the real TLE files in
data/ (falling back to synthetic fleets if the files don't exist), steps
through a few simulation ticks, and prints the rich display at each step.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Path setup (allows running as `python scripts/run_environment_demo.py`
# from the project root without installing the package)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from satcollision.environment import SyntheticEnvironment


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    starlink_tle = os.path.join(data_dir, "starlink.tle")
    kuiper_tle = os.path.join(data_dir, "kuiper.tle")
    oneweb_tle = os.path.join(data_dir, "oneweb.tle")

    # -----------------------------------------------------------------------
    # Build the environment
    # -----------------------------------------------------------------------
    if os.path.exists(starlink_tle):
        tle_files = {"Starlink": starlink_tle}
        if os.path.exists(kuiper_tle):
            tle_files["Kuiper"] = kuiper_tle
        if os.path.exists(oneweb_tle):
            tle_files["OneWeb"] = oneweb_tle

        env = SyntheticEnvironment.from_tle_files(
            tle_files=tle_files,
            inject_encounter=True,   # always guarantee one conjunction exists
            step_size_s=300.0,       # 5-minute steps
            screening_window_s=3600.0,
            screening_step_s=30.0,
        )
    else:
        print("[demo] Real TLE files not found in data/ — using synthetic fleets.")
        env = SyntheticEnvironment.from_synthetic_fleets(
            inject_encounter=True,
            step_size_s=300.0,
        )

    print(f"\n{repr(env)}\n")

    # -----------------------------------------------------------------------
    # Take an initial snapshot without advancing time
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("  INITIAL SNAPSHOT  (t = 0, clock not advanced)")
    print("=" * 60)
    snap0 = env.snapshot()
    print(snap0.summary())

    # -----------------------------------------------------------------------
    # Step through 3 ticks and display after each
    # -----------------------------------------------------------------------
    for tick in range(1, 4):
        snap = env.step()
        print(f"\n{'=' * 60}")
        print(f"  STEP {tick}  (+{env._step_size_s:.0f}s each)")
        print(f"{'=' * 60}")
        print(snap.summary())

    # -----------------------------------------------------------------------
    # Data extraction examples
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  DATA EXTRACTION EXAMPLES")
    print("=" * 60)

    snap = env.snapshot()

    # 1. Raw positions numpy array
    positions, norad_ids = env.positions_array()
    print(f"\npositions_array() shape: {positions.shape},  first row: {positions[0].round(2)} km")

    # 2. Velocities numpy array
    velocities, _ = env.velocities_array()
    print(f"velocities_array() shape: {velocities.shape},  first row: {velocities[0].round(4)} km/s")

    # 3. Position dict
    pos_dict = snap.positions
    first_id = norad_ids[0]
    print(f"snap.positions[{first_id}]  pos={pos_dict[first_id][0]}")

    # 4. Filter by operator
    operators = snap.operator_names
    if operators:
        op = operators[0]
        op_sats = snap.get_operator_satellites(op)
        print(f"\nOperator '{op}' has {len(op_sats)} satellites in this snapshot.")
        if op_sats:
            s = op_sats[0]
            print(f"  Example: {s.label}  alt={s.altitude_km:.1f} km  speed={s.speed_km_s:.3f} km/s")

    # 5. Active conjunctions
    active = snap.active_conjunctions
    print(f"\nActive conjunctions (Pc >= 1e-4): {len(active)}")
    for c in active:
        print(f"  {c.pair_label:50s}  d={c.miss_distance_km:.3f} km  Pc={c.pc:.2e}")

    # 6. Seek to a future time without stepping
    print("\nSeeking to +2h from epoch...")
    env.seek(7200.0)
    snap_future = env.snapshot()
    print(f"  sim_time after seek: {snap_future.sim_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  conjunctions found : {len(snap_future.conjunctions)}")

    # 7. Conjunctions as DataFrame (if pandas available)
    try:
        df = env.conjunctions_dataframe()
        if hasattr(df, "to_string"):
            print(f"\nconjunctions_dataframe() -> {len(df)} rows, columns: {list(df.columns)}")
            print(df.head(3).to_string(index=False))
    except Exception as exc:
        print(f"\n[demo] conjunctions_dataframe(): {exc}")

    print("\n[demo] Done.\n")


if __name__ == "__main__":
    main()
