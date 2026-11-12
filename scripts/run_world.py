#!/usr/bin/env python3
"""
Run one always-running world until stopped.

    python scripts/run_world.py --world worlds/sandbox.json
    python scripts/run_world.py --world worlds/sandbox.json --reset        # start over
    python scripts/run_world.py --world worlds/sandbox.json --ticks 500 --fast   # batch, no pacing

The world resumes from runs/<name>/checkpoint.json if it exists. It ticks
every ``tick_sim_s / speed`` wall seconds; if a tick takes longer than that,
the next one starts straight away and the world simply runs behind (the
dashboard shows the lag). Ctrl+C (or the launcher stopping it) saves and
exits; a hard kill loses at most the tick in progress, because every tick
ends with a checkpoint.

Normally started by scripts/run_worlds.py, which runs all worlds at once
and serves the live dashboard.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from satcollision.world import World, WorldConfig  # noqa: E402

_stop = False


def _request_stop(signum, frame):  # noqa: ARG001
    global _stop
    _stop = True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world", required=True, help="path to a worlds/<name>.json config")
    parser.add_argument("--runs", default=str(ROOT / "runs"), help="directory holding every world's run folder")
    parser.add_argument("--reset", action="store_true", help="discard the saved run and start from scratch")
    parser.add_argument("--ticks", type=int, default=0, help="stop after this many ticks (0 = run forever)")
    parser.add_argument("--fast", action="store_true", help="don't wait between ticks")
    parser.add_argument("--speed", type=float, default=None, help="override the config's speed")
    args = parser.parse_args(argv)

    cfg = WorldConfig.load(args.world)
    if args.speed:
        cfg.speed = args.speed
    world = World.open(cfg, Path(args.runs) / cfg.name, repo_root=ROOT, reset=args.reset)
    interval = cfg.tick_sim_s / cfg.speed

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    if hasattr(signal, "SIGBREAK"):  # Windows: Ctrl+Break and console close
        signal.signal(signal.SIGBREAK, _request_stop)

    print(f"[{cfg.name}] {len(world._objects):,} objects, tick every {interval:.1f}s "
          f"({cfg.tick_sim_s:.0f}s simulated, {cfg.speed:g}x), resuming at tick {world.tick_index}, "
          f"ledger {'intact' if world.ledger_verified else 'TAMPERED'}", flush=True)
    world.save(interval)

    done = 0
    next_at = time.monotonic()
    while not _stop:
        world.tick()
        world.save(interval)
        done += 1
        if done % 50 == 0:
            c = world.counters
            print(f"[{cfg.name}] tick {world.tick_index} {world.sim_time:%Y-%m-%d %H:%M}Z "
                  f"shared={c['encounters_shared']} plans={c['maneuvers_agreed']} "
                  f"rounds={c['federation_rounds']} ({world.last_tick_wall_s:.2f}s/tick)", flush=True)
        if args.ticks and done >= args.ticks:
            break
        if args.fast:
            continue
        next_at += interval
        delay = next_at - time.monotonic()
        if delay < -5 * interval:
            next_at = time.monotonic()  # far behind: don't try to catch up in a burst
        while delay > 0 and not _stop:
            time.sleep(min(delay, 0.5))
            delay = next_at - time.monotonic()

    world.save(interval)
    print(f"[{cfg.name}] stopped at tick {world.tick_index} ({world.sim_time:%Y-%m-%d %H:%M}Z)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
