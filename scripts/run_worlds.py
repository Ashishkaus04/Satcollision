#!/usr/bin/env python3
"""
Start every world, keep them running, and serve the live dashboard.

    python scripts/run_worlds.py                    # all worlds in worlds/, dashboard on :8765
    python scripts/run_worlds.py sandbox adversarial
    python scripts/run_worlds.py --reset            # start every world from scratch
    python scripts/run_worlds.py --port 9000 --no-browser

Then open http://127.0.0.1:8765/ (it opens by itself unless --no-browser).

Each world runs as its own OS process (scripts/run_world.py), so one world
crashing or running slow never stalls another. This launcher is the
supervisor: it restarts a world that exits unexpectedly, backing off
2, 4, 8 ... up to 60 seconds if it keeps failing, and writes
runs/index.json (which worlds exist, whether each is running, how often it
has been restarted, the tail of its log) for the dashboard's world switcher.

Ctrl+C stops everything; each world saves its checkpoint on the way out and
carries on from the same tick next time. "Always running" means: for as
long as this launcher is running. Leave it open in a terminal (or set it to
start at login) and the worlds keep going; close it and they pause, then
resume where they stopped.

The dashboard server binds to 127.0.0.1 only, so nothing is exposed beyond
this computer.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from satcollision.world import WorldConfig, _write_atomic  # noqa: E402

DASHBOARD = "/docs/live_worlds.html"


class _Handler(SimpleHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", DASHBOARD)
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self):
        # The state files change every tick: never let the browser cache them.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):  # quiet: the dashboard polls every few seconds
        pass


class WorldProcess:
    def __init__(self, name: str, config_path: Path, runs: Path, reset: bool):
        self.name = name
        self.config_path = config_path
        self.runs = runs
        self.reset_next = reset
        self.proc: subprocess.Popen | None = None
        self.restarts = 0
        self.started_at = 0.0
        self.backoff = 2.0
        self.retry_at = 0.0
        self.last_exit: int | None = None
        self.log_path = runs / name / "runner.log"

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "scripts" / "run_world.py"),
               "--world", str(self.config_path), "--runs", str(self.runs)]
        if self.reset_next:
            cmd.append("--reset")
            self.reset_next = False
        log = open(self.log_path, "a", encoding="utf-8")
        log.write(f"\n--- started {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z ---\n")
        log.flush()
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"})
        log.close()  # the child holds its own handle
        self.started_at = time.monotonic()

    def check(self, stopping: bool) -> None:
        """Restart the world if it died, with exponential backoff."""
        if self.proc is None or stopping:
            return
        code = self.proc.poll()
        if code is None:
            if time.monotonic() - self.started_at > 60:
                self.backoff = 2.0  # it has been healthy for a while
            return
        if self.retry_at == 0.0:
            self.last_exit = code
            self.retry_at = time.monotonic() + self.backoff
            print(f"[launcher] {self.name} exited with code {code}; restarting in {self.backoff:.0f}s", flush=True)
            self.backoff = min(60.0, self.backoff * 2)
        elif time.monotonic() >= self.retry_at:
            self.retry_at = 0.0
            self.restarts += 1
            self.start()

    def status(self) -> dict:
        running = self.proc is not None and self.proc.poll() is None
        tail = []
        if self.log_path.exists():
            try:
                tail = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-6:]
            except OSError:
                pass
        return {"running": running, "restarts": self.restarts, "last_exit": self.last_exit,
                "pid": self.proc.pid if running else None, "log_tail": tail}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="worlds to run (default: every worlds/*.json)")
    parser.add_argument("--worlds-dir", default=str(ROOT / "worlds"))
    parser.add_argument("--runs", default=str(ROOT / "runs"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reset", action="store_true", help="start every world from scratch")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    runs = Path(args.runs).resolve()
    runs.mkdir(parents=True, exist_ok=True)
    configs = {}
    for path in sorted(Path(args.worlds_dir).glob("*.json")):
        cfg = WorldConfig.load(path)
        if not args.names or cfg.name in args.names:
            configs[cfg.name] = (cfg, path.resolve())
    missing = set(args.names) - set(configs)
    if missing:
        parser.error(f"no such world(s): {', '.join(sorted(missing))}")
    if not configs:
        parser.error(f"no world configs found in {args.worlds_dir}")

    workers = {name: WorldProcess(name, path, runs, args.reset) for name, (cfg, path) in configs.items()}
    for worker in workers.values():
        worker.start()

    handler = functools.partial(_Handler, directory=str(ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}/"
    print(f"[launcher] {len(workers)} world(s) running: {', '.join(workers)}", flush=True)
    print(f"[launcher] live dashboard: {url}   (Ctrl+C to stop; worlds resume next time)", flush=True)
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    stopping = False

    def _stop(signum, frame):  # noqa: ARG001
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    rel_runs = os.path.relpath(runs, ROOT).replace(os.sep, "/")
    try:
        while not stopping:
            for worker in workers.values():
                worker.check(stopping)
            index = {
                "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "runs_path": "/" + rel_runs,
                "worlds": [{"name": name, "title": cfg.title or name, "description": cfg.description,
                            **workers[name].status()} for name, (cfg, _) in configs.items()],
            }
            _write_atomic(runs / "index.json", json.dumps(index))
            time.sleep(1.0)
    finally:
        print("[launcher] stopping worlds...", flush=True)
        for worker in workers.values():
            if worker.proc and worker.proc.poll() is None:
                if os.name == "nt":
                    # Windows children share the console and already got Ctrl+C;
                    # give them a moment to save before forcing them.
                    pass
                else:
                    worker.proc.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 15
        for worker in workers.values():
            if worker.proc:
                try:
                    worker.proc.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    worker.proc.kill()
        server.shutdown()
        print("[launcher] stopped. Run again to resume every world where it left off.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
