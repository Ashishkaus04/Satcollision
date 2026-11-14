"""
Run one world as separate OS processes: one per operator twin, one for
the federation, plus the sky (ground truth) in the calling process.

    group = ProcessWorld.start(config, run_dir, repo_root)
    group.tick()                       # sky, then every twin, then the federation
    group.operator_view("Alpha")       # served by Alpha's own process
    group.federation_view()            # served by the federation's process
    group.close()

Why bother, when ``World`` can play every part in one process? Because in
one process "the server can't see your orbit" is a promise kept by
filtering code. Here it is a property of the system: the federation is a
different program, with its own memory, that never imports a propagator
and is only ever sent what operators chose to sign and send. An operator's
private key exists only inside that operator's process. tests/test_nodes.py
checks both.

Who talks to whom (all HTTP on 127.0.0.1, with a per-start internal token):

    this process (sky + app) --tick--> each twin --signed signals/reports--> federation
    this process ------------settle/round------------------------------------> federation
    federation ---------attest: "which of these encounter ids are yours?"---> each twin
    twin --signed feed request--> federation --alerts cut to its tier--> twin

Lockstep per tick: the sky ticks; the federation is told the time; every
twin ticks to the same tick (screening, deciding, signing, sending); the
federation asks the twins to attest new encounter ids; and when a reporting
period ends it closes the round. Each process checkpoints its own state in
its own folder (``<run_dir>/sky``, ``nodes/<operator>``, ``federation``), so
the group resumes after a restart: a twin that is behind just catches up.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from .node_http import TOKEN_ENV, NodeError, call
from .views import (
    _world_header,
    compare_from_views,
    encounter_path,
    ground_track,
    observer_view,
)
from .world import World, WorldConfig, _iso

STARTUP_TIMEOUT_S = 300.0   # a 10,000-satellite catalogue takes a while to load


class NodeProcess:
    def __init__(self, name: str, args: List[str], env: dict, cwd: Path):
        self.name = name
        self.proc = subprocess.Popen(
            [sys.executable, "-u", *args], cwd=str(cwd), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.url: Optional[str] = None
        self._stderr: List[str] = []
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self):
        for line in self.proc.stderr:
            self._stderr.append(line.rstrip())
            del self._stderr[:-50]

    def wait_ready(self, timeout: float) -> None:
        result: Dict[str, str] = {}

        def read():
            result["line"] = self.proc.stdout.readline()

        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(timeout)
        line = result.get("line", "")
        if not line.startswith("LISTENING "):
            self.stop()
            raise RuntimeError(f"{self.name} did not start: {line.strip() or 'no output'}; "
                               + " | ".join(self._stderr[-8:]))
        parts = line.split()
        self.url = f"http://127.0.0.1:{int(parts[1])}"
        # The node's own pid: on Windows a venv's python.exe is a launcher
        # that starts the real interpreter as a child, so Popen's pid is not it.
        self.node_pid = int(parts[2]) if len(parts) > 2 else self.proc.pid

    @property
    def pid(self) -> int:
        return getattr(self, "node_pid", None) or self.proc.pid

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self, timeout: float = 10.0) -> None:
        if self.proc.poll() is not None:
            return
        try:
            self.proc.stdin.close()          # the node shuts down cleanly on EOF
            self.proc.wait(timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
            self.proc.wait(timeout=timeout)


class ProcessWorld:
    """Same surface the app needs from a world: tick, save, the role views."""

    def __init__(self, sky: World, federation: NodeProcess, twins: Dict[str, NodeProcess], token: str):
        self.sky = sky
        self.federation = federation
        self.twins = twins
        self.token = token

    # ---- start / stop ----

    @classmethod
    def start(cls, config: WorldConfig, run_dir: str | Path, repo_root: str | Path,
              reset: bool = False, world_file: Optional[str | Path] = None) -> "ProcessWorld":
        run_dir, repo_root = Path(run_dir), Path(repo_root).resolve()
        if reset:
            for sub in ("sky", "nodes", "federation"):
                shutil.rmtree(run_dir / sub, ignore_errors=True)
        sky = World.open(config, run_dir / "sky", repo_root, reset=reset, mode="sky")
        token = secrets.token_hex(24)
        env = {**os.environ, TOKEN_ENV: token,
               "PYTHONPATH": os.pathsep.join([str(repo_root / "src"), os.environ.get("PYTHONPATH", "")])}
        if world_file is None:
            world_file = run_dir / "sky" / "world.json"
            import dataclasses
            import json
            world_file.write_text(json.dumps(dataclasses.asdict(config)), encoding="utf-8")
        flag = ["--reset"] if reset else []
        fed = NodeProcess("federation", ["-m", "satcollision.federation_node", "--world", config.name,
                                         "--operators", ",".join(sky.operators),
                                         "--run-dir", str((run_dir / "federation").resolve()), *flag],
                          env, repo_root)
        twins: Dict[str, NodeProcess] = {}
        try:
            fed.wait_ready(30)
            for op in sky.operators:
                twins[op] = NodeProcess(f"{op} twin", [
                    "-m", "satcollision.twin_node", "--world-file", str(Path(world_file).resolve()),
                    "--operator", op, "--run-dir", str((run_dir / "nodes" / op).resolve()),
                    "--repo-root", str(repo_root), "--federation", fed.url, *flag], env, repo_root)
            for twin in twins.values():
                twin.wait_ready(STARTUP_TIMEOUT_S)
        except Exception:
            for p in [fed, *twins.values()]:
                p.stop()
            raise
        group = cls(sky, fed, twins, token)
        group.sync()   # a twin left behind by a crash catches up here
        return group

    def close(self) -> None:
        for p in [*self.twins.values(), self.federation]:
            p.stop()

    # ---- driving ----

    def _call(self, node: NodeProcess, method: str, path: str, body=None, timeout: float = 120.0) -> dict:
        if not node.alive():
            raise RuntimeError(f"{node.name} process has exited: " + " | ".join(node._stderr[-5:]))
        return call(node.url, method, path, body, token=self.token, timeout=timeout)

    def tick(self) -> None:
        started = _iso(self.sky.sim_time)
        self.sky.tick()
        self.sync(started)

    def sync(self, tick_started: Optional[str] = None) -> None:
        target = self.sky.tick_index
        self._call(self.federation, "POST", "/clock",
                   {"tick": target, "sim_time": tick_started or _iso(self.sky.sim_time)})
        # Separate processes, so the twins screen in parallel on separate cores.
        with ThreadPoolExecutor(max_workers=len(self.twins)) as pool:
            list(pool.map(lambda twin: self._call(twin, "POST", "/tick", {"to_tick": target}, timeout=600),
                          self.twins.values()))
        self._call(self.federation, "POST", "/settle", {})
        last = self._call(self.federation, "GET", "/health")["last_round"]
        if self.sky.round_index > last:
            self._call(self.federation, "POST", "/clock", {"tick": target, "sim_time": _iso(self.sky.sim_time)})
            for r in range(last + 1, self.sky.round_index + 1):
                self._call(self.federation, "POST", "/round", {"round": r})

    def save(self, wall_interval_s: Optional[float] = None) -> None:
        self.sky.save(wall_interval_s)

    # ---- views ----

    def processes(self) -> dict:
        return {"app": os.getpid(), "federation": self.federation.pid,
                **{op: t.pid for op, t in self.twins.items()}}

    def operator_view(self, operator: str) -> dict:
        view = self._call(self.twins[operator], "GET", "/view")
        view["world"] = _world_header(self.sky)   # the shared clock, as the app shows it
        return view

    def federation_view(self) -> dict:
        view = self._call(self.federation, "GET", "/view")
        view["world"] = _world_header(self.sky)
        return view

    def observer_view(self) -> dict:
        view = observer_view(self.sky, federation=self.federation_view())
        view["processes"] = self.processes()
        return view

    def compare(self, encounter_id: str) -> dict:
        truth = next((e for e in self.sky.encounters if e["id"] == encounter_id), None)
        if truth is None:
            raise KeyError(encounter_id)
        views = {op: self.operator_view(op) for op in dict.fromkeys(truth["operators"]) if op in self.twins}
        return compare_from_views(self.sky, truth, views, self.federation_view())

    def encounter_path(self, encounter_id: str, operator: Optional[str]) -> dict:
        """The observer's comes from the sky; an operator's from its own
        process, which only ever knows its own burn."""
        if operator is None:
            return encounter_path(self.sky, encounter_id)
        from urllib.parse import quote
        try:
            out = self._call(self.twins[operator], "GET", f"/path?id={quote(encounter_id)}")
        except NodeError as exc:
            if exc.status == 404:
                raise KeyError(encounter_id)
            raise
        out["world"] = _world_header(self.sky)
        return out

    def track(self, norad_id: int, minutes: int, operator: Optional[str]) -> dict:
        """``operator=None`` is the observer (ground truth, from the sky);
        an operator's request goes to its own process, which refuses
        anything it doesn't own."""
        if operator is None:
            return ground_track(self.sky, norad_id, minutes)
        try:
            return self._call(self.twins[operator], "GET", f"/track?id={int(norad_id)}&minutes={int(minutes)}")
        except NodeError as exc:
            if exc.status == 403:
                raise PermissionError(exc.message)
            if exc.status == 404:
                raise KeyError(norad_id)
            raise
