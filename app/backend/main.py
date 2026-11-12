"""
Satcollision full-stack app: backend.

    python -m app.backend                 # from the repo root; open http://127.0.0.1:8000
    APP_HOST=0.0.0.0 APP_SECRET=... APP_USERS_FILE=users.json python -m app.backend   # for a server

One FastAPI process runs every world in ``worlds/`` on its own background
thread (the same ``World`` class scripts/run_worlds.py uses, so a world's
history in ``runs/`` carries over between the two), and serves:

* a JSON + WebSocket API under ``/api`` whose every response is cut down to
  what the logged-in party may see (``satcollision.views``), and
* the built React frontend (``app/frontend/dist``) at ``/``.

Roles: an **operator** login (alpha, beta, gamma, starlink, ...) sees its
own digital twin; **federation** sees what the central server holds;
**observer** sees simulation ground truth and the side-by-side comparison.
The filtering is server-side: an operator token asking for another
operator's view, or for the observer endpoints, gets a 403, not a hidden
panel (tests/test_app.py checks this).

Configuration is by environment variable, so the same code runs on a laptop
and on a server:

==================  =========================================================
APP_HOST            bind address (default 127.0.0.1: this computer only)
APP_PORT            default 8000
APP_SECRET          token signing key (default: random per start, so tokens
                    die on restart; set it for a deployment)
APP_USERS_FILE      JSON of users (default: demo users, password "demo")
APP_RUNS_DIR        where worlds keep their state (default runs/)
APP_WORLDS          comma-separated world names (default: all in worlds/)
APP_SPEED           multiply every world's speed (default 1)
APP_RESET           "1" to start every world from scratch
APP_PROCESSES       "1" (default): every operator's twin and the federation
                    run as separate OS processes (satcollision.processes);
                    "0": one process plays every part (faster, same results)
==================  =========================================================

Do not run this and scripts/run_worlds.py on the same runs/ folder at the
same time: both would advance the same worlds.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import secrets
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jwt  # noqa: E402
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from satcollision.processes import ProcessWorld  # noqa: E402
from satcollision.views import (  # noqa: E402
    compare_encounter,
    federation_view,
    ground_track,
    observer_view,
    operator_view,
)
from satcollision.world import World, WorldConfig, run_is_stale  # noqa: E402

log = logging.getLogger("satcollision.app")

# ---------------------------------------------------------------------------
# Settings and users
# ---------------------------------------------------------------------------

DEMO_USERS = {
    "alpha": {"role": "operator", "operator": "Alpha"},
    "beta": {"role": "operator", "operator": "Beta"},
    "gamma": {"role": "operator", "operator": "Gamma"},
    "starlink": {"role": "operator", "operator": "Starlink"},
    "kuiper": {"role": "operator", "operator": "Kuiper"},
    "oneweb": {"role": "operator", "operator": "OneWeb"},
    "planet": {"role": "operator", "operator": "Planet"},
    "spire": {"role": "operator", "operator": "Spire"},
    "iceye": {"role": "operator", "operator": "ICEYE"},
    "federation": {"role": "federation"},
    "observer": {"role": "observer"},
}
DEMO_PASSWORD = "demo"
TOKEN_HOURS = 12


class Settings:
    def __init__(self, **overrides):
        env = {**os.environ, **{k: str(v) for k, v in overrides.items()}}
        self.host = env.get("APP_HOST", "127.0.0.1")
        self.port = int(env.get("APP_PORT", "8000"))
        self.secret = env.get("APP_SECRET") or secrets.token_hex(32)
        self.runs_dir = Path(env.get("APP_RUNS_DIR", str(ROOT / "runs")))
        self.worlds_dir = Path(env.get("APP_WORLDS_DIR", str(ROOT / "worlds")))
        self.world_names = [w for w in env.get("APP_WORLDS", "").split(",") if w]
        self.speed = float(env.get("APP_SPEED", "1"))
        self.reset = env.get("APP_RESET", "") == "1"
        self.run_worlds = env.get("APP_RUN_WORLDS", "1") == "1"   # tests drive ticks by hand
        self.processes = env.get("APP_PROCESSES", "1") == "1"
        self.frontend_dist = Path(env.get("APP_FRONTEND_DIST", str(ROOT / "app" / "frontend" / "dist")))
        users_file = env.get("APP_USERS_FILE")
        if users_file:
            self.users = json.loads(Path(users_file).read_text(encoding="utf-8"))
        else:
            self.users = {name: {**u, "password": DEMO_PASSWORD} for name, u in DEMO_USERS.items()}
        self.using_demo_users = not users_file


# ---------------------------------------------------------------------------
# World threads
# ---------------------------------------------------------------------------

class WorldRunner:
    """Owns one world: ticks it on a background thread at its configured
    pace and hands out role views, computed once per tick and cached.

    The world is either a ``World`` playing every part in this process, or
    a ``ProcessWorld`` whose operators and federation are separate
    processes (``self.group``); ``self.world`` is then its sky, which holds
    the clock and the ground truth."""

    def __init__(self, engine, speed_factor: float = 1.0, restart=None):
        self.group: Optional[ProcessWorld] = engine if isinstance(engine, ProcessWorld) else None
        self.world: World = engine.sky if self.group else engine
        self._restart = restart   # how to bring a ProcessWorld back if a process dies
        self.lock = threading.RLock()
        self.interval = self.world.cfg.tick_sim_s / (self.world.cfg.speed * speed_factor)
        self.version = 0
        self.error: Optional[str] = None
        self._cache: Dict[str, tuple] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def separate_processes(self) -> bool:
        return self.group is not None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name=f"world-{self.world.cfg.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        with self.lock:
            self.world.save(self.interval)
            if self.group:
                self.group.close()

    def step(self, n: int = 1) -> None:
        """Advance synchronously (used by the loop and by tests)."""
        for _ in range(n):
            with self.lock:
                (self.group or self.world).tick()
                self.world.save(self.interval)
                self.version += 1
                self._cache.clear()

    def _recover(self) -> None:
        """A twin or the federation process died: restart the group. Each
        process resumes from its own checkpoint and catches up."""
        if not self.group or not self._restart:
            return
        with self.lock:
            self.group.close()
            self.group = self._restart()
            self.world = self.group.sky
            self._cache.clear()

    def _loop(self) -> None:
        next_at = time.monotonic()
        failures = 0
        while not self._stop.is_set():
            try:
                self.step()
                self.error, failures = None, 0
            except Exception as exc:  # keep the server up; report on /api/worlds
                failures += 1
                self.error = f"{type(exc).__name__}: {exc}"
                log.exception("world %s tick failed", self.world.cfg.name)
                self._stop.wait(min(60, 2 ** failures))
                try:
                    self._recover()
                except Exception:
                    log.exception("world %s: restarting its processes failed", self.world.cfg.name)
                next_at = time.monotonic()
                continue
            next_at += self.interval
            delay = next_at - time.monotonic()
            if delay < -5 * self.interval:
                next_at = time.monotonic()
            elif delay > 0:
                self._stop.wait(delay)

    def view(self, key: str, build) -> dict:
        with self.lock:
            cached = self._cache.get(key)
            if cached and cached[0] == self.version:
                return cached[1]
            data = build()
            data["version"] = self.version
            self._cache[key] = (self.version, data)
            return data

    # ---- role views, from whichever kind of world this is ----

    def operator_view(self, operator: str) -> dict:
        if self.group:
            return self.group.operator_view(operator)
        return operator_view(self.world, operator)

    def federation_view(self) -> dict:
        return self.group.federation_view() if self.group else federation_view(self.world)

    def observer_view(self) -> dict:
        return self.group.observer_view() if self.group else observer_view(self.world)

    def compare(self, encounter_id: str) -> dict:
        with self.lock:
            if self.group:
                return self.group.compare(encounter_id)
            return compare_encounter(self.world, encounter_id)

    def track(self, norad_id: int, minutes: int, operator: Optional[str]) -> dict:
        """``operator=None`` for the observer. Raises KeyError / PermissionError."""
        with self.lock:
            if self.group:
                return self.group.track(norad_id, minutes, operator)
            obj = next((o for o in self.world._objects if o.norad_id == norad_id), None)
            if obj is None:
                raise KeyError(norad_id)
            if operator is not None and obj.operator != operator:
                raise PermissionError("only a satellite's own operator can see its orbit")
            return ground_track(self.world, norad_id, minutes)


class AppState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.runners: Dict[str, WorldRunner] = {}

    def load(self) -> None:
        for path in sorted(self.settings.worlds_dir.glob("*.json")):
            cfg = WorldConfig.load(path)
            if self.settings.world_names and cfg.name not in self.settings.world_names:
                continue
            run_dir = self.settings.runs_dir / cfg.name
            if not self.settings.reset and run_is_stale(cfg, run_dir):
                # The world's definition changed (new fleet or rules): its
                # old history can't be resumed. Keep it aside, start fresh.
                archived = run_dir.with_name(f"{cfg.name}.old-{time.strftime('%Y%m%d-%H%M%S')}")
                run_dir.rename(archived)
                log.warning("world %s: config changed, old run moved to %s", cfg.name, archived)
            try:
                if self.settings.processes:
                    def restart(cfg=cfg, path=path, run_dir=run_dir):
                        return ProcessWorld.start(cfg, run_dir, ROOT, world_file=path)
                    engine = ProcessWorld.start(cfg, run_dir, ROOT, reset=self.settings.reset, world_file=path)
                    runner = WorldRunner(engine, self.settings.speed, restart=restart)
                else:
                    engine = World.open(cfg, run_dir, ROOT, reset=self.settings.reset)
                    runner = WorldRunner(engine, self.settings.speed)
            except FileNotFoundError as exc:
                log.warning("skipping world %s: %s", cfg.name, exc)
                continue
            self.runners[cfg.name] = runner
        if self.settings.run_worlds:
            for r in self.runners.values():
                r.start()

    def shutdown(self) -> None:
        for r in self.runners.values():
            r.stop()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class LoginBody(BaseModel):
    username: str
    password: str


class User(BaseModel):
    username: str
    role: str
    operator: Optional[str] = None


def _issue(settings: Settings, username: str, user: dict) -> str:
    payload = {"sub": username, "role": user["role"], "op": user.get("operator"),
               "exp": int(time.time()) + TOKEN_HOURS * 3600}
    return jwt.encode(payload, settings.secret, algorithm="HS256")


def _decode(settings: Settings, token: str) -> User:
    try:
        p = jwt.decode(token, settings.secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "invalid or expired token")
    return User(username=p["sub"], role=p["role"], operator=p.get("op"))


def _can_see_world(user: User, runner: WorldRunner) -> bool:
    return user.role in ("federation", "observer") or user.operator in runner.world.operators


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings()
    state = AppState(settings)
    bearer = HTTPBearer(auto_error=False)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await run_in_threadpool(state.load)
        log.info("worlds running: %s", ", ".join(state.runners) or "none")
        if settings.using_demo_users:
            log.warning("demo users enabled (password 'demo'); set APP_USERS_FILE before exposing this server")
        yield
        await run_in_threadpool(state.shutdown)

    app = FastAPI(title="Satcollision federation", version="1.0", lifespan=lifespan)
    app.state.app_state = state

    def current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> User:
        if creds is None:
            raise HTTPException(401, "login required")
        return _decode(settings, creds.credentials)

    def runner_for(name: str, user: User) -> WorldRunner:
        runner = state.runners.get(name)
        if runner is None:
            raise HTTPException(404, f"no world named {name!r}")
        if not _can_see_world(user, runner):
            raise HTTPException(403, f"{user.operator} is not an operator in {name}")
        return runner

    def build_view(runner: WorldRunner, user: User) -> dict:
        if user.role == "operator":
            return runner.view(f"op:{user.operator}", lambda: runner.operator_view(user.operator))
        if user.role == "federation":
            return runner.view("federation", runner.federation_view)
        return runner.view("observer", runner.observer_view)

    # ---- routes ----

    @app.get("/api/health")
    def health():
        return {"ok": True, "worlds": {n: {"tick": r.world.tick_index, "error": r.error,
                                           "separate_processes": r.separate_processes}
                                       for n, r in state.runners.items()}}

    @app.post("/api/login")
    def login(body: LoginBody):
        user = settings.users.get(body.username.strip().lower())
        if not user or not hmac.compare_digest(str(user.get("password", "")), body.password):
            raise HTTPException(401, "wrong username or password")
        return {"token": _issue(settings, body.username.strip().lower(), user),
                "role": user["role"], "operator": user.get("operator"),
                "username": body.username.strip().lower()}

    @app.get("/api/demo-users")
    def demo_users():
        """Only answers when the built-in demo accounts are active, so the
        login page can offer one-click demo logins and a deployment with
        real accounts reveals nothing."""
        if not settings.using_demo_users:
            return {"enabled": False, "users": []}
        return {"enabled": True, "password": DEMO_PASSWORD,
                "users": [{"username": n, **u} for n, u in DEMO_USERS.items()]}

    @app.get("/api/me")
    def me(user: User = Depends(current_user)):
        return user

    @app.get("/api/worlds")
    def worlds(user: User = Depends(current_user)):
        out = []
        for name, r in state.runners.items():
            if not _can_see_world(user, r):
                continue
            w = r.world
            out.append({"name": name, "title": w.cfg.title or name, "description": w.cfg.description,
                        "operators": w.operators, "tick": w.tick_index, "error": r.error,
                        "separate_processes": r.separate_processes,
                        "source": "synthetic fleets" if w.cfg.fleet["type"] == "synthetic" else "real TLE catalogue"})
        return out

    @app.get("/api/worlds/{name}/view")
    async def view(name: str, user: User = Depends(current_user)):
        runner = runner_for(name, user)
        return await run_in_threadpool(build_view, runner, user)

    @app.get("/api/worlds/{name}/compare/{encounter_id}")
    async def compare(name: str, encounter_id: str, user: User = Depends(current_user)):
        if user.role != "observer":
            raise HTTPException(403, "only the observer can put several parties' views side by side")
        runner = runner_for(name, user)
        try:
            return await run_in_threadpool(runner.compare, encounter_id)
        except KeyError:
            raise HTTPException(404, f"no encounter {encounter_id} in {name}")

    @app.get("/api/worlds/{name}/satellites/{norad_id}/track")
    async def track(name: str, norad_id: int, minutes: int = Query(100, ge=10, le=300),
                    user: User = Depends(current_user)):
        """Ground track for the next ``minutes``: the owner's own orbit, or
        anything for the observer. Nobody else may ask for an orbit."""
        runner = runner_for(name, user)
        if user.role not in ("observer", "operator"):
            raise HTTPException(403, "only a satellite's own operator can see its orbit")
        operator = None if user.role == "observer" else user.operator
        try:
            return await run_in_threadpool(runner.track, norad_id, minutes, operator)
        except KeyError:
            raise HTTPException(404, "no such satellite")
        except PermissionError as exc:
            raise HTTPException(403, str(exc))

    @app.websocket("/api/worlds/{name}/live")
    async def live(ws: WebSocket, name: str, token: str = Query(...)):
        try:
            user = _decode(settings, token)
        except HTTPException:
            await ws.close(code=4401)
            return
        runner = state.runners.get(name)
        if runner is None or not _can_see_world(user, runner):
            await ws.close(code=4403)
            return
        await ws.accept()
        sent = -1
        try:
            while True:
                if runner.version != sent:
                    data = await run_in_threadpool(build_view, runner, user)
                    sent = data["version"]
                    await ws.send_json(data)
                await asyncio.sleep(0.5)
        except (WebSocketDisconnect, RuntimeError):
            return

    # ---- frontend ----
    dist = settings.frontend_dist
    if (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/earth.jpg", include_in_schema=False)
        def earth():
            for p in (ROOT / "data" / "earth_texture.jpg", ROOT / "data" / "earth_texture.jpeg"):
                if p.exists():
                    return FileResponse(p)
            raise HTTPException(404)

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            target = (dist / path).resolve()
            if path and target.is_file() and dist.resolve() in target.parents:
                return FileResponse(target)
            return FileResponse(dist / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def no_frontend():
            return JSONResponse({"message": "API is running. Build the frontend: cd app/frontend && npm install && npm run build"})

    return app


def main() -> None:
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    if settings.host not in ("127.0.0.1", "localhost") and settings.using_demo_users:
        log.warning("binding %s with demo users: anyone who can reach this port can log in", settings.host)
    print(f"Satcollision app: http://{'127.0.0.1' if settings.host == '0.0.0.0' else settings.host}:{settings.port}/", flush=True)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
