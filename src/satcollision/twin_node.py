"""
One operator's digital twin as its own OS process.

    python -m satcollision.twin_node --world-file worlds/adversarial.json \\
        --operator Alpha --run-dir runs/adversarial/nodes/Alpha \\
        --federation http://127.0.0.1:PORT

Normally started by the app (satcollision.processes). What lives only in
this process, and nowhere else:

* the operator's **private signing key**, generated here on first start and
  kept in ``<run-dir>/identity.key`` (a real operator would keep it in a
  hardware key store). Only the public half is sent to the federation,
  which pins it;
* its twin's **screening results**: every close approach involving its
  satellites, with the exact miss distance and Pc, including the ones below
  the sharing threshold that nobody else ever hears about;
* its **decisions**: whether to share (a scripted free-rider or spoofer
  makes that choice here, where a real dishonest operator would), what to
  put in its round report, and the manoeuvre plan it derives.

It screens its own satellites against the public catalogue (TLEs are
public in reality too) with the same World code the single-process mode
uses, in ``mode="operator"``, so the two modes stay comparable.

Routes (all need the internal node token):

=================  ==========================================================
GET  /health       operator, tick, pid
POST /tick         advance to ``to_tick``; signed messages go to the federation
POST /attest       which of these encounter ids involve my satellites?
GET  /view         this operator's dashboard (asks the federation for its feed)
GET  /track        ground track of one of *my* satellites (?id=&minutes=)
=================  ==========================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from .identity import OperatorIdentity
from .node_http import TOKEN_ENV, Forbidden, NotFound, call, run_forever, serve
from .reputation import attest_participation
from .views import ground_track, operator_view
from .world import World, WorldConfig


def load_or_create_identity(operator: str, path: Path) -> OperatorIdentity:
    if path.exists():
        return OperatorIdentity.from_private_key_bytes(operator, path.read_bytes())
    identity = OperatorIdentity.generate(operator)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(identity.private_key_bytes())
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return identity


class TwinNode:
    def __init__(self, cfg: WorldConfig, operator: str, run_dir: Path, repo_root: Path,
                 federation_url: str, token: str, reset: bool = False):
        self.cfg, self.operator, self.run_dir, self.repo_root = cfg, operator, run_dir, repo_root
        self.federation_url, self.token = federation_url, token
        if reset:
            (run_dir / "identity.key").unlink(missing_ok=True)
        self.identity = load_or_create_identity(operator, run_dir / "identity.key")
        self.lock = threading.RLock()
        self.world = self._open(reset)

    def _open(self, reset: bool = False) -> World:
        return World.open(self.cfg, self.run_dir / "twin", self.repo_root, reset=reset,
                          mode="operator", operator=self.operator, identity=self.identity)

    def _fed(self, method: str, path: str, body=None) -> dict:
        return call(self.federation_url, method, path, body, token=self.token)

    def register(self, my_url: str) -> dict:
        return self._fed("POST", "/register", {"operator": self.operator, "url": my_url,
                                               "public_key": self.identity.public_key_bytes().hex()})

    # ---- routes ----

    def health(self, body, query):
        return {"ok": True, "role": "operator", "operator": self.operator, "pid": os.getpid(),
                "tick": self.world.tick_index, "round": self.world.round_index}

    def tick(self, body, query):
        target = int(body["to_tick"])
        sent = 0
        with self.lock:
            while self.world.tick_index < target:
                try:
                    self.world.tick()
                    for message in self.world.outbox:
                        self._fed("POST", "/message", message)
                        sent += 1
                    self.world.outbox = []
                    self.world.save()
                except Exception:
                    # Roll back to the last saved tick; the federation drops
                    # anything it already accepted when the tick is redone.
                    self.world = self._open()
                    raise
        return {"tick": self.world.tick_index, "sent": sent}

    def attest(self, body, query):
        # Read-only and lock-free on purpose: the federation asks while it
        # holds its own lock, and a view on this node may be waiting on it.
        owned = set(self.world.ownership.get(self.operator, ()))
        mine = []
        for eid in body["ids"]:
            try:
                if attest_participation(eid, owned):
                    mine.append(eid)
            except ValueError:
                continue
        return {"mine": mine}

    def view(self, body, query):
        ts = time.time()
        feed = self._fed("POST", "/feed", {
            "operator": self.operator, "ts": ts,
            "signature": self.identity.sign_bytes(f"feed|{self.operator}|{ts}".encode()).hex()})
        with self.lock:
            return operator_view(self.world, self.operator, feed=feed)

    def track(self, body, query):
        norad_id = int(query["id"])
        minutes = max(10, min(300, int(query.get("minutes", 100))))
        with self.lock:
            obj = next((o for o in self.world._objects if o.norad_id == norad_id), None)
            if obj is None:
                raise NotFound("no such satellite")
            if obj.operator != self.operator:
                raise Forbidden("only a satellite's own operator can see its orbit")
            return ground_track(self.world, norad_id, minutes)

    def routes(self):
        return {
            ("GET", "/health"): self.health,
            ("POST", "/tick"): self.tick,
            ("POST", "/attest"): self.attest,
            ("GET", "/view"): self.view,
            ("GET", "/track"): self.track,
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--world-file", required=True)
    ap.add_argument("--operator", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--federation", required=True, help="the federation node's URL")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args(argv)
    token = os.environ.get(TOKEN_ENV)
    if not token:
        print(f"{TOKEN_ENV} must be set", file=sys.stderr)
        return 2
    node = TwinNode(WorldConfig.load(args.world_file), args.operator, Path(args.run_dir),
                    Path(args.repo_root), args.federation, token, reset=args.reset)
    server = serve(node.routes(), token)
    node.register(f"http://127.0.0.1:{server.server_address[1]}")
    run_forever(server)
    return 0


if __name__ == "__main__":
    sys.exit(main())
