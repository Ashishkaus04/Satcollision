"""
The federation server as its own OS process.

    python -m satcollision.federation_node --world adversarial \\
        --operators Alpha,Beta,Gamma --run-dir runs/adversarial/federation

Normally started by the app (satcollision.processes), not by hand. This
process imports no orbital code (no sgp4, scipy, fleets, twin or world;
GET /health lists what it has loaded so a test can check), holds no
catalogue and never receives a position: only what operators' twins send
it, signed with keys it has pinned.

Routes (all need the internal node token):

=================  ==========================================================
GET  /health       pid, last round, which modules are loaded
POST /register     an operator's node announces its public key and address
POST /clock        the current simulated time (ledger timestamps)
POST /message      a signed signal or report from an operator's node
POST /settle       ask every node which new encounter ids involve it
POST /round        close a reporting round: attest, score, aggregate
POST /feed         one operator's alerts, reputation and reports; the request
                   must be signed by that operator's registered key
GET  /view         the federation's own dashboard
=================  ==========================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from typing import Dict

from .federation_service import FederationService, Refused
from .node_http import TOKEN_ENV, NodeError, call, run_forever, serve

HEAVY_MODULES = ("sgp4", "scipy", "satcollision.twin", "satcollision.fleets", "satcollision.world",
                 "satcollision.views")


def build_routes(service: FederationService, token: str):
    lock = threading.RLock()
    nodes: Dict[str, str] = {}   # operator -> its node's URL (re-announced on every start)

    def health(body, query):
        return {"ok": True, "role": "federation", "pid": os.getpid(), "world": service.world,
                "last_round": service.last_round, "registered": sorted(nodes),
                "heavy_modules_loaded": [m for m in HEAVY_MODULES if m in sys.modules]}

    def register(body, query):
        with lock:
            out = service.register(body["operator"], body["public_key"])
            nodes[body["operator"]] = body["url"]
            return out

    def clock(body, query):
        with lock:
            service.set_clock(body["tick"], body["sim_time"])
            return {"ok": True}

    def message(body, query):
        with lock:
            return service.receive(body)

    def _settle() -> dict:
        asked = service.unattested_ids()
        if not asked:
            return {"asked": 0}
        answers = {}
        for op in service.operators:
            url = nodes.get(op)
            if url is None:
                return {"asked": 0, "waiting_for": op}   # try again once it is back
            try:
                answers[op] = call(url, "POST", "/attest", {"ids": asked}, token=token)["mine"]
            except NodeError:
                return {"asked": 0, "waiting_for": op}
        service.record_attestations(asked, answers)
        return {"asked": len(asked)}

    def settle(body, query):
        with lock:
            out = _settle()
            service.save()
            return out

    def round_(body, query):
        with lock:
            _settle()
            summary = service.run_round(int(body["round"]))
            service.save()
            return {"summary": summary, "last_round": service.last_round}

    def feed(body, query):
        op = body["operator"]
        service.verify_request(op, "feed", float(body["ts"]), body["signature"])
        with lock:
            return service.feed(op)

    def view(body, query):
        with lock:
            return service.view()

    return {
        ("GET", "/health"): health,
        ("POST", "/register"): register,
        ("POST", "/clock"): clock,
        ("POST", "/message"): message,
        ("POST", "/settle"): settle,
        ("POST", "/round"): round_,
        ("POST", "/feed"): feed,
        ("GET", "/view"): view,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--world", required=True)
    ap.add_argument("--operators", required=True, help="comma-separated member operators")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args(argv)
    token = os.environ.get(TOKEN_ENV)
    if not token:
        print(f"{TOKEN_ENV} must be set", file=sys.stderr)
        return 2
    service = FederationService(args.world, args.operators.split(","), args.run_dir, reset=args.reset)
    server = serve(build_routes(service, token), token, refused=(Refused,))
    run_forever(server, on_exit=service.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
