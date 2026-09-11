#!/usr/bin/env python3
"""
One operator's standalone process: connects to the federation hub over a
real TCP socket, generates a real signed conjunction signal from actual
SGP4-propagated physics, sends it, and verifies whatever its peers send
back -- all as a genuinely separate OS process from every other operator
and from the hub, exchanging bytes over ``127.0.0.1`` rather than sharing
Python objects in one script's memory.

Two run modes, both real, deliberately not hidden behind one code path
that "decides" which is honest:

* **Legitimate operator** (default): loads its own private key from a
  file (:func:`~satcollision.identity.OperatorIdentity.from_private_key_bytes`)
  and a pre-distributed public-key registry, builds a real conjunction
  from its own synthetic fleet, signs the resulting
  :class:`~satcollision.signal.AbstractedSignal` with its own key, and
  verifies every incoming message against the registry it was handed --
  never against a key the message itself supplies.
* **Attacker** (``--attacker``): generates its *own* fresh keypair but
  claims (via ``--name``) to be a legitimate operator it has no real key
  for, and signs a fabricated signal under that claimed name. This
  succeeds at the signing step -- ``identity.sign_signal`` only checks
  that an identity is internally self-consistent (the operator field on
  the object matches the operator field on the signal), which an
  attacker's own freshly generated identity trivially satisfies. What
  catches it is entirely on the *receiving* end: every legitimate peer
  verifies against its own pre-distributed registry, which holds the
  real operator's key, not the attacker's -- so ``verify_signed`` returns
  False and the forged message is logged as rejected, never appended to
  that peer's tamper-evident log. This is the same guarantee
  ``identity.py``'s own unit tests exercise in-process; here it survives
  a real process boundary and a real socket.

Startup rendezvous: to avoid the classic distributed-systems race where
one node's real broadcast arrives before a slower-starting peer has even
connected (the hub never replays history to a late joiner), every
legitimate node sends a "ready" ping and waits until it has seen one from
every name in ``--peers`` before sending anything else. The attacker
does not participate in this and does not need to -- nothing in its
success depends on precise timing, only on arriving before its targets
give up and exit.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time

from .identity import OperatorIdentity, SignedLog, sign_signal, verify_signed
from .signal import AbstractedSignal
from .wire import decode_signed_report, decode_signed_signal, encode_ready, \
    encode_signed_signal, read_message, send_message

# `fleets`/`twin`'s propagation pipeline (SGP4 propagation, the Nelder-Mead
# encounter-geometry solver -- the actually expensive part of the physics
# stack) is imported and run lazily, inside `_build_real_signal`, rather
# than at module level. The attacker path (`run_attacker`) never calls that
# function -- it forges a signal by hand, on purpose, since an attacker has
# no real orbital data to propagate, and only needs the plain
# `AbstractedSignal` dataclass imported above. Paying for a real SGP4
# propagation it never uses would only slow the attacker's own process
# startup, which matters here: this project's distributed demo launches
# five real OS processes at once on one machine (see `distributed_demo.py`),
# and under CPU contention between them, a slower-starting attacker is a
# slower-arriving attacker -- exactly the race this module's ``--linger``
# grace period exists to absorb. Keeping the attacker's own startup light
# is a small, honest way to reduce how much that grace period has to cover.

_OTHER_SHELL_FALLBACK = {"Alpha": "Beta", "Beta": "Gamma", "Gamma": "Alpha"}


def _build_real_signal(own_name: str, index: int, seed: int) -> AbstractedSignal:
    """A real, SGP4-propagated, Pc-screened, threshold-crossing conjunction
    signal for ``own_name`` to send -- not a hand-built stand-in. Reuses
    exactly the same simulation/twin/signal-abstraction pipeline every
    other demo in this project runs.
    """
    from .fleets import _epoch_from_calendar, build_synthetic_fleets, inject_actor_encounter
    from .signal import abstract_conjunction
    from .twin import propagate_window, screen_conjunctions

    partner = _OTHER_SHELL_FALLBACK.get(own_name, "Beta")
    if partner == own_name:
        partner = "Alpha"
    objects = build_synthetic_fleets(seed=seed)
    # crossing_angle_deg=15 (rather than inject_actor_encounter's own
    # default of 55) keeps actor B's inclination close to actor A's own --
    # for a near-polar watch operator (Beta/Gamma sit at ~87 degrees) a
    # wide crossing angle can ask for an orbital plane whose maximum
    # latitude reach is geometrically smaller than the point on actor A's
    # own near-polar orbit the encounter needs to hit, which the function's
    # local optimizer cannot solve around. A small angle sidesteps that
    # without touching fleets.py's shared implementation.
    actor_a, actor_b = inject_actor_encounter(
        objects, watch_operator_a=own_name, new_operator_b=partner,
        tca_offset_min=20.0 + 5.0 * index, crossing_angle_deg=15.0, seed=seed,
    )
    _, jd0, fr0 = _epoch_from_calendar(2026, 9, 9, 0.0)
    prop = propagate_window(objects, jd0, fr0, duration_s=(25.0 + 5.0 * index) * 60, step_s=10.0)
    conjunctions = screen_conjunctions(objects, prop, screening_distance_km=25.0)
    actor_conj = next(
        c for c in conjunctions
        if {c.object_a.norad_id, c.object_b.norad_id} == {actor_a.norad_id, actor_b.norad_id}
    )
    signals = abstract_conjunction(actor_conj)
    if signals is None:
        raise RuntimeError("engineered encounter did not cross the Pc threshold -- unexpected")
    sig_own = next(s for s in signals if s.sender_operator == own_name)
    return sig_own


async def _connect_with_retry(host: str, port: int, attempts: int = 40, delay: float = 0.25):
    last_error = None
    for _ in range(attempts):
        try:
            return await asyncio.open_connection(host, port)
        except (ConnectionRefusedError, OSError) as exc:
            last_error = exc
            await asyncio.sleep(delay)
    raise ConnectionError(f"could not connect to hub at {host}:{port} after {attempts} attempts") from last_error


class OperatorNode:
    def __init__(self, name: str, peers: list[str], trusted_public_keys: dict[str, bytes] | None):
        self.name = name
        self.peers = peers
        self.trusted_public_keys = trusted_public_keys or {}
        self.ready_seen: set[str] = set()
        self.verified_from_peer: dict[str, int] = {p: 0 for p in peers}
        self.rejected_count = 0
        self.log = SignedLog()
        # Every inbound message this node has seen, of any type -- used to
        # decide, deterministically, when it's safe to stop listening (see
        # `--expected-inbound` in `run_legitimate_operator`), as opposed to
        # guessing from a fixed wall-clock delay.
        self.messages_received = 0

    async def read_loop(self, reader: asyncio.StreamReader) -> None:
        while True:
            message = await read_message(reader)
            if message is None:
                return
            msg_type = message.get("type")
            self.messages_received += 1
            if msg_type == "ready":
                self.ready_seen.add(message["operator"])
            elif msg_type == "signal":
                signed = decode_signed_signal(message)
                self._handle_signed(signed)
            elif msg_type == "report":
                signed = decode_signed_report(message)
                self._handle_signed(signed)

    def _handle_signed(self, signed) -> None:
        if verify_signed(signed, self.trusted_public_keys):
            self.verified_from_peer[signed.sender_operator] = (
                self.verified_from_peer.get(signed.sender_operator, 0) + 1
            )
            self.log.append(signed)
            print(f"[{self.name}] ACCEPTED a verified message from '{signed.sender_operator}'", flush=True)
        else:
            self.rejected_count += 1
            print(f"[{self.name}] REJECTED a message claiming to be from "
                  f"'{signed.sender_operator}' -- signature did not verify against the "
                  f"trusted registry", flush=True)

    def total_verified(self) -> int:
        return sum(self.verified_from_peer.values())


async def run_legitimate_operator(args) -> dict:
    with open(args.private_key_file, "rb") as fh:
        raw_key = fh.read()
    identity = OperatorIdentity.from_private_key_bytes(args.name, raw_key)

    with open(args.registry_file, "r", encoding="utf-8") as fh:
        registry = json.load(fh)
    trusted_public_keys = {op: base64.b64decode(key) for op, key in registry.items()}

    peers = [p for p in args.peers.split(",") if p] if args.peers else []
    node = OperatorNode(args.name, peers, trusted_public_keys)

    reader, writer = await _connect_with_retry(args.host, args.port)
    read_task = asyncio.create_task(node.read_loop(reader))

    await send_message(writer, encode_ready(args.name))
    barrier_deadline = time.monotonic() + args.timeout
    while not set(peers).issubset(node.ready_seen) and time.monotonic() < barrier_deadline:
        await asyncio.sleep(0.02)
    if not set(peers).issubset(node.ready_seen):
        print(f"[{args.name}] WARNING: rendezvous timed out waiting for "
              f"{set(peers) - node.ready_seen} -- proceeding anyway", flush=True)

    for i in range(args.n_signals):
        real_signal = _build_real_signal(args.name, i, seed=args.seed + i)
        signed = sign_signal(identity, real_signal)
        await send_message(writer, encode_signed_signal(signed))
        print(f"[{args.name}] sent real signed signal {i + 1}/{args.n_signals} "
              f"(encounter_id={real_signal.encounter_id})", flush=True)

    expected_total = args.n_signals * len(peers)
    quota_deadline = time.monotonic() + args.timeout

    if args.expected_inbound is not None:
        # Deterministic exit condition, used by the orchestrated demo
        # (distributed_demo.py), which knows exactly how many messages
        # will ever cross the wire: len(peers) "ready" pings + len(peers)
        # real signals + however many forged messages an impersonator
        # sends -- every one of them broadcast to every other currently
        # connected client, this node included. Waiting for the *full*
        # inbound count, not just this node's own quota of verified peer
        # signals, is what makes catching an impersonator's message a real
        # guarantee rather than a race: this node simply does not exit
        # until it has literally seen everything there was to see (or
        # --timeout is reached as a safety net, e.g. if a peer crashes).
        # This replaced an earlier fixed-length "linger" sleep after quota
        # was met, which turned out not to be safe -- under enough CPU
        # contention from launching five real OS processes at once, a
        # straggling attacker could still arrive after any fixed delay,
        # so a wall-clock guess was never going to be reliable. Counting
        # actual messages is.
        while node.messages_received < args.expected_inbound and time.monotonic() < quota_deadline:
            await asyncio.sleep(0.02)
        if node.messages_received < args.expected_inbound:
            print(f"[{args.name}] WARNING: only saw {node.messages_received}/"
                  f"{args.expected_inbound} expected inbound messages before --timeout "
                  f"({args.timeout}s) -- proceeding anyway", flush=True)
    else:
        # Standalone fallback (no orchestrator-supplied expected count):
        # wait for this node's own quota, then linger briefly so an
        # unannounced straggler still has a short window to be seen.
        while node.total_verified() < expected_total and time.monotonic() < quota_deadline:
            await asyncio.sleep(0.02)
        await asyncio.sleep(args.linger)

    read_task.cancel()
    writer.close()

    chain_ok, chain_reason = node.log.verify_chain(trusted_public_keys)
    result = {
        "operator": args.name,
        "role": "legitimate",
        "sent": args.n_signals,
        "expected_verified_total": expected_total,
        "verified_from_peer": node.verified_from_peer,
        "rejected_count": node.rejected_count,
        "log_entries": len(node.log.entries),
        "tail_hash": node.log.tail_hash(),
        "verify_chain_ok": chain_ok,
        "verify_chain_reason": chain_reason,
        "quota_met": node.total_verified() >= expected_total,
    }
    return result


async def run_attacker(args) -> dict:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    # The attacker's own, freshly generated keypair -- claimed under
    # `args.name`, a real operator it has no legitimate key for. This is
    # the exact same construction identity.py's own unit tests use for
    # impersonation, now happening in a genuinely separate OS process.
    forger = OperatorIdentity(operator=args.name, _private_key=Ed25519PrivateKey.generate())

    reader, writer = await _connect_with_retry(args.host, args.port)
    sent = []
    for i in range(args.n_signals):
        forged = AbstractedSignal(
            encounter_id=f"ENC-FORGED-{i}",
            sender_operator=args.name,
            local_object_label=f"{args.name}-99999",
            tca_offset_s=999.0,
            geometry_class="head-on",
            severity_tier="critical",
        )
        signed = sign_signal(forger, forged)
        await send_message(writer, encode_signed_signal(signed))
        sent.append(forged.encounter_id)
        print(f"[ATTACKER claiming '{args.name}'] sent forged signal {i + 1}/{args.n_signals} "
              f"signed with its OWN key (not the real {args.name}'s)", flush=True)

    await asyncio.sleep(0.2)  # give the hub a moment to relay before this process exits
    writer.close()
    return {"operator": args.name, "role": "attacker", "forged_encounter_ids": sent}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="operator name this node claims to be")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--attacker", action="store_true", help="run in attacker/impersonation mode")
    parser.add_argument("--private-key-file")
    parser.add_argument("--registry-file")
    parser.add_argument("--peers", default="", help="comma-separated names of other legitimate operators")
    parser.add_argument("--n-signals", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--linger", type=float, default=2.0,
                         help="(standalone fallback only, ignored when --expected-inbound is "
                              "given) seconds to keep reading after this node's own quota is "
                              "met, so an unannounced straggler still has a short window to "
                              "arrive before this process exits")
    parser.add_argument("--expected-inbound", type=int, default=None,
                         help="exit only once this many inbound messages of any type (ready "
                              "pings, real signals, forged signals) have been received -- the "
                              "deterministic alternative to --linger, used by the orchestrated "
                              "demo (distributed_demo.py), which knows this count exactly")
    parser.add_argument("--report-file", help="write the final JSON result here")
    args = parser.parse_args(argv)

    if args.attacker:
        result = asyncio.run(run_attacker(args))
    else:
        if not args.private_key_file or not args.registry_file:
            parser.error("--private-key-file and --registry-file are required for a legitimate operator")
        result = asyncio.run(run_legitimate_operator(args))

    print(f"[{args.name}] FINAL RESULT: {json.dumps(result)}", flush=True)
    if args.report_file:
        with open(args.report_file, "w", encoding="utf-8") as fh:
            json.dump(result, fh)

    if result.get("role") == "legitimate" and not result.get("quota_met", True):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
