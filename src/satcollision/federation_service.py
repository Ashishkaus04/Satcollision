"""
The federation server's state and rules, with no orbital code at all.

This is what runs inside the federation's own process (federation_node.py)
when each operator's digital twin is a separate process. It is written so
that it *cannot* look at an orbit even by accident: nothing here imports
sgp4, scipy, fleets, twin or world (tests/test_nodes.py checks that the
federation process never loads them). What it holds:

* a **pinned key registry**: the first public key an operator registers is
  kept, and a later registration with a different key is refused, so
  nobody can take over an operator's identity after the fact;
* the **signed ledger** (identity.SignedLog, hash-chained, mirrored to
  ``ledger.jsonl`` and re-verified on restart) of every signal and report
  operators sent;
* **attestations**: for each encounter id it has heard of, which operators
  said "that's mine" when asked (one boolean per operator and encounter);
* **reputation** and the round history.

From those it answers two questions: what goes on the federation's own
dashboard (:meth:`FederationService.view`), and what one operator is told
(:meth:`FederationService.feed`: alerts about its own encounters, cut down
to the tier its reputation has earned).
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .federation import (
    OperatorReport,
    krum_aggregate,
    mean_aggregate,
    trimmed_mean_aggregate,
)
from .identity import GENESIS_HASH, SignedLog, SignedReport, SignedSignal, verify_signed
from .reputation import (
    ReputationState,
    alert_tier,
    evaluate_round,
    reputation_weighted_aggregate,
    tailor_alert,
    update_scores,
)
from .signal import AbstractedSignal, assert_no_raw_state
from .wire import decode_signed_report, decode_signed_signal

MAX_ROUNDS_KEPT = 200
MAX_REPUTATION_HISTORY = 200
MAX_ATTESTED_KEPT = 4000

FEDERATION_WITHHELD = [
    "Any satellite position, velocity or orbit",
    "Miss distances and exact collision probabilities",
    "Which encounters are real until operators attest to them",
    "Close approaches below the sharing threshold or inside one operator's own fleet",
]


# ---------------------------------------------------------------------------
# Small helpers shared with views.py
# ---------------------------------------------------------------------------

def iso(when: datetime) -> str:
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_from_ts(ts: float) -> str:
    return iso(datetime.fromtimestamp(ts, tz=timezone.utc))


def ts_from_iso(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def ledger_signal_rows(entries, limit: int = 400) -> List[dict]:
    """The signed signals in a ledger, newest first, as the dashboards show them."""
    out = []
    for entry in reversed(entries):
        if isinstance(entry.signed, SignedSignal):
            s = entry.signed.signal
            out.append({
                "ledger_index": entry.index,
                "received": iso_from_ts(entry.timestamp),
                "encounter_id": s.encounter_id,
                "sender_operator": s.sender_operator,
                "local_object_label": s.local_object_label,
                "severity_tier": s.severity_tier,
                "geometry_class": s.geometry_class,
                "tca_offset_s": s.tca_offset_s,
                "signature": entry.signed.signature.hex()[:16] + "…",
                "entry_hash": entry.entry_hash[:16] + "…",
            })
            if len(out) >= limit:
                break
    return out


def ledger_report_rows(entries, operator: Optional[str] = None, limit: int = 30) -> List[dict]:
    out = []
    for entry in reversed(entries):
        if isinstance(entry.signed, SignedReport):
            rep = entry.signed.report
            if operator is not None and rep.operator != operator:
                continue
            out.append({"received": iso_from_ts(entry.timestamp), "operator": rep.operator,
                        "counts": [int(round(c)) for c in rep.counts], "ledger_index": entry.index})
            if len(out) >= limit:
                break
    return out


def encounter_rows(signal_rows: List[dict]) -> List[dict]:
    """Group signal rows by encounter: who has signed so far."""
    by_encounter: Dict[str, dict] = {}
    for s in reversed(signal_rows):
        rec = by_encounter.setdefault(s["encounter_id"], {
            "encounter_id": s["encounter_id"], "first_received": s["received"],
            "senders": [], "severity_tier": s["severity_tier"], "geometry_class": s["geometry_class"],
        })
        if s["sender_operator"] not in rec["senders"]:
            rec["senders"].append(s["sender_operator"])
    rows = sorted(by_encounter.values(), key=lambda r: r["first_received"], reverse=True)
    for rec in rows:
        rec["status"] = "both sides signed" if len(rec["senders"]) >= 2 else "one side only"
    return rows


def reputation_block(rep: ReputationState, operators: List[str]) -> dict:
    return {
        "scores": {op: round(rep.score(op), 3) for op in operators},
        "tiers": {op: alert_tier(rep.score(op), rep.params) for op in operators},
        "history": [{op: round(s.get(op, float("nan")), 3) for op in operators}
                    for s in rep.history[-100:]],
    }


def ledger_line(entry, signed) -> str:
    if isinstance(signed, SignedSignal):
        kind, payload = "signal", dataclasses.asdict(signed.signal)
    else:
        kind = "report"
        payload = {"operator": signed.report.operator,
                   "counts": [float(c) for c in signed.report.counts],
                   "is_adversarial": signed.report.is_adversarial}
    return json.dumps({
        "index": entry.index, "timestamp": entry.timestamp, "kind": kind, "payload": payload,
        "signature": signed.signature.hex(), "public_key": signed.public_key.hex(),
        "prev_hash": entry.prev_hash, "entry_hash": entry.entry_hash,
    }, sort_keys=True)


def signed_from_line(rec: dict):
    if rec["kind"] == "signal":
        return SignedSignal(signal=AbstractedSignal(**rec["payload"]),
                            signature=bytes.fromhex(rec["signature"]),
                            public_key=bytes.fromhex(rec["public_key"]))
    p = rec["payload"]
    return SignedReport(
        report=OperatorReport(operator=p["operator"], counts=np.array(p["counts"], dtype=float),
                              is_adversarial=p["is_adversarial"]),
        signature=bytes.fromhex(rec["signature"]), public_key=bytes.fromhex(rec["public_key"]))


def _write_atomic(path: Path, text: str, retries: int = 20) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))


class Refused(Exception):
    """A message the federation will not accept (bad signature, unknown key...)."""


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------

class FederationService:
    def __init__(self, world: str, operators: List[str], run_dir: str | Path, reset: bool = False):
        self.world = world
        self.operators = sorted(operators)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if reset:
            for name in ("checkpoint.json", "ledger.jsonl"):
                (self.run_dir / name).unlink(missing_ok=True)
        self.registry: Dict[str, bytes] = {}
        self.ledger = SignedLog()
        self.ledger_verified = True
        self.attested: Dict[str, List[str]] = {}     # encounter id -> operators who said "mine"
        self.period: List[dict] = []                 # signals since the last round
        self.round_reports: Dict[str, dict] = {}     # reports for the round now open
        self.reputation = ReputationState()
        self.rounds: List[dict] = []
        self.last_round = 0
        self.clock = {"tick": 0, "sim_time": None}
        self.counters = {"signals_accepted": 0, "reports_accepted": 0, "refused": 0,
                         "duplicates": 0, "attestation_queries": 0}
        self._pending_lines: List[str] = []
        self._load()

    # ---- identity ----

    def register(self, operator: str, public_key_hex: str) -> dict:
        if operator not in self.operators:
            raise Refused(f"{operator} is not a member of the {self.world} federation")
        key = bytes.fromhex(public_key_hex)
        if len(key) != 32:
            raise Refused("an Ed25519 public key is 32 bytes")
        pinned = self.registry.get(operator)
        if pinned is not None and pinned != key:
            self.counters["refused"] += 1
            raise Refused(f"{operator} already registered a different key; refusing to replace it")
        self.registry[operator] = key
        self.save()
        return {"operator": operator, "key_fingerprint": key.hex()[:16], "pinned": pinned is not None}

    def verify_request(self, operator: str, purpose: str, ts: float, signature_hex: str,
                       max_skew_s: float = 120.0) -> None:
        """Check that a request really comes from ``operator``'s node: it
        signed ``purpose|operator|ts`` with the key the federation pinned."""
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        key = self.registry.get(operator)
        if key is None:
            raise Refused(f"{operator} has not registered")
        if abs(time.time() - ts) > max_skew_s:
            raise Refused("request timestamp too far from now")
        try:
            Ed25519PublicKey.from_public_bytes(key).verify(
                bytes.fromhex(signature_hex), f"{purpose}|{operator}|{ts}".encode())
        except (InvalidSignature, ValueError):
            raise Refused("request signature does not match the registered key")

    # ---- clock ----

    def set_clock(self, tick: int, sim_time: str) -> None:
        self.clock = {"tick": int(tick), "sim_time": sim_time}

    def _now_ts(self) -> float:
        return round(ts_from_iso(self.clock["sim_time"]), 3) if self.clock["sim_time"] else 0.0

    # ---- receiving ----

    def receive(self, message: dict) -> dict:
        kind = message.get("type")
        try:
            if kind == "signal":
                signed = decode_signed_signal(message)
                assert_no_raw_state(signed.signal)
            elif kind == "report":
                signed = decode_signed_report(message)
            else:
                raise Refused(f"unknown message type {kind!r}")
        except (KeyError, TypeError, ValueError, AssertionError) as exc:
            self.counters["refused"] += 1
            raise Refused(f"malformed message: {exc}")
        # verify_signed checks the signature *and* that the embedded public
        # key is the one pinned for the claimed sender.
        if not verify_signed(signed, self.registry):
            self.counters["refused"] += 1
            raise Refused(f"signature from {signed.sender_operator} does not verify against its pinned key")

        if kind == "signal":
            sig = signed.signal
            key = (sig.encounter_id, sig.sender_operator)
            if any((p["encounter_id"], p["sender_operator"]) == key for p in self.period):
                self.counters["duplicates"] += 1
                return {"accepted": False, "duplicate": True}
            self._append(signed)
            self.period.append(dataclasses.asdict(sig))
            self.counters["signals_accepted"] += 1
            return {"accepted": True}

        round_no = int(message.get("round", self.last_round + 1))
        op = signed.report.operator
        if round_no != self.last_round + 1 or op in self.round_reports:
            self.counters["duplicates"] += 1
            return {"accepted": False, "duplicate": True}
        self._append(signed)
        self.round_reports[op] = {"operator": op, "counts": [float(c) for c in signed.report.counts]}
        self.counters["reports_accepted"] += 1
        return {"accepted": True}

    def _append(self, signed) -> None:
        entry = self.ledger.append(signed, timestamp=self._now_ts())
        self._pending_lines.append(ledger_line(entry, signed))

    # ---- attestation ----

    def unattested_ids(self) -> List[str]:
        return sorted({s["encounter_id"] for s in self.period} - set(self.attested))

    def record_attestations(self, asked: List[str], answers: Dict[str, List[str]]) -> None:
        """``answers``: operator -> the subset of ``asked`` it says involves it."""
        self.counters["attestation_queries"] += len(asked) * len(answers)
        for eid in asked:
            self.attested[eid] = sorted(op for op, ids in answers.items() if eid in set(ids))
        if len(self.attested) > MAX_ATTESTED_KEPT:
            for eid in list(self.attested)[: len(self.attested) - MAX_ATTESTED_KEPT]:
                del self.attested[eid]

    # ---- rounds ----

    def run_round(self, round_no: int) -> Optional[dict]:
        if round_no <= self.last_round:
            return None  # already done (a retried call after a restart)
        signals = [AbstractedSignal(**s) for s in self.period]
        reports = [OperatorReport(operator=r["operator"], counts=np.array(r["counts"], dtype=float))
                   for r in self.round_reports.values()]
        attested = {op: {eid for eid, ops in self.attested.items() if op in ops} for op in self.operators}
        before = {op: self.reputation.tier(op) for op in self.operators}
        evidence = evaluate_round(self.operators, signals, reports, None, self.reputation.params,
                                  attested=attested)
        self.reputation = update_scores(self.reputation, evidence)
        self.reputation.history = self.reputation.history[-MAX_REPUTATION_HISTORY:]
        after = {op: self.reputation.tier(op) for op in self.operators}
        aggregates = {}
        if reports:
            aggregates = {
                "mean": mean_aggregate(reports),
                "trimmed_mean": trimmed_mean_aggregate(reports),
                "reputation_weighted": reputation_weighted_aggregate(reports, self.reputation),
            }
            if len(reports) >= 4:
                aggregates["krum"] = krum_aggregate(reports)
        self.last_round = round_no
        summary = {
            "round": round_no,
            "sim_time": self.clock["sim_time"],
            "n_signals": len(signals),
            "reports": {r.operator: [int(round(c)) for c in r.counts] for r in reports},
            "missing_reports": [op for op in self.operators if op not in self.round_reports],
            "aggregates": {k: [round(float(x), 3) for x in v] for k, v in aggregates.items()},
            # Error against the honest operators' truth needs to know who is
            # honest, which the federation by definition cannot: the
            # observer's view adds it from the simulation script.
            "errors": {},
            "scores": {op: round(self.reputation.score(op), 3) for op in self.operators},
            "tiers": after,
            "tier_changes": {op: [before[op], after[op]] for op in self.operators if before[op] != after[op]},
            "evidence": {op: {"involved": ev.n_involved, "matched": ev.n_matched,
                              "missed": ev.n_missed, "unbacked": ev.n_unbacked}
                         for op, ev in evidence.items()},
        }
        self.rounds.append(summary)
        del self.rounds[:-MAX_ROUNDS_KEPT]
        self.period = []
        self.round_reports = {}
        return summary

    # ---- what each party is told ----

    def feed(self, operator: str) -> dict:
        """Everything the federation tells one operator: alerts about
        encounters it attested to, tailored to its tier, who has signed
        each of those encounters, its reputation, and its own reports."""
        tier = self.reputation.tier(operator)
        mine = {eid for eid, ops in self.attested.items() if operator in ops}
        alerts, senders = [], {}
        for entry in reversed(self.ledger.entries):
            if not isinstance(entry.signed, SignedSignal):
                continue
            sig = entry.signed.signal
            if sig.encounter_id not in mine:
                continue
            senders.setdefault(sig.encounter_id, set()).add(sig.sender_operator)
            if sig.sender_operator != operator and len(alerts) < 40:
                alert = tailor_alert(sig, tier)
                alert["received"] = iso_from_ts(entry.timestamp)
                alerts.append(alert)
        return {
            "operator": operator,
            "alerts": alerts,
            "senders": {eid: sorted(ops) for eid, ops in senders.items()},
            "reputation": {"score": round(self.reputation.score(operator), 3), "tier": tier},
            "my_reports": [{"received": r["received"], "counts": r["counts"]}
                           for r in ledger_report_rows(self.ledger.entries, operator, limit=10)],
        }

    def view(self) -> dict:
        signals = ledger_signal_rows(self.ledger.entries)
        return {
            "role": "federation",
            "positions": [],     # deliberately empty: this process never receives one
            "signals": signals[:120],
            "encounters": encounter_rows(signals)[:60],
            "reports": ledger_report_rows(self.ledger.entries),
            "rounds": self.rounds[-20:][::-1],
            "reputation": reputation_block(self.reputation, self.operators),
            "ledger": {"entries": len(self.ledger.entries), "tail_hash": self.ledger.tail_hash(),
                       "verified_on_load": self.ledger_verified},
            "registry": {op: key.hex()[:16] for op, key in sorted(self.registry.items())},
            "counters": self.counters,
            "process": {"pid": os.getpid(), "separate": True},
            "withheld": FEDERATION_WITHHELD,
        }

    # ---- persistence ----

    def save(self) -> None:
        if self._pending_lines:
            with open(self.run_dir / "ledger.jsonl", "a", encoding="utf-8") as fh:
                fh.write("\n".join(self._pending_lines) + "\n")
            self._pending_lines = []
        _write_atomic(self.run_dir / "checkpoint.json", json.dumps({
            "world": self.world, "operators": self.operators,
            "registry": {op: k.hex() for op, k in self.registry.items()},
            "attested": self.attested, "period": self.period, "round_reports": self.round_reports,
            "reputation": {"scores": self.reputation.scores, "rounds": self.reputation.rounds,
                           "history": self.reputation.history},
            "rounds": self.rounds, "last_round": self.last_round, "clock": self.clock,
            "counters": self.counters,
            "ledger_entries": len(self.ledger.entries), "ledger_tail": self.ledger.tail_hash(),
        }))

    def _load(self) -> None:
        ck_path = self.run_dir / "checkpoint.json"
        if not ck_path.exists():
            (self.run_dir / "ledger.jsonl").unlink(missing_ok=True)
            return
        ck = json.loads(ck_path.read_text(encoding="utf-8"))
        if ck.get("world") != self.world or ck.get("operators") != self.operators:
            raise RuntimeError(f"{self.run_dir} belongs to another federation; start with --reset")
        self.registry = {op: bytes.fromhex(k) for op, k in ck["registry"].items()}
        self.attested = ck["attested"]
        self.period = ck["period"]
        self.round_reports = ck["round_reports"]
        rep = ck["reputation"]
        self.reputation = ReputationState(scores=dict(rep["scores"]), rounds=rep["rounds"],
                                          history=list(rep["history"]))
        self.rounds = ck["rounds"]
        self.last_round = ck["last_round"]
        self.clock = ck["clock"]
        self.counters.update(ck["counters"])

        path = self.run_dir / "ledger.jsonl"
        from .jsonl import read_valid_lines
        lines = read_valid_lines(path)        # a torn tail from a killed process is dropped
        n = ck["ledger_entries"]
        if len(lines) > n:   # written after the last checkpoint: that work is redone
            lines = lines[:n]
            path.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
        ok = len(lines) == n
        for line in lines:
            rec = json.loads(line)
            entry = self.ledger.append(signed_from_line(rec), timestamp=rec["timestamp"])
            if entry.entry_hash != rec["entry_hash"] or entry.prev_hash != rec["prev_hash"]:
                ok = False
        chain_ok, _ = self.ledger.verify_chain(self.registry)
        tail = self.ledger.tail_hash() if self.ledger.entries else GENESIS_HASH
        self.ledger_verified = bool(ok and chain_ok and tail == ck["ledger_tail"])
