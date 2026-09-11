"""
Wire protocol: how a :class:`~satcollision.identity.SignedSignal` /
:class:`~satcollision.identity.SignedReport` actually gets serialized onto
a real TCP socket, for ``hub.py`` and ``operator_node.py``.

Two concerns, kept separate on purpose:

* **Framing** — TCP is a byte stream, not a message stream, so every
  message is sent as a 4-byte big-endian length prefix followed by that
  many bytes of UTF-8 JSON (:func:`send_message`/:func:`read_message`).
  This is the same length-prefixing every real line protocol needs; the
  point of building it here instead of reaching for a higher-level RPC
  library is that the framing itself should be visible and auditable.
* **Serialization** — a :class:`~satcollision.signal.AbstractedSignal` /
  :class:`~satcollision.federation.OperatorReport` and their signatures
  are plain dataclasses and raw bytes; JSON can carry neither directly,
  so :func:`encode_signed_signal`/:func:`decode_signed_signal` (and the
  ``_report`` counterparts) convert bytes fields to/from base64 and numpy
  arrays to/from plain lists. Nothing here re-derives or checks a
  signature -- that stays entirely inside ``identity.verify_signed``,
  called by whoever receives a decoded message. The wire format's only
  job is getting the exact same bytes that were signed to the other side
  intact; trusting them is a decision made one layer up.
"""

from __future__ import annotations

import asyncio
import base64
import json

import numpy as np

from .federation import OperatorReport
from .identity import SignedReport, SignedSignal
from .signal import AbstractedSignal

_LENGTH_PREFIX_BYTES = 4
MAX_MESSAGE_BYTES = 1_000_000  # generous cap; real messages here are a few hundred bytes


async def send_message(writer: asyncio.StreamWriter, message: dict) -> None:
    payload = json.dumps(message).encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"message of {len(payload)} bytes exceeds MAX_MESSAGE_BYTES")
    writer.write(len(payload).to_bytes(_LENGTH_PREFIX_BYTES, "big"))
    writer.write(payload)
    await writer.drain()


async def read_message(reader: asyncio.StreamReader) -> dict | None:
    """Read one length-prefixed JSON message, or ``None`` on a clean
    end-of-stream (the peer closed the connection)."""
    try:
        length_prefix = await reader.readexactly(_LENGTH_PREFIX_BYTES)
    except asyncio.IncompleteReadError:
        return None
    length = int.from_bytes(length_prefix, "big")
    if length > MAX_MESSAGE_BYTES:
        raise ValueError(f"peer announced a {length}-byte message, exceeding MAX_MESSAGE_BYTES")
    try:
        payload = await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        return None
    return json.loads(payload.decode("utf-8"))


def _b64encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64decode(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def encode_signed_signal(signed: SignedSignal) -> dict:
    signal = signed.signal
    return {
        "type": "signal",
        "encounter_id": signal.encounter_id,
        "sender_operator": signal.sender_operator,
        "local_object_label": signal.local_object_label,
        "tca_offset_s": signal.tca_offset_s,
        "geometry_class": signal.geometry_class,
        "severity_tier": signal.severity_tier,
        "threshold_crossed": signal.threshold_crossed,
        "signature": _b64encode(signed.signature),
        "public_key": _b64encode(signed.public_key),
    }


def decode_signed_signal(message: dict) -> SignedSignal:
    signal = AbstractedSignal(
        encounter_id=message["encounter_id"],
        sender_operator=message["sender_operator"],
        local_object_label=message["local_object_label"],
        tca_offset_s=message["tca_offset_s"],
        geometry_class=message["geometry_class"],
        severity_tier=message["severity_tier"],
        threshold_crossed=message["threshold_crossed"],
    )
    return SignedSignal(
        signal=signal,
        signature=_b64decode(message["signature"]),
        public_key=_b64decode(message["public_key"]),
    )


def encode_signed_report(signed: SignedReport) -> dict:
    report = signed.report
    return {
        "type": "report",
        "operator": report.operator,
        "counts": list(report.counts.tolist()),
        "is_adversarial": report.is_adversarial,
        "signature": _b64encode(signed.signature),
        "public_key": _b64encode(signed.public_key),
    }


def decode_signed_report(message: dict) -> SignedReport:
    report = OperatorReport(
        operator=message["operator"],
        counts=np.array(message["counts"], dtype=float),
        is_adversarial=message["is_adversarial"],
    )
    return SignedReport(
        report=report,
        signature=_b64decode(message["signature"]),
        public_key=_b64decode(message["public_key"]),
    )


def encode_ready(operator: str) -> dict:
    """A liveness/rendezvous ping -- carries no signature and proves
    nothing about content; it exists only so a node can tell its peers
    "I'm connected and reading" before anyone sends a real signed
    message, closing the startup race where an early sender's broadcast
    would be lost to a not-yet-connected peer (the hub never replays
    history to a late joiner)."""
    return {"type": "ready", "operator": operator}
