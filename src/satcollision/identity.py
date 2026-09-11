"""
Cryptographic identity, signing, and tamper-evident logging layer.

Signal abstraction (``signal.py``) guarantees *what* crosses the operator
boundary is safe to share. It says nothing about *who* actually sent it, or
whether what arrives is what was actually sent. This layer adds both, with
real primitives rather than a placeholder:

* **Real Ed25519 keypairs** (via the ``cryptography`` library), one per
  operator, generated once (:func:`OperatorIdentity.generate`) and reused
  for the life of the federation -- exactly how a real deployment would
  provision identity (a PKI/directory handing out public keys, private
  keys never leaving each operator's own process).
* **Signing and verification** (:func:`sign_signal`, :func:`sign_report`,
  :func:`verify_signed`) wrap an :class:`~satcollision.signal.AbstractedSignal`
  or an :class:`~satcollision.federation.OperatorReport` in a genuine
  digital signature over a canonical byte encoding of its fields. This buys
  two properties neither the signal-abstraction layer nor the federation
  layer has on its own:
    - **Authenticity** -- a receiver can verify a message actually came
      from the operator it claims to (checked against a registry of
      *known* public keys the verifier already trusts, never against a key
      bundled with the message itself -- otherwise an attacker could just
      attach their own key and "self-verify").
    - **Integrity** -- if a single bit of the signed content changes after
      signing (in transit, by a bug, or by a man-in-the-middle), the
      signature fails to verify.
* **A hash-chained, append-only log** (:class:`SignedLog`) -- the
  blockchain-style integrity structure, minus the parts (consensus, mining,
  a shared ledger replicated across mutually untrusted nodes) this project
  has no need for, since every entry here is already individually signed
  by its origin operator. What hash-chaining adds *on top of* individual
  signatures is ordering and completeness: one signature proves one
  message is authentic and unaltered, but says nothing about whether a
  past entry was later deleted, reordered, or had a fabricated entry
  spliced in. Chaining each entry's hash into the next (:func:`_hash_entry`)
  makes any such change detectable by :meth:`SignedLog.verify_chain`, by
  any party who kept even just the latest hash (:meth:`SignedLog.tail_hash`).

This is a layer *underneath* the existing ones -- ``AbstractedSignal`` and
``OperatorReport`` are unchanged, and ``assert_no_raw_state`` still applies
to whatever gets signed. Nothing here changes what data crosses the
boundary; it only adds authenticity and tamper-evidence to the fact that it
crossed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Union

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .federation import OperatorReport
from .signal import AbstractedSignal, assert_no_raw_state


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"cannot canonicalize object of type {type(obj)!r} for signing")


def _canonical_bytes(obj) -> bytes:
    """Deterministic byte encoding of a dataclass's fields -- the exact
    message that gets signed and, later, re-derived and checked at verify
    time. Sorted-key JSON over the dataclass's own field values (never
    ``repr``/``pickle``, which are not guaranteed stable across processes
    or Python versions) so two independent parties always agree on what
    bytes a given signature covers.
    """
    payload = dataclasses.asdict(obj)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")


@dataclass(frozen=True)
class OperatorIdentity:
    """One operator's real Ed25519 keypair.

    In a genuine deployment ``_private_key`` never leaves the operator's
    own process -- every other party only ever receives
    :meth:`public_key_bytes`, mirroring how real key distribution works
    (a public directory / PKI). Nothing in this module hands the private
    key anywhere else.
    """
    operator: str
    _private_key: Ed25519PrivateKey = field(repr=False)

    @staticmethod
    def generate(operator: str) -> "OperatorIdentity":
        return OperatorIdentity(operator=operator, _private_key=Ed25519PrivateKey.generate())

    @staticmethod
    def from_private_key_bytes(operator: str, raw_bytes: bytes) -> "OperatorIdentity":
        """Reconstruct an identity from its 32-byte raw private key --
        the counterpart to :meth:`private_key_bytes`, used when an
        operator's identity needs to survive being handed to a genuinely
        separate OS process (see ``operator_node.py``) rather than living
        only inside one Python object for the process's lifetime. Loading
        this from a file on disk is itself a simplification stated
        plainly here: a real deployment would pull this from a hardware
        key store or a secrets manager, never a plain file -- this
        project's distributed demo uses a file only because it is
        already running as separate OS processes on one machine.
        """
        return OperatorIdentity(operator=operator, _private_key=Ed25519PrivateKey.from_private_bytes(raw_bytes))

    def private_key_bytes(self) -> bytes:
        """The 32 raw private key bytes -- deliberately named and
        documented as loudly as possible, since exposing this at all is
        already a departure from "the private key never leaves the
        operator's own process" (see class docstring). Exists only so a
        genuinely separate OS process can be handed this operator's
        identity across a process boundary; nothing in this codebase
        sends these bytes over the network.
        """
        return self._private_key.private_bytes_raw()

    def public_key_bytes(self) -> bytes:
        return self._private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )


def build_operator_identities(operator_names: list[str]) -> dict[str, OperatorIdentity]:
    """Provision one fresh real keypair per named operator."""
    return {name: OperatorIdentity.generate(name) for name in operator_names}


def public_key_registry(identities: dict[str, OperatorIdentity]) -> dict[str, bytes]:
    """The registry every verifier checks incoming signatures against --
    the trusted-key directory a real PKI would hand out, containing only
    public keys."""
    return {name: identity.public_key_bytes() for name, identity in identities.items()}


@dataclass(frozen=True)
class SignedSignal:
    """A real Ed25519 signature bound to one immutable
    :class:`~satcollision.signal.AbstractedSignal`."""
    signal: AbstractedSignal
    signature: bytes
    public_key: bytes

    @property
    def sender_operator(self) -> str:
        return self.signal.sender_operator

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.signal)


@dataclass(frozen=True)
class SignedReport:
    """A real Ed25519 signature bound to one immutable
    :class:`~satcollision.federation.OperatorReport`."""
    report: OperatorReport
    signature: bytes
    public_key: bytes

    @property
    def sender_operator(self) -> str:
        return self.report.operator

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.report)


Signed = Union[SignedSignal, SignedReport]


def sign_signal(identity: OperatorIdentity, signal: AbstractedSignal) -> SignedSignal:
    """Sign one outgoing AbstractedSignal with an operator's own key.

    Refuses to sign a signal claiming a different ``sender_operator`` --
    an operator may only ever sign its own outgoing traffic, never
    manufacture a signature on someone else's behalf.
    """
    if signal.sender_operator != identity.operator:
        raise ValueError(
            f"identity for '{identity.operator}' cannot sign a signal claiming "
            f"sender_operator='{signal.sender_operator}' -- an operator may only "
            "sign its own outgoing signals."
        )
    assert_no_raw_state(signal)
    content = _canonical_bytes(signal)
    signature = identity._private_key.sign(content)
    return SignedSignal(signal=signal, signature=signature, public_key=identity.public_key_bytes())


def sign_report(identity: OperatorIdentity, report: OperatorReport) -> SignedReport:
    """Sign one outgoing federation-period OperatorReport with an
    operator's own key. Same own-traffic-only restriction as
    :func:`sign_signal`."""
    if report.operator != identity.operator:
        raise ValueError(
            f"identity for '{identity.operator}' cannot sign a report claiming "
            f"operator='{report.operator}' -- an operator may only sign its own "
            "outgoing reports."
        )
    content = _canonical_bytes(report)
    signature = identity._private_key.sign(content)
    return SignedReport(report=report, signature=signature, public_key=identity.public_key_bytes())


def verify_signed(signed: Signed, trusted_public_keys: dict[str, bytes]) -> bool:
    """Verify a SignedSignal or SignedReport against a registry of known-
    good public keys keyed by operator name.

    Deliberately checked against ``trusted_public_keys`` -- never against
    ``signed.public_key``, which travels with the message and is entirely
    attacker-controlled data; trusting it would let an attacker forge a
    message for any operator simply by attaching a key of their own
    choosing. Verification fails if:

    * the claimed ``sender_operator`` has no entry in
      ``trusted_public_keys`` (unknown operator), or
    * the trusted key on file for that operator does not match the key
      that actually produced the signature (impersonation attempt using a
      key that isn't the real one on file), or
    * the signature does not verify against the message's exact canonical
      content (tampering with any field after signing).
    """
    trusted_key_bytes = trusted_public_keys.get(signed.sender_operator)
    if trusted_key_bytes is None:
        return False
    if trusted_key_bytes != signed.public_key:
        return False
    try:
        public_key = Ed25519PublicKey.from_public_bytes(trusted_key_bytes)
        public_key.verify(signed.signature, signed.canonical_bytes())
        return True
    except InvalidSignature:
        return False


GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class LogEntry:
    index: int
    timestamp: float
    signed: object  # SignedSignal | SignedReport
    prev_hash: str
    entry_hash: str


def _hash_entry(index: int, timestamp: float, signed: Signed, prev_hash: str) -> str:
    """Every byte that must stay unchanged for this entry to still be
    "the same entry" goes into the hash: its position, when it was logged,
    who sent it and what they signed, and the link to the entry before it.
    Changing any of these -- including splicing in a different prior
    entry -- produces a different hash.
    """
    material = (
        str(index).encode("utf-8")
        + repr(timestamp).encode("utf-8")
        + prev_hash.encode("utf-8")
        + signed.sender_operator.encode("utf-8")
        + signed.canonical_bytes()
        + signed.signature
        + signed.public_key
    )
    return hashlib.sha256(material).hexdigest()


class SignedLog:
    """Append-only, hash-chained log of signed signals/reports exchanged in
    the federation.

    A single signature (see :func:`verify_signed`) already proves one
    message is authentic and unaltered. What it cannot do alone is prove
    something about the *sequence* -- that no message was later deleted,
    that two messages weren't silently swapped, that a fabricated entry
    wasn't inserted into the recorded history. Hash-chaining closes that
    gap: each entry's hash commits to the previous entry's hash plus its
    own signed content, so :meth:`verify_chain` can detect any such change
    by simply re-deriving every hash from scratch and confirming the chain
    still links up -- exactly the tamper-evidence property real audit logs
    and (minus the consensus machinery) blockchains rely on.
    """

    def __init__(self):
        self._entries: list[LogEntry] = []

    def append(self, signed: Signed, timestamp: float | None = None) -> LogEntry:
        index = len(self._entries)
        prev_hash = self._entries[-1].entry_hash if self._entries else GENESIS_HASH
        ts = time.time() if timestamp is None else timestamp
        entry_hash = _hash_entry(index, ts, signed, prev_hash)
        entry = LogEntry(index=index, timestamp=ts, signed=signed, prev_hash=prev_hash, entry_hash=entry_hash)
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> list[LogEntry]:
        return list(self._entries)

    def tail_hash(self) -> str:
        """The single value any party can independently checkpoint -- one
        hash that commits to the entire history so far. Anyone holding it
        can later detect if anything before it was altered, without
        needing to keep the whole log themselves."""
        return self._entries[-1].entry_hash if self._entries else GENESIS_HASH

    def verify_chain(self, trusted_public_keys: dict[str, bytes]) -> tuple[bool, str | None]:
        """Re-derive every entry's hash and re-check every signature from
        scratch, in order. Returns ``(True, None)`` if the whole chain is
        intact, or ``(False, reason)`` naming the first problem found:
        a broken hash link (an entry was deleted, reordered, or a
        fabricated entry was spliced in), a stored hash that no longer
        matches its recomputed value (an entry's content was edited after
        being logged), or a signature that no longer verifies (unknown
        signer, or signed content tampered independently of the log
        itself).
        """
        expected_prev = GENESIS_HASH
        for expected_index, entry in enumerate(self._entries):
            if entry.index != expected_index:
                return False, f"entry at position {expected_index}: stored index is {entry.index} -- entries were reordered"
            if entry.prev_hash != expected_prev:
                return False, f"entry {entry.index}: prev_hash does not match the preceding entry's hash -- chain link broken"
            recomputed = _hash_entry(entry.index, entry.timestamp, entry.signed, entry.prev_hash)
            if recomputed != entry.entry_hash:
                return False, f"entry {entry.index}: stored hash does not match recomputed hash -- entry content was modified after logging"
            if not verify_signed(entry.signed, trusted_public_keys):
                return False, f"entry {entry.index}: signature invalid -- unknown signer or signed content tampered"
            expected_prev = entry.entry_hash
        return True, None
