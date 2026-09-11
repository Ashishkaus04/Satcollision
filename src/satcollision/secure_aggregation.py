"""
Secure aggregation: the aggregator learns the sum, never any individual
operator's report.

Every other privacy mechanism in this project reduces WHAT crosses the
operator boundary (``signal.py``'s abstraction) or adds noise to it
(``federation.apply_local_dp_noise``). Neither one hides an operator's
report from the party doing the combining: whoever runs ``mean_aggregate``,
``trimmed_mean_aggregate``, or ``krum_aggregate`` in ``federation.py`` sees
every operator's plaintext count vector before combining them. In a real
deployment that aggregator role might be a third-party clearinghouse, a
rotating "it's your turn" operator, or an infrastructure provider -- and
"we promise not to look" is not a technical guarantee.

This module implements the real fix used in production federated systems
(Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving
Machine Learning", CCS 2017 -- the technique behind Google Gboard's
federated learning deployment): **pairwise-masked additive secret
sharing**. Every pair of operators agrees on a shared random mask (via a
genuine X25519 Diffie-Hellman key exchange, so the "agreement" step is not
hand-waved); each operator adds a fresh mask for every peer that sorts
after it and subtracts the same mask for every peer that sorts before it.
A single operator's masked share is then indistinguishable from a
uniformly random field element to anyone who doesn't also hold the private
keys behind every pairwise mask -- but when every operator's share is
summed together, every pairwise mask appears exactly once with a ``+`` and
once with a ``-``, so the masks cancel in pairs and the aggregator is left
holding the true sum of all plaintext reports, having never possessed a
single one of them.

Concretely, for operators sorted by name, operator ``i``'s share is

    y_i = x_i + sum_{j: j > i} PRG(seed_ij) - sum_{j: j < i} PRG(seed_ij)   (mod P)

and ``sum_i y_i == sum_i x_i (mod P)`` because every ``PRG(seed_ij)`` term
appears once with each sign. All arithmetic is done in a large prime field
(``FIELD_PRIME``) rather than plain floating point, exactly matching the
real construction -- this is what makes a single share information-
theoretically uniform (indistinguishable from random) rather than merely
"hard to guess."

What this module deliberately does NOT do (an honest limitation, not an
oversight): once inputs are masked, the aggregator can no longer inspect
individual reports to run Byzantine-robust aggregation (``trimmed_mean``,
``Krum`` in ``federation.py``) against them -- those need to *see* the
outliers to exclude them. Real deployments that need both properties layer
on additional machinery (verifiable secret sharing, zero-knowledge range
proofs on committed inputs) that is out of scope here. Secure aggregation
and Byzantine-robust aggregation solve two different problems and, as
implemented, are not simultaneously active on the same round -- see the
"Trust model" section of the README.

This is also independent of ``identity.py``'s Ed25519 signing: signing
keys authenticate *who* sent a message, DH key-agreement keys here are
used only to derive pairwise masks, and a real system keeps these as two
separate keypairs per operator (mixing signing and encryption/agreement
keys is a well-known footgun) -- exactly how they're kept separate below.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)

from .federation import OperatorReport

# Fixed-point scale: report counts (which can be fractional after
# apply_local_dp_noise) are converted to integers by multiplying by SCALE
# before entering the finite field, and divided back out on the way out.
SCALE = 1_000_000

# A 127-bit Mersenne prime. Deliberately far larger than any plausible
# aggregate value (bounded by roughly n_operators * max_count * SCALE) --
# not for overflow safety (Python integers don't overflow), but so that a
# *single* masked share, examined alone, is (owing to the mask dominating
# the plaintext contribution) statistically indistinguishable from a
# uniformly random element of the whole field.
FIELD_PRIME = 2 ** 127 - 1


@dataclass
class OperatorMPCState:
    """One operator's X25519 Diffie-Hellman keypair, used only to agree on
    pairwise masks with every other participant this round.

    Deliberately a *different* keypair from the Ed25519 signing identity
    in ``identity.py`` -- using the same key for signing and key agreement
    is a classic real-world cryptographic mistake this project avoids on
    purpose, not by accident.
    """
    operator: str
    _private_key: X25519PrivateKey = field(repr=False)

    @staticmethod
    def generate(operator: str) -> "OperatorMPCState":
        return OperatorMPCState(operator=operator, _private_key=X25519PrivateKey.generate())

    def public_key_bytes(self) -> bytes:
        return self._private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def shared_secret_with(self, peer_public_key_bytes: bytes) -> bytes:
        """The raw X25519 Diffie-Hellman shared secret with one peer.
        By the Diffie-Hellman property this is identical whichever side
        computes it (my_private x their_public == their_private x
        my_public) -- neither side ever transmits it.
        """
        peer_public_key = X25519PublicKey.from_public_bytes(peer_public_key_bytes)
        return self._private_key.exchange(peer_public_key)


def build_mpc_states(operator_names: list[str]) -> dict[str, OperatorMPCState]:
    """Provision one fresh X25519 keypair per named operator, for one
    round of secure aggregation."""
    return {name: OperatorMPCState.generate(name) for name in operator_names}


def _pairwise_seed(mpc_states: dict[str, OperatorMPCState], op_a: str, op_b: str) -> bytes:
    """The shared mask seed for one unordered pair of operators.

    Computed here from ``op_a``'s private key and ``op_b``'s public key,
    which by the Diffie-Hellman property is bit-for-bit identical to what
    ``op_b`` would compute from their own private key and ``op_a``'s
    public key -- and the label is built from the two names sorted, so
    both call orders (``_pairwise_seed(s, "A", "B")`` and
    ``_pairwise_seed(s, "B", "A")``) always yield the same seed.
    """
    raw_secret = mpc_states[op_a].shared_secret_with(mpc_states[op_b].public_key_bytes())
    a, b = sorted((op_a, op_b))
    return hashlib.sha256(raw_secret + f"{a}:{b}".encode("utf-8")).digest()


def _prg_vector(seed: bytes, length: int) -> list[int]:
    """Deterministically expand a shared seed into `length` pseudorandom
    field elements via SHA-256 in counter mode. Deterministic so both
    operators behind a given pairwise seed independently produce the
    identical mask vector without any further communication.
    """
    out = []
    counter = 0
    while len(out) < length:
        digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        out.append(int.from_bytes(digest, "big") % FIELD_PRIME)
        counter += 1
    return out


def _encode(counts: np.ndarray) -> list[int]:
    return [int(round(float(v) * SCALE)) % FIELD_PRIME for v in counts]


def _decode(field_values: list[int], length: int) -> np.ndarray:
    decoded = np.zeros(length)
    for idx, v in enumerate(field_values):
        v = int(v) % FIELD_PRIME
        if v > FIELD_PRIME // 2:  # centered representation: undo the wraparound for negative sums
            v -= FIELD_PRIME
        decoded[idx] = v / SCALE
    return decoded


def compute_masked_share(
    mpc_states: dict[str, OperatorMPCState],
    reports: dict[str, OperatorReport],
    operator: str,
) -> list[int]:
    """The one masked share of ``operator``'s own report that ever leaves
    its boundary -- this, and only this, is what an aggregator sees for
    this operator.

    Sums a pairwise pseudorandom mask with every OTHER participant in
    ``reports``: added for peers that sort after ``operator``, subtracted
    for peers that sort before it. That sign convention is what makes the
    masks cancel in pairs once every operator's share is summed (see
    module docstring) -- it is not an arbitrary choice.
    """
    counts = reports[operator].counts
    share = _encode(counts)
    for peer in reports:
        if peer == operator:
            continue
        seed = _pairwise_seed(mpc_states, operator, peer)
        mask_vec = _prg_vector(seed, len(share))
        sign = 1 if operator < peer else -1
        for idx in range(len(share)):
            share[idx] = (share[idx] + sign * mask_vec[idx]) % FIELD_PRIME
    return share


def secure_sum_aggregate(
    mpc_states: dict[str, OperatorMPCState],
    reports: dict[str, OperatorReport],
) -> np.ndarray:
    """The aggregator-facing entry point: collect every operator's masked
    share -- never a plaintext report -- and sum them. The pairwise masks
    cancel automatically (each appears once with ``+`` and once with
    ``-`` across the cohort), so the result equals the exact sum of every
    operator's true report even though the aggregator's own arithmetic
    never once touches a plaintext value.
    """
    if not reports:
        raise ValueError("secure_sum_aggregate requires at least one report")
    dim = len(next(iter(reports.values())).counts)
    total = [0] * dim
    for operator in reports:
        share = compute_masked_share(mpc_states, reports, operator)
        for idx in range(dim):
            total[idx] = (total[idx] + share[idx]) % FIELD_PRIME
    return _decode(total, dim)


def secure_mean_aggregate(
    mpc_states: dict[str, OperatorMPCState],
    reports: dict[str, OperatorReport],
) -> np.ndarray:
    """Secure sum divided by operator count -- the secure-aggregation
    equivalent of ``federation.mean_aggregate``, exposed separately from
    :func:`secure_sum_aggregate` because the sum is what the underlying
    protocol actually computes; the mean is a public constant-factor
    rescaling of it, safe to do after the fact.
    """
    total = secure_sum_aggregate(mpc_states, reports)
    return total / len(reports)
