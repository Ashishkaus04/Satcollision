import dataclasses

import numpy as np
import pytest

from satcollision.federation import OperatorReport
from satcollision.signal import AbstractedSignal
from satcollision.identity import (
    OperatorIdentity,
    build_operator_identities,
    public_key_registry,
    sign_signal,
    sign_report,
    verify_signed,
    SignedLog,
)


def _signal(operator="Alpha", encounter_id="ENC-1-2-100"):
    return AbstractedSignal(
        encounter_id=encounter_id,
        sender_operator=operator,
        local_object_label=f"{operator}-1",
        tca_offset_s=100.0,
        geometry_class="crossing",
        severity_tier="elevated",
    )


def _report(operator="Alpha"):
    return OperatorReport(operator=operator, counts=np.array([2.0, 1.0, 0.0]))


def test_sign_and_verify_roundtrip_signal():
    identities = build_operator_identities(["Alpha", "Beta"])
    registry = public_key_registry(identities)

    signed = sign_signal(identities["Alpha"], _signal("Alpha"))
    assert verify_signed(signed, registry) is True


def test_sign_and_verify_roundtrip_report():
    identities = build_operator_identities(["Alpha", "Beta"])
    registry = public_key_registry(identities)

    signed = sign_report(identities["Alpha"], _report("Alpha"))
    assert verify_signed(signed, registry) is True


def test_sign_signal_refuses_to_sign_on_behalf_of_another_operator():
    identities = build_operator_identities(["Alpha", "Beta"])
    with pytest.raises(ValueError):
        sign_signal(identities["Alpha"], _signal("Beta"))


def test_verify_fails_when_signed_content_is_tampered():
    identities = build_operator_identities(["Alpha", "Beta"])
    registry = public_key_registry(identities)

    signed = sign_signal(identities["Alpha"], _signal("Alpha"))
    # Swap in a signal with different content but keep the original
    # signature/public_key -- simulates a man-in-the-middle editing the
    # payload after it was signed.
    tampered_signal = dataclasses.replace(signed.signal, tca_offset_s=999.0)
    tampered = dataclasses.replace(signed, signal=tampered_signal)
    assert verify_signed(tampered, registry) is False


def test_verify_fails_for_impersonation_with_attackers_own_key():
    identities = build_operator_identities(["Alpha", "Beta"])
    registry = public_key_registry(identities)

    # An attacker with no legitimate identity for "Alpha" signs a signal
    # claiming to be Alpha, using their own (different) keypair.
    attacker_identity = OperatorIdentity.generate("Alpha")
    forged = sign_signal(attacker_identity, _signal("Alpha"))

    assert verify_signed(forged, registry) is False


def test_verify_fails_for_unknown_sender():
    identities = build_operator_identities(["Alpha"])
    registry = public_key_registry(identities)
    signed = sign_signal(identities["Alpha"], _signal("Alpha"))

    # Verifier only trusts a registry that doesn't mention Alpha at all.
    assert verify_signed(signed, {}) is False


def test_hash_chain_verifies_when_untampered():
    identities = build_operator_identities(["Alpha", "Beta"])
    registry = public_key_registry(identities)
    log = SignedLog()
    log.append(sign_signal(identities["Alpha"], _signal("Alpha", "ENC-1")))
    log.append(sign_signal(identities["Beta"], _signal("Beta", "ENC-2")))
    log.append(sign_report(identities["Alpha"], _report("Alpha")))

    ok, reason = log.verify_chain(registry)
    assert ok is True
    assert reason is None


def test_hash_chain_detects_content_tampering():
    identities = build_operator_identities(["Alpha", "Beta"])
    registry = public_key_registry(identities)
    log = SignedLog()
    log.append(sign_signal(identities["Alpha"], _signal("Alpha", "ENC-1")))
    log.append(sign_signal(identities["Beta"], _signal("Beta", "ENC-2")))

    # Mutate the frozen entry in place to simulate someone editing history
    # after the fact -- the stored entry_hash no longer matches.
    entries = log.entries
    tampered_signal = dataclasses.replace(entries[0].signed.signal, geometry_class="head-on")
    tampered_signed = dataclasses.replace(entries[0].signed, signal=tampered_signal)
    object.__setattr__(log._entries[0], "signed", tampered_signed)

    ok, reason = log.verify_chain(registry)
    assert ok is False
    assert "entry 0" in reason


def test_hash_chain_detects_reordering():
    identities = build_operator_identities(["Alpha", "Beta"])
    registry = public_key_registry(identities)
    log = SignedLog()
    log.append(sign_signal(identities["Alpha"], _signal("Alpha", "ENC-1")))
    log.append(sign_signal(identities["Beta"], _signal("Beta", "ENC-2")))
    log.append(sign_report(identities["Alpha"], _report("Alpha")))

    log._entries[0], log._entries[1] = log._entries[1], log._entries[0]

    ok, reason = log.verify_chain(registry)
    assert ok is False


def test_hash_chain_detects_deletion():
    identities = build_operator_identities(["Alpha", "Beta"])
    registry = public_key_registry(identities)
    log = SignedLog()
    log.append(sign_signal(identities["Alpha"], _signal("Alpha", "ENC-1")))
    log.append(sign_signal(identities["Beta"], _signal("Beta", "ENC-2")))
    log.append(sign_report(identities["Alpha"], _report("Alpha")))

    del log._entries[1]

    ok, reason = log.verify_chain(registry)
    assert ok is False


def test_tail_hash_changes_as_entries_are_appended():
    identities = build_operator_identities(["Alpha"])
    log = SignedLog()
    genesis = log.tail_hash()
    log.append(sign_signal(identities["Alpha"], _signal("Alpha", "ENC-1")))
    after_one = log.tail_hash()
    assert after_one != genesis
    log.append(sign_signal(identities["Alpha"], _signal("Alpha", "ENC-2")))
    after_two = log.tail_hash()
    assert after_two != after_one
