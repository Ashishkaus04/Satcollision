import numpy as np

from satcollision.federation import OperatorReport
from satcollision.identity import OperatorIdentity, sign_report, sign_signal, verify_signed
from satcollision.signal import AbstractedSignal
from satcollision.wire import (
    decode_signed_report,
    decode_signed_signal,
    encode_ready,
    encode_signed_report,
    encode_signed_signal,
)


def test_signed_signal_survives_wire_round_trip():
    identity = OperatorIdentity.generate("Alpha")
    signal = AbstractedSignal(
        encounter_id="ENC-1-2-100", sender_operator="Alpha", local_object_label="Alpha-1",
        tca_offset_s=100.0, geometry_class="crossing", severity_tier="elevated",
    )
    signed = sign_signal(identity, signal)

    wire_message = encode_signed_signal(signed)
    # every value must be JSON-safe (no bytes, no numpy scalars)
    import json
    json.dumps(wire_message)

    decoded = decode_signed_signal(wire_message)
    assert decoded.signal == signed.signal
    assert decoded.signature == signed.signature
    assert decoded.public_key == signed.public_key
    assert verify_signed(decoded, {"Alpha": identity.public_key_bytes()}) is True


def test_signed_report_survives_wire_round_trip():
    identity = OperatorIdentity.generate("Beta")
    report = OperatorReport(operator="Beta", counts=np.array([3.0, 1.0, 0.0]))
    signed = sign_report(identity, report)

    wire_message = encode_signed_report(signed)
    import json
    json.dumps(wire_message)

    decoded = decode_signed_report(wire_message)
    assert np.array_equal(decoded.report.counts, report.counts)
    assert decoded.report.operator == report.operator
    assert verify_signed(decoded, {"Beta": identity.public_key_bytes()}) is True


def test_wire_round_trip_detects_tampering():
    identity = OperatorIdentity.generate("Alpha")
    signal = AbstractedSignal(
        encounter_id="ENC-1-2-100", sender_operator="Alpha", local_object_label="Alpha-1",
        tca_offset_s=100.0, geometry_class="crossing", severity_tier="elevated",
    )
    signed = sign_signal(identity, signal)
    wire_message = encode_signed_signal(signed)

    wire_message["tca_offset_s"] = 999.0  # simulate corruption/tampering in transit
    decoded = decode_signed_signal(wire_message)
    assert verify_signed(decoded, {"Alpha": identity.public_key_bytes()}) is False


def test_encode_ready_is_json_safe():
    import json
    message = encode_ready("Alpha")
    json.dumps(message)
    assert message["type"] == "ready"
    assert message["operator"] == "Alpha"
