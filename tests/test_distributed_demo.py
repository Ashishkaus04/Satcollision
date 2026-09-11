"""
Integration test for the real distributed demo: actually spawns the hub
and operator processes described in satcollision.distributed_demo (real
subprocesses, real TCP sockets on localhost) and checks the security
property end to end. Slower than the rest of the suite (spawns 5 real OS
processes) and touches the network stack (loopback only) -- kept in its
own file so it's easy to skip in an environment where that's unwelcome.
"""

from satcollision.distributed_demo import ATTACKER_TARGET, LEGIT_OPERATORS, run_demo, summarize

# No test-level timeout marker: run_demo() already bounds itself via each
# subprocess's own communicate(timeout=...) calls (25s per operator, 10s
# for the hub), so a hang inside the demo can't hang the test suite.


def test_distributed_demo_end_to_end():
    result = run_demo()

    assert set(result["reports"].keys()) == set(LEGIT_OPERATORS) | {"ATTACKER"}

    for name in LEGIT_OPERATORS:
        report = result["reports"][name]
        assert report["quota_met"] is True, f"{name} did not receive its expected peer signals in time"
        assert report["verify_chain_ok"] is True, f"{name}'s local tamper-evident log failed to verify"
        assert report["log_entries"] == len(LEGIT_OPERATORS) - 1  # one verified signal per peer

    # The central security claim: every peer other than the impersonated
    # operator itself must have rejected the attacker's forged message.
    for name in LEGIT_OPERATORS:
        if name == ATTACKER_TARGET:
            continue
        report = result["reports"][name]
        assert report["rejected_count"] >= 1, (
            f"{name} should have rejected the attacker's forged "
            f"'{ATTACKER_TARGET}' message, signed with a key that isn't the real "
            f"{ATTACKER_TARGET}'s"
        )
        # and the forged message must not have been counted as a genuine
        # verified signal from the real ATTACKER_TARGET
        assert report["verified_from_peer"].get(ATTACKER_TARGET, 0) == 1  # only the real one

    all_ok, _lines = summarize(result)
    assert all_ok is True

    for name in LEGIT_OPERATORS:
        assert result["outputs"][name]["returncode"] == 0
    assert result["outputs"]["hub"]["returncode"] == 0
