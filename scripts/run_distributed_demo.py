#!/usr/bin/env python3
"""
Real distributed demo: each operator (Alpha, Beta, Gamma) as its own OS
process, exchanging real Ed25519-signed conjunction signals with a fourth
process (a federation hub) over actual TCP sockets on localhost -- plus a
fifth process, an attacker impersonating "Alpha," to prove the signature
verification from identity.py still catches a forgery when it has to
survive a genuine process/network boundary instead of one shared script.

The orchestration logic lives in ``satcollision.distributed_demo`` (see
that module's docstring for the full picture); this script just runs it
and prints a human-readable report.

Usage:
    PYTHONPATH=src python3 scripts/run_distributed_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from satcollision.distributed_demo import (  # noqa: E402
    ATTACKER_TARGET, LEGIT_OPERATORS, run_demo, summarize,
)


def main() -> int:
    result = run_demo()

    print("=" * 78)
    print("REAL DISTRIBUTED DEMO: separate OS processes, real TCP sockets, real Ed25519 signing")
    print("=" * 78)
    for name in [*LEGIT_OPERATORS, "ATTACKER", "hub"]:
        out = result["outputs"].get(name)
        if out:
            print(f"\n----- {name} (exit code {out['returncode']}) -----")
            print(out["stdout"].strip())

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    all_ok, lines = summarize(result)
    print("\n".join(lines))
    print(f"\n{'PASS' if all_ok else 'FAIL'}: every legitimate operator received and verified its real "
          f"peers' genuine signals over the real network, and rejected the attacker's forged "
          f"'{ATTACKER_TARGET}' message signed with a key that isn't the real {ATTACKER_TARGET}'s.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
