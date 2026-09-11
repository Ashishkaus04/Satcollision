"""
Orchestrates the real distributed demo: spawns the federation hub, three
legitimate operators, and one attacker as genuinely separate OS processes
(``subprocess.Popen``, real ``python3`` interpreters, not threads or
asyncio tasks sharing one process), wires them together over real TCP
sockets, waits for them to finish, and returns a structured result.

This module holds the orchestration *logic*; ``scripts/run_distributed_demo.py``
is the thin CLI wrapper that calls :func:`run_demo` and prints a report --
kept separate, matching the rest of this project's convention, so
``run_demo`` is directly importable and testable (see
``tests/test_distributed_demo.py``) without scraping a script's stdout.

Every earlier layer in this project (signing, secure aggregation, Kalman
filtering, federated learning) runs inside a single Python process calling
plain functions on shared objects. This is the one place that's actually
architecturally distributed: the hub (``hub.py``) and every operator
(``operator_node.py``) run as independent OS processes exchanging real
bytes over ``127.0.0.1``, with the same wire framing (``wire.py``) and
signature verification (``identity.py``) a genuine multi-organization
deployment across separate machines would use.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .identity import OperatorIdentity

LEGIT_OPERATORS = ["Alpha", "Beta", "Gamma"]
N_SIGNALS_PER_OPERATOR = 1
ATTACKER_TARGET = "Alpha"  # the legitimate operator the attacker impersonates
ATTACKER_N_SIGNALS = 1
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_PATH = str(_PROJECT_ROOT / "src")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, attempts: int = 60, delay: float = 0.25) -> None:
    for _ in range(attempts):
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(delay)
    raise RuntimeError(f"hub never opened {host}:{port}")


def run_demo(work_dir: Path | None = None) -> dict:
    """Run the full distributed demo once and return a structured result:
    ``{"outputs": {name: {"returncode", "stdout"}}, "reports": {name: dict},
    "expected_messages": int}``. Each legitimate operator's report dict
    (see ``operator_node.run_legitimate_operator``) is exactly what that
    process wrote to its own ``--report-file`` -- this function reads
    those files rather than parsing anyone's stdout for the numbers that
    matter.
    """
    own_tmp = work_dir is None
    tmp_path = Path(tempfile.mkdtemp(prefix="satcollision_dist_")) if own_tmp else work_dir
    tmp_path.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "PYTHONPATH": _SRC_PATH}

    # ---- Provision real keypairs and a pre-distributed public registry ----
    # This is the honest bootstrap step: in a real deployment this comes
    # from a PKI / certificate authority, established out-of-band before
    # any of this runs -- not negotiated over the same socket the
    # operators later talk on (see operator_node.py's module docstring).
    identities = {name: OperatorIdentity.generate(name) for name in LEGIT_OPERATORS}
    registry = {name: base64.b64encode(ident.public_key_bytes()).decode("ascii")
                for name, ident in identities.items()}
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(json.dumps(registry))

    key_files = {}
    for name, ident in identities.items():
        key_file = tmp_path / f"{name}.key"
        key_file.write_bytes(ident.private_key_bytes())
        key_files[name] = key_file

    port = _find_free_port()
    host = "127.0.0.1"

    # ---- Launch the hub as a real OS process ----
    expected_messages = len(LEGIT_OPERATORS) + len(LEGIT_OPERATORS) * N_SIGNALS_PER_OPERATOR + 1
    hub_proc = subprocess.Popen(
        [sys.executable, "-m", "satcollision.hub", "--host", host, "--port", str(port),
         "--expected-messages", str(expected_messages), "--max-duration", "30"],
        cwd=str(_PROJECT_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    _wait_for_port(host, port)

    # ---- Launch the three legitimate operators, each its own process ----
    # Each legitimate operator is told exactly how many inbound messages
    # to expect before it's safe to stop listening: one "ready" ping and
    # one real signal from each of its peers, plus the attacker's forged
    # message (broadcast to every connected client, this one included).
    # See operator_node.py's `--expected-inbound` for why this deterministic
    # count replaced an earlier fixed-delay "linger" -- under the CPU
    # contention of launching five real OS processes at once, a wall-clock
    # guess occasionally let a node exit before a slow-starting attacker's
    # message ever arrived, which made the demo's central security check
    # flaky without the guarantee itself actually being broken.
    peer_lists = {name: ",".join(p for p in LEGIT_OPERATORS if p != name) for name in LEGIT_OPERATORS}
    report_files = {name: tmp_path / f"{name}.report.json" for name in LEGIT_OPERATORS}
    expected_inbound = len(LEGIT_OPERATORS) - 1  # ready pings from peers
    expected_inbound += (len(LEGIT_OPERATORS) - 1) * N_SIGNALS_PER_OPERATOR  # real signals from peers
    expected_inbound += ATTACKER_N_SIGNALS  # the forged message, broadcast to everyone
    legit_procs = {}
    for i, name in enumerate(LEGIT_OPERATORS):
        legit_procs[name] = subprocess.Popen(
            [sys.executable, "-m", "satcollision.operator_node",
             "--name", name, "--host", host, "--port", str(port),
             "--private-key-file", str(key_files[name]), "--registry-file", str(registry_file),
             "--peers", peer_lists[name], "--n-signals", str(N_SIGNALS_PER_OPERATOR),
             "--seed", str(100 + i), "--timeout", "15",
             "--expected-inbound", str(expected_inbound), "--report-file", str(report_files[name])],
            cwd=str(_PROJECT_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

    # ---- Launch the attacker as its own process, impersonating ATTACKER_TARGET ----
    attacker_report_file = tmp_path / "attacker.report.json"
    attacker_proc = subprocess.Popen(
        [sys.executable, "-m", "satcollision.operator_node",
         "--name", ATTACKER_TARGET, "--host", host, "--port", str(port),
         "--attacker", "--n-signals", str(ATTACKER_N_SIGNALS), "--report-file", str(attacker_report_file)],
        cwd=str(_PROJECT_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    # ---- Wait for everyone, real process exit codes and all ----
    outputs = {}
    for name, proc in {**legit_procs, "ATTACKER": attacker_proc}.items():
        try:
            stdout, _ = proc.communicate(timeout=25)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()
        outputs[name] = {"returncode": proc.returncode, "stdout": stdout}

    try:
        hub_stdout, _ = hub_proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        hub_proc.kill()
        hub_stdout, _ = hub_proc.communicate()
    outputs["hub"] = {"returncode": hub_proc.returncode, "stdout": hub_stdout}

    reports = {}
    for name, report_file in {**report_files, "ATTACKER": attacker_report_file}.items():
        if report_file.exists():
            reports[name] = json.loads(report_file.read_text())

    if own_tmp:
        shutil.rmtree(tmp_path, ignore_errors=True)

    return {"outputs": outputs, "reports": reports, "expected_messages": expected_messages}


def summarize(result: dict) -> tuple[bool, list[str]]:
    """Turn a :func:`run_demo` result into a pass/fail verdict plus the
    human-readable lines explaining it -- shared between the CLI script
    and tests so both judge the same run the same way.
    """
    lines = []
    all_ok = True
    for name in LEGIT_OPERATORS:
        report = result["reports"].get(name)
        if not report:
            lines.append(f"  {name}: NO REPORT -- something went wrong")
            all_ok = False
            continue
        lines.append(
            f"  {name}: quota_met={report['quota_met']}  "
            f"verified_from_peer={report['verified_from_peer']}  "
            f"rejected={report['rejected_count']}  "
            f"verify_chain_ok={report['verify_chain_ok']}"
        )
        if not report["quota_met"] or not report["verify_chain_ok"]:
            all_ok = False
        if name != ATTACKER_TARGET and report["rejected_count"] < 1:
            lines.append(f"  WARNING: {name} did not report rejecting the attacker's forged "
                          f"'{ATTACKER_TARGET}' message")
            all_ok = False
    return all_ok, lines
