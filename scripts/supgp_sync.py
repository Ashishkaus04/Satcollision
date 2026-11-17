"""Laptop side of the SupGP collection: pull what the cloud job saved, then
download whatever it could not get.

    python scripts/supgp_sync.py         # run by the "Satcollision SupGP collector" task

The GitHub Actions job (private repo satcollision-supgp) reliably gets Planet,
Kuiper and CelesTrak's comparison table, but CelesTrak refuses its Starlink and
OneWeb downloads (HTTP 403, most likely because the request comes from a cloud
data-centre address). So this script first pulls the cloud files
(pull_supgp.py) and then runs the normal collector (collect_supgp.py), which
skips any group fetched less than 4 h ago, from the cloud or from here, and
downloads the rest from the laptop's own connection.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    rc_pull = subprocess.run([sys.executable, str(HERE / "pull_supgp.py")]).returncode
    rc_collect = subprocess.run([sys.executable, str(HERE / "collect_supgp.py")]).returncode
    return rc_pull or rc_collect


if __name__ == "__main__":
    sys.exit(main())
