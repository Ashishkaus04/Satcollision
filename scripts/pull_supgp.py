"""Bring SupGP files collected in the cloud onto the laptop.

The GitHub Actions job in the private repo satcollision-supgp saves
CelesTrak's SupGP sets every 6 h. This script (run by a Windows scheduled
task at log-on, on wake and every 6 h) pulls that repo, cloned next to this
one, and copies any new files into data/historical/supgp/, where the
analyses read them. It never deletes anything.

    python scripts/pull_supgp.py
"""
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLONE = ROOT.parent / "satcollision-supgp"
DEST = ROOT / "data" / "historical" / "supgp"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    now = f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ}"
    r = subprocess.run(["git", "-C", str(CLONE), "pull", "--ff-only", "-q"], capture_output=True, text=True)
    if r.returncode != 0:
        msg = f"{now} pull FAILED: {(r.stderr or r.stdout).strip()[:200]}"
        status = 1
    else:
        new = 0
        for f in sorted((CLONE / "supgp").glob("*.tle")):
            if not (DEST / f.name).exists():
                shutil.copy2(f, DEST / f.name)
                new += 1
        msg = f"{now} pulled: {new} new file(s)"
        status = 0
    with open(DEST / "pull.log", "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")
    print(msg)
    return status


if __name__ == "__main__":
    sys.exit(main())
