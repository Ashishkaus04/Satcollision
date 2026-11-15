"""Save CelesTrak's supplemental GP element sets (SupGP) every few hours.

SupGP sets are fitted by CelesTrak to the ephemerides the operators
themselves publish (Starlink, OneWeb, Planet, Kuiper), so they are much
closer to each operator's own precise orbit than the radar-based public
TLEs. Kept as a history, they let the hindcast use operator-quality truth as
the private labels for owners' error models (research/RESULTS.md section 10:
with TLE-derived labels a public model nearly matches owners' models, and
only better labels can show whether private data adds anything).

    python scripts/collect_supgp.py            # run by a scheduled task every 6 h

Each run writes data/historical/supgp/<group>_<YYYYmmddTHHMM>.tle (git-ignored)
and appends a line to data/historical/supgp/collect.log. A group fetched less
than MIN_GAP_H ago is skipped, so a catch-up run after the laptop wakes does
not hammer CelesTrak. ICEYE and Spire have no SupGP file.
"""
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

GROUPS = ["starlink", "oneweb", "planet", "kuiper"]
URL = "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE={}&FORMAT=tle"
MIN_GAP_H = 4.0
OUT = Path(__file__).resolve().parents[1] / "data" / "historical" / "supgp"


def last_fetch_age_h(group: str) -> float:
    files = sorted(OUT.glob(f"{group}_*.tle"))
    if not files:
        return float("inf")
    stamp = files[-1].stem.split("_", 1)[1]
    t = datetime.strptime(stamp, "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - t).total_seconds() / 3600.0


def fetch(group: str) -> bytes:
    req = urllib.request.Request(URL.format(group), headers={"User-Agent": "satcollision-research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    lines = []
    status = 0
    for g in GROUPS:
        age = last_fetch_age_h(g)
        if age < MIN_GAP_H:
            lines.append(f"{now:%Y-%m-%dT%H:%MZ} {g}: skipped (last fetch {age:.1f} h ago)")
            continue
        try:
            data = fetch(g)
            n = data.count(b"\n1 ")
            if n == 0 or b"Invalid query" in data:
                raise ValueError(f"no element sets ({data[:80]!r})")
            (OUT / f"{g}_{now:%Y%m%dT%H%M}.tle").write_bytes(data)
            lines.append(f"{now:%Y-%m-%dT%H:%MZ} {g}: {n} sets")
        except Exception as e:  # keep going with the other groups
            status = 1
            lines.append(f"{now:%Y-%m-%dT%H:%MZ} {g}: FAILED {e}")
        time.sleep(2)
    with open(OUT / "collect.log", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(main())
