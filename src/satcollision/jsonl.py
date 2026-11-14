"""
Reading the append-only JSON-lines files (ledgers, event logs) after a crash.

A process killed mid-write (a laptop sleeping, Docker stopping a container)
can leave the end of an appended file torn: half a line, or a run of NUL
bytes where the filesystem had reserved space the data never reached. That
tail was never covered by a checkpoint, so it is safe to drop, and the
caller's checkpoint logic then decides what is valid (a ledger that comes up
short of its checkpoint is reported as not verified, not silently repaired).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

log = logging.getLogger("satcollision.jsonl")

NUL = chr(0)


def _clean(line: str) -> str:
    return line.strip().strip(NUL).strip()


def read_valid_lines(path: Path, rewrite: bool = True) -> List[str]:
    """The lines of ``path`` up to the first one that is not valid JSON.
    If anything was cut (or NUL bytes were found), the file is rewritten
    without it (``rewrite``)."""
    if not path.exists():
        return []
    raw = path.read_bytes().decode("utf-8", errors="replace").splitlines()
    good: List[str] = []
    dropped = 0
    dirty = any(NUL in line for line in raw)
    for i, line in enumerate(raw):
        text = _clean(line)
        if not text:
            continue
        try:
            json.loads(text)
        except ValueError:
            dropped = sum(1 for rest in raw[i:] if _clean(rest))
            break
        good.append(text)
    if dropped or dirty:
        log.warning("%s: dropped a torn tail (%d bad line(s), NUL bytes: %s) after %d good ones "
                    "(an interrupted write)", path, dropped, dirty, len(good))
        if rewrite:
            path.write_text("".join(ln + "\n" for ln in good), encoding="utf-8")
    return good
