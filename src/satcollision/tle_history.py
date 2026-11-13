"""
A TLE *history*: many element sets per object, as Space-Track's gp_history
returns them, and the three things a realistic world needs from it.

1. **Truth.** At time t the best estimate of where a satellite really was
   comes from the element set whose epoch is nearest to t, including one
   published a little after t (an orbit fit uses observations on both sides
   of its epoch). Switching sets would make the position jump by the TLE
   error, so across each switch the two sets' positions are blended linearly
   over ``blend_s`` seconds, centred on the midpoint between the two epochs.

2. **Prediction.** An operator screening at time t only has element sets
   whose epoch is at or before t (the as-of rule historical_replay.py uses).
   The newest of those is what it propagates; how old that set will be at
   the time of closest approach decides how wrong the prediction is.

3. **Measured error.** :func:`fit_tle_errors` measures that wrongness from
   the file itself: each element set propagated forward is compared with a
   later set at the later set's epoch, the difference expressed in the later
   set's radial / along-track / cross-track (RTN) frame, and summarised by
   age bin with a robust scale (1.4826 x the median absolute deviation, so a
   manoeuvre between two sets doesn't inflate the sigma). :class:`ErrorModel`
   turns the table into an RTN sigma for any age, which the world feeds to
   the Foster Pc. :func:`detect_jumps` uses the same model to flag element
   sets that disagree with the previous one by more than ``n_sigma`` robust
   sigmas: almost always a manoeuvre.

The file can hold ~340,000 element sets, and every operator's twin loads it
in its own process, so the lines are kept as fixed-width byte arrays and a
``Satrec`` is only built for a set when one is actually propagated.

Times are Julian dates (whole days plus fraction in one float64, good to
~20 microseconds, far below anything here).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sgp4.api import Satrec, SatrecArray

ALPHA5 = "ABCDEFGHJKLMNPQRSTUVWXYZ"   # Space-Track's alpha-5 letters (no I, no O)
DEFAULT_BINS_H = (0.0, 3.0, 6.0, 12.0, 24.0, 36.0, 48.0, 72.0)
MAD_SCALE = 1.4826
SECONDS_PER_DAY = 86400.0


def parse_satnum(field: str) -> int:
    """NORAD id from columns 3-7 of line 1, including alpha-5 (``A0000`` = 100000)."""
    s = field.strip()
    if s and s[0].isalpha():
        return (ALPHA5.index(s[0].upper()) + 10) * 10000 + int(s[1:])
    return int(s)


def tle_epoch_jd(line1: str) -> float:
    yy = int(line1[18:20])
    doy = float(line1[20:32])
    year = 2000 + yy if yy < 57 else 1900 + yy
    # JD of 0 January of ``year`` at 00:00 UTC, then add the day of year.
    a = (14 - 1) // 12
    y = year + 4800 - a
    jdn = 1 + (153 * (1 + 12 * a - 3) + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    return jdn - 0.5 - 1.0 + doy


def group_of(name: str) -> str:
    """Which error-model group an object belongs to. Starlink is fitted on
    its own: its satellites manoeuvre often and its TLEs age differently."""
    n = (name or "").strip().upper()
    if n.startswith("0 "):
        n = n[2:]
    return "starlink" if n.startswith("STARLINK") else "other"


def _rtn_basis(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(..., 3, 3): rows are R, T, C unit vectors in ECI."""
    r_hat = r / np.linalg.norm(r, axis=-1, keepdims=True)
    h = np.cross(r, v)
    c_hat = h / np.linalg.norm(h, axis=-1, keepdims=True)
    t_hat = np.cross(c_hat, r_hat)
    return np.stack([r_hat, t_hat, c_hat], axis=-2)


def to_rtn(delta: np.ndarray, r_ref: np.ndarray, v_ref: np.ndarray) -> np.ndarray:
    """Express ECI vector(s) ``delta`` in the RTN frame of (r_ref, v_ref)."""
    basis = _rtn_basis(r_ref, v_ref)
    return np.einsum("...ij,...j->...i", basis, delta)


def from_rtn(rtn: np.ndarray, r_ref: np.ndarray, v_ref: np.ndarray) -> np.ndarray:
    basis = _rtn_basis(r_ref, v_ref)
    return np.einsum("...ji,...j->...i", basis, rtn)


def _split_jd(jd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    whole = np.floor(jd - 0.5) + 0.5
    return whole, jd - whole


# ---------------------------------------------------------------------------
# The history
# ---------------------------------------------------------------------------

class TLEHistory:
    """Every element set in one file, grouped by object and sorted by epoch.

    Objects are rows ``0..n-1`` in ascending NORAD id. Element sets live in
    one flat array, object ``i`` owning ``start[i]:start[i+1]``. Two sets of
    the same object with the same epoch (Space-Track re-issues) keep the one
    that comes last in the file.
    """

    def __init__(self, ids: np.ndarray, names: List[str], start: np.ndarray,
                 epochs: np.ndarray, line1: np.ndarray, line2: np.ndarray, source: str = ""):
        self.ids = ids
        self.names = names
        self.start = start
        self.epochs = epochs
        self.line1 = line1
        self.line2 = line2
        self.source = source
        self.row_of = {int(n): i for i, n in enumerate(ids)}
        self.row_of_set = np.repeat(np.arange(len(ids)), np.diff(start))
        self._cache: Dict[int, Satrec] = {}
        # Flat key that sorts by (row, epoch), so one searchsorted call finds
        # the bracketing element sets of every object at once.
        self.jd0 = float(epochs.min()) - 1.0
        self._span = float(epochs.max() - self.jd0) + 1000.0
        self._keys = self.row_of_set * self._span + (epochs - self.jd0)

    # ---- loading ----

    @classmethod
    def load(cls, path: str | Path) -> "TLEHistory":
        path = Path(path)
        raw = path.read_text(encoding="ascii", errors="replace").splitlines()
        lines = [ln.rstrip() for ln in raw if ln.strip()]
        recs: Dict[int, Dict[float, Tuple[str, str, str]]] = {}
        names: Dict[int, str] = {}
        i, n = 0, len(lines)
        while i < n:
            if lines[i].startswith("1 ") and i + 1 < n and lines[i + 1].startswith("2 "):
                name, l1, l2 = None, lines[i], lines[i + 1]
                i += 2
            elif i + 2 < n and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
                name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
                i += 3
            else:
                i += 1
                continue
            if len(l1) < 69 or len(l2) < 69:
                continue   # a cut-off line: skip it rather than mis-parse it
            sat = parse_satnum(l1[2:7])
            if name is not None:
                name = name[2:].strip() if name.startswith("0 ") else name.strip()
                names[sat] = name
            recs.setdefault(sat, {})[tle_epoch_jd(l1)] = (l1[:69], l2[:69])
        if not recs:
            raise ValueError(f"no element sets in {path}")
        ids = np.array(sorted(recs), dtype=np.int64)
        start = [0]
        epochs, l1s, l2s = [], [], []
        for sat in ids:
            by_epoch = recs[int(sat)]
            for ep in sorted(by_epoch):
                epochs.append(ep)
                l1s.append(by_epoch[ep][0])
                l2s.append(by_epoch[ep][1])
            start.append(len(epochs))
        return cls(ids=ids, names=[names.get(int(s), "") for s in ids],
                   start=np.array(start, dtype=np.int64), epochs=np.array(epochs, dtype=float),
                   line1=np.array(l1s, dtype="S69"), line2=np.array(l2s, dtype="S69"),
                   source=str(path))

    # ---- basic access ----

    @property
    def n_objects(self) -> int:
        return len(self.ids)

    @property
    def n_sets(self) -> int:
        return len(self.epochs)

    @property
    def first_epoch(self) -> float:
        return float(self.epochs.min())

    @property
    def last_epoch(self) -> float:
        return float(self.epochs.max())

    def satrec(self, k: int) -> Satrec:
        s = self._cache.get(k)
        if s is None:
            if len(self._cache) > 4 * self.n_objects + 1000:
                self._cache.clear()
            s = Satrec.twoline2rv(self.line1[k].decode(), self.line2[k].decode())
            self._cache[k] = s
        return s

    def sets_of(self, row: int) -> range:
        return range(int(self.start[row]), int(self.start[row + 1]))

    def first_set_satrec(self, row: int) -> Satrec:
        return self.satrec(int(self.start[row]))

    # ---- which set, when ----

    def _bracket(self, rows: np.ndarray, jd):
        """For each row (and each time, if ``jd`` is an array: result shape
        (n_rows, n_times)): index of its newest set with epoch <= jd (or -1)
        and of its oldest set with epoch > jd (or -1)."""
        rows = np.asarray(rows, dtype=np.int64)
        jd = np.asarray(jd, dtype=float)
        r = rows if jd.ndim == 0 else rows[:, None]
        key = r * self._span + (jd - self.jd0)
        pos = np.searchsorted(self._keys, key, side="right")
        lo = self.start[r]
        hi = self.start[r + 1]
        prev = np.where(pos - 1 >= lo, pos - 1, -1)
        nxt = np.where(pos < hi, pos, -1)
        return prev, nxt

    def asof(self, rows: np.ndarray, jd: float) -> np.ndarray:
        """Newest element set with epoch <= jd for each row (-1: none yet)."""
        return self._bracket(np.asarray(rows, dtype=np.int64), jd)[0]

    def nearest(self, rows: np.ndarray, jd: float) -> np.ndarray:
        prev, nxt = self._bracket(np.asarray(rows, dtype=np.int64), jd)
        ep_prev = np.where(prev >= 0, self.epochs[np.maximum(prev, 0)], -np.inf)
        ep_next = np.where(nxt >= 0, self.epochs[np.maximum(nxt, 0)], np.inf)
        return np.where(jd - ep_prev <= ep_next - jd, prev, nxt)

    def truth_weights(self, rows: np.ndarray, jd, blend_s: float):
        """(set_a, set_b, w): truth = (1 - w) * set_a + w * set_b at ``jd``
        (a scalar, or an array of times: results then (n_rows, n_times)).

        Away from a switch, set_a == set_b and w == 0. Around the midpoint m
        between consecutive epochs the weight ramps 0 -> 1 over ``blend_s``
        (or the whole gap, if the sets are closer together than that)."""
        rows = np.asarray(rows, dtype=np.int64)
        prev, nxt = self._bracket(rows, jd)
        a = np.where(prev >= 0, prev, nxt)
        b = np.where(nxt >= 0, nxt, prev)
        both = (prev >= 0) & (nxt >= 0)
        ep_p = self.epochs[np.maximum(prev, 0)]
        ep_n = self.epochs[np.maximum(nxt, 0)]
        mid = 0.5 * (ep_p + ep_n)
        half = np.maximum(np.minimum(blend_s / SECONDS_PER_DAY / 2.0, 0.5 * (ep_n - ep_p)), 1e-9)
        w = np.where(both, np.clip((np.asarray(jd) - (mid - half)) / (2.0 * half), 0.0, 1.0), 0.0)
        # Collapse the trivial blends so most objects need one propagation.
        a = np.where(w >= 1.0, b, a)
        b = np.where(w <= 0.0, a, b)
        w = np.where((w <= 0.0) | (w >= 1.0), 0.0, w)
        return a, b, w

    # ---- propagation ----

    def _propagate_sets(self, sets: np.ndarray, jd: np.ndarray):
        """Propagate each distinct set in ``sets`` at every time in ``jd``.
        Returns (sorted distinct sets, err, r, v), rows in that order."""
        uniq = np.unique(sets[sets >= 0])
        if uniq.size == 0:
            z = np.zeros((0, len(jd), 3))
            return uniq, np.zeros((0, len(jd)), dtype=int), z, z
        whole, frac = _split_jd(np.asarray(jd, dtype=float))
        arr = SatrecArray([self.satrec(int(k)) for k in uniq])
        err, r, v = arr.sgp4(whole, frac)
        return uniq, err, r, v

    def predicted_states(self, rows: Sequence[int], jd_now: float, jd: np.ndarray):
        """Where an operator at ``jd_now`` predicts each object will be at
        each time in ``jd``: the as-of set at ``jd_now``, propagated.
        Returns r, v (n, m, 3), valid (n, m), and the set used per row
        (-1 where the object has no element set yet)."""
        rows = np.asarray(rows, dtype=np.int64)
        jd = np.asarray(jd, dtype=float)
        sets = self.asof(rows, jd_now)
        uniq, err, r, v = self._propagate_sets(sets, jd)
        n, m = len(rows), len(jd)
        R = np.zeros((n, m, 3))
        V = np.zeros((n, m, 3))
        ok = np.zeros((n, m), dtype=bool)
        have = sets >= 0
        if have.any():
            idx = np.searchsorted(uniq, sets[have])
            R[have] = r[idx]
            V[have] = v[idx]
            ok[have] = err[idx] == 0
        return R, V, ok, sets

    def truth_states(self, rows: Sequence[int], jd: np.ndarray, blend_s: float,
                     valid_margin_days: float = 0.5):
        """Blended nearest-set ephemeris of each row at each time in ``jd``.
        An object is invalid more than ``valid_margin_days`` before its first
        element set or after its last (not launched yet / decayed)."""
        rows = np.asarray(rows, dtype=np.int64)
        jd = np.asarray(jd, dtype=float)
        n, m = len(rows), len(jd)
        A, B, W = self.truth_weights(rows, jd, blend_s)
        uniq, err, r, v = self._propagate_sets(np.concatenate([A.ravel(), B.ravel()]), jd)
        if uniq.size == 0:
            return np.zeros((n, m, 3)), np.zeros((n, m, 3)), np.zeros((n, m), dtype=bool)
        ia = np.searchsorted(uniq, A)
        ib = np.searchsorted(uniq, B)
        cols = np.broadcast_to(np.arange(m), (n, m))
        w = W[..., None]
        R = (1.0 - w) * r[ia, cols] + w * r[ib, cols]
        V = (1.0 - w) * v[ia, cols] + w * v[ib, cols]
        ok = (err[ia, cols] == 0) & (err[ib, cols] == 0)
        first = self.epochs[self.start[rows]]
        last = self.epochs[self.start[rows + 1] - 1]
        ok &= (jd[None, :] >= first[:, None] - valid_margin_days) & (jd[None, :] <= last[:, None] + valid_margin_days)
        return R, V, ok

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(str(self.n_sets).encode())
        h.update(self.epochs.tobytes())
        h.update(self.ids.tobytes())
        return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Measured error
# ---------------------------------------------------------------------------

@dataclass
class ErrorModel:
    """Robust RTN sigma (km) per age bin, per group."""

    bins_h: List[float]
    table: Dict[str, List[dict]]       # group -> one row per bin
    floor_km: float = 0.05

    def sigma_rtn(self, group: str, age_h: float) -> Tuple[float, float, float]:
        rows = [r for r in self.table.get(group) or self.table.get("other") or [] if r["n"] >= 20]
        if not rows:
            return (self.floor_km,) * 3
        ages = np.array([r["age_mid_h"] for r in rows])
        out = []
        for comp in ("r", "t", "c"):
            # The robust spread plus the systematic offset (the median), in
            # quadrature: the Pc covariance is centred on the prediction, so
            # a bias is error too. Starlink's along-track median is several km.
            vals = np.array([math.hypot(r[f"sigma_{comp}_km"], r[f"median_{comp}_km"]) for r in rows])
            if age_h <= ages[0]:
                s = vals[0]
            elif age_h >= ages[-1]:
                # Beyond the last bin, keep growing at the last bin's rate per hour.
                s = vals[-1] * max(1.0, age_h / ages[-1])
            else:
                s = float(np.interp(age_h, ages, vals))
            out.append(max(float(s), self.floor_km))
        return tuple(out)

    def to_dict(self) -> dict:
        return {"bins_h": self.bins_h, "table": self.table, "floor_km": self.floor_km}

    @classmethod
    def from_dict(cls, d: dict) -> "ErrorModel":
        return cls(bins_h=list(d["bins_h"]), table=d["table"], floor_km=d.get("floor_km", 0.05))


def _pair_differences(hist: TLEHistory, row: int, max_age_days: float):
    """(age_h, rtn diff) for every (older set, later set) pair of one object
    no more than ``max_age_days`` apart."""
    sets = list(hist.sets_of(row))
    if len(sets) < 2:
        return np.zeros(0), np.zeros((0, 3))
    ep = hist.epochs[sets]
    # Reference state of every set at its own epoch.
    ref_r, ref_v = [], []
    for k in sets:
        s = hist.satrec(k)
        e, r, v = s.sgp4(s.jdsatepoch, s.jdsatepochF)
        ref_r.append(r if e == 0 else (np.nan,) * 3)
        ref_v.append(v if e == 0 else (np.nan,) * 3)
    ref_r = np.array(ref_r)
    ref_v = np.array(ref_v)
    ages, diffs = [], []
    for i, k in enumerate(sets[:-1]):
        later = [j for j in range(i + 1, len(sets)) if 0 < ep[j] - ep[i] <= max_age_days]
        if not later:
            continue
        s = hist.satrec(k)
        whole, frac = _split_jd(ep[later])
        e, r, _ = s.sgp4_array(whole, frac)
        good = e == 0
        if not good.any():
            continue
        later = np.array(later)[good]
        d = to_rtn(r[good] - ref_r[later], ref_r[later], ref_v[later])
        ages.append((ep[later] - ep[i]) * 24.0)
        diffs.append(d)
    if not ages:
        return np.zeros(0), np.zeros((0, 3))
    ages = np.concatenate(ages)
    diffs = np.concatenate(diffs)
    ok = np.all(np.isfinite(diffs), axis=1)
    return ages[ok], diffs[ok]


def _robust_sigma(x: np.ndarray) -> float:
    return float(MAD_SCALE * np.median(np.abs(x - np.median(x))))


def fit_tle_errors(hist: TLEHistory, bins_h: Sequence[float] = DEFAULT_BINS_H,
                   rows: Optional[Iterable[int]] = None, max_objects_per_group: int = 1500,
                   seed: int = 0, min_age_h: float = 1.0 / 60.0) -> ErrorModel:
    """Fit the robust RTN error table. ``rows`` restricts the objects used;
    each group is sampled down to ``max_objects_per_group`` (seeded)."""
    bins_h = list(bins_h)
    candidates = list(range(hist.n_objects)) if rows is None else list(rows)
    groups: Dict[str, List[int]] = {}
    for row in candidates:
        groups.setdefault(group_of(hist.names[row]), []).append(row)
    table: Dict[str, List[dict]] = {}
    for group, members in sorted(groups.items()):
        rng = random.Random(f"{seed}:{group}")
        if len(members) > max_objects_per_group:
            members = sorted(rng.sample(members, max_objects_per_group))
        all_age, all_d = [], []
        for row in members:
            a, d = _pair_differences(hist, row, bins_h[-1] / 24.0)
            all_age.append(a)
            all_d.append(d)
        age = np.concatenate(all_age) if all_age else np.zeros(0)
        d = np.concatenate(all_d) if all_d else np.zeros((0, 3))
        out = []
        for lo, hi in zip(bins_h[:-1], bins_h[1:]):
            sel = (age > max(lo, min_age_h)) & (age <= hi)
            n = int(sel.sum())
            row = {"age_lo_h": lo, "age_hi_h": hi, "n": n,
                   "age_mid_h": float(np.median(age[sel])) if n else 0.5 * (lo + hi)}
            for i, comp in enumerate(("r", "t", "c")):
                row[f"sigma_{comp}_km"] = round(_robust_sigma(d[sel, i]), 4) if n else float("nan")
                row[f"median_{comp}_km"] = round(float(np.median(d[sel, i])), 4) if n else float("nan")
            out.append(row)
        table[group] = out
    return ErrorModel(bins_h=bins_h, table=table)


def detect_jumps(hist: TLEHistory, model: ErrorModel, n_sigma: float = 5.0,
                 rows: Optional[Iterable[int]] = None) -> List[dict]:
    """Element sets that disagree with the previous set of the same object
    by more than ``n_sigma`` robust sigmas in some RTN component, at the
    later set's epoch. Sorted by that epoch."""
    out = []
    for row in (range(hist.n_objects) if rows is None else rows):
        sets = list(hist.sets_of(row))
        if len(sets) < 2:
            continue
        group = group_of(hist.names[row])
        r1, r2, v2, keep = [], [], [], []
        for k_prev, k in zip(sets[:-1], sets[1:]):
            s_prev, s = hist.satrec(k_prev), hist.satrec(k)
            e1, a, _ = s_prev.sgp4(s.jdsatepoch, s.jdsatepochF)
            e2, b, vb = s.sgp4(s.jdsatepoch, s.jdsatepochF)
            if e1 or e2:
                continue
            r1.append(a); r2.append(b); v2.append(vb); keep.append((k_prev, k))
        if not keep:
            continue
        r2a = np.array(r2)
        d_all = to_rtn(np.array(r1) - r2a, r2a, np.array(v2))
        for (k_prev, k), d in zip(keep, d_all):
            gap_h = (hist.epochs[k] - hist.epochs[k_prev]) * 24.0
            sig = np.array(model.sigma_rtn(group, gap_h))
            z = np.abs(d) / sig
            worst = int(np.argmax(z))
            if z[worst] > n_sigma:
                out.append({"norad_id": int(hist.ids[row]), "name": hist.names[row],
                            "epoch_jd": float(hist.epochs[k]), "gap_h": round(gap_h, 2),
                            "rtn_km": [round(float(x), 3) for x in d],
                            "n_sigma": round(float(z[worst]), 1), "axis": "RTC"[worst]})
    out.sort(key=lambda j: (j["epoch_jd"], j["norad_id"]))
    return out


def cached_analysis(hist: TLEHistory, cache_dir: Path, bins_h: Sequence[float],
                    n_sigma: float, jump_rows: Optional[Sequence[int]] = None,
                    seed: int = 0, max_objects_per_group: int = 1500) -> Tuple[ErrorModel, List[dict]]:
    """The error fit and the jump list, computed once per file and cached
    (every process of a world needs them; only the first one pays)."""
    key = hashlib.sha256(json.dumps({
        "hist": hist.fingerprint(), "bins": list(bins_h), "n_sigma": n_sigma, "seed": seed,
        "max_objects": max_objects_per_group,
        "rows": None if jump_rows is None else hashlib.sha256(
            np.asarray(sorted(jump_rows), dtype=np.int64).tobytes()).hexdigest(),
    }, sort_keys=True).encode()).hexdigest()[:20]
    path = Path(cache_dir) / f"tle_history_{key}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ErrorModel.from_dict(data["model"]), data["jumps"]
        except (ValueError, KeyError):
            pass
    model = fit_tle_errors(hist, bins_h, seed=seed, max_objects_per_group=max_objects_per_group)
    jumps = detect_jumps(hist, model, n_sigma, rows=jump_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"model": model.to_dict(), "jumps": jumps}), encoding="utf-8")
    try:
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
    return model, jumps


def format_table(model: ErrorModel) -> str:
    lines = []
    for group, rows in model.table.items():
        lines.append(f"{group}:")
        lines.append("  age (h)    n        sigma R    sigma T    sigma C  (km)   median T")
        for r in rows:
            lines.append(f"  {r['age_lo_h']:>4g}-{r['age_hi_h']:<4g} {r['n']:>8d}   "
                         f"{r['sigma_r_km']:>8.3f}   {r['sigma_t_km']:>8.3f}   {r['sigma_c_km']:>8.3f}"
                         f"        {r['median_t_km']:>8.3f}")
    return "\n".join(lines)
