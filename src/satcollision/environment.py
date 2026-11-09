"""
Synthetic satellite simulation environment.

A Jupyter-friendly, step-controlled live simulation that:

* Loads satellites from real TLE files (``data/starlink.tle``, etc.) and/or
  the built-in synthetic fleets (Alpha/Beta/Gamma + debris), optionally with
  an engineered close-approach pair injected for guaranteed conjunction demo.
* Propagates every satellite to a configurable simulation clock on demand —
  calling :meth:`SyntheticEnvironment.step` advances time by a fixed amount
  and recomputes positions; calling :meth:`SyntheticEnvironment.snapshot`
  returns the current state without advancing.
* Exposes data at three levels of detail:
    - **Raw ECI state** — 3-D position/velocity for every satellite in the
      TEME frame (km, km/s), as a :class:`SatelliteSnapshot` or a flat
      :class:`EnvironmentSnapshot`.
    - **Conjunctions** — active close approaches screened above a configurable
      distance threshold, with Pc estimates.
    - **Rich Jupyter display** — calling ``display()`` (or letting IPython
      auto-render the object) prints a formatted table of active conjunctions
      and a summary stats block that updates as time advances.

Usage (Jupyter notebook)::

    from satcollision.environment import SyntheticEnvironment

    # --- Option A: real TLE files ---
    env = SyntheticEnvironment.from_tle_files(
        tle_files={
            "Starlink": "data/starlink.tle",
            "Kuiper":   "data/kuiper.tle",
        },
        inject_encounter=True,   # engineer a guaranteed conjunction for demo
        step_size_s=60.0,        # each step() advances by 1 minute
    )

    # --- Option B: built-in synthetic fleets only ---
    # env = SyntheticEnvironment.from_synthetic_fleets(inject_encounter=True)

    snap = env.snapshot()        # current state (does NOT advance time)
    env.step()                   # advance by step_size_s
    env.step(n=10)               # advance 10 steps at once
    env.display()                # rich Jupyter table

    # Direct data access
    print(snap.positions)        # dict {norad_id: (pos_km, vel_km_s)}
    print(snap.conjunctions)     # list[Conjunction] above Pc threshold
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sgp4.api import jday

from .fleets import (
    TrackedObject,
    build_synthetic_fleets,
    catalogue_epoch,
    inject_actor_encounter,
    load_tle_file,
    _epoch_from_calendar,
)
from .twin import (
    Conjunction,
    PropagationResult,
    propagate_window,
    screen_conjunctions,
    DEFAULT_PC_THRESHOLD,
    DEFAULT_SCREENING_DISTANCE_KM,
    DEFAULT_COMBINED_SIGMA_KM,
    DEFAULT_COMBINED_HBR_KM,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SatelliteSnapshot:
    """State of a single satellite at the current simulation instant."""
    norad_id: int
    operator: str
    label: str
    is_debris: bool
    is_actor: bool
    position_km: Tuple[float, float, float]   # TEME frame ECI position
    velocity_km_s: Tuple[float, float, float]  # TEME frame ECI velocity
    altitude_km: float                          # approximate altitude above WGS84 ellipsoid

    @property
    def speed_km_s(self) -> float:
        return float(np.linalg.norm(self.velocity_km_s))


@dataclass
class EnvironmentSnapshot:
    """Complete state of the simulation at one instant in time."""
    sim_time: datetime                             # simulation clock (UTC)
    sim_time_jd: Tuple[float, float]              # (jd, fr) for further SGP4 calls
    step_index: int                                # how many steps have been taken
    satellites: List[SatelliteSnapshot]           # all tracked objects
    conjunctions: List[Conjunction]               # pairs above the screening threshold
    n_objects: int
    n_debris: int
    n_operators: int
    operator_names: List[str]

    # ---- convenience accessors ----------------------------------------

    @property
    def positions(self) -> Dict[int, Tuple[Tuple, Tuple]]:
        """``{norad_id: (position_km, velocity_km_s)}`` — same shape as
        :func:`satcollision.fleets.propagate_all` returns."""
        return {s.norad_id: (s.position_km, s.velocity_km_s) for s in self.satellites}

    pc_threshold: float = DEFAULT_PC_THRESHOLD

    @property
    def active_conjunctions(self) -> List[Conjunction]:
        """Conjunctions at or above the environment's Pc threshold."""
        return [c for c in self.conjunctions if c.pc >= self.pc_threshold]

    def get_operator_satellites(self, operator: str) -> List[SatelliteSnapshot]:
        return [s for s in self.satellites if s.operator == operator]

    def get_satellite(self, norad_id: int) -> Optional[SatelliteSnapshot]:
        for s in self.satellites:
            if s.norad_id == norad_id:
                return s
        return None

    def summary(self) -> str:
        lines = [
            f"  Sim time     : {self.sim_time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"  Step index   : {self.step_index}",
            f"  Objects      : {self.n_objects}  ({self.n_debris} debris)",
            f"  Operators    : {', '.join(self.operator_names)}",
            f"  Conjunctions : {len(self.conjunctions)} screened, "
            f"{len(self.active_conjunctions)} above Pc threshold",
        ]
        if self.active_conjunctions:
            lines.append("  Active conjunctions:")
            for c in self.active_conjunctions[:5]:
                lines.append(
                    f"    {c.pair_label:40s}  "
                    f"d={c.miss_distance_km:.3f} km  "
                    f"Pc={c.pc:.2e}  "
                    f"geom={c.geometry_class}"
                )
        return "\n".join(lines)


# The calendar epoch the built-in synthetic fleets and the engineered encounter
# are constructed at (``fleets.build_synthetic_fleets`` / ``inject_actor_encounter``
# default). The encounter only happens relative to this instant, so a
# synthetic environment has to start its clock here to ever show it.
SYNTHETIC_EPOCH = (2026, 9, 9, 0.0)


def _jd_fr_from_datetime(when: datetime) -> Tuple[float, float]:
    return jday(when.year, when.month, when.day,
                when.hour, when.minute, when.second + when.microsecond / 1e6)


def _calendar_tuple(when: datetime) -> Tuple[int, int, int, float]:
    return (when.year, when.month, when.day,
            when.hour + when.minute / 60.0 + when.second / 3600.0)


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class SyntheticEnvironment:
    """
    Jupyter-friendly satellite simulation environment.

    Time is controlled manually: :meth:`step` advances by
    ``step_size_s`` seconds; :meth:`snapshot` reads the current state
    without advancing.

    Parameters
    ----------
    objects : list[TrackedObject]
        All satellites and debris to track.
    epoch_jd : tuple[float, float]
        Simulation epoch as (Julian day integer part, fractional day).
        Defaults to now (UTC) at construction time.
    step_size_s : float
        How many seconds each call to :meth:`step` advances the clock.
    screening_distance_km : float
        Broad-phase screening radius for conjunction detection.
    pc_threshold : float
        Only conjunctions at or above this Pc are reported as "active".
    screening_window_s : float
        The time window (centred on the current instant ± half-window)
        used to find the closest approach for each screened pair.
        A wider window catches more of each encounter's geometry but
        costs more compute.
    screening_step_s : float
        Time resolution inside the screening window.
    """

    def __init__(
        self,
        objects: List[TrackedObject],
        epoch_jd: Tuple[float, float] | None = None,
        step_size_s: float = 60.0,
        screening_distance_km: float = DEFAULT_SCREENING_DISTANCE_KM,
        pc_threshold: float = DEFAULT_PC_THRESHOLD,
        screening_window_s: float = 3600.0,
        screening_step_s: float = 30.0,
    ):
        self._objects = objects
        self._step_size_s = step_size_s
        self._screening_distance_km = screening_distance_km
        self._pc_threshold = pc_threshold
        self._screening_window_s = screening_window_s
        self._screening_step_s = screening_step_s
        self._step_index: int = 0
        self._last_snapshot: Optional[EnvironmentSnapshot] = None

        # Initialise simulation clock
        if epoch_jd is None:
            now = datetime.now(tz=timezone.utc)
            jd, fr = jday(now.year, now.month, now.day,
                          now.hour, now.minute, now.second + now.microsecond / 1e6)
            self._jd0: float = jd
            self._fr0: float = fr
        else:
            self._jd0, self._fr0 = epoch_jd

        # Elapsed time accumulator (seconds since epoch)
        self._elapsed_s: float = 0.0

    # ------------------------------------------------------------------ #
    # Factory constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_tle_files(
        cls,
        tle_files: Dict[str, str],
        inject_encounter: bool = True,
        step_size_s: float = 60.0,
        screening_window_s: float = 3600.0,
        epoch: datetime | str | None = "catalogue",
        **kwargs,
    ) -> "SyntheticEnvironment":
        """Build an environment from one or more real TLE files.

        Parameters
        ----------
        tle_files : dict
            Mapping of operator name → path to ``.tle`` file, e.g.
            ``{"Starlink": "data/starlink.tle", "Kuiper": "data/kuiper.tle"}``.
        inject_encounter : bool
            If ``True``, also inject the engineered close-approach pair
            from the built-in synthetic fleets so that at least one
            guaranteed conjunction exists in the environment regardless
            of what the real data produces. The pair is built at the
            environment's own epoch, so the encounter happens 30 minutes
            after the clock starts.
        epoch : datetime, ``"catalogue"`` or ``"now"``
            Where the simulation clock starts. The default, ``"catalogue"``,
            is the newest TLE epoch in the loaded files: a TLE is only
            accurate for a few days around its epoch, so starting "now" with
            a catalogue downloaded weeks ago would show orbits that are
            kilometres off.
        """
        objects: List[TrackedObject] = []
        for op_name, path in tle_files.items():
            loaded = load_tle_file(str(path), operator_name=op_name)
            objects.extend(loaded)
            print(f"[env] Loaded {len(loaded):,} satellites from '{path}' as operator '{op_name}'")

        if not objects:
            raise ValueError("no satellites loaded from tle_files")
        if epoch == "catalogue":
            start = catalogue_epoch(objects)
        elif epoch is None or epoch == "now":
            start = datetime.now(tz=timezone.utc).replace(microsecond=0)
        elif isinstance(epoch, datetime):
            start = epoch if epoch.tzinfo else epoch.replace(tzinfo=timezone.utc)
        else:
            raise ValueError(f"epoch must be a datetime, 'catalogue' or 'now', not {epoch!r}")
        print(f"[env] Clock starts {start.strftime('%Y-%m-%d %H:%M UTC')}")

        if inject_encounter:
            # Inject the synthetic close-approach pair into a minimal
            # synthetic set so the actor objects are available even when
            # the real TLE fleets don't happen to produce a conjunction
            # in the current time window. Built at the environment's own
            # epoch: the encounter is defined relative to that instant.
            synthetic = build_synthetic_fleets(epoch=_calendar_tuple(start))
            actor_a, actor_b = inject_actor_encounter(synthetic, epoch=_calendar_tuple(start))
            objects.append(actor_a)
            objects.append(actor_b)
            print(f"[env] Injected engineered encounter: {actor_a.label} <-> {actor_b.label}")

        print(f"[env] Total objects: {len(objects):,}")
        return cls(
            objects=objects,
            epoch_jd=_jd_fr_from_datetime(start),
            step_size_s=step_size_s,
            screening_window_s=screening_window_s,
            **kwargs,
        )

    @classmethod
    def from_synthetic_fleets(
        cls,
        inject_encounter: bool = True,
        step_size_s: float = 60.0,
        screening_window_s: float = 3600.0,
        seed: int = 42,
        n_debris: int = 25,
        **kwargs,
    ) -> "SyntheticEnvironment":
        """Build an environment from the built-in synthetic fleets only
        (Alpha/Beta/Gamma + ambient debris). No TLE files required.

        The clock starts at :data:`SYNTHETIC_EPOCH`, the instant the fleets
        and the engineered encounter are built at, so the encounter shows up
        30 minutes in.
        """
        objects = build_synthetic_fleets(epoch=SYNTHETIC_EPOCH, seed=seed, n_debris=n_debris)
        if inject_encounter:
            actor_a, actor_b = inject_actor_encounter(objects, epoch=SYNTHETIC_EPOCH)
            print(f"[env] Injected engineered encounter: {actor_a.label} <-> {actor_b.label}")
        print(f"[env] Synthetic fleet: {len(objects)} objects")
        y, m, d, h = SYNTHETIC_EPOCH
        return cls(
            objects=objects,
            epoch_jd=jday(y, m, d, h, 0, 0.0),
            step_size_s=step_size_s,
            screening_window_s=screening_window_s,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Time management
    # ------------------------------------------------------------------ #

    @property
    def sim_time(self) -> datetime:
        """Current simulation clock as a UTC datetime."""
        base = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # JD 2451545.0 = 2000-01-01 12:00 TT ≈ UTC for our purposes
        jd_now = self._jd0 + self._fr0 + self._elapsed_s / 86400.0
        jd_j2000 = 2451545.0
        delta_days = jd_now - jd_j2000
        return base + timedelta(days=delta_days)

    def _current_jd_fr(self) -> Tuple[float, float]:
        """Return (jd, fr) for the current simulation instant."""
        fr_raw = self._fr0 + self._elapsed_s / 86400.0
        jd = self._jd0 + math.floor(fr_raw)
        fr = fr_raw - math.floor(fr_raw)
        return jd, fr

    def reset(self) -> None:
        """Reset the simulation clock to the initial epoch."""
        self._elapsed_s = 0.0
        self._step_index = 0
        self._last_snapshot = None

    def seek(self, offset_s: float) -> None:
        """Jump the simulation clock to ``offset_s`` seconds after the epoch."""
        self._elapsed_s = float(offset_s)
        self._last_snapshot = None

    # ------------------------------------------------------------------ #
    # Core simulation step
    # ------------------------------------------------------------------ #

    def step(self, n: int = 1) -> "EnvironmentSnapshot":
        """Advance the simulation clock by ``n × step_size_s`` seconds
        and return a fresh :class:`EnvironmentSnapshot`.

        Parameters
        ----------
        n : int
            Number of steps to advance. Defaults to 1.
        """
        for _ in range(n):
            self._elapsed_s += self._step_size_s
            self._step_index += 1
        return self.snapshot()

    def snapshot(self) -> "EnvironmentSnapshot":
        """Return an :class:`EnvironmentSnapshot` at the current simulation
        time **without** advancing the clock.

        This propagates every tracked object to the current instant and
        runs the KD-tree conjunction screening over a ±(window/2) time
        band around that instant.
        """
        jd, fr = self._current_jd_fr()

        # --- Propagate over a short window centred on current time ------
        # The screening pass needs a window (not just a single instant)
        # to find the true closest approach rather than just the
        # instantaneous distance (which could be mid-approach).
        half = self._screening_window_s / 2.0

        # Shift the window start back by half the window so the
        # current simulation time sits near the middle.
        fr_start = fr - half / 86400.0
        jd_start = jd + math.floor(fr_start)
        fr_start = fr_start - math.floor(fr_start)

        prop: PropagationResult = propagate_window(
            objects=self._objects,
            jd0=jd_start,
            fr0=fr_start,
            duration_s=self._screening_window_s,
            step_s=self._screening_step_s,
        )

        conjunctions = screen_conjunctions(
            objects=self._objects,
            propagation=prop,
            screening_distance_km=self._screening_distance_km,
        )

        # --- Single-instant positions for the snapshot ------------------
        # Propagate to the exact current instant (centre of the window)
        centre_idx = prop.positions.shape[1] // 2
        satellites: List[SatelliteSnapshot] = []
        by_id = {o.norad_id: o for o in self._objects}

        for obj_idx, obj in enumerate(self._objects):
            if not prop.valid[obj_idx, centre_idx]:
                continue  # decayed or SGP4 error — skip silently
            pos = tuple(float(x) for x in prop.positions[obj_idx, centre_idx])
            vel = tuple(float(x) for x in prop.velocities[obj_idx, centre_idx])
            r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            alt = r - 6378.137  # approximate altitude (WGS84 equatorial radius)
            satellites.append(SatelliteSnapshot(
                norad_id=obj.norad_id,
                operator=obj.operator,
                label=obj.label,
                is_debris=obj.is_debris,
                is_actor=obj.is_actor,
                position_km=pos,
                velocity_km_s=vel,
                altitude_km=alt,
            ))

        operator_names = sorted({s.operator for s in satellites if not s.is_debris})
        n_debris = sum(1 for s in satellites if s.is_debris)

        snap = EnvironmentSnapshot(
            sim_time=self.sim_time,
            sim_time_jd=(jd, fr),
            step_index=self._step_index,
            satellites=satellites,
            conjunctions=conjunctions,
            n_objects=len(satellites),
            n_debris=n_debris,
            n_operators=len(operator_names),
            operator_names=operator_names,
            pc_threshold=self._pc_threshold,
        )
        self._last_snapshot = snap
        return snap

    # ------------------------------------------------------------------ #
    # Display / Jupyter integration
    # ------------------------------------------------------------------ #

    def display(self, max_conjunctions: int = 20) -> None:
        """Print a rich formatted view of the current simulation state.

        Call this in a Jupyter cell after one or more :meth:`step` calls
        to see what the environment looks like right now. Automatically
        calls :meth:`snapshot` if no snapshot exists yet.
        """
        snap = self._last_snapshot or self.snapshot()
        _render_display(snap, max_conjunctions=max_conjunctions,
                        step_size_s=self._step_size_s)

    def _repr_html_(self) -> str:
        """Auto-renders in Jupyter when the object is the last expression
        in a cell (Jupyter calls ``_repr_html_`` automatically)."""
        snap = self._last_snapshot or self.snapshot()
        return _render_html(snap, step_size_s=self._step_size_s)

    def __repr__(self) -> str:
        return (
            f"SyntheticEnvironment("
            f"objects={len(self._objects)}, "
            f"step={self._step_index}, "
            f"sim_time={self.sim_time.strftime('%Y-%m-%dT%H:%M:%SZ')}, "
            f"step_size_s={self._step_size_s})"
        )

    # ------------------------------------------------------------------ #
    # Bulk data export helpers
    # ------------------------------------------------------------------ #

    def positions_array(self) -> Tuple[np.ndarray, List[int]]:
        """Return all satellite positions as a numpy array.

        Returns
        -------
        positions : np.ndarray
            Shape ``(n_satellites, 3)`` — ECI position in km.
        norad_ids : list[int]
            Row-order NORAD IDs matching the array rows.
        """
        snap = self._last_snapshot or self.snapshot()
        norad_ids = [s.norad_id for s in snap.satellites]
        positions = np.array([s.position_km for s in snap.satellites], dtype=float)
        return positions, norad_ids

    def velocities_array(self) -> Tuple[np.ndarray, List[int]]:
        """Return all satellite velocities as a numpy array.

        Returns
        -------
        velocities : np.ndarray
            Shape ``(n_satellites, 3)`` — ECI velocity in km/s.
        norad_ids : list[int]
            Row-order NORAD IDs matching the array rows.
        """
        snap = self._last_snapshot or self.snapshot()
        norad_ids = [s.norad_id for s in snap.satellites]
        velocities = np.array([s.velocity_km_s for s in snap.satellites], dtype=float)
        return velocities, norad_ids

    def conjunctions_dataframe(self):
        """Return the current conjunction list as a ``pandas.DataFrame``.

        Requires ``pandas`` (not in the project's core requirements, but
        commonly available in any Jupyter environment). Falls back to a
        plain list-of-dicts if pandas is not installed.
        """
        snap = self._last_snapshot or self.snapshot()
        rows = [
            {
                "encounter_id": (
                    f"ENC-{min(c.object_a.norad_id, c.object_b.norad_id)}"
                    f"-{max(c.object_a.norad_id, c.object_b.norad_id)}"
                    f"-{int(c.tca_offset_s)}"
                ),
                "object_a": c.object_a.label,
                "object_b": c.object_b.label,
                "operator_a": c.object_a.operator,
                "operator_b": c.object_b.operator,
                "miss_distance_km": round(c.miss_distance_km, 4),
                "pc": c.pc,
                "pc_above_threshold": c.pc >= self._pc_threshold,
                "geometry": c.geometry_class,
                "tca_offset_s": c.tca_offset_s,
                "rel_speed_km_s": round(c.relative_speed_km_s, 4),
            }
            for c in snap.conjunctions
        ]
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows


# ---------------------------------------------------------------------------
# Rendering helpers (plain-text + HTML for Jupyter)
# ---------------------------------------------------------------------------

_TIER_COLORS = {
    "critical":  "\033[91m",  # red
    "elevated":  "\033[93m",  # yellow
    "threshold": "\033[96m",  # cyan
    "below":     "\033[90m",  # grey
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"


def _pc_tier(pc: float) -> str:
    if pc >= 100 * DEFAULT_PC_THRESHOLD:
        return "critical"
    if pc >= 10 * DEFAULT_PC_THRESHOLD:
        return "elevated"
    if pc >= DEFAULT_PC_THRESHOLD:
        return "threshold"
    return "below"


def _render_display(snap: EnvironmentSnapshot, max_conjunctions: int, step_size_s: float) -> None:
    """Print a colour-formatted environment summary to stdout."""
    sep = "-" * 72

    print(f"\n{_BOLD}{'=' * 72}{_RESET}")
    print(f"{_BOLD}  [SAT] SYNTHETIC SATELLITE ENVIRONMENT  --  Step {snap.step_index}{_RESET}")
    print(f"{'=' * 72}")
    print(f"  Sim time  : {_BOLD}{snap.sim_time.strftime('%Y-%m-%d  %H:%M:%S  UTC')}{_RESET}")
    print(f"  Step size : {step_size_s:.0f} s  ({step_size_s/60:.1f} min)")
    print(f"  Objects   : {snap.n_objects:,}  ({snap.n_debris} debris)")
    print(f"  Operators : {', '.join(snap.operator_names)}")
    print(sep)

    # --- Operator breakdown ---
    print(f"\n{_BOLD}  Operator Satellite Counts{_RESET}")

    op_counts: Dict[str, int] = {}
    for s in snap.satellites:
        op_counts[s.operator] = op_counts.get(s.operator, 0) + 1
    for op, cnt in sorted(op_counts.items()):
        bar = "#" * min(cnt // 5 + 1, 40)
        print(f"    {op:12s}  {cnt:5d}  {bar}")

    # --- Conjunctions ---
    print(f"\n{sep}")
    n_active = len(snap.active_conjunctions)
    n_total  = len(snap.conjunctions)
    print(f"{_BOLD}  Conjunctions  (screened: {n_total}  |  above Pc threshold: {n_active}){_RESET}")

    if not snap.conjunctions:
        print("  No conjunctions found in current screening window.\n")
    else:
        print(f"\n  {'#':>3}  {'Object A':20s}  {'Object B':20s}  "
              f"{'Miss dist':>10}  {'Pc':>10}  {'Tier':>10}  {'Geometry':12s}  {'TCA offset':>12}")
        print(f"  {'-'*3}  {'-'*20}  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*12}")
        for i, c in enumerate(snap.conjunctions[:max_conjunctions], 1):
            tier = _pc_tier(c.pc)
            color = _TIER_COLORS.get(tier, "")
            tca_str = f"{c.tca_offset_s/60:.1f} min"
            print(
                f"  {i:>3}  {c.object_a.label:20s}  {c.object_b.label:20s}  "
                f"{color}{c.miss_distance_km:>9.3f}km  {c.pc:>10.2e}  "
                f"{tier:>10}  {c.geometry_class:12s}  {tca_str:>12}{_RESET}"
            )
        if n_total > max_conjunctions:
            print(f"  ... {n_total - max_conjunctions} more (use max_conjunctions= to show all)")

    print(f"\n{'=' * 72}\n")


def _render_html(snap: EnvironmentSnapshot, step_size_s: float) -> str:
    """Return an HTML string for Jupyter rich display (``_repr_html_``)."""
    tier_badge = {
        "critical":  "background:#ff4d4d;color:#fff;",
        "elevated":  "background:#ffaa00;color:#000;",
        "threshold": "background:#00c0c0;color:#000;",
        "below":     "background:#444;color:#aaa;",
    }

    rows_html = ""
    for i, c in enumerate(snap.conjunctions[:50], 1):
        tier = _pc_tier(c.pc)
        badge_style = tier_badge.get(tier, "")
        tca_str = f"{c.tca_offset_s/60:.1f} min"
        rows_html += (
            f"<tr>"
            f"<td style='text-align:center'>{i}</td>"
            f"<td><code>{c.object_a.label}</code></td>"
            f"<td><code>{c.object_b.label}</code></td>"
            f"<td style='text-align:right'>{c.miss_distance_km:.3f} km</td>"
            f"<td style='text-align:right'><code>{c.pc:.2e}</code></td>"
            f"<td><span style='padding:2px 6px;border-radius:4px;font-size:0.8em;{badge_style}'>{tier}</span></td>"
            f"<td>{c.geometry_class}</td>"
            f"<td style='text-align:right'>{tca_str}</td>"
            f"</tr>"
        )

    op_counts: Dict[str, int] = {}
    for s in snap.satellites:
        op_counts[s.operator] = op_counts.get(s.operator, 0) + 1
    op_rows = "".join(
        f"<tr><td>{op}</td><td style='text-align:right'>{cnt:,}</td></tr>"
        for op, cnt in sorted(op_counts.items())
    )

    n_active = len(snap.active_conjunctions)

    html = f"""
<div style="font-family:monospace;background:#1a1a2e;color:#e0e0e0;
            padding:16px;border-radius:10px;max-width:900px;border:1px solid #333">
  <div style="font-size:1.3em;font-weight:bold;margin-bottom:8px;color:#7eb8f7">
    🛰&nbsp; Synthetic Satellite Environment &mdash; Step {snap.step_index}
  </div>
  <div style="display:flex;gap:40px;margin-bottom:12px;font-size:0.9em;color:#bbb">
    <div>🕐 <b style="color:#e0e0e0">{snap.sim_time.strftime('%Y-%m-%d %H:%M:%S UTC')}</b></div>
    <div>⏩ Step size: <b style="color:#e0e0e0">{step_size_s:.0f} s</b></div>
    <div>🌍 Objects: <b style="color:#e0e0e0">{snap.n_objects:,}</b> ({snap.n_debris} debris)</div>
    <div>🏢 Operators: <b style="color:#e0e0e0">{snap.n_operators}</b></div>
  </div>

  <details open>
    <summary style="cursor:pointer;font-weight:bold;color:#7eb8f7;margin-bottom:6px">
      Fleet breakdown
    </summary>
    <table style="border-collapse:collapse;font-size:0.85em;margin-bottom:12px">
      <tr style="color:#888"><th style="text-align:left;padding:2px 12px 2px 0">Operator</th>
          <th style="text-align:right;padding:2px">Satellites</th></tr>
      {op_rows}
    </table>
  </details>

  <details open>
    <summary style="cursor:pointer;font-weight:bold;color:#f7a77e;margin-bottom:6px">
      Conjunctions &mdash; screened: {len(snap.conjunctions)}&nbsp;&nbsp;
      <span style="background:#ff4d4d;color:#fff;padding:1px 6px;border-radius:4px;font-size:0.8em">
        {n_active} above threshold
      </span>
    </summary>
    {'<p style="color:#888;font-size:0.85em">No conjunctions in current window.</p>'
     if not snap.conjunctions else f'''
    <table style="border-collapse:collapse;font-size:0.82em;width:100%">
      <thead><tr style="color:#888;border-bottom:1px solid #444">
        <th>#</th><th>Object A</th><th>Object B</th>
        <th style="text-align:right">Miss dist</th>
        <th style="text-align:right">Pc</th>
        <th>Tier</th><th>Geometry</th>
        <th style="text-align:right">TCA offset</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>'''}
  </details>
</div>
"""
    return html
