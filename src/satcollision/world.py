"""
Always-running satellite worlds.

A *world* is one self-contained simulated orbital environment: a set of
operator fleets (synthetic or real TLE catalogues), a simulation clock that
keeps moving, and the project's full pipeline running on every step:

    propagate -> screen -> Pc -> abstract -> sign -> ledger
              -> deconflict -> federation round -> reputation

``SyntheticEnvironment`` (environment.py) answers "what does the sky look like
at time t?" for a notebook. A ``World`` answers a different question: "what
happens if this federation is left running?" So it adds the things a
long-lived process needs and a notebook does not:

* **Look-ahead screening.** Each tick screens a thin slice of time
  ``lookahead_s`` in the future, so every encounter is found once, with a
  fixed warning time, and then counts down until it passes.
* **State on disk.** Every tick ends with an atomic checkpoint. A world that
  is stopped (or crashes) resumes from the exact tick it reached, and the
  run is deterministic: N ticks straight through and N ticks with a restart
  in the middle produce the same ledger, hash for hash (see tests).
* **A persistent signed ledger.** Every signal and report that crosses the
  operator boundary is Ed25519-signed and appended to the hash-chained
  ``SignedLog`` from identity.py, and mirrored to ``ledger.jsonl``. On resume
  the chain is rebuilt from the file and re-verified, so editing the file
  by hand is detected.
* **A live state file** (``state.json``) that the live dashboard polls.

What a world deliberately does *not* do: planned manoeuvres are agreed
(deconfliction.py) but never applied to the orbits, and the miss distance is
the closest 30 s grid sample, exactly as in the rest of the project.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sgp4.api import SatrecArray, WGS72, Satrec, jday

from .deconfliction import negotiate_maneuver
from .federation import (
    OperatorReport,
    build_operator_report,
    krum_aggregate,
    mean_aggregate,
    trimmed_mean_aggregate,
)
from .fleets import (
    TrackedObject,
    build_synthetic_fleets,
    catalogue_epoch,
    inject_actor_encounter,
    load_tle_file,
)
from .identity import (
    GENESIS_HASH,
    OperatorIdentity,
    SignedLog,
    SignedReport,
    SignedSignal,
    public_key_registry,
    sign_report,
    sign_signal,
    verify_signed,
)
from .reputation import (
    ReputationState,
    evaluate_round,
    reputation_weighted_aggregate,
    update_scores,
)
from .signal import AbstractedSignal, abstract_conjunction, assert_no_raw_state, severity_tier
from .summaries import summarize_maneuver_plan
from .twin import Conjunction, PropagationResult, propagate_window, screen_conjunctions
from .wire import encode_signed_report, encode_signed_signal

STATE_VERSION = 1
BEHAVIOURS = ("honest", "free_rider", "spoofer", "on_off")
# How a World instance takes part (see the class docstring).
MODES = ("all", "sky", "operator")
SYNTHETIC_START = datetime(2026, 9, 9, 0, 0, 0, tzinfo=timezone.utc)

# How much history the live state keeps. The full history is on disk
# (events.jsonl, ledger.jsonl); these only bound what the dashboard polls.
MAX_EVENTS_IN_STATE = 150
MAX_PASSED_IN_STATE = 40
MAX_ROUNDS_KEPT = 200
MAX_REPUTATION_HISTORY = 200


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WorldConfig:
    """One world's definition. Loaded from ``worlds/<name>.json``.

    Time settings are in *simulated* seconds. ``speed`` is simulated seconds
    per wall-clock second, so a tick happens every ``tick_sim_s / speed``
    wall seconds.
    """

    name: str
    title: str = ""
    description: str = ""
    fleet: dict = field(default_factory=lambda: {"type": "synthetic"})
    speed: float = 60.0
    tick_sim_s: float = 120.0
    screening_step_s: float = 30.0
    lookahead_s: float = 1800.0
    pc_threshold: float = 1.0e-4
    pc_model: str = "circular"           # "circular" or "foster"
    watch_miss_km: float = 2.0            # below-threshold approaches kept privately
    federation_period_s: float = 7200.0
    injection: Optional[dict] = None      # see _maybe_inject
    behaviours: Dict[str, str] = field(default_factory=dict)  # default honest
    defect_after_round: int = 6           # for "on_off"
    cycle_days: Optional[float] = None    # restart the clock (stale TLEs)
    seed: int = 42
    max_tracks: int = 900                 # satellites drawn on the live map

    @classmethod
    def from_dict(cls, data: dict) -> "WorldConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown world config keys: {sorted(unknown)}")
        cfg = cls(**data)
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "WorldConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        step = self.screening_step_s
        for name in ("tick_sim_s", "lookahead_s", "federation_period_s"):
            value = getattr(self, name)
            if value <= 0 or abs(value / step - round(value / step)) > 1e-9:
                # Keeping every time a whole number of screening steps puts
                # engineered encounters exactly on the screening grid.
                raise ValueError(f"{name}={value} must be a positive multiple of screening_step_s={step}")
        if self.pc_model not in ("circular", "foster"):
            raise ValueError(f"pc_model must be 'circular' or 'foster', not {self.pc_model!r}")
        for op, behaviour in self.behaviours.items():
            if behaviour not in BEHAVIOURS:
                raise ValueError(f"behaviour for {op} must be one of {BEHAVIOURS}, not {behaviour!r}")
        if self.speed <= 0:
            raise ValueError("speed must be positive")
        if self.fleet.get("type") not in ("synthetic", "tle", "catalogue", "history"):
            raise ValueError("fleet.type must be 'synthetic', 'tle', 'catalogue' or 'history'")
        if self.fleet["type"] in ("catalogue", "history") and not self.fleet.get("operators"):
            raise ValueError(f"a {self.fleet['type']} fleet needs 'operators': {{name: [name prefixes]}}")
        if self.fleet["type"] == "history":
            if self.pc_model != "foster":
                raise ValueError("a history world uses the measured RTN covariance: pc_model must be 'foster'")
            if self.cycle_days is not None or self.injection:
                raise ValueError("a history world runs through its file's own dates: "
                                 "no cycle_days and no injection (use fleet.at_end)")
            if self.fleet.get("at_end", "stop") not in ("stop", "loop"):
                raise ValueError("fleet.at_end must be 'stop' or 'loop'")

    def fingerprint(self) -> str:
        """Hash of the settings that change what the world *is*. A checkpoint
        made under a different fingerprint cannot be resumed (the fleet or
        the rules changed under it); ``speed`` and ``max_tracks`` only change
        pacing and display, so they are left out and can be edited freely."""
        data = dataclasses.asdict(self)
        for display_only in ("speed", "max_tracks", "title", "description"):
            data.pop(display_only, None)
        blob = json.dumps(data, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _calendar_tuple(when: datetime):
    return (when.year, when.month, when.day,
            when.hour + when.minute / 60.0 + (when.second + when.microsecond / 1e6) / 3600.0)


def _jd_fr(when: datetime):
    return jday(when.year, when.month, when.day, when.hour, when.minute,
                when.second + when.microsecond / 1e6)


def _gmst_rad(jd: float, fr: float) -> float:
    """Greenwich mean sidereal time (IAU 1982, low precision). Good to well
    under a degree, which is all a map dot needs."""
    d = (jd - 2451545.0) + fr
    return math.radians((280.46061837 + 360.98564736629 * d) % 360.0)


def _iso(when: datetime) -> str:
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rng(*parts) -> random.Random:
    """Deterministic RNG keyed by a description of the decision being made.
    Keying by content (not by call order) keeps a resumed run identical to
    an uninterrupted one without having to persist RNG state."""
    return random.Random(":".join(str(p) for p in parts))


def _identity_for(world: str, seed: int, operator: str) -> OperatorIdentity:
    """Simulated operators get keys derived from the world name and seed, so
    a world's signatures are reproducible across restarts. A real operator
    generates its key locally and never derives it from anything shared."""
    raw = hashlib.sha256(f"satcollision-world:{world}:{seed}:{operator}".encode()).digest()
    return OperatorIdentity.from_private_key_bytes(operator, raw)


def _signal_to_dict(sig: AbstractedSignal) -> dict:
    return dataclasses.asdict(sig)


def _signal_from_dict(data: dict) -> AbstractedSignal:
    return AbstractedSignal(**data)


def _write_atomic(path: Path, text: str, retries: int = 20) -> bool:
    """Write-then-rename so a reader never sees half a file. On Windows the
    rename fails while another process (the dashboard server) has the target
    open, so retry briefly; give up quietly if the reader holds on."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    return False


def _shift_satrec(satrec: Satrec, norad_id: int, delta_mean_anomaly_rad: float) -> Satrec:
    """Copy of ``satrec`` moved along its own orbit by a small angle."""
    out = Satrec()
    epoch_days = (satrec.jdsatepoch + satrec.jdsatepochF) - 2433281.5
    out.sgp4init(
        WGS72, "i", norad_id, epoch_days, satrec.bstar, 0.0, 0.0,
        satrec.ecco, satrec.argpo, satrec.inclo,
        satrec.mo + delta_mean_anomaly_rad, satrec.no_kozai, satrec.nodeo,
    )
    return out


NON_MEMBER = "NON-MEMBER"


def _mean_altitude_km(satrec: Satrec) -> float:
    n = satrec.no_kozai / 60.0                       # rad/s
    a = (398600.4418 / (n * n)) ** (1.0 / 3.0)       # km
    return a - 6378.137


def group_catalogue(objects: List[TrackedObject], operators: Dict[str, List[str]],
                    max_altitude_km: Optional[float] = None) -> List[TrackedObject]:
    """Split one mixed catalogue (CelesTrak's "active" group, for example)
    into operators by satellite name: ``{"Planet": ["FLOCK", "SKYSAT"]}``
    makes every object whose name starts with FLOCK or SKYSAT a Planet
    satellite. Everything else stays in the sky as a tracked object that
    belongs to no federation member (handled like debris: whoever it comes
    close to deals with it alone). ``max_altitude_km`` drops objects whose
    mean altitude is above it (MEO and GEO, which never meet LEO)."""
    prefixes = [(p.upper(), op) for op, ps in operators.items() for p in ps]
    prefixes.sort(key=lambda t: -len(t[0]))   # longest prefix wins
    out = []
    for o in objects:
        if max_altitude_km is not None and _mean_altitude_km(o.satrec) > max_altitude_km:
            continue
        name = (o.meta.get("name") or "").strip().upper()
        if name.startswith("0 "):          # 3LE files put a "0 " before the name
            name = name[2:]
        owner = next((op for p, op in prefixes if name.startswith(p)), None)
        if owner is None:
            out.append(dataclasses.replace(o, operator=NON_MEMBER, is_debris=True))
        else:
            out.append(dataclasses.replace(o, operator=owner))
    return out


def run_is_stale(config: WorldConfig, run_dir: str | Path) -> bool:
    """True if any checkpoint under ``run_dir`` (single-process, sky, or an
    operator's twin) was made under a different config fingerprint."""
    run_dir = Path(run_dir)
    paths = [run_dir / "checkpoint.json", run_dir / "sky" / "checkpoint.json",
             *run_dir.glob("nodes/*/twin/checkpoint.json")]
    for path in paths:
        if path.exists():
            try:
                if json.loads(path.read_text(encoding="utf-8")).get("fingerprint") != config.fingerprint():
                    return True
            except ValueError:
                return True
    return False


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------

class World:
    """One always-running world. Drive it with :meth:`tick`; persist it with
    :meth:`save`; bring it back with :meth:`open`.

    ``mode`` says whose world this instance is:

    * ``"all"`` (default): one process plays every part -- the sky, every
      operator's twin and the federation -- with keys derived from the seed.
      Fast and deterministic; what scripts/run_worlds.py and the tests use.
    * ``"sky"``: ground truth only. Propagates, screens and records what
      really happens (the observer's view) but signs nothing and scores
      nothing; in the separate-process setup the federation does that.
    * ``"operator"``: one operator's digital twin (twin_node.py). It screens
      its own satellites against the public catalogue, keeps only the
      encounters it is party to, decides by itself whether to share, signs
      with its *own* key (``identity``, never derived from anything shared)
      and queues the signed messages in :attr:`outbox` for the caller to
      send to the federation process.
    """

    def __init__(self, config: WorldConfig, run_dir: str | Path, repo_root: str | Path = ".",
                 mode: str = "all", operator: Optional[str] = None,
                 identity: Optional[OperatorIdentity] = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "operator" and (operator is None or identity is None):
            raise ValueError("operator mode needs the operator's name and its own identity")
        self.mode = mode
        self.me = operator if mode == "operator" else None
        self.cfg = config
        self.run_dir = Path(run_dir)
        self.repo_root = Path(repo_root)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.history = None   # a HistoryModel for fleet type "history"
        self._base_objects, self.start_time = self._build_fleet()
        self.operators = sorted({o.operator for o in self._base_objects if not o.is_debris})
        self.ownership = {
            op: {o.norad_id for o in self._base_objects if o.operator == op} for op in self.operators
        }
        if self.me is not None and self.me not in self.operators:
            raise ValueError(f"{self.me} is not an operator in world {config.name}")
        if mode == "all":
            self.identities = {op: _identity_for(config.name, config.seed, op) for op in self.operators}
        elif mode == "operator":
            self.identities = {self.me: identity}   # only its own key, and only here
        else:
            self.identities = {}
        self.registry = public_key_registry(self.identities)
        self.pc_model = self._build_pc_model()
        self._display_ids = self._choose_display_ids()
        self._max_base_id = max(o.norad_id for o in self._base_objects)

        # ---- mutable state (everything below is checkpointed) ----
        self.tick_index = 0
        self.cycle = 0
        self.elapsed_s = 0.0          # seconds into the current cycle
        self.total_sim_s = 0.0        # seconds since the world first started
        self.wall_started = _iso(datetime.now(timezone.utc))
        self.injections: List[dict] = []
        self.injection_counter = 0
        self.next_injection_s = 0.0
        self.next_round_s = config.federation_period_s
        self.round_index = 0
        self.encounters: List[dict] = []       # upcoming, plus recently passed
        self.period_signals: List[dict] = []
        self.reputation = ReputationState()
        self.rounds: List[dict] = []
        self.recent_events: List[dict] = []
        self.counters = {
            "ticks": 0, "encounters_shared": 0, "private_watch": 0, "internal": 0,
            "signals_signed": 0, "signals_withheld": 0, "phantom_signals": 0,
            "maneuvers_agreed": 0, "unilateral": 0, "passed": 0,
            "federation_rounds": 0, "injections": 0, "cycles": 0, "screened_pairs": 0,
        }
        self.ledger = SignedLog()
        self.ledger_verified = True
        self._pending_ledger_lines: List[str] = []
        self._pending_events: List[dict] = []
        self._objects: List[TrackedObject] = list(self._base_objects)
        self.last_tick_wall_s = 0.0
        self._tick_tag = 0  # the tick an event belongs to (the one being run)
        self.outbox: List[dict] = []  # operator mode: signed messages for the federation
        self.maneuvers: List[dict] = []  # history worlds: avoidance burns applied to the truth
        self.finished = False            # history worlds: reached the end of the file
        if self.history is not None:
            self._refresh_sets()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def _build_fleet(self):
        fleet = self.cfg.fleet
        if fleet["type"] == "synthetic":
            start = SYNTHETIC_START
            objects = build_synthetic_fleets(
                epoch=_calendar_tuple(start),
                n_debris=int(fleet.get("n_debris", 25)),
                seed=int(fleet.get("seed", self.cfg.seed)),
            )
            return objects, start
        if fleet["type"] == "history":
            return self._build_history_fleet()
        objects: List[TrackedObject] = []
        files = fleet["files"].items() if fleet["type"] == "tle" else [(None, fleet["file"])]
        for operator, rel_path in files:
            path = Path(rel_path)
            if not path.is_absolute():
                path = self.repo_root / path
            if not path.exists():
                raise FileNotFoundError(
                    f"world '{self.cfg.name}' needs {path} (download the catalogue first; see README)"
                )
            if operator is not None:
                objects.extend(load_tle_file(str(path), operator_name=operator))
            else:
                objects.extend(group_catalogue(load_tle_file(str(path)), fleet["operators"],
                                               max_altitude_km=fleet.get("max_altitude_km")))
        if not objects:
            raise ValueError(f"world '{self.cfg.name}': no satellites loaded")
        # TLEs are only good for a few days around their epoch: start there.
        start = catalogue_epoch(objects).replace(microsecond=0)
        return objects, start

    def _build_history_fleet(self):
        from .history_world import HistoryModel
        try:
            self.history = H = HistoryModel(self.cfg.fleet, self.repo_root)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"world '{self.cfg.name}' {exc}") from None
        hist = H.hist
        objects = [TrackedObject(norad_id=int(hist.ids[row]), operator="", satrec=hist.first_set_satrec(row),
                                 meta={"name": hist.names[row]})
                   for row in range(hist.n_objects)]
        objects = group_catalogue(objects, self.cfg.fleet["operators"],
                                  max_altitude_km=self.cfg.fleet.get("max_altitude_km"))
        if not objects:
            raise ValueError(f"world '{self.cfg.name}': no satellites loaded")
        self._row_of = {int(n): i for i, n in enumerate(hist.ids)}
        self._rows = np.array([self._row_of[o.norad_id] for o in objects], dtype=np.int64)
        self._ids = np.array([o.norad_id for o in objects], dtype=np.int64)
        self._id_set = set(int(i) for i in self._ids)
        return objects, H.start_time()

    def _build_pc_model(self):
        if self.cfg.pc_model == "foster":
            from .pc import FosterPcModel
            return FosterPcModel()
        return None  # twin.screen_conjunctions' circular default

    def _choose_display_ids(self) -> set:
        """A fixed, evenly spread subset per operator for the live map, so a
        10,000-satellite catalogue doesn't ship 10,000 dots every tick."""
        groups: Dict[str, List[int]] = {}
        for o in self._base_objects:
            groups.setdefault(o.operator, []).append(o.norad_id)
        budget = max(1, self.cfg.max_tracks // max(1, len(groups)))
        chosen = set()
        for ids in groups.values():
            ids = sorted(ids)
            stride = max(1, math.ceil(len(ids) / budget))
            chosen.update(ids[::stride])
        return chosen

    # ------------------------------------------------------------------ #
    # Clock
    # ------------------------------------------------------------------ #

    @property
    def sim_time(self) -> datetime:
        return self.start_time + timedelta(seconds=self.elapsed_s)

    def _at(self, cycle_offset_s: float) -> datetime:
        return self.start_time + timedelta(seconds=cycle_offset_s)

    def behaviour(self, operator: str) -> str:
        b = self.cfg.behaviours.get(operator, "honest")
        if b == "on_off":
            return "spoofer" if self.round_index >= self.cfg.defect_after_round else "honest"
        return b

    # ------------------------------------------------------------------ #
    # Events and ledger
    # ------------------------------------------------------------------ #

    def _event(self, kind: str, text: str, **data) -> dict:
        event = {
            "tick": self._tick_tag,
            "cycle": self.cycle,
            "sim_time": _iso(self.sim_time),
            "kind": kind,
            "text": text,
            **data,
        }
        self._pending_events.append(event)
        self.recent_events.append(event)
        del self.recent_events[:-MAX_EVENTS_IN_STATE]
        return event

    def _ledger_timestamp(self) -> float:
        # Simulated time, not wall time: the ledger must come out identical
        # whether or not the world was restarted along the way.
        return round(self.sim_time.timestamp(), 3)

    def _ledger_append(self, signed) -> None:
        entry = self.ledger.append(signed, timestamp=self._ledger_timestamp())
        if isinstance(signed, SignedSignal):
            kind, payload = "signal", _signal_to_dict(signed.signal)
        else:
            kind = "report"
            payload = {
                "operator": signed.report.operator,
                "counts": [float(c) for c in signed.report.counts],
                "is_adversarial": signed.report.is_adversarial,
            }
        self._pending_ledger_lines.append(json.dumps({
            "index": entry.index,
            "timestamp": entry.timestamp,
            "kind": kind,
            "payload": payload,
            "signature": signed.signature.hex(),
            "public_key": signed.public_key.hex(),
            "prev_hash": entry.prev_hash,
            "entry_hash": entry.entry_hash,
        }, sort_keys=True))

    def _emit(self, sig: AbstractedSignal) -> None:
        """An operator decided to share ``sig``: what that means depends on
        whose world this is."""
        if self.mode == "all":
            self._sign_and_log_signal(sig)
        elif self.mode == "sky":
            self.counters["signals_signed"] += 1   # it happens, but not here
        else:
            assert_no_raw_state(sig)
            if sig.sender_operator != self.me:
                raise RuntimeError("an operator node can only sign its own signals")
            signed = sign_signal(self.identities[self.me], sig)
            self.outbox.append(encode_signed_signal(signed))
            self.period_signals.append(_signal_to_dict(sig))
            self.counters["signals_signed"] += 1

    def _sign_and_log_signal(self, sig: AbstractedSignal) -> None:
        assert_no_raw_state(sig)
        signed = sign_signal(self.identities[sig.sender_operator], sig)
        if not verify_signed(signed, self.registry):  # the federation's check
            raise RuntimeError(f"signature from {sig.sender_operator} failed verification")
        self._ledger_append(signed)
        self.period_signals.append(_signal_to_dict(sig))
        self.counters["signals_signed"] += 1

    # ------------------------------------------------------------------ #
    # Engineered encounters
    # ------------------------------------------------------------------ #

    def _maybe_inject(self) -> None:
        """Plant engineered close approaches on a schedule.

        Synthetic fleets laid out on tidy shells almost never meet by chance
        (that's what safe spacing looks like), so without this the sandbox
        would sit quiet forever. Config::

            "injection": {"every_s": 2700, "tca_after_s": 2100,
                          "miss_km": [0.0, 1.4]}

        Each injection builds a new satellite (via fleets.inject_actor_encounter,
        then nudged along its track so the miss distance varies), owned by a
        different operator than the one it meets. The new object is retired
        an hour after its encounter.
        """
        spec = self.cfg.injection
        if not spec:
            return
        every = float(spec.get("every_s", 2700))
        while self.elapsed_s >= self.next_injection_s:
            self._inject_one(spec)
            self.next_injection_s += every

    # Crossing angles to try, best first. fleets.inject_actor_encounter's
    # closed-form inversion is reliable for shallow and near-antiparallel
    # crossings and often misses for steep ones (measured: 15/17 hits at 30
    # degrees, 1/17 at 110), so every attempt is checked and the next angle
    # or satellite is tried until one really meets.
    _INJECTION_ANGLES = (30.0, 170.0, 40.0, 55.0, 150.0, 70.0)

    def _inject_one(self, spec: dict) -> None:
        k = self.injection_counter
        self.injection_counter += 1
        rng = _rng(self.cfg.name, self.cfg.seed, "inject", self.cycle, k)
        ops = list(spec.get("operators") or self.operators)
        if len(ops) < 2:
            raise ValueError("injection needs at least two operators")
        op_a = ops[k % len(ops)]
        op_b = ops[(k + 1 + rng.randrange(len(ops) - 1)) % len(ops)]  # never op_a
        candidates = sorted(o.norad_id for o in self._base_objects if o.operator == op_a)
        lo, hi = spec.get("miss_km", [0.0, 1.4])
        step = self.cfg.screening_step_s
        tca_after = float(spec.get("tca_after_s", self.cfg.lookahead_s + 2 * self.cfg.tick_sim_s))
        tca_after = max(step, round(tca_after / step) * step)  # on the screening grid
        record = {
            "k": k,
            "b_operator": op_b,
            "b_id": 200_000 + k,  # above both the real and the synthetic id ranges
            "made_at_s": self.elapsed_s,
            "tca_s": self.elapsed_s + tca_after,
            "target_miss_km": round(rng.uniform(lo, hi), 3),
        }
        record["retire_s"] = record["tca_s"] + 3600.0
        first = rng.randrange(len(self._INJECTION_ANGLES))
        angles = self._INJECTION_ANGLES[first:] + self._INJECTION_ANGLES[:first]
        satrec = None
        for attempt in range(4):
            record["a_id"] = candidates[rng.randrange(len(candidates))]
            for angle in angles[:3]:
                record["crossing_angle_deg"] = angle
                satrec = self._build_injected_satrec(record, verify=True)
                if satrec is not None:
                    break
            if satrec is not None:
                break
        if satrec is None:
            self._event("injection", f"Could not engineer a crossing for {op_a} this time; skipped.")
            return
        self.injections.append(record)
        self._add_injected(record, satrec)
        self.counters["injections"] += 1
        self._event(
            "injection",
            f"New {op_b} satellite {op_b}-{record['b_id']} launched onto a path that crosses "
            f"{op_a}-{record['a_id']} in {(record['tca_s'] - self.elapsed_s) / 60:.0f} min.",
            operators=[op_a, op_b],
        )

    def _build_injected_satrec(self, record: dict, verify: bool) -> Optional[Satrec]:
        a = next(o for o in self._base_objects if o.norad_id == record["a_id"])
        made_at = self._at(record["made_at_s"])
        tca_after_min = (record["tca_s"] - record["made_at_s"]) / 60.0
        probe = TrackedObject(norad_id=a.norad_id, operator=a.operator, satrec=a.satrec)
        try:
            _, b = inject_actor_encounter(
                [probe], epoch=_calendar_tuple(made_at), watch_operator_a=a.operator,
                new_operator_b=record["b_operator"], tca_offset_min=tca_after_min,
                crossing_angle_deg=record["crossing_angle_deg"],
            )
        except (ValueError, RuntimeError):
            return None
        if verify:
            jd, fr = _jd_fr(self._at(record["tca_s"]))
            ea, ra, _ = a.satrec.sgp4(jd, fr)
            eb, rb, _ = b.satrec.sgp4(jd, fr)
            if ea or eb or math.dist(ra, rb) > 0.05:
                return None
        # inject_actor_encounter lands within metres. Slide B along its own
        # orbit by a fraction of a second so the miss distance is the one we
        # drew: d ~ v * dt, at ~7.5 km/s.
        dt_s = record["target_miss_km"] / 7.5
        delta = b.satrec.no_kozai * dt_s / 60.0  # rad/min * min
        return _shift_satrec(b.satrec, 99000, delta)  # SGP4 ignores the number

    def _add_injected(self, record: dict, satrec: Satrec) -> None:
        self._objects.append(TrackedObject(
            norad_id=record["b_id"], operator=record["b_operator"], satrec=satrec,
            is_actor=True, meta={"injection": record["k"]},
        ))
        self.ownership.setdefault(record["b_operator"], set()).add(record["b_id"])

    def _materialize_injection(self, record: dict) -> None:
        """Rebuild a saved injection on resume (it was verified when made)."""
        satrec = self._build_injected_satrec(record, verify=False)
        if satrec is not None:
            self._add_injected(record, satrec)

    def _retire_injections(self) -> None:
        keep = []
        for rec in self.injections:
            if self.elapsed_s >= rec["retire_s"]:
                self._objects = [o for o in self._objects if o.norad_id != rec["b_id"]]
                self.ownership.get(rec["b_operator"], set()).discard(rec["b_id"])
            else:
                keep.append(rec)
        self.injections = keep

    # ------------------------------------------------------------------ #
    # Screening
    # ------------------------------------------------------------------ #

    def _screen(self) -> None:
        """Screen the slice of time that has just come into view.

        Window = [now + lookahead - step, now + lookahead + tick + step]; a
        closest approach is accepted only strictly inside it (a sample on
        the edge may be halfway through an approach). Consecutive windows
        overlap by two steps, so an encounter can show up twice; the
        duplicate is dropped by pair and time.
        """
        if self.history is not None:
            self._screen_history()
            return
        cfg = self.cfg
        step = cfg.screening_step_s
        start_s = self.elapsed_s + cfg.lookahead_s - step
        duration = cfg.tick_sim_s + 2 * step
        jd, fr = _jd_fr(self._at(start_s))
        prop = propagate_window(self._objects, jd, fr, duration_s=duration, step_s=step)
        conjs = screen_conjunctions(self._objects, prop, pc_model=self.pc_model)
        self.counters["screened_pairs"] += len(conjs)

        for c in sorted(conjs, key=lambda c: (c.tca_offset_s, c.object_a.norad_id, c.object_b.norad_id)):
            if not (0 < c.tca_offset_s < duration):
                continue
            shared = c.pc >= cfg.pc_threshold
            if not shared and c.miss_distance_km > cfg.watch_miss_km:
                continue
            tca_s = start_s + c.tca_offset_s
            lo, hi = sorted((c.object_a.norad_id, c.object_b.norad_id))
            if any(e["pair"] == [lo, hi] and abs(e["tca_s"] - tca_s) <= 2 * step for e in self.encounters):
                continue
            self._handle_conjunction(c, tca_s, shared)

    def _handle_conjunction(self, c: Conjunction, tca_s: float, shared: bool) -> None:
        a, b = c.object_a, c.object_b
        if self.me is not None and self.me not in (a.operator, b.operator):
            return  # someone else's encounter: this twin never learns of it
        lo, hi = sorted((a.norad_id, b.norad_id))
        encounter_id = f"ENC-{lo}-{hi}-{int(round(self.total_sim_s + (tca_s - self.elapsed_s)))}"
        warning_s = tca_s - self.elapsed_s
        cross_operator = (not a.is_debris and not b.is_debris and a.operator != b.operator)
        record = {
            "id": encounter_id,
            "pair": [lo, hi],
            "labels": [a.label, b.label],
            "operators": [a.operator, b.operator],
            "tca_s": tca_s,
            "tca": _iso(self._at(tca_s)),
            "detected": _iso(self.sim_time),
            "warning_s": warning_s,
            "miss_km": round(float(c.miss_distance_km), 4),
            "pc": float(c.pc),
            "geometry": c.geometry_class,
            "rel_speed_km_s": round(float(c.relative_speed_km_s), 3),
            "engineered": bool(a.is_actor or b.is_actor),
            "status": "upcoming",
        }
        if not shared:
            # Below the sharing threshold: stays inside the owners' own twins.
            record["handling"] = "private"
            self.counters["private_watch"] += 1
            self.encounters.append(record)
            self._event(
                "private_watch",
                f"{a.label} and {b.label} will pass {record['miss_km'] * 1000:.0f} m apart; "
                f"Pc {c.pc:.1e} is under the sharing threshold, so nothing leaves the operators' own twins.",
                encounter=encounter_id,
            )
            return

        if not cross_operator:
            # Same-operator or debris encounter: the owner handles it alone,
            # nobody else needs to hear about it.
            owner = a.operator if not a.is_debris else b.operator
            record["handling"] = "internal"
            self.counters["internal"] += 1
            self.encounters.append(record)
            what = "debris" if (a.is_debris or b.is_debris) else "its own satellites"
            self._event(
                "internal",
                f"{owner} sees {a.label} and {b.label} ({what}) {record['miss_km'] * 1000:.0f} m apart, "
                f"Pc {c.pc:.1e}. One operator involved, so it plans its own move and shares nothing.",
                encounter=encounter_id, operators=[owner],
            )
            return

        # Cross-operator: the federated path.
        self.counters["encounters_shared"] += 1
        pair = abstract_conjunction(c, threshold=self.cfg.pc_threshold)
        sig_a, sig_b = (dataclasses.replace(s, encounter_id=encounter_id, tca_offset_s=float(warning_s)) for s in pair)
        if self.mode == "operator":
            self._handle_own_side(record, c, sig_a, sig_b, warning_s)
            return
        sent = []
        for sig in (sig_a, sig_b):
            behaviour = self.behaviour(sig.sender_operator)
            emit = behaviour == "honest" or (
                behaviour == "spoofer"
                and _rng(self.cfg.name, self.cfg.seed, "report", encounter_id, sig.sender_operator).random() < 0.3
            )
            if emit:
                self._emit(sig)
                sent.append(sig)
            else:
                self.counters["signals_withheld"] += 1
        record["handling"] = "federated"
        record["signals_sent"] = [s.sender_operator for s in sent]
        record["tier"] = sig_a.severity_tier

        if len(sent) == 2:
            plan = negotiate_maneuver(sig_a, sig_b)
            record["plan"] = {
                "primary": plan.primary_object, "primary_operator": plan.primary_operator,
                "secondary": plan.secondary_object, "secondary_operator": plan.secondary_operator,
            }
            self.counters["maneuvers_agreed"] += 1
            self._event(
                "maneuver",
                f"{a.label} x {b.label}: both operators signed a '{record['tier']}' signal "
                f"({c.geometry_class}, {warning_s / 3600:.1f} h warning). "
                + summarize_maneuver_plan(plan).split(" Both sides")[0],
                encounter=encounter_id, operators=[a.operator, b.operator],
            )
        else:
            silent = [s.sender_operator for s in (sig_a, sig_b) if s not in sent]
            self.counters["unilateral"] += 1
            self._event(
                "unilateral",
                f"{a.label} x {b.label}: {', '.join(silent)} stayed silent, so no joint plan could be "
                f"agreed. {' and '.join(s.sender_operator for s in sent) or 'Nobody'} must move alone.",
                encounter=encounter_id, operators=[a.operator, b.operator], silent=silent,
            )
        self.encounters.append(record)

    def _would_share(self, sig: AbstractedSignal, encounter_id: str) -> bool:
        behaviour = self.behaviour(sig.sender_operator)
        return behaviour == "honest" or (
            behaviour == "spoofer"
            and _rng(self.cfg.name, self.cfg.seed, "report", encounter_id, sig.sender_operator).random() < 0.3
        )

    def _handle_own_side(self, record: dict, c: Conjunction, sig_a: AbstractedSignal,
                         sig_b: AbstractedSignal, warning_s: float) -> None:
        """Operator mode: decide and sign this twin's half of a cross-operator
        encounter. Whether the partner signed, and so whether there is a
        joint plan, is only known later, from the federation."""
        mine = sig_a if sig_a.sender_operator == self.me else sig_b
        partner = sig_b if mine is sig_a else sig_a
        emit = self._would_share(mine, record["id"])
        if emit:
            self._emit(mine)
        else:
            self.counters["signals_withheld"] += 1
        record["handling"] = "federated"
        record["tier"] = sig_a.severity_tier
        record["i_signalled"] = emit
        record["signals_sent"] = [self.me] if emit else []
        # Both halves, in screening order, so the plan negotiate_maneuver
        # derives here matches the one the partner derives. The partner's
        # half holds only what this twin already knows (public label, its
        # own timing and geometry), nothing the partner sent.
        record["pair_signals"] = [_signal_to_dict(sig_a), _signal_to_dict(sig_b)]
        self.encounters.append(record)
        self._event(
            "federated",
            f"{mine.local_object_label} x {partner.local_object_label}: {self.me} "
            + (f"signed its '{record['tier']}' signal and sent it to the federation "
               f"({c.geometry_class}, {warning_s / 3600:.1f} h warning)."
               if emit else "kept quiet and sent nothing."),
            encounter=record["id"], operators=[a for a in record["operators"]],
        )

    def _mark_passed(self) -> None:
        now_end = self.elapsed_s + self.cfg.tick_sim_s
        for e in self.encounters:
            if e["status"] == "upcoming" and e["tca_s"] <= now_end:
                e["status"] = "passed"
                self.counters["passed"] += 1
        upcoming = [e for e in self.encounters if e["status"] == "upcoming"]
        passed = [e for e in self.encounters if e["status"] == "passed"]
        self.encounters = passed[-MAX_PASSED_IN_STATE:] + upcoming

    # ------------------------------------------------------------------ #
    # Federation round
    # ------------------------------------------------------------------ #

    def _phantom_signals(self, operator: str) -> List[AbstractedSignal]:
        """A spoofer invents encounters between objects nobody in the
        federation owns (the 'cry wolf' attack)."""
        rng = _rng(self.cfg.name, self.cfg.seed, "phantom", self.round_index, operator)
        out = []
        for _ in range(3):
            x = 700000 + rng.randrange(50000)
            eid = f"ENC-{x}-{x + 1 + rng.randrange(9)}-{int(self.total_sim_s) + rng.randrange(3600)}"
            out.append(AbstractedSignal(
                encounter_id=eid, sender_operator=operator, local_object_label=f"{operator}-?",
                tca_offset_s=float(rng.randrange(600, 7200)), geometry_class="head-on",
                severity_tier=rng.choice(["elevated", "critical"]),
            ))
        return out

    def _federation_round(self) -> None:
        if self.mode == "sky":
            self._sky_round()
            return
        if self.mode == "operator":
            self._operator_round()
            return
        self.round_index += 1
        for op in self.operators:
            if self.behaviour(op) == "spoofer":
                for sig in self._phantom_signals(op):
                    self._sign_and_log_signal(sig)
                    self.counters["phantom_signals"] += 1
        signals = [_signal_from_dict(s) for s in self.period_signals]
        reports = []
        for op in self.operators:
            report = build_operator_report(op, signals)
            if self.behaviour(op) == "free_rider":
                report = OperatorReport(operator=op, counts=np.zeros_like(report.counts))
            reports.append(report)
            self._ledger_append(sign_report(self.identities[op], report))

        before = {op: self.reputation.tier(op) for op in self.operators}
        evidence = evaluate_round(self.operators, signals, reports, self.ownership, self.reputation.params)
        self.reputation = update_scores(self.reputation, evidence)
        self.reputation.history = self.reputation.history[-MAX_REPUTATION_HISTORY:]
        after = {op: self.reputation.tier(op) for op in self.operators}

        honest = [r for r in reports if self.behaviour(r.operator) == "honest"]
        aggregates = {
            "mean": mean_aggregate(reports),
            "trimmed_mean": trimmed_mean_aggregate(reports),
            "reputation_weighted": reputation_weighted_aggregate(reports, self.reputation),
        }
        if len(reports) >= 4:
            aggregates["krum"] = krum_aggregate(reports)
        errors = {}
        if honest and len(honest) < len(reports):
            truth = np.mean([r.counts for r in honest], axis=0)
            errors = {k: round(float(np.linalg.norm(v - truth)), 3) for k, v in aggregates.items()}

        summary = {
            "round": self.round_index,
            "sim_time": _iso(self.sim_time),
            "n_signals": len(signals),
            "reports": {r.operator: [int(round(c)) for c in r.counts] for r in reports},
            "aggregates": {k: [round(float(x), 3) for x in v] for k, v in aggregates.items()},
            "errors": errors,
            "scores": {op: round(self.reputation.score(op), 3) for op in self.operators},
            "tiers": after,
            "evidence": {op: {"involved": ev.n_involved, "matched": ev.n_matched,
                              "missed": ev.n_missed, "unbacked": ev.n_unbacked}
                         for op, ev in evidence.items()},
        }
        self.rounds.append(summary)
        del self.rounds[:-MAX_ROUNDS_KEPT]
        self.counters["federation_rounds"] += 1
        self.period_signals = []

        moved = [f"{op} {before[op]} -> {after[op]}" for op in self.operators if before[op] != after[op]]
        text = (f"Federation round {self.round_index}: {len(signals)} signed signals, "
                f"{len(reports)} signed reports. Reputation: "
                + ", ".join(f"{op} {self.reputation.score(op):.2f}" for op in self.operators) + ".")
        if moved:
            text += " Tier change: " + "; ".join(moved) + "."
        if errors:
            text += (f" Error vs honest truth: mean {errors['mean']:.2f}, "
                     f"reputation-weighted {errors['reputation_weighted']:.2f}.")
        self._event("federation_round", text, round=self.round_index)

    def _sky_round(self) -> None:
        self.round_index += 1
        for op in self.operators:
            if self.behaviour(op) == "spoofer":
                self.counters["phantom_signals"] += 3
        self.counters["federation_rounds"] += 1
        self.period_signals = []
        self._event("federation_round",
                    f"Federation round {self.round_index}: each operator's process sends a signed report "
                    f"to the federation process, which scores them.", round=self.round_index)

    def _operator_round(self) -> None:
        self.round_index += 1
        me = self.me
        if self.behaviour(me) == "spoofer":
            for sig in self._phantom_signals(me):
                self._emit(sig)
                self.counters["phantom_signals"] += 1
        signals = [_signal_from_dict(s) for s in self.period_signals]
        report = build_operator_report(me, signals)
        if self.behaviour(me) == "free_rider":
            report = OperatorReport(operator=me, counts=np.zeros_like(report.counts))
        message = encode_signed_report(sign_report(self.identities[me], report))
        message["round"] = self.round_index
        self.outbox.append(message)
        self.counters["federation_rounds"] += 1
        self.period_signals = []
        counts = " / ".join(str(int(round(x))) for x in report.counts)
        self._event("report", f"{me} signed its round {self.round_index} report "
                              f"(threshold / elevated / critical: {counts}) and sent it to the federation.",
                    round=self.round_index)

    # ------------------------------------------------------------------ #
    # History worlds (fleet type "history"; see history_world.py)
    # ------------------------------------------------------------------ #

    def _jd(self, cycle_offset_s: float) -> float:
        from .history_world import jd_of
        return jd_of(self._at(cycle_offset_s))

    def _refresh_sets(self) -> None:
        """Point every object's ``satrec`` at the element set this party
        would draw it from right now (the map, ground tracks, views): the
        nearest set for the truth and for an operator's own satellites, the
        newest published one for everybody else's."""
        hist = self.history.hist
        jd = self._jd(self.elapsed_s)
        sets = hist.nearest(self._rows, jd)
        if self.mode == "operator":
            asof = hist.asof(self._rows, jd)
            others = np.array([o.operator != self.me for o in self._objects])
            sets = np.where(others & (asof >= 0), asof, sets)
        for o, k in zip(self._objects, sets):
            o.satrec = hist.satrec(int(k))

    def _state_fn(self, obj: TrackedObject, kind: str, jd_now: float):
        """``jd_array -> (r, v, ok)`` for one object, as ``kind`` knows it:
        "public" (as-of prediction), "own" (owner's ephemeris, no simulated
        manoeuvres) or "truth" (with them)."""
        H = self.history
        rows = np.array([self._row_of[obj.norad_id]])
        if kind == "public":
            def fn(jd):
                r, v, ok, _ = H.predicted(rows, jd_now, jd)
                return r[0], v[0], ok[0]
        elif kind == "own":
            def fn(jd):
                r, v, ok = H.truth(rows, jd)
                return r[0], v[0], ok[0]
        else:
            ids = np.array([obj.norad_id])

            def fn(jd):
                r, v, ok = H.truth(rows, jd, self.maneuvers, ids)
                return r[0], v[0], ok[0]
        return fn

    def _assess(self, c: Conjunction, tca_jd: float, jd_now: float, viewer: Optional[str]):
        """One party's view of a publicly detected encounter (``viewer=None``:
        public data only). Returns history_world.Assessment or None."""
        from .history_world import Assessment, pc_with, refine_tca
        H = self.history
        objs = (c.object_a, c.object_b)
        own = [viewer is not None and o.operator == viewer for o in objs]
        fns = [self._state_fn(o, "own" if mine else "public", jd_now) for o, mine in zip(objs, own)]
        found = refine_tca(fns[0], fns[1], tca_jd, H.settings["refine_window_s"])
        if found is None:
            return None
        miss, tca, ra, va, rb, vb = found
        sigmas, other = [], None
        for o, mine in zip(objs, own):
            if mine:
                sigmas.append((H.precise,) * 3)
                continue
            row = self._row_of[o.norad_id]
            k = int(H.hist.asof(np.array([row]), jd_now)[0])
            if k < 0:
                return None
            sig, age_h = H.sigma_public(row, k, tca)
            sigmas.append(sig)
            other = (sig, age_h)
        pc = pc_with(ra, va, rb, vb, sigmas[0], sigmas[1])
        if other is not None and any(own):
            return Assessment(miss, pc, tca, other[0], other[1])
        return Assessment(miss, pc, tca)

    def _truth_miss(self, c: Conjunction, tca_jd: float, with_maneuvers: bool = True):
        from .history_world import dt_of, refine_tca
        kind = "truth" if with_maneuvers else "own"
        fns = [self._state_fn(o, kind, 0.0) for o in (c.object_a, c.object_b)]
        found = refine_tca(fns[0], fns[1], tca_jd, self.history.settings["refine_window_s"])
        if found is None:
            return None
        return {"miss_km": round(found[0], 4), "tca": _iso(dt_of(found[1]))}

    def _proxies(self, sets) -> List[TrackedObject]:
        """The world's objects, each drawing from element set ``sets[i]``
        (objects with none yet are left out)."""
        hist = self.history.hist
        return [dataclasses.replace(o, satrec=hist.satrec(int(k)))
                for o, k in zip(self._objects, sets) if k >= 0]

    def _pair_filter(self):
        members = set(self.operators)
        me = self.me

        def keep(a, b):
            if me is not None and me not in (a.operator, b.operator):
                return False
            return a.operator in members or b.operator in members
        return keep

    def _history_window(self):
        """This tick's slice of the look-ahead: approaches whose TCA falls in
        [now + lookahead, now + lookahead + tick) belong to this tick, so
        each one is accepted exactly once. The screened span reaches a step
        either side so nothing at the edges is lost."""
        cfg = self.cfg
        step = cfg.screening_step_s
        lo_s = self.elapsed_s + cfg.lookahead_s
        start_s = lo_s - step
        duration = cfg.tick_sim_s + 2 * step
        jd, fr = _jd_fr(self._at(start_s))
        return start_s, duration, jd, fr, lo_s, lo_s + cfg.tick_sim_s

    def _screen_history(self) -> None:
        """Refined public screening (every party the same: the newest
        published set of every object), then each side's own assessment; in
        the sky, also the truth, to catch what the public catalogue missed.
        screening.screen_conjunctions_refined finds the closest approach
        between samples, so fast crossings are not skipped."""
        from .history_world import jd_of
        from .screening import screen_conjunctions_refined
        cfg = self.cfg
        H = self.history
        step = cfg.screening_step_s
        start_s, duration, jd0, fr0, lo_s, hi_s = self._history_window()
        jd_now = jd_of(self.sim_time)
        objs = self._proxies(H.hist.asof(self._rows, jd_now))
        conjs = screen_conjunctions_refined(
            objs, jd0, fr0, duration, step_s=step, screening_distance_km=float(H.settings["candidate_km"]),
            pair_filter=self._pair_filter(), chunk_s=duration + step)
        self.counters["screened_pairs"] += len(conjs)
        by_id = {o.norad_id: o for o in self._objects}
        for c in sorted(conjs, key=lambda c: (c.tca_offset_s, c.object_a.norad_id, c.object_b.norad_id)):
            tca_s = start_s + c.tca_offset_s
            if not (lo_s <= tca_s < hi_s):
                continue
            c.object_a, c.object_b = by_id[c.object_a.norad_id], by_id[c.object_b.norad_id]
            lo, hi = sorted((c.object_a.norad_id, c.object_b.norad_id))
            if any(e["pair"] == [lo, hi] and abs(e["tca_s"] - tca_s) <= 2 * step and not e.get("observer_only")
                   for e in self.encounters):
                continue
            self._handle_history_conjunction(c, tca_s, jd_now)
        if self.mode != "operator":
            self._screen_truth()

    def _screen_truth(self) -> None:
        """Close approaches that really happen that no public-screening
        record covers. Only the observer sees these: no operator flagged
        them. Found by the refined screen on the nearest element set at the
        middle of the window (the SGP4 refinement needs one set per object);
        the miss is then re-measured on the blended truth, applied
        manoeuvres included."""
        from .screening import screen_conjunctions_refined
        cfg = self.cfg
        start_s, duration, jd0, fr0, lo_s, hi_s = self._history_window()
        objs = self._proxies(self.history.hist.nearest(self._rows, self._jd(start_s + duration / 2)))
        conjs = screen_conjunctions_refined(
            objs, jd0, fr0, duration, step_s=cfg.screening_step_s,
            screening_distance_km=cfg.watch_miss_km + 0.5, pair_filter=self._pair_filter(),
            chunk_s=duration + cfg.screening_step_s, refine=True)
        by_id = {o.norad_id: o for o in self._objects}
        for c in sorted(conjs, key=lambda c: (c.tca_offset_s, c.object_a.norad_id, c.object_b.norad_id)):
            tca_s = start_s + c.tca_offset_s
            if not (lo_s <= tca_s < hi_s):
                continue
            c.object_a, c.object_b = by_id[c.object_a.norad_id], by_id[c.object_b.norad_id]
            lo, hi = sorted((c.object_a.norad_id, c.object_b.norad_id))
            if any(e["pair"] == [lo, hi] and abs(e["tca_s"] - tca_s) <= 900 for e in self.encounters):
                continue
            truth = self._truth_miss(c, self._jd(tca_s))
            if truth is None or truth["miss_km"] > cfg.watch_miss_km:
                continue
            a, b = c.object_a, c.object_b
            self.counters["unseen"] = self.counters.get("unseen", 0) + 1
            self.encounters.append({
                "id": f"TRUTH-{lo}-{hi}-{int(round(self.total_sim_s + (tca_s - self.elapsed_s)))}",
                "pair": [lo, hi], "labels": [a.label, b.label], "operators": [a.operator, b.operator],
                "tca_s": tca_s, "tca": truth["tca"], "detected": _iso(self.sim_time),
                "warning_s": tca_s - self.elapsed_s, "miss_km": truth["miss_km"], "pc": 0.0,
                "geometry": c.geometry_class, "rel_speed_km_s": round(float(c.relative_speed_km_s), 3),
                "engineered": False, "status": "upcoming", "handling": "unseen",
                "observer_only": True, "truth": truth,
            })
            self._event(
                "unseen",
                f"Truth: {a.label} and {b.label} really pass {truth['miss_km'] * 1000:.0f} m apart at "
                f"{truth['tca']} ({c.relative_speed_km_s:.1f} km/s), but nobody's twin flagged it: "
                f"the published element sets put them further apart.",
            )

    def _handle_history_conjunction(self, c: Conjunction, tca_s: float, jd_now: float) -> None:
        cfg = self.cfg
        a, b = c.object_a, c.object_b
        if self.me is not None and self.me not in (a.operator, b.operator):
            return
        members = [op for op in dict.fromkeys((a.operator, b.operator)) if op in self.operators]
        if not members:
            return
        tca_jd = self._jd(tca_s)
        sides = [self.me] if self.mode == "operator" else members
        assessed = {op: self._assess(c, tca_jd, jd_now, op) for op in sides}
        if any(s is None for s in assessed.values()):
            return
        above = {op: s.pc >= cfg.pc_threshold for op, s in assessed.items()}
        near = {op: s.miss_km <= cfg.watch_miss_km for op, s in assessed.items()}
        if not any(above.values()) and not any(near.values()):
            return
        public = self._assess(c, tca_jd, jd_now, None)
        cross = len(members) == 2
        lo, hi = sorted((a.norad_id, b.norad_id))
        encounter_id = f"ENC-{lo}-{hi}-{int(round(self.total_sim_s + (tca_s - self.elapsed_s)))}"
        warning_s = tca_s - self.elapsed_s
        # A real encounter the truth screening logged first is now covered.
        self.encounters = [e for e in self.encounters if not (
            e.get("observer_only") and e["pair"] == [lo, hi] and abs(e["tca_s"] - tca_s) <= 900)]
        side_info = {}
        for op, s in assessed.items():
            handling = ("federated" if cross else "internal") if above[op] else ("private" if near[op] else None)
            side_info[op] = {**s.as_dict(), "handling": handling}
        record = {
            "id": encounter_id, "pair": [lo, hi], "labels": [a.label, b.label],
            "operators": [a.operator, b.operator], "tca_s": tca_s, "tca": _iso(self._at(tca_s)),
            "detected": _iso(self.sim_time), "warning_s": warning_s,
            "miss_km": round(public.miss_km, 4) if public else round(float(c.miss_distance_km), 4),
            "pc": float(public.pc) if public else 0.0,
            "geometry": c.geometry_class, "rel_speed_km_s": round(float(c.relative_speed_km_s), 3),
            "engineered": False, "status": "upcoming", "sides": side_info,
        }
        if self.mode != "operator":
            record["truth"] = self._truth_miss(c, tca_jd)

        def own_text(op):
            s = side_info[op]
            text = f"{op}'s own assessment {s['miss_km'] * 1000:.0f} m, Pc {s['pc']:.1e}"
            if "other_tle_age_h" in s:
                text += (f" (the other object from a {s['other_tle_age_h']:.0f} h old element set, "
                         f"along-track sigma {s['sigma_other_rtn_km'][1]:.1f} km)")
            return text

        if not cross or not any(above.values()):
            owner = members[0] if not cross else (self.me or members[0])
            handling = side_info[owner]["handling"] if not cross else "private"
            if handling is None:
                return
            record["handling"] = handling
            self.encounters.append(record)
            if handling == "internal":
                self.counters["internal"] += 1
                what = "a non-member object" if (a.is_debris or b.is_debris) else "its own satellites"
                self._event("internal", f"{owner} sees {a.label} and {b.label} ({what}): {own_text(owner)}. "
                                        f"One operator involved, so it plans its own move and shares nothing.",
                            encounter=encounter_id, operators=[owner])
            else:
                self.counters["private_watch"] += 1
                self._event("private_watch", f"{a.label} and {b.label}: "
                            + "; ".join(own_text(op) for op in side_info)
                            + ". Under the sharing threshold, so nothing leaves the operators' own twins.",
                            encounter=encounter_id)
            return

        # Cross-operator, at least one side over the threshold: the federated path.
        self.counters["encounters_shared"] += 1

        def signal_for(obj):
            pc = side_info.get(obj.operator, side_info.get(self.me))["pc"]
            return AbstractedSignal(encounter_id=encounter_id, sender_operator=obj.operator,
                                    local_object_label=obj.label, tca_offset_s=float(warning_s),
                                    geometry_class=c.geometry_class,
                                    severity_tier=severity_tier(max(pc, cfg.pc_threshold), cfg.pc_threshold))

        sig_a, sig_b = signal_for(a), signal_for(b)
        if self.mode == "operator":
            self._handle_own_side(record, c, sig_a, sig_b, warning_s)
            return
        sent = []
        for sig in (sig_a, sig_b):
            if not above[sig.sender_operator]:
                continue
            if self._would_share(sig, encounter_id):
                self._emit(sig)
                sent.append(sig)
            else:
                self.counters["signals_withheld"] += 1
        record["handling"] = "federated"
        record["signals_sent"] = [s.sender_operator for s in sent]
        record["below_threshold"] = [op for op in members if not above[op]]
        record["tier"] = (sig_a if above[a.operator] else sig_b).severity_tier
        if len(sent) == 2:
            plan = negotiate_maneuver(sig_a, sig_b)
            record["plan"] = {
                "primary": plan.primary_object, "primary_operator": plan.primary_operator,
                "secondary": plan.secondary_object, "secondary_operator": plan.secondary_operator,
            }
            self.counters["maneuvers_agreed"] += 1
            moved = self._apply_avoidance(record, c, tca_jd)
            self._event(
                "maneuver",
                f"{a.label} x {b.label}: both operators signed ({c.geometry_class}, "
                f"{warning_s / 3600:.1f} h warning). " + summarize_maneuver_plan(plan).split(" Both sides")[0]
                + (f" Applied to the truth: {moved}." if moved else ""),
                encounter=encounter_id, operators=[a.operator, b.operator],
            )
        else:
            quiet = [s.sender_operator for s in (sig_a, sig_b) if s not in sent and above[s.sender_operator]]
            below = record["below_threshold"]
            self.counters["unilateral"] += 1
            reasons = []
            if below:
                reasons.append(f"{', '.join(below)}'s own assessment was under the threshold")
            if quiet:
                reasons.append(f"{', '.join(quiet)} stayed silent")
            self._event(
                "unilateral",
                f"{a.label} x {b.label}: " + " and ".join(reasons) + ", so no joint plan could be agreed. "
                f"{' and '.join(s.sender_operator for s in sent) or 'Nobody'} must move alone. "
                + "; ".join(own_text(op) for op in side_info) + ".",
                encounter=encounter_id, operators=[a.operator, b.operator], silent=quiet,
            )
        self.encounters.append(record)

    def _predicted_miss_with(self, c: Conjunction, tca_jd: float, jd_now: float, maneuvers: List[dict]) -> float:
        from .history_world import apply_maneuvers, refine_tca
        fns = []
        for o in (c.object_a, c.object_b):
            base = self._state_fn(o, "public", jd_now)

            def fn(jd, base=base, nid=o.norad_id):
                r, v, ok = base(jd)
                r, v = r[None].copy(), v[None]
                apply_maneuvers(r, v, np.array([nid]), np.asarray(jd, dtype=float), maneuvers)
                return r[0], v[0], ok
            fns.append(fn)
        found = refine_tca(fns[0], fns[1], tca_jd, self.history.settings["refine_window_s"])
        return found[0] if found else 0.0

    def _apply_avoidance(self, record: dict, c: Conjunction, tca_jd: float) -> str:
        """Give both satellites of an agreed plan their burn (see
        history_world). Returns a description, '' if there was no time."""
        from .history_world import dt_of, jd_of, plan_burn
        burn_jd = jd_of(self.sim_time)
        lead_s = (tca_jd - burn_jd) * 86400.0
        if lead_s < 600:
            return ""
        before = self._truth_miss(c, tca_jd, with_maneuvers=True)
        by_label = {c.object_a.label: c.object_a, c.object_b.label: c.object_b}
        shift = float(self.history.settings["maneuver_shift_km"])

        def burns(flip: bool) -> List[dict]:
            out = []
            for label, direction in ((record["plan"]["primary"], "+"), (record["plan"]["secondary"], "-")):
                if flip:
                    direction = "-" if direction == "+" else "+"
                obj = by_label[label]
                n = obj.satrec.no_kozai / 60.0
                out.append({"norad_id": obj.norad_id, "encounter": record["id"], "burn_jd": burn_jd,
                            "tca_jd": tca_jd, "dv_km_s": plan_burn(n, lead_s, shift, direction),
                            "n_rad_s": n, "direction": direction, "label": label})
            return out

        # The plan fixes only that the two moves are opposite. Which way is
        # "+" is settled from public data alone (both operators can compute
        # it and agree): the orientation that opens up the predicted miss.
        options = [burns(False), burns(True)]
        predicted = [self._predicted_miss_with(c, tca_jd, burn_jd, ms) for ms in options]
        chosen = options[int(predicted[1] > predicted[0])]
        self.maneuvers.extend(chosen)
        dvs = [f"{m['label']} {m['dv_km_s'] * 1e5:+.1f} cm/s" for m in chosen]
        after = self._truth_miss(c, tca_jd, with_maneuvers=True)
        if before and after:
            record["truth"] = {**after, "miss_km_without_maneuver": before["miss_km"]}
            return (f"burns {', '.join(dvs)} along-track at {_iso(dt_of(burn_jd))}; "
                    f"true miss {before['miss_km'] * 1000:.0f} m -> {after['miss_km'] * 1000:.0f} m")
        return f"burns {', '.join(dvs)}"

    def _history_before_tick(self) -> bool:
        """End-of-file handling and housekeeping. False: don't advance."""
        cfg = self.cfg
        H = self.history
        horizon = self._jd(self.elapsed_s + cfg.lookahead_s + cfg.tick_sim_s + 2 * cfg.screening_step_s)
        if horizon > H.hist.last_epoch:
            if H.settings["at_end"] == "loop":
                self._start_new_cycle()
            else:
                if not self.finished:
                    self.finished = True
                    self._event("end", f"End of the element-set file: the last set's epoch is "
                                       f"{_iso(H.end_time())}, so there is no truth beyond it. "
                                       f"The clock has stopped at {_iso(self.sim_time)} (at_end: stop).")
                return False
        now = self._jd(self.elapsed_s)
        self.maneuvers = [m for m in self.maneuvers if 2 * m["tca_jd"] - m["burn_jd"] > now]
        self._refresh_sets()
        return True

    def _report_jumps(self, old_elapsed_s: float) -> None:
        """Once per simulated hour: element sets published in the hour that
        jump more than the error model allows (real manoeuvres)."""
        hour_old, hour_new = int(old_elapsed_s // 3600), int(self.elapsed_s // 3600)
        if hour_new <= hour_old:
            return
        lo, hi = self._jd(hour_old * 3600.0), self._jd(hour_new * 3600.0)
        jumps = self.history.jumps_between(lo, hi, self._id_set)
        if not jumps:
            return
        self.counters["tle_jumps"] = self.counters.get("tle_jumps", 0) + len(jumps)
        owner = {o.norad_id: o for o in self._objects}
        n_members = sum(1 for j in jumps if not owner[j["norad_id"]].is_debris)
        top = sorted(jumps, key=lambda j: -j["n_sigma"])[:3]
        text = ", ".join(f"{j['name'] or owner[j['norad_id']].label} "
                         f"({'/'.join(f'{x:+.1f}' for x in j['rtn_km'])} km R/T/C, {j['n_sigma']:.0f} sigma)"
                         for j in top)
        self._event("tle_jump",
                    f"{len(jumps)} element sets published in the last hour ({n_members} of them federation "
                    f"members') jumped more than {self.history.settings['jump_sigma']:g} robust sigma from the "
                    f"previous set: real manoeuvres, most likely. Largest: {text}.",
                    n=len(jumps))

    # ------------------------------------------------------------------ #
    # Tick
    # ------------------------------------------------------------------ #

    def _start_new_cycle(self) -> None:
        self.cycle += 1
        self.counters["cycles"] += 1
        for rec in self.injections:
            self._objects = [o for o in self._objects if o.norad_id != rec["b_id"]]
            self.ownership.get(rec["b_operator"], set()).discard(rec["b_id"])
        self.injections = []
        self.elapsed_s = 0.0
        self.next_injection_s = 0.0
        self.encounters = [e for e in self.encounters if e["status"] == "passed"]
        if self.history is not None:
            self.maneuvers = []
            self._refresh_sets()
            self._event("cycle", f"Reached the end of the element-set file ({_iso(self.history.end_time())}); "
                                 f"replaying it from {_iso(self.start_time)} (cycle {self.cycle}, at_end: loop).")
            return
        self._event(
            "cycle",
            f"Clock reset to the catalogue epoch {_iso(self.start_time)} (cycle {self.cycle}): "
            f"TLEs drift by kilometres within days, so the world replays its accurate window "
            f"instead of extrapolating stale orbits.",
        )

    def tick(self) -> None:
        """Advance the world by one tick (``tick_sim_s`` simulated seconds)."""
        t0 = time.perf_counter()
        self._tick_tag = self.tick_index + 1
        cycle_s = (self.cfg.cycle_days or 0) * 86400.0
        if cycle_s and self.elapsed_s >= cycle_s:
            self._start_new_cycle()
        if self.history is not None and not self._history_before_tick():
            self._tick_tag = self.tick_index
            return
        self._retire_injections()
        self._maybe_inject()
        self._screen()
        self._mark_passed()
        old_elapsed = self.elapsed_s
        self.elapsed_s += self.cfg.tick_sim_s
        if self.history is not None and self.mode != "operator":
            self._report_jumps(old_elapsed)
        self.total_sim_s += self.cfg.tick_sim_s
        self.tick_index += 1
        self.counters["ticks"] += 1
        if self.total_sim_s >= self.next_round_s:
            self._federation_round()
            self.next_round_s += self.cfg.federation_period_s
        self.last_tick_wall_s = time.perf_counter() - t0
        self._tick_tag = self.tick_index

    # ------------------------------------------------------------------ #
    # Live view
    # ------------------------------------------------------------------ #

    def tracks(self) -> dict:
        """Lat/lon of the displayed subset at the current instant."""
        upcoming_ids = set()
        for e in self.encounters:
            if e["status"] == "upcoming":
                upcoming_ids.update(e["pair"])
        objs = [o for o in self._objects if o.norad_id in self._display_ids or o.is_actor or o.norad_id in upcoming_ids]
        if not objs:
            return {"operators": [], "points": []}
        jd, fr = _jd_fr(self.sim_time)
        err, r, _ = SatrecArray([o.satrec for o in objs]).sgp4(np.array([jd]), np.array([fr]))
        gmst = _gmst_rad(jd, fr)
        names = sorted({o.operator for o in objs})
        index = {n: i for i, n in enumerate(names)}
        points = []
        for o, e, pos in zip(objs, err[:, 0], r[:, 0, :]):
            if e != 0:
                continue
            x, y, z = (float(v) for v in pos)
            rad = math.sqrt(x * x + y * y + z * z)
            lat = math.degrees(math.asin(z / rad))
            lon = (math.degrees(math.atan2(y, x) - gmst) + 540.0) % 360.0 - 180.0
            flag = 2 if o.norad_id in upcoming_ids else (1 if o.is_actor else 0)
            points.append([round(lat, 2), round(lon, 2), index[o.operator], flag, o.norad_id,
                           round(rad - 6378.137)])
        return {"operators": names, "points": points}

    def source_text(self) -> str:
        if self.history is not None:
            return self.history.source_text()
        return "synthetic fleets" if self.cfg.fleet["type"] == "synthetic" else "real TLE catalogue"

    def history_info(self) -> dict:
        H = self.history
        return {
            "end_time": _iso(H.end_time()), "at_end": H.settings["at_end"],
            "truth": f"nearest element set, blended over {H.blend_s:g} s at each switch",
            "prediction": "newest element set published at or before now",
            "precise_sigma_km": H.precise,
            "error_model": H.model.table,
            "maneuvers_active": len(self.maneuvers),
        }

    def state(self, wall_interval_s: float | None = None) -> dict:
        upcoming = sorted((e for e in self.encounters if e["status"] == "upcoming"), key=lambda e: e["tca_s"])
        passed = [e for e in self.encounters if e["status"] == "passed"][::-1]
        for e in upcoming:
            e["time_to_tca_s"] = e["tca_s"] - self.elapsed_s
        fleet_counts: Dict[str, int] = {}
        for o in self._objects:
            fleet_counts[o.operator] = fleet_counts.get(o.operator, 0) + 1
        return {
            "version": STATE_VERSION,
            "world": {
                "name": self.cfg.name, "title": self.cfg.title or self.cfg.name,
                "description": self.cfg.description, "fleet": fleet_counts,
                "operators": self.operators, "pc_model": self.cfg.pc_model,
                "pc_threshold": self.cfg.pc_threshold, "speed": self.cfg.speed,
                "tick_sim_s": self.cfg.tick_sim_s, "lookahead_s": self.cfg.lookahead_s,
                "federation_period_s": self.cfg.federation_period_s,
                "cycle_days": self.cfg.cycle_days,
                "behaviours": {op: self.cfg.behaviours.get(op, "honest") for op in self.operators},
                "start_time": _iso(self.start_time),
                "source": self.source_text(),
                **({"history": self.history_info()} if self.history is not None else {}),
            },
            "clock": {
                "sim_time": _iso(self.sim_time), "tick": self.tick_index, "cycle": self.cycle,
                "cycle_elapsed_s": self.elapsed_s, "total_sim_s": self.total_sim_s,
                "wall_started": self.wall_started, "wall_updated": _iso(datetime.now(timezone.utc)),
                "last_tick_compute_s": round(self.last_tick_wall_s, 3),
                "wall_interval_s": wall_interval_s,
                "finished": self.finished,
                "end_time": _iso(self.history.end_time()) if self.history is not None else None,
                "at_end": self.history.settings["at_end"] if self.history is not None else None,
            },
            "counters": self.counters,
            "upcoming": upcoming[:40],
            "n_upcoming": len(upcoming),
            "passed": passed[:20],
            "events": self.recent_events[::-1],
            "reputation": {
                "scores": {op: round(self.reputation.score(op), 3) for op in self.operators},
                "tiers": {op: self.reputation.tier(op) for op in self.operators},
                "history": [{op: round(s.get(op, float("nan")), 3) for op in self.operators}
                            for s in self.reputation.history[-100:]],
            },
            "rounds": self.rounds[-30:][::-1],
            "ledger": {"entries": len(self.ledger.entries), "tail_hash": self.ledger.tail_hash(),
                       "verified_on_load": self.ledger_verified},
            "tracks": self.tracks(),
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _checkpoint(self) -> dict:
        return {
            "version": STATE_VERSION,
            "fingerprint": self.cfg.fingerprint(),
            "mode": self.mode, "operator": self.me,
            "tick_index": self.tick_index, "cycle": self.cycle,
            "elapsed_s": self.elapsed_s, "total_sim_s": self.total_sim_s,
            "wall_started": self.wall_started,
            "injections": self.injections, "injection_counter": self.injection_counter,
            "next_injection_s": self.next_injection_s, "next_round_s": self.next_round_s,
            "round_index": self.round_index, "encounters": self.encounters,
            "period_signals": self.period_signals,
            "reputation": {"scores": self.reputation.scores, "rounds": self.reputation.rounds,
                           "history": self.reputation.history},
            "rounds": self.rounds, "recent_events": self.recent_events,
            "counters": self.counters,
            "ledger_entries": len(self.ledger.entries), "ledger_tail": self.ledger.tail_hash(),
            "maneuvers": self.maneuvers, "finished": self.finished,
        }

    def save(self, wall_interval_s: float | None = None) -> None:
        """Flush this tick: ledger and events first (append-only), then the
        checkpoint that says how much of them is valid, then the live state."""
        if self._pending_ledger_lines:
            with open(self.run_dir / "ledger.jsonl", "a", encoding="utf-8") as fh:
                fh.write("\n".join(self._pending_ledger_lines) + "\n")
            self._pending_ledger_lines = []
        if self._pending_events:
            with open(self.run_dir / "events.jsonl", "a", encoding="utf-8") as fh:
                fh.write("\n".join(json.dumps(e) for e in self._pending_events) + "\n")
            self._pending_events = []
        _write_atomic(self.run_dir / "checkpoint.json", json.dumps(self._checkpoint()))
        _write_atomic(self.run_dir / "state.json", json.dumps(self.state(wall_interval_s)))

    @classmethod
    def open(cls, config: WorldConfig, run_dir: str | Path, repo_root: str | Path = ".",
             reset: bool = False, **mode) -> "World":
        """Resume the world in ``run_dir`` or start it fresh. ``mode`` is
        passed to the constructor (``mode=``, ``operator=``, ``identity=``)."""
        run_dir = Path(run_dir)
        if reset and run_dir.exists():
            for name in ("checkpoint.json", "state.json", "ledger.jsonl", "events.jsonl"):
                (run_dir / name).unlink(missing_ok=True)
        world = cls(config, run_dir, repo_root, **mode)
        ckpt_path = run_dir / "checkpoint.json"
        if not ckpt_path.exists():
            for name in ("ledger.jsonl", "events.jsonl"):
                (run_dir / name).unlink(missing_ok=True)
            world._event("start", f"World '{config.title or config.name}' created at {_iso(world.start_time)} "
                                  f"with {len(world._objects):,} objects across {len(world.operators)} operators.")
            return world
        world._restore(json.loads(ckpt_path.read_text(encoding="utf-8")))
        return world

    def _restore(self, ck: dict) -> None:
        if (ck.get("mode", "all"), ck.get("operator")) != (self.mode, self.me):
            raise RuntimeError(
                f"world '{self.cfg.name}': {self.run_dir} holds a {ck.get('mode', 'all')} run, "
                f"not a {self.mode} one. Use another folder or --reset."
            )
        if ck.get("fingerprint") != self.cfg.fingerprint():
            raise RuntimeError(
                f"world '{self.cfg.name}': the config changed since this run was saved "
                f"(fleet or rules differ). Start it again with --reset."
            )
        for key in ("tick_index", "cycle", "elapsed_s", "total_sim_s", "wall_started", "injections",
                    "injection_counter", "next_injection_s", "next_round_s", "round_index",
                    "encounters", "period_signals", "rounds", "recent_events"):
            setattr(self, key, ck[key])
        self.counters.update(ck["counters"])
        rep = ck["reputation"]
        self.reputation = ReputationState(scores=dict(rep["scores"]), rounds=rep["rounds"],
                                          history=list(rep["history"]))
        self.maneuvers = list(ck.get("maneuvers", []))
        self.finished = bool(ck.get("finished", False))
        for rec in self.injections:
            self._materialize_injection(rec)
        if self.history is not None:
            self._refresh_sets()
        self._tick_tag = self.tick_index
        self._load_ledger(ck["ledger_entries"], ck["ledger_tail"])
        # Events written after the last checkpoint belong to a tick that will
        # now be redone; drop them so the file has no duplicates.
        self._truncate_jsonl(self.run_dir / "events.jsonl",
                             keep=lambda e: e["tick"] <= self.tick_index)
        self._event("resume", f"Resumed at tick {self.tick_index} ({_iso(self.sim_time)}); "
                              f"ledger of {len(self.ledger.entries)} entries re-verified: "
                              f"{'intact' if self.ledger_verified else 'TAMPERED'}.")

    def _load_ledger(self, n_entries: int, expected_tail: str) -> None:
        path = self.run_dir / "ledger.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        lines = [ln for ln in lines if ln.strip()]
        if len(lines) > n_entries:  # written by a tick that never checkpointed
            lines = lines[:n_entries]
            path.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
        ok = len(lines) == n_entries
        for line in lines:
            rec = json.loads(line)
            if rec["kind"] == "signal":
                signed = SignedSignal(signal=_signal_from_dict(rec["payload"]),
                                      signature=bytes.fromhex(rec["signature"]),
                                      public_key=bytes.fromhex(rec["public_key"]))
            else:
                p = rec["payload"]
                signed = SignedReport(
                    report=OperatorReport(operator=p["operator"], counts=np.array(p["counts"], dtype=float),
                                          is_adversarial=p["is_adversarial"]),
                    signature=bytes.fromhex(rec["signature"]), public_key=bytes.fromhex(rec["public_key"]))
            entry = self.ledger.append(signed, timestamp=rec["timestamp"])
            if entry.entry_hash != rec["entry_hash"] or entry.prev_hash != rec["prev_hash"]:
                ok = False
        chain_ok, _ = self.ledger.verify_chain(self.registry)
        tail = self.ledger.tail_hash() if self.ledger.entries else GENESIS_HASH
        self.ledger_verified = bool(ok and chain_ok and tail == expected_tail)

    @staticmethod
    def _truncate_jsonl(path: Path, keep) -> None:
        if not path.exists():
            return
        kept = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    if keep(json.loads(line)):
                        kept.append(line)
                except (ValueError, KeyError):
                    continue
        path.write_text("".join(ln + "\n" for ln in kept), encoding="utf-8")


def load_world_configs(worlds_dir: str | Path) -> Dict[str, WorldConfig]:
    out = {}
    for path in sorted(Path(worlds_dir).glob("*.json")):
        cfg = WorldConfig.load(path)
        out[cfg.name] = cfg
    return out
