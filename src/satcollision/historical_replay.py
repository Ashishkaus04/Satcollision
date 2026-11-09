"""
Historical replay: would the pipeline have flagged a documented real event?

The project definition (Sec. 4, "Historical backtesting") asks for more than
running on today's catalogue: replay the system against real historical
orbital data around a *documented* past close approach and show it would have
flagged it. This module does that, and does it honestly, by only ever using
element sets that existed *before* the moment it pretends to be screening at.

How a replay works
------------------
You give it a TLE history for the two objects (several element sets each, as
downloaded from Space-Track or CelesTrak's historical archive) and the
documented time of closest approach. For each "as-of" time — 72 h, 48 h, 24 h,
12 h and 6 h before the documented TCA by default — it:

1. picks, for each object, the newest element set whose epoch is at or before
   the as-of time (nothing from the future);
2. propagates both from the as-of time to shortly after the documented TCA,
   on a 1-second grid, and runs the ordinary screening pass from ``twin.py``;
3. records the predicted closest approach, its miss distance and Pc, and
   whether it crossed the sharing threshold — i.e. whether an operator's twin
   would have raised the abstracted signal at that point.

The earliest as-of time at which the encounter is flagged is the warning the
pipeline would have given. That is the number the report should quote.

What it does not claim
----------------------
A documented event is usually a *predicted* conjunction that someone then
manoeuvred away from (the 2021 China Space Station events below are exactly
that), so "the objects never actually came that close" is expected and not a
failure. Element sets published after the avoidance manoeuvre will show the
post-manoeuvre orbit; that is why only pre-as-of element sets are used.

The circular Pc model with the default 0.5 km sigma is used unless a
``pc_model`` is passed, matching the rest of the pipeline.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

from sgp4.api import jday

from .fleets import TrackedObject, tle_epoch_utc
from .signal import abstract_conjunction
from .twin import (
    DEFAULT_COMBINED_HBR_KM,
    DEFAULT_COMBINED_SIGMA_KM,
    DEFAULT_PC_THRESHOLD,
    propagate_window,
    screen_conjunctions,
)

UTC = datetime.timezone.utc

DEFAULT_AS_OF_HOURS = (72.0, 48.0, 24.0, 12.0, 6.0)


@dataclass(frozen=True)
class DocumentedEvent:
    """A real, publicly documented close approach to replay against."""

    key: str
    title: str
    name_a: str      # name as it appears on the TLE name line
    norad_a: int | None
    name_b: str
    norad_b: int | None  # None = look it up by name in the downloaded file
    tca_utc: datetime.datetime
    tca_uncertainty_min: float  # how precisely the public record gives the time
    source: str
    notes: str = ""


# Registry of documented events. NORAD numbers and times come from the cited
# public records; the element sets themselves are NOT stored here (a TLE typed
# by hand or by a model is worse than none) — download them, see
# scripts/run_historical_replay.py.
EVENTS: dict[str, DocumentedEvent] = {
    "css-starlink1095-2021": DocumentedEvent(
        key="css-starlink1095-2021",
        title="China Space Station (Tianhe) vs Starlink-1095, 1 July 2021",
        name_a="CSS (TIANHE)",
        norad_a=48274,   # Tianhe core module, launched 29 Apr 2021
        name_b="STARLINK-1095",
        norad_b=None,
        tca_utc=datetime.datetime(2021, 7, 1, 13, 0, tzinfo=UTC),
        tca_uncertainty_min=12 * 60.0,
        source="UNOOSA document A/AC.105/1262 (note verbale from China, 6 Dec 2021): "
               "https://www.unoosa.org/res/oosadoc/data/documents/2021/aac_105/"
               "aac_1051262_0_html/AAC105_1262E.pdf",
        notes="The note gives the date and says the station made a preventive manoeuvre that "
              "evening. Read the exact time from the note before quoting a lead time, and set "
              "--tca accordingly; the time here is a placeholder within that day.",
    ),
    "css-starlink2305-2021": DocumentedEvent(
        key="css-starlink2305-2021",
        title="China Space Station (Tianhe) vs Starlink-2305, 21 October 2021",
        name_a="CSS (TIANHE)",
        norad_a=48274,
        name_b="STARLINK-2305",
        norad_b=None,
        tca_utc=datetime.datetime(2021, 10, 21, 12, 0, tzinfo=UTC),
        tca_uncertainty_min=12 * 60.0,
        source="UNOOSA document A/AC.105/1262 (as above)",
        notes="Starlink-2305 was manoeuvring continuously, so its TLE history is the hard part: "
              "expect the prediction to move between as-of times. Same caveat on the exact time.",
    ),
}


@dataclass
class AsOfResult:
    hours_before: float
    as_of_utc: datetime.datetime
    epoch_a_utc: datetime.datetime | None
    epoch_b_utc: datetime.datetime | None
    predicted_tca_utc: datetime.datetime | None = None
    miss_distance_km: float | None = None
    pc: float = 0.0
    flagged: bool = False
    geometry_class: str | None = None
    tca_error_min: float | None = None  # predicted minus documented TCA
    note: str = ""

    def as_row(self) -> dict:
        def iso(t):
            return t.isoformat(timespec="seconds") if t else None
        return {
            "hours_before": self.hours_before,
            "as_of_utc": iso(self.as_of_utc),
            "element_set_epoch_a": iso(self.epoch_a_utc),
            "element_set_epoch_b": iso(self.epoch_b_utc),
            "predicted_tca_utc": iso(self.predicted_tca_utc),
            "miss_distance_m": None if self.miss_distance_km is None
            else round(self.miss_distance_km * 1000.0, 1),
            "pc": self.pc,
            "flagged": self.flagged,
            "geometry_class": self.geometry_class,
            "tca_error_min": None if self.tca_error_min is None else round(self.tca_error_min, 1),
            "note": self.note,
        }


@dataclass
class ReplayResult:
    norad_a: int
    norad_b: int
    documented_tca_utc: datetime.datetime
    threshold: float
    rows: list[AsOfResult] = field(default_factory=list)

    @property
    def earliest_flag(self) -> AsOfResult | None:
        flagged = [r for r in self.rows if r.flagged]
        return max(flagged, key=lambda r: r.hours_before) if flagged else None

    @property
    def flagged_at_all(self) -> bool:
        return self.earliest_flag is not None


def find_norad_by_name(objects: list[TrackedObject], name: str) -> int | None:
    """NORAD number of the object whose TLE name line matches ``name``
    (case-insensitive, surrounding spaces ignored), or ``None``."""
    wanted = name.strip().upper()
    ids = {o.norad_id for o in objects if str(o.meta.get("name", "")).strip().upper() == wanted}
    if len(ids) > 1:
        raise ValueError(f"name {name!r} matches several NORAD numbers: {sorted(ids)}")
    return ids.pop() if ids else None


def element_set_history(objects: list[TrackedObject], norad_id: int) -> list[TrackedObject]:
    """Every element set for one object, oldest first."""
    history = [o for o in objects if o.norad_id == norad_id]
    return sorted(history, key=tle_epoch_utc)


def latest_before(history: list[TrackedObject], when: datetime.datetime) -> TrackedObject | None:
    """Newest element set whose epoch is at or before ``when`` (nothing from the future)."""
    eligible = [o for o in history if tle_epoch_utc(o) <= when]
    return eligible[-1] if eligible else None


def _jd_fr(when: datetime.datetime) -> tuple[float, float]:
    return jday(when.year, when.month, when.day, when.hour, when.minute,
                when.second + when.microsecond / 1e6)


def predict_closest_approach(
    obj_a: TrackedObject,
    obj_b: TrackedObject,
    start: datetime.datetime,
    end: datetime.datetime,
    step_s: float = 1.0,
    screening_km: float = 50.0,
    sigma_km: float = DEFAULT_COMBINED_SIGMA_KM,
    hbr_km: float = DEFAULT_COMBINED_HBR_KM,
    pc_model=None,
):
    """Closest approach of two objects between ``start`` and ``end``, or ``None``
    if they never come within ``screening_km``."""
    jd0, fr0 = _jd_fr(start)
    duration_s = (end - start).total_seconds()
    propagation = propagate_window([obj_a, obj_b], jd0, fr0, duration_s=duration_s, step_s=step_s)
    found = screen_conjunctions([obj_a, obj_b], propagation, screening_distance_km=screening_km,
                                combined_sigma_km=sigma_km, combined_hbr_km=hbr_km,
                                include_debris_debris=True, pc_model=pc_model)
    return found[0] if found else None


def replay_event(
    objects: list[TrackedObject],
    norad_a: int,
    norad_b: int,
    documented_tca_utc: datetime.datetime,
    as_of_hours=DEFAULT_AS_OF_HOURS,
    after_tca_h: float = 1.0,
    threshold: float = DEFAULT_PC_THRESHOLD,
    step_s: float = 1.0,
    screening_km: float = 50.0,
    pc_model=None,
) -> ReplayResult:
    """Replay one documented event from several as-of times. See module docstring."""
    history_a = element_set_history(objects, norad_a)
    history_b = element_set_history(objects, norad_b)
    if not history_a or not history_b:
        missing = [n for n, h in ((norad_a, history_a), (norad_b, history_b)) if not h]
        raise ValueError(f"no element sets for NORAD {missing} in the loaded TLE files")

    result = ReplayResult(norad_a, norad_b, documented_tca_utc, threshold)
    end = documented_tca_utc + datetime.timedelta(hours=after_tca_h)
    for hours in sorted(as_of_hours, reverse=True):
        as_of = documented_tca_utc - datetime.timedelta(hours=hours)
        obj_a, obj_b = latest_before(history_a, as_of), latest_before(history_b, as_of)
        row = AsOfResult(hours, as_of,
                         tle_epoch_utc(obj_a) if obj_a else None,
                         tle_epoch_utc(obj_b) if obj_b else None)
        if obj_a is None or obj_b is None:
            row.note = "no element set published yet for one object"
            result.rows.append(row)
            continue
        conjunction = predict_closest_approach(obj_a, obj_b, as_of, end, step_s=step_s,
                                               screening_km=screening_km, pc_model=pc_model)
        if conjunction is None:
            row.note = f"never within {screening_km:.0f} km in this window"
        else:
            row.predicted_tca_utc = as_of + datetime.timedelta(seconds=conjunction.tca_offset_s)
            row.miss_distance_km = conjunction.miss_distance_km
            row.pc = conjunction.pc
            row.geometry_class = conjunction.geometry_class
            row.flagged = abstract_conjunction(conjunction, threshold=threshold) is not None
            row.tca_error_min = (row.predicted_tca_utc - documented_tca_utc).total_seconds() / 60.0
        result.rows.append(row)
    return result


def describe(result: ReplayResult) -> str:
    """One plain sentence on the outcome, for the console and the report."""
    first = result.earliest_flag
    if first is None:
        closest = min((r for r in result.rows if r.miss_distance_km is not None),
                      key=lambda r: r.miss_distance_km, default=None)
        if closest is None:
            return "Not flagged: the two objects never came within the screening radius."
        return (f"Not flagged at any as-of time. Closest predicted approach "
                f"{closest.miss_distance_km * 1000:.0f} m (Pc {closest.pc:.1e}), "
                f"{closest.hours_before:.0f} h before the documented time.")
    return (f"Flagged {first.hours_before:.0f} h before the documented approach: predicted "
            f"miss {first.miss_distance_km * 1000:.0f} m, Pc {first.pc:.1e}, "
            f"{first.geometry_class} geometry, predicted time within "
            f"{abs(first.tca_error_min):.0f} min of the record.")

