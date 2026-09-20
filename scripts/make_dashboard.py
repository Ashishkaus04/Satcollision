#!/usr/bin/env python3
r"""
Build the Conjunction Watch dashboard from the project's own measured output.

The dashboard used to be a hand-scripted demo: every number on the page was
typed in by whoever wrote the HTML. That is fine for explaining a mechanism
and worthless as evidence, and it rots the moment a real run produces a
different figure. This script closes that gap by injecting the actual run
data into the page, so the dashboard cannot disagree with the report.

Inputs (all produced by other scripts in this repo):

* ``docs/figures/real_evaluation.json``       -- scripts/run_real_evaluation.py
* ``docs/figures/fig_reputation_trajectories.csv`` -- scripts/make_figures.py
* ``docs/figures/fig_robustness_sweep.csv``       -- scripts/make_figures.py
* ``data/*.tle`` (optional)                   -- the real catalogues, for the 3D view
* ``data/earth_texture.jpg`` (optional)       -- NASA Blue Marble, inlined as a data URI

The 3D view replays **real propagated positions**: the generator re-runs the
project's own SGP4 propagation over the measured encounter's window and
embeds the sampled state vectors, so the page animates the same geometry the
evaluation measured rather than an artist's impression of it. Without the
TLE files it degrades cleanly: the page keeps every measured figure and the
2D scenario, and says the 3D replay needs the catalogues.

Output: ``docs/conjunction_watch.html``, self-contained (no external data
files, no runtime fetches), built from
``docs/conjunction_watch_template.html`` by replacing the single
``/*__DASHBOARD_DATA__*/`` marker with a JSON literal.

Usage:

    PYTHONPATH=src python3 scripts/make_dashboard.py
    PYTHONPATH=src python3 scripts/make_dashboard.py --check   # verify only

The template keeps the page's markup, CSS and behaviour; this script owns
only the data. Editing the design means editing the template, and the next
build keeps whatever numbers the latest run produced.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = REPO_ROOT / "docs" / "figures"
DATA_DIR = REPO_ROOT / "data"
# Accept whatever the user saved: NASA's downloads land as .jpg or .jpeg, and
# a PNG works just as well once it is a data URI.
EARTH_TEXTURE_NAMES = ("earth_texture.jpg", "earth_texture.jpeg", "earth_texture.png")
TEMPLATE = REPO_ROOT / "docs" / "conjunction_watch_template.html"
OUTPUT = REPO_ROOT / "docs" / "conjunction_watch.html"
MARKER = "/*__DASHBOARD_DATA__*/{}"

# Reputation tier thresholds, mirrored from reputation.ReputationParams so the
# chart's reference lines match the layer they describe. Imported lazily in
# _reputation_thresholds() so this script still runs without the package on
# the path (it only needs the CSVs otherwise).


def _require(path: Path) -> Path:
    if not path.exists():
        print(f"ERROR: {path.relative_to(REPO_ROOT)} is missing.")
        print("Run scripts/run_real_evaluation.py and scripts/make_figures.py first.")
        sys.exit(1)
    return path


def _reputation_thresholds() -> list[dict]:
    try:
        from satcollision.reputation import ReputationParams
    except Exception:  # the dashboard is still buildable without the package
        return [{"value": 0.25, "label": "quarantine"}, {"value": 0.45, "label": "degraded"},
                {"value": 0.70, "label": "full access"}]
    params = ReputationParams()
    return [
        {"value": params.quarantine_threshold, "label": "quarantine"},
        {"value": params.degraded_threshold, "label": "degraded"},
        {"value": params.standard_threshold, "label": "full access"},
    ]


def _load_real_evaluation() -> tuple[dict, dict | None, list[float]]:
    """Headline summary, the cross-operator run, and the warning-time series."""
    payload = json.loads(_require(FIGURE_DIR / "real_evaluation.json").read_text(encoding="utf-8"))
    run = payload["headline"]
    summary = run["summary"]

    headline = {
        "catalog": run["catalog"],
        "n_objects": run["n_objects"],
        "window_hours": run["window_hours"],
        "screening_horizon_h": run["screening_horizon_h"],
        "computed_at_utc": run["computed_at_utc"],
        "n_above_threshold": run["n_above_threshold"],
        "n_scored": run.get("n_scored", summary.get("n_encounters", 0)),
        "n_excluded_burn_in": run.get("n_excluded_burn_in", 0),
        "median_lead_time_nocoop_h": summary["median_lead_time_nocoop_h"],
        "median_lead_time_cooperative_h": summary["median_lead_time_cooperative_h"],
        "median_lead_time_gain_h": summary["median_lead_time_gain_h"],
        "min_lead_time_nocoop_h": summary["min_lead_time_nocoop_h"],
        "max_lead_time_nocoop_h": summary["max_lead_time_nocoop_h"],
        "median_miss_distance_m": summary["median_miss_distance_m"],
    }
    # One decimal is plenty for an hours axis, and it keeps the embedded
    # series small enough that the page stays comfortably under any size cap.
    warning_hours = [round(r["lead_time_nocoop_min"] / 60.0, 2) for r in run["encounters"]]

    negative = payload.get("negative")
    if negative is not None:
        negative = {
            "catalog": negative["catalog"],
            "n_objects": negative["n_objects"],
            "n_cross_operator": negative["n_cross_operator"],
            "n_above_threshold": negative["n_above_threshold"],
            "closest_cross_operator_km": negative["closest_cross_operator_km"],
            "max_cross_operator_pc": negative["max_cross_operator_pc"],
            # Only the encounters that actually crossed the threshold are shown
            # on the page; the rest live in the JSON and the report.
            "encounters": negative["encounters"][:3],
        }
    return headline, negative, warning_hours


def _load_reputation() -> dict:
    rows = list(csv.DictReader(_require(FIGURE_DIR / "fig_reputation_trajectories.csv")
                               .read_text(encoding="utf-8").splitlines()))
    round_keys = [k for k in rows[0] if k.startswith("round_")]
    round_keys.sort(key=lambda k: int(k.split("_")[1]))
    series = [{
        "operator": r["operator"],
        "strategy": r["strategy"],
        "values": [float(r[k]) for k in round_keys],
    } for r in rows]
    return {
        "rounds": len(round_keys),
        "series": series,
        "thresholds": _reputation_thresholds(),
        # Where the on-off operator defects, read off the trajectory rather
        # than hard-coded: it is the round after that series' peak.
        "defect_round": _defect_round(series),
    }


def _defect_round(series: list[dict]) -> int:
    for s in series:
        if s["strategy"] == "on_off":
            peak = max(range(len(s["values"])), key=lambda i: s["values"][i])
            return peak + 1
    return 0


def _load_robustness() -> list[dict]:
    rows = list(csv.DictReader(_require(FIGURE_DIR / "fig_robustness_sweep.csv")
                               .read_text(encoding="utf-8").splitlines()))
    return [{
        "fraction": float(r["adversary_fraction"]),
        "mean": float(r["mean_error"]),
        "trimmed": float(r["trimmed_mean_error"]),
        "krum": float(r["krum_error"]),
    } for r in rows]


# --------------------------------------------------------------------------
# The 3D replay: real propagated positions, not an artist's impression
# --------------------------------------------------------------------------

SCENE_BACKGROUND_SATS = 120      # enough to read as a constellation, small enough to embed
SCENE_BACKGROUND_STEP_S = 60.0   # one sample a minute: the page interpolates between them
SCENE_BACKGROUND_SPAN_S = 5700.0 # ~95 minutes: one low-Earth-orbit period
SCENE_ENCOUNTER_SPAN_S = 900.0   # +/- 15 minutes around closest approach
SCENE_ENCOUNTER_STEP_S = 5.0
SCENE_RIBBON_STEP_S = 30.0       # a full orbit at 30 s resolution, for the wide view


def _featured_encounter(payload: dict) -> dict | None:
    """The encounter the 3D view flies to.

    Preference order is deliberate: a *genuine* cross-operator crossing is
    the most interesting thing the project has ever measured, so it wins;
    otherwise the highest-Pc encounter from the split-catalogue run stands
    in, and the page says which it is showing.
    """
    negative = payload.get("negative") or {}
    if negative.get("encounters"):
        row = dict(negative["encounters"][0])
        row["source"] = "cross_operator"
        row["run_epoch_utc"] = negative["computed_at_utc"]
        row["catalog"] = negative["catalog"]
        return row
    rows = payload["headline"].get("encounters") or []
    if not rows:
        return None
    row = dict(max(rows, key=lambda r: r["pc"]))
    row["source"] = "split_catalog"
    row["run_epoch_utc"] = payload["headline"]["computed_at_utc"]
    row["catalog"] = payload["headline"]["catalog"]
    return row


def _tle_files() -> list[Path]:
    return sorted(DATA_DIR.glob("*.tle")) if DATA_DIR.exists() else []


def _epoch_from_iso(iso: str):
    """(jd, fr) for the run epoch, so the replay reproduces the measured geometry."""
    import datetime
    from satcollision.fleets import _epoch_from_calendar
    stamp = datetime.datetime.fromisoformat(iso)
    _, jd0, fr0 = _epoch_from_calendar(stamp.year, stamp.month, stamp.day,
                                        stamp.hour + stamp.minute / 60.0)
    return jd0, fr0


def _sample_positions(objects, jd0, fr0, start_s: float, span_s: float, step_s: float):
    """Propagate and return positions as [[x,y,z], ...] per object, in km."""
    import numpy as np
    from satcollision.twin import propagate_window
    # propagate_window starts at jd0/fr0, so shift the epoch to the sample start
    jd_start, fr_start = jd0, fr0 + start_s / 86400.0
    result = propagate_window(objects, jd_start, fr_start, duration_s=span_s, step_s=step_s)
    positions = np.asarray(result.positions, dtype=float)
    valid = np.asarray(result.valid, dtype=bool)
    positions[~valid] = np.nan
    return positions, np.asarray(result.t_offsets_s, dtype=float)


def _round_track(track) -> list[list[float]]:
    """Kilometre precision: the globe is 6,371 km across, nobody reads metres off it."""
    out = []
    for point in track:
        if any(p != p for p in point):  # NaN guard, keeps the JSON valid
            out.append(None)
        else:
            out.append([round(float(p), 1) for p in point])
    return out


def build_scene(payload: dict) -> dict | None:
    """Sample the real catalogue and the measured encounter for the 3D replay."""
    tles = _tle_files()
    if not tles:
        return {"available": False, "reason": "no .tle catalogues in data/"}
    try:
        import numpy as np
        from satcollision.fleets import load_tle_file
    except Exception as exc:
        return {"available": False, "reason": f"satcollision package not importable ({exc})"}

    featured = _featured_encounter(payload)
    if featured is None:
        return {"available": False, "reason": "the run found no encounters to replay"}

    catalogue = []
    for path in tles:
        catalogue += load_tle_file(str(path), operator_name=path.stem.capitalize())
    by_norad = {obj.norad_id: obj for obj in catalogue}

    parts = featured["encounter_id"].split("-")
    norad_a, norad_b = int(parts[1]), int(parts[2])
    if norad_a not in by_norad or norad_b not in by_norad:
        return {"available": False,
                "reason": f"objects {norad_a}/{norad_b} are not in today's catalogues — "
                          "the encounter was measured against an older TLE set"}

    obj_a, obj_b = by_norad[norad_a], by_norad[norad_b]
    jd0, fr0 = _epoch_from_iso(featured["run_epoch_utc"])
    tca_s = featured["tca_offset_min"] * 60.0

    # 1. the encounter itself, densely sampled around closest approach
    start = max(0.0, tca_s - SCENE_ENCOUNTER_SPAN_S)
    enc_pos, enc_t = _sample_positions([obj_a, obj_b], jd0, fr0, start,
                                        SCENE_ENCOUNTER_SPAN_S * 2, SCENE_ENCOUNTER_STEP_S)
    separation = np.linalg.norm(enc_pos[0] - enc_pos[1], axis=1)
    tca_index = int(np.nanargmin(separation))

    # 2. one full orbit each, for the ribbons the pair sweeps out
    ribbon_pos, _ = _sample_positions([obj_a, obj_b], jd0, fr0, start,
                                       SCENE_BACKGROUND_SPAN_S, SCENE_RIBBON_STEP_S)

    # 3. a thinned slice of the catalogue, one period, for the constellation
    step = max(1, len(catalogue) // SCENE_BACKGROUND_SATS)
    background_objects = catalogue[::step][:SCENE_BACKGROUND_SATS]
    bg_pos, _ = _sample_positions(background_objects, jd0, fr0, start,
                                   SCENE_BACKGROUND_SPAN_S, SCENE_BACKGROUND_STEP_S)

    return {
        "available": True,
        "source": featured["source"],
        "catalog": featured["catalog"],
        "run_epoch_utc": featured["run_epoch_utc"],
        "encounter": {
            "id": featured["encounter_id"],
            "miss_distance_m": featured["miss_distance_m"],
            "pc": featured["pc"],
            "geometry_class": featured["geometry_class"],
            "tca_offset_min": featured["tca_offset_min"],
            "lead_time_nocoop_min": featured["lead_time_nocoop_min"],
            "lead_time_cooperative_min": featured["lead_time_cooperative_min"],
            "object_a": {"label": obj_a.label, "operator": obj_a.operator, "norad": norad_a},
            "object_b": {"label": obj_b.label, "operator": obj_b.operator, "norad": norad_b},
            "step_s": SCENE_ENCOUNTER_STEP_S,
            "tca_index": tca_index,
            "separation_km": [round(float(v), 3) for v in separation],
            "track_a": _round_track(enc_pos[0]),
            "track_b": _round_track(enc_pos[1]),
            "ribbon_a": _round_track(ribbon_pos[0]),
            "ribbon_b": _round_track(ribbon_pos[1]),
        },
        "background": {
            "n_objects": len(background_objects),
            "step_s": SCENE_BACKGROUND_STEP_S,
            "tracks": [_round_track(track) for track in bg_pos],
        },
    }


def _earth_texture_path() -> Path | None:
    for name in EARTH_TEXTURE_NAMES:
        candidate = DATA_DIR / name
        if candidate.exists():
            return candidate
    return None


def earth_texture_data_uri() -> str | None:
    """Inline the Blue Marble image, if the user dropped one into data/.

    Everything the page needs has to live inside the page: an artifact's
    content-security policy blocks external images outright, so a texture
    either travels as a data URI or does not travel at all.
    """
    path = _earth_texture_path()
    if path is None:
        return None
    import base64
    raw = path.read_bytes()
    if len(raw) > 1_500_000:
        print(f"  note: {path.name} is {len(raw)/1024/1024:.1f} MB — it will add about "
              f"{len(raw)*4/3/1024/1024:.1f} MB to the page. 2048x1024 at JPEG quality ~82 "
              "(roughly 250 KB) is plenty for a globe this size.")
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def build() -> str:
    headline, negative, warning_hours = _load_real_evaluation()
    payload = json.loads((FIGURE_DIR / "real_evaluation.json").read_text(encoding="utf-8"))
    scene = build_scene(payload)
    texture = earth_texture_data_uri()
    data = {
        "headline": headline,
        "negative": negative,
        "warning_hours": warning_hours,
        "reputation": _load_reputation(),
        "robustness": _load_robustness(),
        "scene": scene,
        "earth_texture": texture,
    }
    template = _require(TEMPLATE).read_text(encoding="utf-8")
    if MARKER not in template:
        print(f"ERROR: {TEMPLATE.name} no longer contains the '{MARKER}' marker.")
        sys.exit(1)
    return template.replace(MARKER, json.dumps(data, separators=(",", ":")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="build in memory and report what would be embedded, without writing")
    args = parser.parse_args()

    html = build()
    headline, negative, warning = _load_real_evaluation()
    print(f"Embedded run: {headline['catalog']}, {headline['n_objects']:,} objects, "
          f"{headline['window_hours']:.0f} h window, screened {headline['computed_at_utc']}")
    print(f"  {headline['n_scored']} scored encounters "
          f"({headline['n_excluded_burn_in']} excluded as window-clipped)")
    print(f"  median warning: {headline['median_lead_time_nocoop_h']:.1f} h without cooperation, "
          f"{headline['median_lead_time_cooperative_h']:.0f} h with")
    if negative and negative["n_above_threshold"]:
        print(f"  genuine cross-operator crossings in {negative['catalog']}: "
              f"{negative['n_above_threshold']}")

    payload = json.loads((FIGURE_DIR / "real_evaluation.json").read_text(encoding="utf-8"))
    scene = build_scene(payload)
    if scene.get("available"):
        enc = scene["encounter"]
        print(f"  3D replay: {enc['id']} ({enc['object_a']['label']} vs {enc['object_b']['label']}), "
              f"{len(enc['track_a'])} samples across closest approach, "
              f"{scene['background']['n_objects']} background objects")
    else:
        print(f"  3D replay unavailable: {scene.get('reason')}")
    texture_path = _earth_texture_path()
    print(f"  Earth texture: {'inlined from data/' + texture_path.name if texture_path else 'not found — the globe falls back to the procedural render'}")

    if args.check:
        print(f"\n--check: {len(html):,} bytes would be written to "
              f"{OUTPUT.relative_to(REPO_ROOT)} (not written)")
        return 0

    OUTPUT.write_text(html, encoding="utf-8")
    print(f"\nWrote {OUTPUT.relative_to(REPO_ROOT)} ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
