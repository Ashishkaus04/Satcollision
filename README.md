# Federated Digital Twins for Satellite Constellation Collision Avoidance

Working prototype for the BTech major project defined in
`major_project_final_definition.md`. This README documents what's actually
implemented, what's simplified (and why), and what's future work — an
honest map from code to the project definition's sections, meant to survive
direct panel questioning.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python3 -m pytest tests/ -q        # 12 tests, should all pass
PYTHONPATH=src python3 scripts/run_demo.py         # full pipeline walkthrough
```

The demo script prints a run through every layer below and also writes
`docs/sample_run.md`.

## Layer-by-layer status

| Layer (project definition Sec. 5) | Module | Status |
|---|---|---|
| Simulation | `src/satcollision/fleets.py` | **Built.** Real SGP4 propagation (via the `sgp4` library) of 3 synthetic operator fleets whose shell parameters (altitude/inclination/plane count) mirror publicly documented real constellations, plus an ambient debris population. A `load_tle_file()` loader is included so real CelesTrak TLEs can be dropped in with zero other code changes once run outside this network-restricted sandbox (see "Why synthetic data" below). |
| Per-operator digital twin | `src/satcollision/twin.py` | **Built.** Pairwise conjunction screening across a propagated time grid, and a simplified circular (equal-variance) 2D probability-of-collision estimate — see "Simplifications" below for what a production version would add. |
| Signal abstraction | `src/satcollision/signal.py` | **Built.** Converts a private `Conjunction` into a threshold-gated `AbstractedSignal` (flag + coarse geometry class + coarse severity tier) with a runtime structural check (`assert_no_raw_state`) that fails loudly if any state-vector-shaped field is ever added. |
| Federation (aggregation + DP + Byzantine-robust) | `src/satcollision/federation.py` | **Built.** Mean / trimmed-mean / Krum aggregation over per-operator count vectors, a Laplace-mechanism local-DP step, and 4 attack-model simulators (spoof-high, suppress, sign-flip, free-ride) matching the attack models named in the project definition's tech stack. |
| Maneuver deconfliction | `src/satcollision/deconfliction.py` | **Built.** A deterministic, zero-extra-data-exchange protocol that assigns complementary escape directions, plus a Monte Carlo comparison against uncoordinated independent maneuvering. |
| Historical backtesting | *(not yet built)* | **Future work.** Needs real historical TLE data (see below); the `load_tle_file()` path in `fleets.py` is the intended entry point. |
| Incentive/reputation mechanism | *(not yet built)* | **Future work / stretch goal**, as flagged in the project definition itself. |
| Presentation | `conjunction_watch.html` (project root) | **Built separately** as an illustrative, scripted interactive demo for panel presentations — see that file and Section 12 of the project definition. This codebase's `scripts/run_demo.py` is the *technical* presentation layer (plain-language console output); wiring its real numbers into a live visual dashboard is listed as future work below. |
| 3-scenario evaluation harness | `src/satcollision/evaluate.py` | **Built.** Compares No-Cooperation / Full-Sharing / Federated on detection lead time and raw-data exposure on one engineered conjunction, plus a separate robustness sweep of aggregation error vs. adversary fraction. |

## Why synthetic (not downloaded) orbital data

This sandbox's outbound network access is restricted to package registries
(pypi, npm, etc.) — `celestrak.org` is not reachable from here (confirmed:
a direct request returns a proxy policy denial, not a site error). Rather
than fabricate "real" TLE data or rely on an LLM to transcribe exact
orbital numbers from a fetched page (a real correctness risk — a single
wrong digit in a TLE checksum or mean-motion field silently produces a
wrong orbit), `fleets.py` constructs orbits directly from Keplerian
elements via `sgp4`'s `Satrec.sgp4init`, using each shell's *publicly
documented* approximate altitude/inclination (Starlink ~550km/53°, OneWeb
~1200km/87.4°, Iridium NEXT ~780km/86.4°). This is legitimate synthetic
scenario generation (the same technique a mission-design tool uses), not
scraped data presented as real.

`load_tle_file()` is the drop-in real-data path: on a machine with normal
internet access, download a TLE set from
`https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle`
(or CelesTrak's historical/supplemental bundles for backtesting), save it
under `data/`, and pass its path to `load_tle_file()` in place of
`build_synthetic_fleets()` — every downstream layer (`twin.py` onward)
works on `TrackedObject` instances identically either way.

## Simplifications (and why they're defensible for a final-year prototype)

- **Pc formula.** `twin.compute_pc` uses the simplified circular
  (equal-variance) 2D approximation `Pc ~= (HBR^2/2*sigma^2) * exp(-d^2/2*sigma^2)`
  rather than the full non-circular Foster/Estes double integral with each
  operator's actual covariance. The simplified form is standard for
  illustrative/first-pass work and responds correctly to the variables
  that matter (miss distance, combined uncertainty, object size); the full
  integral is listed as a natural "if time permits" refinement.
- **Detection lead time model** (`evaluate.sigma_no_cooperation`). Models
  no-cooperation's degraded tracking precision as linearly improving
  toward TCA — a simplification of the real, lumpier cadence of public
  CDM updates (18th Space Defense Squadron screens roughly 3x/day). The
  qualitative result (federated matches full-sharing's lead time at zero
  data exposure, both beat no-cooperation) is robust to this
  simplification; only the exact numbers would change with a more
  detailed model.
- **Geometry classification** (`twin._classify_geometry`) buckets the
  relative-velocity angle into 3 coarse classes by fixed thresholds
  (30°/150°). A production classifier might use encounter-plane geometry
  more precisely; the coarse bucketing is in any case the *point* of the
  abstraction layer (finer geometry is exactly the kind of detail that
  should not cross the operator boundary).
- **Reporting-period vectors** (`federation.build_operator_report`) use a
  3-dimensional (severity-tier) count vector as the thing that gets
  aggregated. A richer version would break this down further (by
  geometry class, by altitude band) — a straightforward extension of the
  same code, not a redesign.

## Project layout

```
src/satcollision/
  fleets.py          orbital simulation layer
  twin.py             per-operator digital twin (screening + Pc)
  signal.py            signal abstraction layer
  federation.py         federation aggregation + DP + Byzantine-robust
  deconfliction.py       maneuver deconfliction negotiation
  evaluate.py          3-scenario comparison + robustness sweep
tests/                 pytest suite (12 tests) covering the above
scripts/run_demo.py    end-to-end walkthrough, writes docs/sample_run.md
requirements.txt        sgp4, numpy, scipy, matplotlib, pytest
```

## Next steps (see the Semester Implementation Plan, Sec. 10 of the project definition)

1. Historical backtest against a real documented close-approach period
   (needs real TLE data — see "Why synthetic" above).
2. Incentive/reputation layer (stretch goal).
3. Wire `evaluate.py`'s real computed numbers into a live version of
   `conjunction_watch.html` (currently a standalone, hand-scripted demo).
4. Full non-circular Pc integral with per-operator covariance, if time
   permits.
