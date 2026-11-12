# Paper experiments: first results (12 Nov 2026)

All numbers: Starlink catalogue of 10 Sep 2026 (10,713 objects), 168 h from the
catalogue epoch, 30 s grid, NORAD-parity split into two operators, Pc threshold
T = 1e-4, 72 h screening horizon (first 72 h are burn-in, 4 days scored),
circular Pc (HBR 20 m). Foster and more epochs are next.

## 1. The old screening missed most fast encounters

`twin.screen_conjunctions` keeps a pair only if it is within 5 km *at a grid
instant*. Crossing satellites (7 to 10 km/s relative) pass through 5 km in about
a second, so a 30 s grid almost never samples them inside it. The new
`screening.screen_conjunctions_refined` solves the closest approach between
samples and refines it with direct SGP4 calls (agreement within 0.3 m on every
sub-km approach checked).

| 168 h, Starlink split | Grid screen (old) | Refined screen (new) |
| --- | --- | --- |
| Cross-operator approaches < 5 km | 11,835 pairs | 316,699 approaches |
| Above Pc 1e-4 | 312 | 23,541 |
| Scored (after burn-in) | 237 | 16,752 |
| Geometry (overtaking / crossing / head-on) | 154 / 81 / 2 | 3,669 / 12,580 / 503 |
| Median miss distance | 785 m | 699 m |
| Median warning gain, cooperative vs none (old deterministic model) | 49.3 h | 48.5 h |

The headline gain barely moves (it depends only on miss distance in that
model), but the encounter sample was 70x too small and biased towards slow,
overtaking geometry. Any paper number must use the refined screen.

Caveat: 4,188 "dangerous" encounters a day between two halves of Starlink is
more than SpaceX reports acting on. TLE-based positions are noisy and the 0.5 km
combined sigma is a modelling choice; this must be said in the paper and is a
reason to sweep the sigma values.

## 2. Two-stage federated screening (Monte Carlo, 3 seeds)

Each regime sees an *estimated* miss vector (truth plus Gaussian error with that
regime's sigma, redrawn at each 8-hourly update from 72 h to 8 h before TCA).
Stage 1: each operator, knowing its own satellite precisely (0.35 km) and the
other only from public tracking (2.1 km far out, 0.35 km at TCA), raises an
abstracted alert when its Pc estimate reaches tau. Stage 2: for alerted pairs
only, a secure computation (MPC or homomorphic encryption, cost modelled, not
implemented) gives the full-sharing Pc. Ablation: the same scheme triggered by
public data only. "Dangerous" = true miss < 1.02 km (full-sharing Pc on the
truth >= 1e-4): 4,188 a day.

| Regime | Recall | Warned >= 24 h | Median warning | Missed / day | Secure cluster-hours / day |
| --- | --- | --- | --- | --- | --- |
| No cooperation | 75.2% | 9.9% | 8 h | 1,040 | 0 |
| Two-stage, tau = 1e-4 | 91.8% | 63.4% | 24 h | 344 | 2.6 |
| **Two-stage, tau = 3e-5** | **99.7%** | **99.0%** | **64 h** | **12** | **18.6** |
| Two-stage, tau = 1e-5 | 99.85% | 99.5% | 72 h | 6 | 33.5 |
| Public trigger, tau = 3e-5 | 96.6% | 85.3% | 32 h | 142 | 8.3 |
| Public trigger, tau = 1e-5 | 99.7% | 99.0% | 64 h | 13 | 32.7 |
| Full sharing (all orbits exposed) | 99.85% | 99.5% | 72 h | 6 | n/a |
| Secure Pc on every cross-operator pair | (same as full sharing) | | | | about 1,594 per screening pass |

Secure cost assumes 18,000 pairs per hour per three-server MPC cluster
(Valdma & Veskus, ESA SDC9 2025) and a re-run every 8 h per alerted pair. The
all-pairs figure is 5,356 x 5,357 = 28.7 million pairs per pass. Seed-to-seed
standard deviation of recall is under 0.1 percentage points.

What this says, honestly:

- The two-stage scheme reaches full-sharing safety (99.5% warned a day ahead)
  for about 1% of the secure compute that one all-pairs pass would need (18.6 vs about 1,594 cluster-hours), and
  exposes no orbit to anyone.
- The operators' private alerts beat a public-only trigger, but by less than
  one might hope: at the same safety (about 99% warned a day ahead) they need
  18.6 instead of 32.7 cluster-hours a day, a 43% saving. At small budgets the
  curves are close (see the figure): the alerts' advantage is concentrated around
  the setting that matters, tau = 3e-5. This is the
  measured value of the federated alert in this model.
- False actions (about 5,400 a day) are the same for full sharing and the
  two-stage scheme, because both act on the same full-precision estimate.

Figure: `figures/fig_two_stage_tradeoff.png` (and `.pdf`).

## 3. Second epoch: same result two months later

Same pipeline on the Starlink objects in the 12 Nov 2026 `active.tle` (10,615
objects, 1.12 million cross-operator approaches under 10 km, 4,427 dangerous a day):

| Regime | Warned >= 24 h, 10 Sep | Warned >= 24 h, 12 Nov | Cluster-hours / day, 10 Sep / 12 Nov |
| --- | --- | --- | --- |
| No cooperation | 9.9% | 9.9% | 0 / 0 |
| Two-stage, tau = 3e-5 | 99.0% | 99.1% | 18.6 / 19.8 |
| Public trigger, tau = 3e-5 | 85.3% | 84.7% | 8.3 / 8.9 |
| Public trigger, tau = 1e-5 | 99.0% | 99.1% | 32.7 / 35.2 |
| Full sharing | 99.5% | 99.6% | n/a |

Two epochs are not yet a distribution; three to five are planned.

## 4. Robustness: Pc model and tracking-uncertainty sweep

`research/sweep.py` reruns the Monte Carlo with one implementation for every
case: each object's RTN covariance (along-track ratio k, k = 1 is exactly the
circular model) projected into each encounter plane, Pc by the small-hard-body
form of Foster's integral (median 0.09% from `pc.foster_pc`, worst 16% on 316
checks; `research/aniso.py`). Grid: k in {1, 4, 8, 16}, precise sigma in
{0.25, 0.5, 1.0} km, public sigma far out in {1.5, 3, 6} km, both epochs: 72 cases.

Metric: the MPC cluster-hours a day each trigger needs to warn 99% as many
dangerous encounters at least 24 h ahead as full sharing does.

Findings:

1. **The two-stage scheme always gets there cheaply.** Across all 72 cases it needs
   3.7 to 67 cluster-hours a day, against about 1,594 for a single all-pairs secure
   pass. This is the robust claim.
2. **The operators' alerts beat a public-only trigger in 67 of 72 cases, by a median
   factor of 1.35 (range 0.84 to 2.0).** The advantage is larger when public
   tracking is poor (median 1.41 when no cooperation warns under 90%) and nearly
   vanishes when it is good (median 1.15). In 5 cases, all along-track ×16 or
   precise 1 km, the public trigger is slightly cheaper.
3. **How much cooperation matters at all depends strongly on the covariance
   model.** With the default assumptions (precise 0.5 km, public 3 km), no
   cooperation warns 10% of dangerous encounters a day ahead under the circular
   model but 92% under Foster ×8. The gap to full sharing (99.8%) is then about
   8%, roughly 160 encounters a day across this split catalogue. The paper must
   present the motivation as a range, not the circular 10%.

Figure: `figures/fig_sweep.png` (precise 0.5 km, 10 Sep). Full tables:
`results/*__sweep.json`.

| k | public far (km) | No coop, warned >= 24 h | Two-stage h/day | Public trigger h/day |
| --- | --- | --- | --- | --- |
| 1 (circular) | 1.5 / 3 / 6 | 90% / 10% / 0% | 7.4 / 16.7 / 43.1 | 11.2 / 30.1 / 60.6 |
| 4 | 1.5 / 3 / 6 | 99% / 56% / 1% | 8.2 / 14.5 / 32.7 | 9.5 / 21.2 / 45.9 |
| 8 | 1.5 / 3 / 6 | 100% / 92% / 14% | 6.4 / 9.7 / 21.4 | 6.5 / 15.3 / 30.1 |
| 16 | 1.5 / 3 / 6 | 100% / 100% / 68% | 4.2 / 8.6 / 14.1 | 4.7 / 9.4 / 19.7 |

(10 Sep epoch, precise sigma 0.5 km; the 12 Nov epoch is within 13% of these in every case.)

## 5. Five catalogue dates

Starlink from Space-Track `gp_history` (newest element set per satellite on the
day; `research/latest_per_object.py`) for 28 Feb, 31 May and 31 Jul 2026, plus
the 10 Sep and 12 Nov catalogues. Circular model, precise 0.5 km, public 3 km,
tau = 3e-5, 3 seeds each.

| Date | Satellites | Dangerous / day | No coop, warned >= 24 h | Two-stage, warned >= 24 h | Two-stage cluster-h / day | Public trigger, warned >= 24 h | Full sharing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 28 Feb 2026 | 9,737 | 2,327 | 9.6% | 99.0% | 8.4 | 84.6% | 99.5% |
| 31 May 2026 | 10,073 | 2,373 | 9.9% | 99.0% | 9.6 | 84.8% | 99.5% |
| 31 Jul 2026 | 10,777 | 3,228 | 9.6% | 99.0% | 13.6 | 84.9% | 99.6% |
| 10 Sep 2026 | 10,713 | 4,188 | 9.9% | 99.0% | 18.6 | 85.3% | 99.5% |
| 12 Nov 2026 | 10,615 | 4,427 | 9.7% | 99.0% | 19.8 | 84.9% | 99.5% |

The safety numbers are stable across dates to within 0.3 percentage points. The
secure load is not: it grows with the number of dangerous encounters, which
nearly doubled between February and November as the shells filled, so it should
also be reported per dangerous encounter: 65 to 80 secure Pc evaluations each,
most of them spent on alerted pairs that turn out safe.

Over all five dates the sweep has 180 cases. The two-stage scheme needs 2.0 to
67 cluster-hours a day and the operators' alerts beat the public trigger in 165
of 180, by a median factor of 1.32 (range 0.82 to 2.0). Under Foster ×8 with
the default sigmas, no cooperation warns 92% to 93% of dangerous encounters a day
ahead on every date, against 99.6% to 99.8% for full sharing.

## 6. The two documented 2021 China Space Station events

`research/replay_2021.py`, using `data/historical/css_2021.tle` (Space-Track
`gp_history`, NORAD 48274, 44971, 47989). Every 6 h, each object's newest element
set *published before that moment* is propagated over the event day and the
closest predicted approach found. UN document A/AC.105/1262 gives only the dates
and says the station manoeuvred "in the evening" (1 July) and "on the same day"
(21 October); it gives no times or distances.

**21 October 2021, Starlink-2305: reproduced.** From element sets published by
20 Oct 00:00 UTC, the predicted approach is 471 m at 21 Oct 10:10 UTC (18:10
Beijing time, the evening), stable over the next two updates (432 m). Pc with
both orbits precise: 5.5e-4 circular, 9.9e-4 Foster ×8, above the 1e-4 threshold.
From public tracking alone the Pc at that point is 6.8e-5, under the threshold,
and first crosses it about 22 h ahead.

| Scheme | First warning before the predicted approach |
| --- | --- |
| No cooperation (public-only Pc >= 1e-4) | about 22 h |
| Two-stage (public-level alert >= 3e-5, then secure Pc >= 1e-4) | about 34 h |
| Full sharing | about 34 h |

By the 20 Oct 18:00 update the prediction jumps to 6.4 km, consistent with
Starlink-2305 manoeuvring, as the UN note says it was continuously. Both schemes
are limited here by when the public data first showed the approach (34 h),
which is a real-world limit on any trigger.

**1 July 2021, Starlink-1095: not reproduced.** Public element sets never predict
an approach closer than 3.6 km on 1 July. The closest late predictions are 4.8 km
at 07:10 and 10:14 UTC (15:10 and 18:14 Beijing). With these orbits no Pc model
here reaches the threshold. A crewed station plausibly uses a stricter rule (a
keep-out box) or better tracking of its own orbit, which these data cannot show.
The paper should report this as a negative result, not drop it.

## Reproduce

```bash
PYTHONPATH=src python research/run_refined_eval.py --hours 168 --screening grid
PYTHONPATH=src python research/run_refined_eval.py --hours 168 --screening refined
PYTHONPATH=src python research/cache_encounters.py --tle data/starlink.tle --hours 168 --screen-km 10
PYTHONPATH=src:research python research/two_stage.py research/cache/<cache>.npz      # deterministic
PYTHONPATH=src:research python research/two_stage_mc.py research/cache/<cache>.npz --seeds 3
python research/fig_two_stage.py research/results/<cache>__mc_circular_far3.json
PYTHONPATH=src:research python research/sweep.py research/cache/<cache>.npz --k 1,4,8,16 --sp 0.25,0.5,1.0 --far 1.5,3,6
python research/latest_per_object.py <space-track 3le> data/historical/<name>_latest.tle
PYTHONPATH=src python research/replay_2021.py data/historical/css_2021.tle
python research/fig_sweep.py research/results/<cache>__sweep.json
```

The cache file (125 MB) is not committed; it regenerates in about 7 minutes.

## Next

1. Leakage: what an observer of the alerts learns about each operator's orbits.
2. Adaptive and colluding attackers against the alert layer.
