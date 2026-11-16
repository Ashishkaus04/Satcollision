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

## 7. Privacy: what a rival operator learns from the federation's messages

`research/leakage.py`. Attacker: operator A, which knows its own satellites
precisely and everything else from the public catalogue. Target: a satellite S
of operator B. Secret: B's private correction to S's public orbit (the public
catalogue error, a 3-vector in S's RTN frame, prior 1-sigma 452 m radial,
3,618 m along-track, 452 m cross-track, i.e. 3 km combined with along-track ×8).
The correction is held fixed over the observation window, which favours the
attacker. A sees every message about S's approaches with A's own satellites
(under 10 km, every 8-hourly update) and computes a Bayesian posterior over the
correction (importance sampling in two passes; B's estimation noise integrated
by Monte Carlo). 10 Sep catalogue, 100 targets for one day, 75 for one week
(25 for the hardened design).

Remaining uncertainty, median over targets, as a share of the public 1-sigma
(along-track / radial):

| What A sees | 1 day | 1 week |
| --- | --- | --- |
| "Act now" decisions only (hardened design) | 28% / 26% (1.0 km / 118 m) | 18% / 18% |
| Alerts | 18% / 18% (660 m / 80 m) | 8% / 7% |
| Alerts + severity tier | 13% / 9% | 4% / 3% |
| + TCA in whole seconds | 10% / 12% | 3% / 3% |
| + secure step's yes/no | 8% / 6% (300 m / 28 m) | 3% / 2% |
| Secure step returns the Pc value | about 0 (orbit recovered) | about 0 |

Cross-track is barely learned from alerts alone (96% left after a day) but the
TCA channel exposes it (55%). Per target and day there are about 300
approach-updates and 9 to 21 alerts. Figure: `figures/fig_leakage.png`.

Caveats. The one-week posteriors for the richer channels are poorly resolved
(effective sample size 12 to 28), so treat those percentages as approximate;
the error of the posterior mean is a steadier measure, at about 10% of the prior
there. The attacker knows the noise models exactly, which is a strong
assumption. A real public-catalogue error drifts from one element set to the
next, so the one-day numbers are the more realistic ones.

What this means for the design and the paper:

1. **Coarse signals are not privacy.** Hundreds of yes/no outcomes a day act like
   measurements. The claim "never shares raw orbits" is true but must not be
   presented as "reveals nothing about orbits".
2. **The secure step must return only the decision, never the Pc value.** The
   value pins the orbit exactly.
3. **Hardened design:** feed the stage-1 alerts into the secure computation
   instead of announcing them, drop tiers, and round or omit TCAs. A then learns
   only "act now", the floor any warning service must reveal.
4. **The floor itself is large in a constellation this dense:** a day of
   act/no-act decisions still cuts the along-track uncertainty to about a
   quarter. Adding noise to alerts cannot go below this floor, because the act
   decisions cannot be randomised without costing safety. This is a limit of
   conjunction warning in general, and the paper should say so.

## 8. Dishonest operators against the two-stage scheme

`research/adversarial.py`, 10 Sep catalogue, precise 0.5 km, public 3 km, tau = 3e-5.
Warned = share of dangerous encounters warned at least 24 h ahead.

| Scenario | Circular: warned / MPC h per day | Foster ×8: warned / MPC h per day |
| --- | --- | --- |
| Both honest | 99.1% / 18.6 | 99.8% / 18.5 |
| B withholds all its alerts | 97.6% / 13.7 | 99.7% / 14.6 |
| A alerts on every approach (flood) | 99.6% / 84.0 | 99.8% / 84.0 |
| Flood, but secure step needs B's own Pc >= 1e-5 | 99.4% / 26.2 | 99.7% / 19.7 |
| Flood, but secure step needs B's own Pc >= 1e-6 | 99.5% / 46.2 | 99.7% / 26.9 |
| Both honest, both checks >= 1e-5 | 98.2% / 10.9 | 99.7% / 11.6 |
| Both honest, both checks >= 1e-6 | 99.0% / 15.4 | 99.7% / 15.3 |

- **Withholding is self-limiting:** the honest side's own alerts still cover
  almost every dangerous encounter, because each operator can detect the
  approach from its own precise orbit. Loss: 1.5 points (circular), 0.1 (Foster).
- **Flooding is the real threat.** It multiplies the secure compute by 4.5 (a lower
  bound: the cache only holds approaches under 10 km, and a flooder could name
  any pair) and, worse, lets the flooder query the secure step's decision on
  every approach. That is the probing route to the leakage in section 7.
- **Defence: a plausibility check.** Run the secure step only if the *other*
  operator's own stage-1 Pc reaches a floor. With a floor of 1e-6 an honest
  federation loses nothing measurable (99.0% vs 99.1%) and a flooder's extra
  load falls from +65 to +28 cluster-hours a day (circular). Caveat: if the
  flooder can see whether the secure step ran, it learns the outcome of B's check,
  an alert-like bit at a lower threshold. The check should therefore also run
  inside the secure computation, so only the final decision comes out.
- Collusion: with two operators, collusion only harms the colluders' own pairs.
  The secure step itself assumes its servers do not collude (the standard MPC
  assumption); that belongs in the paper's threat model, not in an experiment.

## 9. Rerun with the measured public-catalogue error (no assumed sigmas)

`research/measured.py`. Sections 2 to 5 assumed the public 1-sigma falls from
3 km (combined) at 72 h to 0.5 km at TCA. We replaced that with the RTN error
measured on one week of Space-Track history (6 to 12 Nov 2026, 11,204 Starlink
objects, TLE vs later TLE, robust 1.4826 x MAD; `research/tle_error_fit.py`):

| Age of the public set | Radial | Along-track | Cross-track |
| --- | --- | --- | --- |
| 0 to 3 h | 0.11 km | 0.82 km | 0.05 km |
| 6 to 12 h | 0.20 km | 2.0 km | 0.14 km |
| 12 to 24 h | 0.28 km | 7.9 km | 0.19 km |
| 24 to 36 h | 0.38 km | 20 km | 0.24 km |
| 48 to 72 h | 1.2 km | 87 km | 0.45 km |

Per object, at age = lead time + 4 h (mean age of the newest set at an update),
interpolated log-log. "Biased" adds the median along-track drift (+3.7 km at
12 to 24 h, +44 km at 48 to 72 h) in quadrature. The operator's own precise
sigma is still an assumption (0.25, 0.5 or 1.0 km combined, constant).

Precise 0.5 km, spread only (the biased variant differs by under 1.5 MPC h/day):

| Date | No coop, warned >= 24 h | Full sharing | Two-stage MPC h/day for 99% of full | Public trigger, same | Ratio |
| --- | --- | --- | --- | --- | --- |
| 28 Feb | 0.0% | 99.6% | 18.6 | 20.9 | 1.12 |
| 31 May | 0.03% | 99.4% | 21.6 | 24.6 | 1.14 |
| 31 Jul | 0.05% | 99.6% | 31.2 | 36.0 | 1.15 |
| 10 Sep | 0.08% | 99.6% | 43.4 | 51.3 | 1.18 |
| 12 Nov | 0.18% | 99.5% | 46.0 | 54.8 | 1.19 |

Over precise 0.25 to 1.0 km the ratio is 1.06 to 1.30 (30 cases, alerts cheaper in all 30).

What changes:

1. **No cooperation is far worse than either assumed model.** With measured
   errors almost no dangerous encounter (under 0.2%) is warned a day ahead from
   public data. The "10% (circular) to 92% (Foster x8)" range of section 4 was
   an artefact of the assumed 3 km. The motivation for sharing is now strong
   and no longer model-dependent.
2. **The scheme still matches full sharing**, but needs a much lower alert
   threshold (tau about 3e-7, not 3e-5): with one side's along-track error at
   tens of km, a single operator's Pc is diluted.
3. **Secure compute roughly doubles to triples**: 19 to 46 cluster-hours/day
   (was 8 to 20), still 35 to 85 times below all-pairs (about 1,594).
4. **Operator alerts help less than before**: 1.12 to 1.19 times cheaper than a
   public trigger (was a median 1.32). When the public error is this large,
   one side's precise orbit narrows the search less. The honest claim is that
   alerts are a modest improvement over a public trigger; the large gain is the
   two-stage structure itself.

Caveats: the error table is Starlink-only and from one week; TLE-vs-TLE
consistency understates the true error for objects with shared observations;
the precise sigma is still assumed. Results: `results/*__measured.json`.

## 10. Hindcast on four weeks of real TLE history: baseline, +Foster, +Foster+FL

`research/hindcast_fl.py` (modules `src/satcollision/hindcast.py`,
`src/satcollision/error_learning.py`). Sections 2 to 9 draw the prediction
error from a Gaussian. This section removes that: every error is a real one.

**Data.** Space-Track `gp_history`, 16 Oct to 12 Nov 2026 (four weeks, 1.64
million element sets, 28,732 objects, mean motion > 11.25 rev/day; the Oct 16
and Oct 30 weeks were downloaded in two NORAD halves to stay under the
500,000-row query cap). Members are the six operators of the
`real_leo_history` world (Starlink 11,234, OneWeb 654, Kuiper 392, Planet 148,
Spire 96, ICEYE 61); everything else (payloads and debris) is unowned.

**Set-up.**

* *Truth* at time t is the first element set published after t, propagated
  back (blended over 10 min across switches). A prediction made at update u
  uses sets with epoch <= u, so it never shares a set with its truth, and its
  error at age a is exactly the TLE-vs-later-TLE error the public table
  measures. (The world's nearest-epoch truth would give zero error to every
  prediction within half a publication gap of its epoch.)
* *Encounters*: every true approach under 10 km between a member's satellite
  and an object that member does not fly, plus a thinned sample of 10-30 km
  (1 in 10) and 30-100 km (1 in 100) approaches, weighted back, because an
  over-confident model raises false alarms on wide approaches. 513,301
  approaches kept (18.3 million weighted) with TCA in the last 4 days.
* *Assessments*: at every 8-hourly update in the 72 h before TCA, each member
  side predicts the other object from the public catalogue and knows its own
  satellite exactly (precise sigma 0.354 km). Predicted miss = true miss minus
  the in-plane part of the real prediction error (linear relative motion near
  TCA, Foster's own assumption). 5.13 million assessments.
* *Dangerous*: Pc on the true miss with both orbits precise >= 1e-4 (miss
  under about 1.02 km): 447 a day. *Caught*: Pc >= threshold at some update.
  *False alarm*: an alerted (approach, side) that is not dangerous.
* *Train / test split in time*: the error table and every model see only
  data before day 19.9; every test update is after it (TCAs on days 22.9 to
  26.9). Three seeds (models and sample draws); the encounters are the same.

**Error models of the other object's public prediction.**

| Model | Trained on | Used for |
| --- | --- | --- |
| Public table (section 9) | TLE vs later TLE, all objects, Starlink / other, by age | +Foster |
| Public model | the same public labels, a small network on set features (age, altitude, inclination, eccentricity, B*, ndot, publication cadence) | unowned objects in every +Foster+model row |
| Local-only | one operator's own fleet (own ephemeris vs public TLE) | members' satellites, the assessor's own model |
| FedAvg | all six fleets, model weights only (sqrt-of-sample weighting, early stopping on each operator's held-out objects) | members' satellites |
| Pooled | all six fleets' data in one place (privacy-violating upper bound) | members' satellites |
| Owners' models (suggested) | each operator shares its own fleet's model | that owner's satellites |

The network predicts per RTN axis a bias and the scale of a Student-t (3 dof,
for manoeuvre tails); the Pc uses the bias-corrected prediction and the robust
sigma. 6,296 bytes per model update.

**Error-model quality** (held-out NLL on each fleet's satellites in the test
period, lower is better; mean of 3 seeds):

| Fleet | Public table | Other operators' local models | FedAvg | Pooled | Owner's own model |
| --- | --- | --- | --- | --- | --- |
| Starlink | 1.99 | 2.66 | 1.58 | 1.27 | 1.25 |
| Kuiper | 2.93 | 1.42 | 0.58 | 0.46 | 0.45 |
| OneWeb | 0.17 | 5.2 | -0.47 | -0.79 | -0.78 |
| Planet | 0.20 | 0.36 | 0.04 | -0.48 | -0.61 |
| Spire | -0.67 | 0.12 | -0.29 | -1.04 | -1.15 |
| ICEYE | -0.09 | 0.20 | -0.18 | -0.67 | -0.75 |

On unowned objects the public table is badly calibrated where it matters: it
covers 54% of along-track errors at its 1-sigma overall and only 11% at 36-72 h
for the objects that actually conjunct (lower, more drag than the average
catalogue object). The public model covers 69-73% (NLL -1.36 vs -1.02).

**Screening.** Two views, because a better-calibrated covariance changes how
many alerts a fixed threshold raises (honest, larger covariances dilute the Pc).
At the fixed threshold 1e-4 (mean of 3 seeds):

| Step | Caught | Caught >= 24 h | Median warning | False alarms / day | Member-vs-member: caught / FA per day |
| --- | --- | --- | --- | --- | --- |
| Baseline: circular, 0.5 km | 87.0% | 60.4% | 36.0 h | 1,333 | 63.3% / 309 |
| + Foster, public table | 73.6% | 51.6% | 36.3 h | 1,046 | 53.1% / 197 |
| + Foster, public model (unowned) | 65.7% | 39.6% | 31.3 h | 771 | 46.5% / 101 |
| + Foster + local-only models | 63.2% | 37.9% | 31.2 h | 804 | 31.6% / 134 |
| **+ Foster + FedAvg** | **65.4%** | **39.3%** | **31.2 h** | **809** | **44.6% / 139** |
| + Foster + DP-FedAvg (eps 13-15) | 64.9% | 38.1% | 30.3 h | 795 | 41.9% / 125 |
| + Foster + pooled | 65.7% | 39.6% | 31.4 h | 787 | 46.3% / 117 |
| + Foster + owners' models | 67.0% | 41.5% | 32.1 h | 826 | 54.3% / 156 |

(The "+ Foster + model" rows all use the public model for unowned objects; they
differ only on members' satellites.)

At equal false-alarm budgets (threshold set per scheme; share of dangerous
encounters caught):

| Step | All, 30 FA/day | All, 100 FA/day | All, 300 FA/day | Member-vs-member, 3 / 10 / 30 FA/day |
| --- | --- | --- | --- | --- |
| Baseline | 21.3% | 22.4% | 33.0% | 14.2% / 15.0% / 15.7% |
| + Foster, public table | 12.5% | 19.6% | 37.3% | 6.9% / 7.3% / 8.9% |
| + Foster, public model | 25.1% | 35.7% | 49.9% | 14.6% / 20.8% / 29.0% |
| + Foster + local-only | 20.9% | 31.6% | 46.0% | 2.1% / 4.1% / 8.6% |
| **+ Foster + FedAvg** | **24.3%** | **34.7%** | **49.2%** | **10.7% / 14.8% / 24.8%** |
| + Foster + pooled | 24.7% | 35.2% | 49.6% | 11.6% / 17.9% / 27.6% |
| + Foster + owners' models | 25.2% | 36.0% | 50.5% | 14.8% / 21.1% / 32.0% |

Small operators (assessor's own encounters, threshold 1e-4, caught / false
alarms a day, local-only vs FedAvg vs pooled): Spire 46% / 29.0, 46% / 21.5,
45% / 11.6; ICEYE 59% / 19.8, 54% / 16.7, 50% / 12.3; Planet 38% / 81.6,
45% / 71.7, 38% / 49.0. With 48 to 131 dangerous encounters each over 4 days,
these are noisy.

Figure: `figures/fig_hindcast_w4.png` (caught vs false alarms for both subsets,
and the model-quality bars).

What this says, honestly:

1. **Foster with the public table is not, by itself, an improvement.** At the
   fixed threshold it trades 13 points of recall for 22% fewer false alarms;
   at equal false-alarm budgets it is worse than the circular baseline below
   about 200 a day. The reason is calibration: the table is a population
   average and under-states the along-track error of the objects that actually
   conjunct. The anisotropic Pc is only as good as its covariance.
2. **A calibrated, feature-based error model is what makes Foster pay off**:
   60% more dangerous encounters caught at 100 false alarms a day (35.7% vs
   22.4%), and 42% fewer false alarms at the fixed threshold (771 vs 1,333 a
   day). The cost is recall at very large budgets (above about 1,000 false
   alarms a day the circular baseline catches more) and about 5 h of median
   warning: the probability-dilution effect, an honest wide covariance caps the
   Pc of a long-range encounter.
3. **Federation versus training alone.** For screening another member's
   satellite, an operator's local model is the worst option (member-vs-member:
   4% caught at 10 false alarms a day, against 15% for FedAvg and 18% pooled);
   other operators' local models predict a fleet worse than even the public
   table (NLL 5.2 vs 0.17 on OneWeb). FedAvg recovers most of the pooled
   model's benefit without moving any data; for the small operators it mostly
   shows up as fewer false alarms at the same recall (Spire 29.0 -> 21.5 a day,
   pooled 11.6). FedAvg stays short of pooled training (non-IID fleets, one
   dominant operator; sqrt weighting helps).
4. **But in this data a public model does as well as federation.** Here the
   operators' "own ephemeris" is the later TLE (the only truth we have), so the
   federated labels are public information and the public model, trained on
   every object, matches or beats FedAvg and pooled. The experiment validates
   the protocol (FedAvg reaches near-pooled quality without pooling; secure
   aggregation gives the same weights to 1e-3; DP at eps 13-15 costs 2-3
   points), not the value of private labels. That value depends on operators'
   precise ephemerides being better than TLE consistency, which these data
   cannot show.
5. **Suggested improvement: owners' models** (adopted; see 10b). Each member shares one model of
   its *own* fleet's public-catalogue error, and a twin uses the owner's model
   for that owner's satellites. It is the best option on every measure here
   (member-vs-member 21% at 10 FA/day vs 15% FedAvg, 18% pooled), costs one
   model per operator instead of rounds of training, and avoids FedAvg's
   non-IID drift. The privacy cost is that one operator's model is not
   averaged with others'; DP on that model, or evaluating it only inside the
   secure step, is the mitigation.

Caveats. Four days of test encounters (one test window, three model seeds; a
rolling multi-window evaluation is the next step). The truth is TLE-derived,
so "dangerous" is relative to a later TLE, not to a precise ephemeris. Linear
relative motion near TCA; approaches slower than 10 m/s dropped. The DP
epsilon is operator-level per training run, by RDP composition, with no
amplification by sampling.

Reproduce (data in `data/historical/`, not in git):

```bash
D=data/historical
PYTHONPATH=src python research/hindcast_fl.py $D/gp_history_oct16a.tle $D/gp_history_oct16b.tle \
  $D/gp_history_oct23.tle $D/gp_history_oct30a.tle $D/gp_history_oct30b.tle $D/gp_history_week.tle \
  --name w4 --seed 0     # screen ~10 min (then cached), ~8 min per seed
PYTHONPATH=src python research/hindcast_table.py research/results/hindcast_w4_s*.json
python research/fig_hindcast.py research/results/hindcast_w4_s*.json
```

### 10b. Adopted design: owners' models (decided 15 Nov 2026)

The project now uses owners' models as the third step: **baseline -> +Foster ->
+Foster + owners' models.** Each member trains one model of its own fleet's
public-catalogue error and shares it with the federation (6.3 kB per operator,
37.8 kB for all six; no training rounds). A twin uses the owner's model for that
owner's satellites and the public model for objects no member flies
(`error_learning.OwnerModels`). FedAvg and local-only stay in the paper as
comparisons.

Because a shared owner model is not averaged with anyone else's, it is trained
with DP-SGD (`error_learning.train_owner_dp`, opacus, RDP accountant, 5 epochs,
clip 1.0). The guarantee is per training sample (one error of one element set
at one age); a satellite with k samples is covered at about k x epsilon by
group privacy. (DP-FedAvg's epsilon 13 to 15 is per operator and per run, a
different unit, so the two epsilons are not comparable.)

Mean of 3 seeds:

| Step | Caught at 1e-4 | Warned >= 24 h | False alarms / day | Caught at 100 FA/day | Member vs member at 10 FA/day |
| --- | --- | --- | --- | --- | --- |
| Baseline: circular, 0.5 km | 87.0% | 60.4% | 1,333 | 22.4% | 15.0% |
| + Foster, public table | 73.6% | 51.6% | 1,046 | 19.6% | 7.3% |
| **+ Foster + owners' models** | **67.0%** | **41.5%** | **826** | **36.0%** | **21.1%** |
| + Foster + owners' models, DP epsilon 8 | 67.2% | 41.7% | 914 | 33.7% | 15.7% |
| + Foster + owners' models, DP epsilon 2 | 67.1% | 41.6% | 912 | 32.8% | 14.7% |
| (comparison) FedAvg, no DP | 65.4% | 39.3% | 809 | 34.7% | 14.8% |
| (comparison) DP-FedAvg | 64.9% | 38.1% | 795 | 34.5% | 12.2% |
| (comparison) local-only | 63.2% | 37.9% | 804 | 31.6% | 4.1% |

Held-out NLL of the DP owner models (epsilon 2) against the same models without
DP: Starlink 1.33 vs 1.25, Kuiper 0.52 vs 0.45, OneWeb -0.75 vs -0.78, Planet
-0.43 vs -0.60, Spire -1.06 vs -1.15, ICEYE -0.51 vs -0.75.

Small operators' own encounters at the fixed threshold (caught / false alarms
a day; local-only, FedAvg, owners' models, owners' models with DP epsilon 2):
Spire 46% / 29.0, 46% / 21.5, 49% / 12.6, 42% / 17.8; ICEYE 59% / 19.8,
54% / 16.7, 51% / 12.8, 52% / 14.3; Planet 38% / 81.6, 45% / 71.7, 40% / 49.3,
40% / 59.3.

Reading:

1. Without DP, owners' models are the best option on every measure in this window (but see 10c: over four windows the public model is level or better).
2. DP costs most of their advantage on member-vs-member screening (21.1% ->
   14.7% at 10 false alarms a day), leaving them level with non-private
   FedAvg and ahead of DP-FedAvg (12.2%). Over all encounters the cost is
   3 points (36.0% -> 32.8%).
3. Epsilon 2 and 8 differ by under 1 point, so the loss comes from DP-SGD's
   clipping and short fixed training, not from the noise level. Better DP
   training (more epochs at the same epsilon, tuned clipping) is the obvious
   next step and may recover much of the gap.
4. The paper's claim is therefore: owners' models with DP at epsilon 2 come
   close to non-private FedAvg (14.7% vs 14.8% member-vs-member, 32.8% vs
   34.7% overall) and beat DP-FedAvg, at one 6.3 kB message per operator and
   with a formal per-sample guarantee. Without DP they are clearly best.

### 10c. Rolling evaluation: four test windows

`research/hindcast_rolling.py`. The same hindcast on four consecutive 4-day test
windows (TCAs on days 10.9-14.9, 14.9-18.9, 18.9-22.9 and 22.9-26.9 of the four
weeks), each with the error table and every model retrained only on the days
before it (expanding window: Spire has 11,474 training samples for the first
window and 26,383 for the last). One model seed per window; 447 to 523
dangerous encounters a day. Mean +- standard deviation over the four windows:

| Step | Caught at 1e-4 | False alarms / day at 1e-4 | Caught at 100 FA/day | Member vs member, caught at 10 FA/day |
| --- | --- | --- | --- | --- |
| Baseline: circular, 0.5 km | 85.7 +- 1.3% | 1,190 +- 145 | 22.9 +- 1.0% | 13.8 +- 1.8% |
| + Foster, public table | 72.2 +- 1.4% | 900 +- 101 | 24.7 +- 3.4% | 11.8 +- 3.2% |
| **+ Foster + owners' models** | **66.3 +- 1.3%** | **792 +- 68** | **35.1 +- 3.8%** | **18.7 +- 1.8%** |
| + Foster + owners' models, DP epsilon 2 | 66.6 +- 1.1% | 835 +- 82 | 34.2 +- 4.2% | 15.7 +- 2.5% |
| (comparison) public model only | 65.8 +- 1.4% | 766 +- 62 | 35.1 +- 4.1% | 20.2 +- 1.9% |
| (comparison) FedAvg | 65.2 +- 1.5% | 805 +- 59 | 34.2 +- 3.8% | 13.2 +- 1.6% |
| (comparison) local-only | 63.7 +- 1.6% | 814 +- 42 | 32.0 +- 2.5% | 3.6 +- 1.2% |

Paired differences per window (percentage points, windows in time order):

| Comparison | All, caught at 100 FA/day | Member vs member, caught at 10 FA/day |
| --- | --- | --- |
| Owners' models - baseline | +18.3, +9.4, +9.1, +12.0 | +5.9, +3.4, +4.6, +6.0 |
| Owners' models - FedAvg | +0.7, +0.9, +0.8, +1.3 | +6.5, +4.5, +4.9, +6.4 |
| Owners' models with DP eps 2 - FedAvg | +0.5, +0.7, -0.2, -1.1 | +6.0, +3.1, +1.3, -0.1 |
| Owners' models - public model | -0.6, 0.0, 0.0, +0.7 | -3.0, -3.1, -2.2, +2.5 |

What holds and what changes:

1. **Holds in every window:** owners' models beat the circular baseline (by 9
   to 18 points overall at 100 false alarms a day) and beat FedAvg on
   member-against-member screening (by 4.5 to 6.5 points). Local-only models are
   always the worst for screening another member's satellite.
2. **With DP at epsilon 2**, owners' models stay ahead of FedAvg in three of
   four windows on member-against-member screening, but the margin shrinks
   from 6 points to zero over time.
3. **Correction to 10b:** owners' models are not the best option on every
   measure. Over four windows the public model, trained only on public TLE data,
   is level overall and better on member-against-member screening in three of
   four windows (by 2 to 3 points). The single-window result in 10b was within
   window-to-window noise. This strengthens the point of section 10: with
   TLE-derived labels, private data adds nothing measurable here, and only
   operator-quality labels (the SupGP collection started on 15 Nov) can show
   whether owners' models earn their place.
4. **Foster with the public table** varies by window (+6.2, +1.6, +1.7 and
   -2.6 points against the baseline at 100 false alarms a day, mean +1.8). The
   calibrated model, not the formula, is what moves the result (+12 on average).

SupGP collection. `scripts/collect_supgp.py` saves CelesTrak's supplemental
element sets (fitted to the operators' own ephemerides) for Starlink, OneWeb,
Planet and Kuiper every 6 h, from a Windows scheduled task ("Satcollision
SupGP collector") that catches up after the laptop sleeps. ICEYE and Spire have
no SupGP. Two to three weeks of it will give operator-quality labels.

## 11. Ground truth from precise orbits: how wrong is the TLE-derived truth?

`research/precise_orbits.py`, `research/precise_vs_tle.py`. Sections 9 and 10 measure
public-catalogue error against the *next published TLE*. That has a floor: every
TLE is itself a fit with its own error. Here two satellites with independent,
centimetre-level orbits give real ground truth: ESA's Sentinel-1C and -1D
(693 km, sun-synchronous), precise POEORB orbits to 26 Oct and restituted RESORB
orbits (about 10 cm) after that, from ESA's public SNAP server (no login),
12 Oct to 13 Nov 2026. State vectors are rotated from ITRF to TEME with IAU-82
sidereal time (polar motion and the equation of the equinoxes ignored: tens of
metres). Sanity check: the TLE fit error at each set's own epoch is 0.08 to
0.11 km radial and 0.58 to 0.70 km along-track (robust sigma), the size
usually quoted for TLEs.

**1. TLE-vs-later-TLE understates the error, badly at short ages.** Along-track
robust sigma of the public prediction, against the precise orbit and against the
next TLE, at the same instants (Sentinel-1C / 1D):

| Age of the public set | Real error (vs precise) | TLE-derived error (vs next TLE) | Understated by |
| --- | --- | --- | --- |
| 0-3 h | 0.49 / 0.62 km | 0.03 / 0.07 km | 17x / 8x |
| 3-6 h | 0.64 / 0.86 km | 0.05 / 0.09 km | 14x / 9x |
| 6-12 h | 0.86 / 0.81 km | 0.07 / 0.12 km | 12x / 7x |
| 12-24 h | 0.75 / 0.71 km | 0.14 / 0.11 km | 6x / 6x |
| 24-36 h | 0.73 / 1.03 km | 0.22 / 0.37 km | 3.4x / 2.8x |
| 36-48 h | 1.53 / 0.70 km | 0.33 / 0.28 km | 4.6x / 2.5x |
| 48-72 h | 1.35 / 1.48 km | 0.86 / 0.83 km | 1.6x / 1.8x |

Consecutive TLEs agree with each other far better than either agrees with the
real orbit (they share observations and fitting), so the TLE-derived truth
itself is off by 0.5 to 0.9 km along-track at every age. For these two
well-tracked, non-manoeuvring satellites the real error of a fresh TLE is about
0.5 to 0.6 km along-track, not 30 to 70 m.

**2. An owner with precise labels gains a lot.** Treating the two Sentinels as one
operator's fleet: models trained on errors before day 19.9 (1,439 samples) and
scored on held-out errors after it (452 samples), against the precise orbit.
Mean of 3 seeds:

| Model | Held-out NLL | Along-track 1-sigma coverage (target ~0.68) |
| --- | --- | --- |
| Public model (TLE labels, every object) | 1.67 +- 0.18 | 0.19 |
| Owner's model trained on TLE labels | 1.66 +- 0.14 | 0.46 |
| **Owner's model trained on precise labels** | **-0.03 +- 0.06** | **0.76** |
| Owner's model, precise labels, DP-SGD epsilon 2 | 0.07 +- 0.05 | 0.65 |

The gain is calibration, not bias: the median error after bias correction is the
same for all four (0.43 to 0.44 km). A model trained on TLE consistency is
over-confident by roughly an order of magnitude at short ages; one trained on the
owner's precise orbit is not, and DP at epsilon 2 keeps nearly all of it.

What this means:

1. **It answers the question sections 10 and 10c left open.** With TLE labels, private
   data added nothing because the labels were public in disguise. With real precise
   labels, the owner's model is far better calibrated than anything public data
   can train. This is the measured case for owners' models (10b), on two
   satellites.
2. **It qualifies every hindcast number.** Sections 9 and 10 score against a truth
   with a 0.5 to 0.9 km along-track error of its own, and their public-error
   models are over-confident at short ages. "Dangerous" (true miss under about
   1 km) is defined against that noisy truth, so the absolute catch rates are
   uncertain. The comparisons between schemes share the same truth, so their
   ordering is more robust than their levels.
3. **Limits.** Two satellites of one type (well tracked, no manoeuvres, little
   drag at 693 km); Starlink and other low, manoeuvring fleets will differ. A few
   hundred test samples. The SupGP collection (and CelesTrak's archive, if they
   send it) is the way to repeat this for the member fleets.

Reproduce:

```bash
python research/precise_orbits.py --start 2026-10-12 --end 2026-11-13     # ~500 files, a few minutes
PYTHONPATH=src python research/precise_vs_tle.py data/precise/sentinel1_2026-10-12_2026-11-13.npz \n  data/historical/gp_history_oct16a.tle data/historical/gp_history_oct16b.tle data/historical/gp_history_oct23.tle \n  data/historical/gp_history_oct30a.tle data/historical/gp_history_oct30b.tle data/historical/gp_history_week.tle
```

## 12. Trying to recover the lost warning: a maximum-Pc rule (negative result)

`research/hindcast_dilution.py`, `hindcast.pc_max`. Honest covariances lower the
Pc of long-range encounters (probability dilution), and section 10 found the
calibrated schemes warn later and catch fewer than the circular baseline at
high false-alarm budgets. The standard remedy is to also look at the maximum Pc
any scaling of the covariance could give (Alfano's maximum probability):
alert if Pc >= 1e-4 OR Pc_max >= T2, with T2 set so the rule raises a chosen
number of false alarms. Four rolling windows (10c), mean +- sd:

| Rule | False alarms / day | Caught | Caught >= 24 h | Median warning |
| --- | --- | --- | --- | --- |
| Baseline (circular), Pc >= 1e-4 | 1,190 +- 145 | 85.7 +- 1.3% | 65.3 +- 5.4% | 44.3 h |
| + Foster, table, Pc >= 1e-4 | 900 +- 101 | 72.2 +- 1.4% | 55.8 +- 4.1% | 47.6 h |
| + Foster, table, + max-Pc at the baseline's false alarms | 1,190 | 80.8 +- 1.2% | 62.5 +- 3.5% | 47.2 h |
| + Foster + owners' models, Pc >= 1e-4 | 792 +- 68 | 66.3 +- 1.3% | 45.7 +- 5.7% | 38.6 h |
| ... + max-Pc, false alarms +25% | 990 | 66.9 +- 1.2% | 46.3 +- 5.5% | 39.1 h |
| ... + max-Pc at the baseline's false alarms | 1,190 | 69.2 +- 0.7% | 47.4 +- 5.1% | 39.0 h |
| Owners' models with DP eps 2, + max-Pc at the baseline's false alarms | 1,190 | 69.2 +- 0.9% | 47.6 +- 4.9% | 39.3 h |

The rule does not recover the gap for owners' models: 50% more false alarms buy
3 points of recall and half an hour of warning. It helps the public table more
(72% -> 81%), but still falls short of the baseline at the same false alarms.

Why: the maximum Pc favours encounters whose predicted miss lies along the
large-uncertainty (along-track) direction, while what makes an encounter
dangerous *in this hindcast* is a small true miss, and here the predicted miss is
already close to the true one, because the truth is the next TLE (section 11:
that truth understates the real error 3 to 17 times). A rule that ranks by plain
predicted miss distance, which is what the 0.5 km circular baseline amounts to,
is flattered by that truth. So the baseline's advantage at large budgets is, at
least partly, an artefact of TLE-derived truth, and should not be "fixed" by
tuning against it. The fair test is against precise or operator-fitted truth
(section 11, the SupGP collection).

## 13. Real secure computation: timing the secure Pc step

`research/mpc/secure_pc.mpc`, `research/mpc/make_inputs.py`, results in
`results/mpc_timing.json`. Sections 2 to 9 priced the secure step with a
throughput from the literature (18,000 pairs an hour per three-server cluster,
Valdma & Veskus, SDC9 2025). Here it is implemented and timed.

**Design.** Three parties: the two operators and an independent server,
replicated secret sharing over a 128-bit ring (MP-SPDZ 0.4.4, semi-honest,
honest majority). The encounter geometry is public: the plane basis from the
public relative velocity, each object's RTN axes, a public reference point. Each
operator projects its *private* precise position into the plane relative to that
point (2 numbers) and its private covariance into the plane (3 numbers), locally,
and secret-shares them. The program computes m = m_a - m_b and C = C_a + C_b,
then the decision Pc >= 1e-4 in the equivalent form q + ln det C <= -2 ln(2T/R^2),
and reveals only the act/no-act bit, never Pc, m or C (as section 7 requires).
Fixed point with 20 fractional bits.

**Correctness.** On 61,000 encounters drawn from the four-week hindcast (half
near misses, so both answers occur), the secure bits match the plaintext decision
in every case.

**Cost** (one laptop, all three parties on localhost under WSL2):

| Encounters per batch | Time | Rounds | Data sent, all parties |
| --- | --- | --- | --- |
| 1,000 | 0.46 s | 231 | 105 MB |
| 10,000 | 3.9 s | 1,073 | 1.05 GB |
| 50,000 | 22.0 s | 4,815 | 5.26 GB |

So about 0.44 ms of computation and 105 kB of traffic (35 kB per party) per
encounter. Localhost hides the network, so we add it from the measured rounds and
traffic:

| Setting | Per encounter | Encounters per hour per cluster |
| --- | --- | --- |
| Localhost (measured) | 0.44 ms | 8.2 million |
| LAN: 1 Gbit/s, 1 ms round trip (modelled) | about 0.8 ms | about 4.4 million |
| WAN: 100 Mbit/s, 50 ms round trip (modelled) | about 8 ms | about 450,000 |
| Literature figure used so far | | 18,000 |

What changes in the paper:

1. **The secure step is cheaper than assumed, by 25 times even over a slow WAN.** The
   two-stage loads of section 9 (19 to 46 cluster-hours a day at the old figure,
   i.e. 0.34 to 0.83 million secure evaluations a day) become 0.8 to 1.8 hours a
   day over the WAN and 5 to 11 minutes on a LAN. A single all-pairs pass
   (28.7 million pairs) takes about 64 h over the WAN instead of about 1,594 h.
2. **The ratio is unchanged.** The two-stage scheme still needs about 1% of the
   secure work of all-pairs screening; that is a count, not a speed.
3. **Not like for like.** The literature figure covers a heavier secure
   computation (secure state handling and the full Pc); ours keeps the geometry
   public and computes only the decision from locally projected inputs. The
   public geometry leaks nothing beyond the public catalogue, but the paper must
   say which design the numbers are for. Semi-honest security only; a maliciously
   secure protocol (e.g. SPDZ-wise replicated) costs several times more.
4. The network rows are modelled from measured rounds and bytes, not measured on a
   real network.

**Against parties that cheat.** The numbers above are for semi-honest parties. Maliciously
secure 3-party protocols in MP-SPDZ need exact truncation, so the same program was rerun
with it (10,000 encounters, localhost):

| Protocol | Security | Time | Per encounter | Data, all parties | Rounds |
| --- | --- | --- | --- | --- | --- |
| ring, probabilistic truncation | semi-honest | 3.9 s | 0.39 ms | 1.05 GB | 1,073 |
| ring, exact truncation | semi-honest | 13.6 s | 1.4 ms | 2.5 GB | about 8,000 |
| sy-rep-ring (SPDZ-wise replicated) | malicious, honest majority | 139 s | 13.9 ms | 13.1 GB | 201,261 |
| ps-rep-ring | malicious, honest majority | 272 s | 27 ms | 12.0 GB | |
| mal-rep-ring | malicious, honest majority | 284 s | 28 ms | 13.6 GB | |

The fastest maliciously secure protocol costs 10 times the semi-honest one with exact
truncation, and 35 times the probabilistic version: about 260,000 encounters an hour on
one laptop. Its decisions matched the plaintext ones on 9,999 of 10,000 encounters; the one
difference sits on the threshold, within fixed-point rounding. The catch is rounds: about
20 per encounter here, because the compiled exact comparison and division are not batched
well. On a LAN (1 ms) that adds about 20 ms per encounter (roughly 100,000 an hour); on a
50 ms WAN, run one batch at a time, it would be latency-bound at a few thousand an hour,
below the literature figure, unless many batches run in parallel or the program is
restructured to batch the comparison. The paper should quote the semi-honest numbers as the
design point and the malicious ones as the current cost of stronger security, with this
caveat.

## 14. Manoeuvres as private knowledge

`research/hindcast_manoeuvre.py`, results `results/hindcast_manoeuvre_w4*.json`. An operator
knows its own manoeuvres; others see them only when the next TLE appears. A shared
error model cannot take "a manoeuvre is planned" as an input without revealing the
plan, so we measure the knowledge where it naturally lives: in the operator's own
assessment, which in the hindcast uses its real (manoeuvred) orbit.

Manoeuvres: jumps between consecutive element sets of member satellites (more than
10 robust sigma and 1 km, the world's rule), dated halfway between the sets: 50,385
on 8,855 member satellites in the four weeks. They make the public prediction of the
manoeuvring satellite about three times worse: median along-track error 19.4 km
(90th percentile 206 km), against 6.6 km (91 km) without a manoeuvre in the 72 h
before TCA.

Member-vs-member encounters where one satellite manoeuvred in the 72 h before TCA
(37 to 53 dangerous per 4-day window), four rolling windows, mean (sd), owners' models
with Foster Pc at 1e-4:

| Who assesses | Caught | Caught >= 24 h | False alarms / day |
| --- | --- | --- | --- |
| The operator that did not manoeuvre, alone (public view of the manoeuvre) | 34% (7) | 16% | 23 |
| The operator that manoeuvred, alone (knows its own orbit) | 44% (10) | 29% | 13 |
| **Both, either alert counts (two-stage scheme)** | **62% (9)** | **43%** | **34** |
| One side alone, threshold lowered to both sides' false alarms | 52 to 56% | 30 to 34% | 37 to 43 |

Without a manoeuvre (100 to 115 dangerous per window): one side 42% / 61%, both 76%,
one side at equal false alarms 62 to 70%. The baseline and the public-table Foster
show the same pattern.

What this says:

1. **The operator that manoeuvred sees its encounters better**: 10 points more
   recall, twice the 24-hour warnings and half the false alarms of the operator that
   only sees the manoeuvre in the public catalogue. The direction holds on average
   but varies between windows (sd 7 to 10 points on 37 to 53 encounters).
2. **The robust gain is combining both operators' views**, with or without a
   manoeuvre: +6 to +10 points over the better single side at the same false-alarm
   rate, and more early warnings. That is the two-stage scheme's alert union: each
   operator contributes what only it knows (its own orbit, including its manoeuvres),
   without sharing it.
3. So manoeuvre knowledge is best used through the federation (each side alerts on
   its own satellites, the secure step uses the owners' precise, manoeuvre-including
   orbits), not as an input to shared error models, which would leak plans.

Caveats: the manoeuvre times are inferred from TLE jumps (some jumps are bad fits,
and small station-keeping burns are missed); truth is TLE-derived (section 11); small
counts per window.

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
PYTHONPATH=src:research python research/leakage.py research/cache/<cache>.npz --targets 25 --seed 1 --window-h 24
python research/fig_leakage.py
PYTHONPATH=src python research/mpc/make_inputs.py 50000 research/mpc/run50000   # then, in MP-SPDZ (WSL): ./compile.py -R 128 secure_pc 50000 show; Scripts/ring.sh secure_pc-50000-show
PYTHONPATH=src:research python research/adversarial.py research/cache/<cache>.npz --k 8
python research/fig_sweep.py research/results/<cache>__sweep.json
```

The cache file (125 MB) is not committed; it regenerates in about 7 minutes.

## Next

1. Paper draft written (Claude Doc); sections 9 and 10 folded in.
2. Rolling evaluation of section 10 over several test windows; operator ephemerides (for example CelesTrak SupGP) as precise labels.
