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
PYTHONPATH=src python3 -m pytest tests/ -q        # 107 tests, should all pass
PYTHONPATH=src python3 scripts/run_demo.py         # full pipeline walkthrough
```

The demo script prints a run through every layer below and also writes
`docs/sample_run.md`.

## Layer-by-layer status

| Layer (project definition Sec. 5) | Module | Status |
|---|---|---|
| Simulation | `src/satcollision/fleets.py` | **Built.** Real SGP4 propagation (via the `sgp4` library) of 3 synthetic operator fleets whose shell parameters (altitude/inclination/plane count) mirror publicly documented real constellations, plus an ambient debris population. A `load_tle_file()` loader takes real CelesTrak TLEs with zero other code changes, and the backtest scripts already run on real Starlink/OneWeb/Kuiper catalogs (see "Why the walkthrough runs on synthetic fleets" below). |
| Per-operator digital twin | `src/satcollision/twin.py` | **Built.** Pairwise conjunction screening across a propagated time grid, and a simplified circular (equal-variance) 2D probability-of-collision estimate — see "Simplifications" below for what a production version would add. |
| Signal abstraction | `src/satcollision/signal.py` | **Built.** Converts a private `Conjunction` into a threshold-gated `AbstractedSignal` (flag + coarse geometry class + coarse severity tier) with a runtime structural check (`assert_no_raw_state`) that fails loudly if any state-vector-shaped field is ever added. |
| Federation (aggregation + DP + Byzantine-robust) | `src/satcollision/federation.py` | **Built.** Mean / trimmed-mean / Krum aggregation over per-operator count vectors, a Laplace-mechanism local-DP step, and 4 attack-model simulators (spoof-high, suppress, sign-flip, free-ride) matching the attack models named in the project definition's tech stack. |
| Maneuver deconfliction | `src/satcollision/deconfliction.py` | **Built.** A deterministic, zero-extra-data-exchange protocol that assigns complementary escape directions, plus a Monte Carlo comparison against uncoordinated independent maneuvering. |
| Cryptographic identity, signing & tamper-evident log | `src/satcollision/identity.py` | **Built.** Real Ed25519 keypairs per operator (`cryptography` library, not a placeholder), genuine sign/verify of every `AbstractedSignal` and `OperatorReport` against a trusted-key registry (never against a key bundled with the message itself — that would let an attacker self-verify), and a hash-chained append-only log (`SignedLog`) that detects tampering, reordering, or deletion of past entries. `scripts/run_demo.py` step `[3b]` exercises all three failure modes live (tamper, impersonation, post-hoc log edit) and shows each one caught. See "Trust model" below for exactly what this layer does and does not guarantee. |
| Secure aggregation (MPC) | `src/satcollision/secure_aggregation.py` | **Built.** Pairwise-masked additive secret sharing (Bonawitz et al., CCS 2017 — the technique behind Google's production federated learning) over a 127-bit prime field, with real X25519 Diffie-Hellman key agreement (not a shared-out-of-band secret) deriving each pair's mask. The aggregator combines masked shares that are individually indistinguishable from random field elements and still recovers the exact plaintext sum/mean — `scripts/run_demo.py` step `[4b]` runs the same 6 reports from `[4]` through it side-by-side with `mean_aggregate`, printing both the masked shares an aggregator would actually see and the reconstructed mean, which matches exactly. See "Trust model" for why this does not compose for free with the Byzantine-robust aggregation in `federation.py`. |
| Kalman-filter state estimation | `src/satcollision/kalman.py` | **Built.** A real linear Kalman filter (constant-velocity process model, position-only measurement model, standard predict/update equations) tracking each object against its own SGP4 ground truth with simulated periodic noisy measurements, producing a combined sigma that actually grows between tracking updates and shrinks right after one — a genuine replacement for `twin.py`'s fixed `DEFAULT_COMBINED_SIGMA_KM=0.5km` constant. `refine_pc_for_conjunction` re-evaluates an already-screened conjunction's Pc with this time-varying sigma; `scripts/run_demo.py` step `[2b]` runs it on the walkthrough's flagged encounter and prints both Pc values side by side (they differ by two orders of magnitude on that encounter — see the module docstring for why this is applied per-encounter rather than inside the broad-phase screening pass). |
| Historical backtesting | `scripts/run_backtest.py`, `scripts/run_cross_operator_backtest.py` | **Built.** Runs the full pipeline against real, independently-downloaded CelesTrak TLE catalogs via `fleets.load_tle_file()` — see "Historical backtesting: real results so far" below for actual numbers from a real Starlink run. |
| Real federated learning (FedAvg + DP-SGD) | `src/satcollision/federated_learning.py` | **Built.** A real PyTorch risk classifier trained across operators via FedAvg (McMahan et al.) with DP-SGD (Abadi et al., via `opacus`) in the local training step — genuine multi-round weight averaging and a formally computed per-operator (epsilon, delta), not count-vector aggregation dressed up as "federated learning." `scripts/run_demo.py` step `[8]` runs centralized / local-only / federated-no-DP / federated-DP side by side on the same held-out test set. See "Trust model" for what the reported epsilon does and doesn't mean, and the module docstring for why the label is a synthetic composite risk index rather than a direct call to `twin.compute_pc`. |
| Real distributed system | `src/satcollision/wire.py`, `hub.py`, `operator_node.py`, `distributed_demo.py` | **Built.** Every earlier layer above runs inside one Python process calling plain functions on shared objects — this is the one place that's actually architecturally distributed. Each operator (Alpha, Beta, Gamma) and a federation hub run as genuinely separate OS processes (`subprocess.Popen`, real `python3` interpreters, not threads or asyncio tasks sharing a process), exchanging real bytes over real TCP sockets on `127.0.0.1` with a length-prefixed wire format (`wire.py`). The hub (`hub.py`) is a deliberately "dumb pipe" — it relays every message verbatim and never checks a signature; each operator (`operator_node.py`) builds a real SGP4-propagated, Pc-screened conjunction, signs it with its own Ed25519 key loaded from a file, and verifies everything it receives against a pre-distributed public-key registry, exactly as `identity.py`'s in-process tests do — except this time the guarantee has to survive a real process and network boundary. A fifth process, an attacker impersonating "Alpha" with its own freshly generated keypair, proves the forgery is still caught: `scripts/run_distributed_demo.py` runs the whole thing and reports every legitimate operator rejecting the forged message (the real Alpha included, when it receives a message claiming to be itself signed by someone else's key). See "Trust model" below for why the hub replays full history to every client rather than only live-broadcasting. |
| Incentive/reputation mechanism | `src/satcollision/reputation.py` | **Built.** Per-round reputation built only from behaviour the federation can already observe: *corroboration* (an encounter id names both NORAD objects, so every operator can answer one boolean — "am I a party to this?" — locally, which makes both withholding and fabrication detectable without any catalogue crossing a boundary), *consensus agreement* (distance from the median report, penalised only in excess of the round's own median deviation), and *participation* (contributing something another operator vouches for, so neither an empty report nor fabricated volume earns credit). Scores update as an asymmetric EWMA — rising at 0.10, falling at 0.35 — and feed back in two ways: `reputation_weighted_aggregate` gives influence over the federated result in proportion to score (quarantining anyone below 0.25), and `alert_tier`/`tailor_alert` make the *detail* an operator receives back proportional to its score, which is what makes it an incentive rather than a scoreboard. `scripts/run_demo.py` step `[9]` runs 20 reporting periods with honest / free-rider / spoofer / on-off strategies. See "Reputation: what it buys and what it must never do" below. |
| Plain-language incident summaries | `src/satcollision/summaries.py` | **Built.** Deterministic template rendering (no language model anywhere — see "Why the summaries are templates" below) of every layer's output into sentences a duty officer, regulator or non-technical reader can act on: `Pc=8.00e-04` becomes "about a 1 in 1,200 chance", `tca_offset_s=1800.0` becomes "in 30 minutes", and every quoted probability carries the caveat that it is a model estimate rather than a measurement. Cross-operator summaries are rendered from the *tailored alert dict* produced by `reputation.tailor_alert`, never from the underlying signal, so a lower-tier reader's summary structurally cannot mention what that reader was not sent; `assert_no_undisclosed_terms` checks the rendered prose against the alert it came from, the prose-level counterpart of `signal.assert_no_raw_state`. `build_incident_report` assembles the four-section narrative (what happened / what was shared / what happens now / how to check), with the detecting operator's private view passed in only for its own copy. `scripts/run_demo.py` step `[10]` prints all of it. |
| Presentation | `conjunction_watch.html` (project root) | **Built separately** as an illustrative, scripted interactive demo for panel presentations — see that file and Section 12 of the project definition. This codebase's `scripts/run_demo.py` is the *technical* presentation layer (plain-language console output); wiring its real numbers into a live visual dashboard is listed as future work below. |
| 3-scenario evaluation harness | `src/satcollision/evaluate.py` | **Built.** Compares No-Cooperation / Full-Sharing / Federated on detection lead time and raw-data exposure on one engineered conjunction, plus a separate robustness sweep of aggregation error vs. adversary fraction. |

## Why the walkthrough runs on synthetic fleets (and where the real data is)

Both paths are live. The backtest scripts run on **real** CelesTrak
catalogs downloaded into `data/` (Starlink, OneWeb and Kuiper — see
"Historical backtesting: real results so far" below for the numbers those
runs produced), while `scripts/run_demo.py` deliberately keeps the
synthetic fleets, because the walkthrough needs one *engineered*
threshold-crossing encounter to carry through every layer, and real
catalogs mostly do not hand you one on demand inside a 40-minute window.

The synthetic fleets are constructed honestly rather than fabricated. Real
TLE sets cannot be transcribed by hand or by a model — a single wrong digit
in a checksum or mean-motion field silently produces a wrong orbit — so
`fleets.py` constructs orbits directly from Keplerian
elements via `sgp4`'s `Satrec.sgp4init`, using each shell's *publicly
documented* approximate altitude/inclination (Starlink ~550km/53°, OneWeb
~1200km/87.4°, Iridium NEXT ~780km/86.4°). This is legitimate synthetic
scenario generation (the same technique a mission-design tool uses), not
scraped data presented as real.

`load_tle_file()` is the real-data path the backtests already use: download
a TLE set from
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
  integral is listed as a natural "if time permits" refinement. The other
  half of this simplification — where the constant `sigma` itself comes
  from — is no longer purely assumed: `kalman.py` derives a real
  time-varying sigma from an actual tracking history (see the layer table
  above), callable per-encounter via `refine_pc_for_conjunction`. It is not
  wired into the default `screen_conjunctions` broad-phase pass, for the
  performance reason explained in `kalman.py`'s module docstring.
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
  identity.py          Ed25519 signing + hash-chained tamper-evident log
  secure_aggregation.py  MPC secure aggregation (pairwise-masked secret sharing)
  kalman.py            Kalman-filter per-object state estimation (evolving sigma)
  federated_learning.py  real FedAvg + DP-SGD training (PyTorch + Opacus)
  reputation.py        incentive/reputation layer (scores, weighted aggregation, alert tiers)
  summaries.py         plain-language incident summaries (tier-aware, template-based)
  evaluate.py          3-scenario comparison + robustness sweep
  wire.py              length-prefixed TCP message framing for the distributed demo
  hub.py               federation hub process (dumb relay, real asyncio TCP server)
  operator_node.py       one operator's standalone process (legitimate or attacker mode)
  distributed_demo.py    orchestrates hub + 3 operators + 1 attacker as real OS processes
tests/                 pytest suite (107 tests) covering the above
scripts/run_demo.py    end-to-end walkthrough, writes docs/sample_run.md
scripts/make_figures.py  renders the evaluation figures into docs/figures/ (PNG + PDF + CSV)
scripts/run_distributed_demo.py  thin CLI wrapper around distributed_demo.run_demo()
requirements.txt        sgp4, numpy, scipy, matplotlib, pytest, cryptography, torch, opacus
```

## Trust model: what signing does and does not buy you

Worth being precise about this, since it's an easy thing to overclaim on a
panel. `identity.py` gives every operator a real keypair and makes every
shared signal/report **authenticated** (you know who really sent it) and
**tamper-evident** (you know if it was altered after signing, or if the
recorded history was edited, reordered, or had entries deleted). It does
**not** make anyone's *reported content* honest — `scripts/run_demo.py`
deliberately shows the adversarial operator's spoofed counts still passing
signature verification, because a valid signature only proves "the real
Adversary sent exactly this," not "what the real Adversary sent is true."
Catching a dishonest-but-authenticated report is what the Byzantine-robust
aggregation in `federation.py` (trimmed-mean, Krum) is for — the two layers
are complementary, not redundant: signing stops an *outsider* from forging
or altering messages; robust aggregation stops an *authenticated insider*
from skewing the federation's conclusions.

`secure_aggregation.py` closes a third, different gap: even with signing
and robust aggregation in place, the party doing the combining still sees
every operator's plaintext report before folding it into the result.
Pairwise-masked secret sharing removes that — the aggregator never
possesses a plaintext report, only a share that is individually
indistinguishable from noise (`scripts/run_demo.py` step `[4b]` prints the
actual masked values an aggregator sees). The honest limitation worth
stating plainly: secure aggregation and Byzantine-robust aggregation don't
combine automatically. `trimmed_mean_aggregate`/`krum_aggregate` need to
inspect individual reports to identify and exclude outliers; once those
reports are masked, the aggregator can no longer do that, so a masked
adversarial report still contributes to the secure result (see the
adversary's spoofed counts still landing in `secure_mean_aggregate()`'s
output in step `[4b]`'s output, identical to plain `mean_aggregate`). A
production system wanting both properties at once needs additional
machinery this project doesn't build — verifiable secret sharing or
zero-knowledge range proofs on the committed inputs, so the aggregator can
reject an out-of-range masked share without ever unmasking it. Choosing
which property a given federation round needs (hide inputs from the
aggregator, or protect the aggregate from a lying insider) is itself a
policy decision, not something this codebase resolves for you.

`federated_learning.py`'s DP-SGD adds a fourth, again distinct, guarantee
on top of all of the above: even the *model weight updates* FedAvg shares
carry information about the local data they were trained on (this is what
model-inversion and membership-inference attacks exploit — a threat
signing, robust aggregation, and secure aggregation none of the above
address, since all three assume the thing being protected is a hand-built
report vector, not a gradient computed from raw examples). The
`epsilon` value `scripts/run_demo.py` step `[8]` prints per operator is a
real, formally computed differential-privacy budget (via Opacus's
accountant) on that operator's *contribution to the shared model this
round* — smaller means a stronger guarantee, and it costs measurable
accuracy (`federated_dp` scores below `federated_no_dp` in that step's
output). What it does not do: bound anything about the plaintext local
dataset's contents directly (that would need the earlier layers), or
survive being reused across unboundedly many rounds without composing
into a larger, weaker epsilon — a real production deployment would need
to budget a total epsilon across the whole federation's lifetime, which
this project reports per-round but does not itself cap.

`hub.py`/`operator_node.py`/`distributed_demo.py` don't add a new trust
guarantee — they re-run `identity.py`'s existing one (authenticity,
tamper-evidence) across an actual process and network boundary instead of
inside one shared script, which is the more demanding environment any of
these guarantees would actually have to hold up in. One implementation
detail is worth stating plainly since it's easy to get wrong: the hub
replays its *full* message history to every newly connecting client,
not just rendezvous pings. An earlier version replayed only "ready"
messages on the theory that real signals "need to be seen live" — that
was broken, not just less realistic: an attacker that connects and sends
before every legitimate operator has joined would have its forged message
broadcast only to whoever was already online, so a later-joining peer
would never receive it to reject at all (not a timing race a longer wait
could fix — the message was never delivered). Replaying full history
makes delivery to every connected client happen exactly once, regardless
of join order, which is closer to how a durable message bus behaves than
a live-only broadcast, and it's what makes the demo's central claim (every
legitimate operator rejects the forgery) a deterministic guarantee rather
than something that merely usually happens.

## Reputation: what it buys and what it must never do

Every guarantee above protects the federation from a *message*. None of
them give an operator a reason to keep participating honestly next period:
Byzantine-robust aggregation quietly discards a liar's report and then
greets it identically next round, and a free-rider that contributes nothing
receives exactly the same alerts as an operator paying real tracking costs.
`reputation.py` is the layer that makes honest participation individually
rational rather than merely collectively desirable, and it is deliberately
built only on evidence the federation already has — no new data crosses an
operator boundary to compute a score. The corroboration signal is the one
worth explaining on a panel: an encounter id is `ENC-<norad_a>-<norad_b>-<tca>`
and catalogue numbers are public, so each operator can answer *locally*
whether one of those two objects is its own and return a single boolean.
That one bit is enough to catch both withholding (you were a party to an
encounter someone else reported, and said nothing) and fabrication (you
reported an encounter no one — not even you — owns an object in), while
revealing nothing an honest report would not already have revealed.

Two design choices carry most of the weight. Scores fall about 3.5x faster
than they rise, which is what kills the "on-off" attack of banking honest
rounds to spend on one big lie: step `[9]`'s `Zeta` behaves perfectly for
12 rounds, peaks around 0.84, and is back below its starting score within
about 6 rounds of defecting. And new identities start at 0.50 —
deliberately below where an honest operator settles (~0.85-0.90) — so
abandoning a damaged identity for a fresh one is a downgrade rather than an
escape.

The feedback loop is also where reputation-weighted aggregation earns its
place next to trimmed mean and Krum, which are *memoryless*: they re-decide
who this round's outlier is from scratch, so an adversary pays nothing for
its plausible rounds. Weighting by a score that accumulates across rounds
cuts the mean aggregation error over the last 5 rounds of step `[9]` from
0.417 (trimmed mean) to 0.221.

The hard limit, stated plainly because it is a design decision and not an
oversight: **a `critical`-tier alert is always delivered in full, to
everyone, whatever their reputation.** Reputation governs influence over
the aggregate and the level of operational detail an operator receives —
it must never become a mechanism for withholding a collision warning,
because the debris from a collision harms third parties who had no part in
anyone's misbehaviour. `tailor_alert`'s safety override is that floor in
code.

What this layer does *not* solve. Sybil attacks: reputation is only as
strong as the cost of a fresh identity, and here that cost comes entirely
from `identity.py`'s registry being permissioned (a real federation would
anchor admission in launch licensing, which is already a heavyweight
regulatory identity). Collusion: a coalition that mutually attests each
other's phantom encounters can raise its own corroboration, bounded by the
fact that encounter ids name catalogued objects, so a coalition can only
fabricate encounters between objects it genuinely owns. And reputation is
a *behavioural* signal, not a correctness proof — a well-meaning operator
with a mis-calibrated Pc pipeline looks, from the outside, somewhat like a
liar, which is exactly why the consequences are graduated (weight, then
detail) rather than binary expulsion.

## Why the summaries are templates, not a language model

The obvious way to build Enhancement #5 in 2026 is to hand the encounter
data to an LLM and ask for a paragraph. `summaries.py` deliberately does
not, and the reasoning is worth having ready for a panel, because "we
didn't have an API key" is not the answer.

A summary attached to a collision warning is a safety artifact, and it has
to be three things a generative model cannot guarantee. **Reproducible**:
the same encounter must render the same words every time, so two operators
comparing "their" copies of an incident are comparing the same text, and so
a summary quoted in an investigation months later can be regenerated
exactly. **Auditable**: every figure in the output traces to the field it
came from, and `describe_odds`/`describe_duration` are unit-tested
conversions rather than a model's paraphrase. **Incapable of inventing a
number**: a template can only render values it was passed, while a model
asked for fluent prose will happily supply a plausible miss distance that
was never computed. There is also a privacy argument specific to this
project: sending encounter data to a third-party inference API would
re-open exactly the data-exposure hole the federated design exists to
close.

The cost is real and worth stating: the output has the cadence of a
template, and it cannot answer a follow-up question. The mitigation is
that the templates are structured per audience (`operator` keeps the
technical figures, `executive` drops catalogue numbers and scientific
notation entirely) rather than trying to be one voice for every reader.

The other half of the design is the disclosure boundary. The summary layer
is the most likely place for a leak to reappear after `signal.py`'s
careful work, because it is the one layer whose whole job is turning
structured fields into free text — and free text is where a well-meaning
edit ("just mention the timing, it's harmless") quietly undoes a boundary.
So cross-operator summaries render from the tailored alert dict rather
than the signal, a plan is withheld from any reader whose tier withheld
the counterparty, and `assert_no_undisclosed_terms` re-checks the rendered
string against the alert that produced it. `tests/test_summaries.py`
includes a test that deliberately appends a timing sentence to a degraded
reader's summary and asserts the guard rejects it.

## Historical backtesting: real results so far

`scripts/run_backtest.py` (single real operator) and
`scripts/run_cross_operator_backtest.py` (two real operators — the version
that actually exercises signal abstraction, federation, and deconfliction
on genuine data) are both working, and have been run against three real
catalogs:

| Run | Window | Result |
|---|---|---|
| Starlink against itself (10,713 objects) | 24h | 173,356 raw conjunctions inside the 25 km screening volume, 19 of them above the operational Pc threshold (1e-4) |
| Starlink × OneWeb | 24h | Closest approach 11.35 km, Pc ~1e-115 — an honest negative: the two shells are genuinely separated in altitude |
| Starlink × Kuiper | 24h | Closest approach 3.1 km, Pc ~3.4e-12 |
| Starlink × Kuiper | 168h (1 week) | Closest approach 1.57 km, Pc ~5.7e-6 — still below threshold, but a real distribution of close approaches between two genuine megaconstellations |

Nothing above crossed the operational threshold across operators, which is
the correct and expected result for today's catalogs: genuine
cross-operator conjunctions above 1e-4 are rare events, and reporting one
that wasn't there would be the failure mode to worry about. The single-
operator Starlink run is what demonstrates the pipeline does flag real
threshold crossings when they exist. TLE files change daily and `data/` is
gitignored, so re-running these scripts will produce different numbers;
`docs/sample_run.md` is the reproducible synthetic-data walkthrough.

Note: the original brute-force screening implementation did not scale to a
real multi-thousand-object catalog (O(n^2) pairwise comparison per
timestep). `twin.py` now uses vectorized SGP4 propagation
(`sgp4.api.SatrecArray`) and a KD-tree broad-phase screening pass
(`scipy.spatial.cKDTree.query_pairs`) instead — a 24h/30s-step backtest
against ~8,000-10,000 objects now runs in well under a minute.

## Figures for the report

```bash
PYTHONPATH=src python3 scripts/make_figures.py                 # figures 1-5
PYTHONPATH=src python3 scripts/make_figures.py --skip-fl       # skip the slow PyTorch one
PYTHONPATH=src python3 scripts/make_figures.py \
    --backtest data/starlink.tle Starlink data/kuiper.tle Kuiper --hours 24   # + the real-data figure
```

Each figure is written to `docs/figures/` three ways: a 300-dpi **PNG** for
slides, a vector **PDF** for the report, and a **CSV** of the exact series
behind it, so any number quoted in the text can be checked against the plot
and the data survives even if the figure is later redrawn.

| Figure | Shows |
|---|---|
| `fig_reputation_trajectories` | Reputation over 20 periods for honest / free-rider / spoofer / on-off operators, with the tier thresholds marked. The on-off line is the one to talk about: honest for 12 rounds, then below its own starting score within about 6 rounds of defecting. |
| `fig_robustness_sweep` | Aggregation error vs. the share of dishonest operators, for plain mean / trimmed mean / Krum. |
| `fig_weighted_aggregation_error` | The same error across rounds for plain mean, trimmed mean and reputation-weighted aggregation — the case for carrying reputation across rounds rather than re-deciding each round. |
| `fig_three_scenarios` | No Cooperation vs. Full Data Sharing vs. Federated, on detection lead time and raw-data exposure. |
| `fig_federated_learning` | Centralized / local-only / FedAvg / FedAvg+DP-SGD accuracy and F1 on the shared held-out set. |
| `fig_backtest_<a>_<b>` | The real cross-operator miss-distance distribution from a genuine TLE catalog run. |

Three decisions in that script are worth being able to defend, because
they are the kind of thing an examiner notices:

- **No dual-axis charts.** Detection lead time (minutes) and data exposure
  (per cent) are different scales, so `fig_three_scenarios` gives them one
  panel each instead of two y-axes on one plot. A dual axis lets the author
  choose the visual conclusion by choosing the scales.
- **The federated-learning figure is a dot plot, not bars.** Every score
  sits between 0.96 and 0.99. Bars from zero would be four
  indistinguishable full-height columns; bars from 0.95 would be a
  truncated bar axis, which overstates the differences because a bar
  encodes by *length*. A dot encodes by position, so a zoomed axis is
  honest for it.
- **The colours were validated, not chosen by eye.** The four series hues
  clear the standard colour-vision-deficiency separation thresholds, and
  every series also carries its own line style or a direct value label —
  so nothing in the report depends on colour alone once it is printed in
  greyscale.

The backtest figure caches its screening result as JSON next to the figure.
That is deliberate: TLE files are republished daily, so a figure
regenerated next week would otherwise quietly disagree with the numbers
quoted in the report text. The cache pins both the distances and the epoch
they came from; delete it to force a fresh run.

## Next steps (see the Semester Implementation Plan, Sec. 10 of the project definition)

1. Wire the Kalman-derived sigma from `kalman.py` into `twin.screen_conjunctions`
   itself (rather than only the per-encounter `refine_pc_for_conjunction`
   path), likely by maintaining one running filter per tracked object
   across the whole propagation window instead of re-deriving one on
   demand.
2. Emit incident summaries from `operator_node.py` in the distributed demo,
   so each process says in words what it received, accepted and rejected —
   `summaries.py` is currently exercised only from the single-process
   walkthrough.
3. Extend the reputation layer's evidence set with post-maneuver
   compliance (did an operator actually fly the deconfliction role it
   agreed to), which needs an attestation path that doesn't exist yet, and
   stress it against a colluding coalition rather than independent
   adversaries.
4. Wire `evaluate.py`'s real computed numbers into a live version of
   `conjunction_watch.html` (currently a standalone, hand-scripted demo).
5. Full non-circular Pc integral with per-operator covariance, if time
   permits.
6. Verifiable secret sharing / range proofs on top of `secure_aggregation.py`
   so secure aggregation and Byzantine-robust aggregation can run in the
   same round (see "Trust model" above for why they currently can't).
7. A total privacy-budget accountant across the whole federation's
   lifetime in `federated_learning.py`, rather than the current per-round
   epsilon — real deployments cap cumulative epsilon across many rounds,
   not just report each round's individually.
8. Run the distributed demo's operators and hub across genuinely separate
   machines on a real LAN (currently all on `127.0.0.1`) — the wire
   protocol and signature verification don't change, only `--host`/`--port`
   would need to point somewhere other than loopback.
