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
PYTHONPATH=src python3 -m pytest tests/ -q        # 269 tests, should all pass
PYTHONPATH=src python3 scripts/run_demo.py         # full pipeline walkthrough
python3 scripts/run_worlds.py                      # always-running worlds + live dashboard
docker compose up --build                          # full-stack app: operator / federation / observer views
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
| Historical backtesting | `scripts/run_backtest.py`, `scripts/run_cross_operator_backtest.py` | **Built.** Runs the full pipeline against real, independently-downloaded CelesTrak TLE catalogs via `fleets.load_tle_file()` — see "Historical backtesting: real results so far" below for actual numbers from a real Starlink run. Every real-data run now starts at the catalogue's own epoch (see "Start the clock at the catalogue epoch" below). |
| Documented-event replay | `src/satcollision/historical_replay.py`, `scripts/run_historical_replay.py` | **Built, waiting on data.** Replays a documented past close approach (the two 2021 China Space Station / Starlink events reported to the UN in A/AC.105/1262 are registered) from several as-of times, using only element sets published before each one, and reports how early the pipeline would have flagged it. Needs the historical element sets from Space-Track's `gp_history` (free account); the script's docstring has the exact query. Verified on the engineered encounter in `tests/test_historical_replay.py`. |
| Refined Pc model | `src/satcollision/pc.py`, `scripts/run_pc_model_comparison.py` | **Built and run.** Foster/Estes 2D encounter-plane Pc with a full 3x3 covariance per object, validated against the closed-form non-central chi-squared answer and Monte Carlo. Selectable everywhere through a `pc_model` argument; the circular model remains the default. See "Refined Pc: what changes" below for the four-way comparison on the real catalogue. |
| Step-through simulation environment | `src/satcollision/environment.py`, `notebooks/satellite_environment.ipynb`, `scripts/run_environment_demo.py` | **Built.** A Jupyter-friendly clock you step, seek and reset, with snapshots, NumPy arrays and a conjunction table at each instant. Starts at the fleet epoch (synthetic) or the catalogue epoch (real TLEs), so the engineered encounter appears 30 minutes in. |
| Always-running worlds | `src/satcollision/world.py`, `scripts/run_worlds.py`, `scripts/run_world.py`, `worlds/*.json`, `docs/live_worlds.html` | **Built.** Three persistent worlds (sandbox, adversarial, real LEO), each its own supervised process running the full pipeline on every tick, with per-tick checkpoints, a signed ledger that is re-verified on resume, deterministic restarts and a live dashboard with a world switcher. See "Always-running worlds" below. |
| Full-stack app (per-party views) | `src/satcollision/views.py`, `app/backend/`, `app/frontend/`, `Dockerfile` | **Built.** Logins for each operator, the federation server and an observer; every API response is filtered server-side to what that party may see (the server's view contains no positions at all), live over WebSocket, with a side-by-side comparison of one encounter from every party. Each operator's twin and the federation run as separate OS processes (the federation never loads orbit code; each private key exists only in its operator's process). Runs with `docker compose up`. See "Full-stack app" below. |
| Real federated learning (FedAvg + DP-SGD) | `src/satcollision/federated_learning.py` | **Built.** A real PyTorch risk classifier trained across operators via FedAvg (McMahan et al.) with DP-SGD (Abadi et al., via `opacus`) in the local training step — genuine multi-round weight averaging and a formally computed per-operator (epsilon, delta), not count-vector aggregation dressed up as "federated learning." `scripts/run_demo.py` step `[8]` runs centralized / local-only / federated-no-DP / federated-DP side by side on the same held-out test set. See "Trust model" for what the reported epsilon does and doesn't mean, and the module docstring for why the label is a synthetic composite risk index rather than a direct call to `twin.compute_pc`. |
| Real distributed system | `src/satcollision/wire.py`, `hub.py`, `operator_node.py`, `distributed_demo.py` | **Built.** Every earlier layer above runs inside one Python process calling plain functions on shared objects — this is the one place that's actually architecturally distributed. Each operator (Alpha, Beta, Gamma) and a federation hub run as genuinely separate OS processes (`subprocess.Popen`, real `python3` interpreters, not threads or asyncio tasks sharing a process), exchanging real bytes over real TCP sockets on `127.0.0.1` with a length-prefixed wire format (`wire.py`). The hub (`hub.py`) is a deliberately "dumb pipe" — it relays every message verbatim and never checks a signature; each operator (`operator_node.py`) builds a real SGP4-propagated, Pc-screened conjunction, signs it with its own Ed25519 key loaded from a file, and verifies everything it receives against a pre-distributed public-key registry, exactly as `identity.py`'s in-process tests do — except this time the guarantee has to survive a real process and network boundary. A fifth process, an attacker impersonating "Alpha" with its own freshly generated keypair, proves the forgery is still caught: `scripts/run_distributed_demo.py` runs the whole thing and reports every legitimate operator rejecting the forged message (the real Alpha included, when it receives a message claiming to be itself signed by someone else's key). See "Trust model" below for why the hub replays full history to every client rather than only live-broadcasting. |
| Incentive/reputation mechanism | `src/satcollision/reputation.py` | **Built.** Per-round reputation built only from behaviour the federation can already observe: *corroboration* (an encounter id names both NORAD objects, so every operator can answer one boolean — "am I a party to this?" — locally, which makes both withholding and fabrication detectable without any catalogue crossing a boundary), *consensus agreement* (distance from the median report, penalised only in excess of the round's own median deviation), and *participation* (contributing something another operator vouches for, so neither an empty report nor fabricated volume earns credit). Scores update as an asymmetric EWMA — rising at 0.10, falling at 0.35 — and feed back in two ways: `reputation_weighted_aggregate` gives influence over the federated result in proportion to score (quarantining anyone below 0.25), and `alert_tier`/`tailor_alert` make the *detail* an operator receives back proportional to its score, which is what makes it an incentive rather than a scoreboard. `scripts/run_demo.py` step `[9]` runs 20 reporting periods with honest / free-rider / spoofer / on-off strategies. See "Reputation: what it buys and what it must never do" below. |
| Plain-language incident summaries | `src/satcollision/summaries.py` | **Built.** Deterministic template rendering (no language model anywhere — see "Why the summaries are templates" below) of every layer's output into sentences a duty officer, regulator or non-technical reader can act on: `Pc=8.00e-04` becomes "about a 1 in 1,200 chance", `tca_offset_s=1800.0` becomes "in 30 minutes", and every quoted probability carries the caveat that it is a model estimate rather than a measurement. Cross-operator summaries are rendered from the *tailored alert dict* produced by `reputation.tailor_alert`, never from the underlying signal, so a lower-tier reader's summary structurally cannot mention what that reader was not sent; `assert_no_undisclosed_terms` checks the rendered prose against the alert it came from, the prose-level counterpart of `signal.assert_no_raw_state`. `build_incident_report` assembles the four-section narrative (what happened / what was shared / what happens now / how to check), with the detecting operator's private view passed in only for its own copy. `scripts/run_demo.py` step `[10]` prints all of it. |
| Presentation | `conjunction_watch.html` (project root) | **Built separately** as an illustrative, scripted interactive demo for panel presentations — see that file and Section 12 of the project definition. This codebase's `scripts/run_demo.py` is the *technical* presentation layer (plain-language console output); wiring its real numbers into a live visual dashboard is listed as future work below. |
| 3-scenario evaluation harness | `src/satcollision/evaluate.py`, `src/satcollision/real_evaluation.py` | **Built.** Compares No-Cooperation / Full-Sharing / Federated on detection lead time and raw-data exposure on one engineered conjunction, plus a separate robustness sweep of aggregation error vs. adversary fraction. |

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
  that matter (miss distance, combined uncertainty, object size). The full
  integral is now implemented in `pc.py` and compared against this one in
  "Refined Pc: what changes" below; the circular model stays the default so
  earlier results remain reproducible. The other
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
  real_evaluation.py   the 3-scenario comparison run across a real catalogue
  evaluate.py          3-scenario comparison + robustness sweep
  pc.py                refined Foster/Estes Pc model + selectable Pc models
  environment.py       step-through simulation environment (Jupyter-friendly)
  historical_replay.py replay a documented past close approach from as-of times
  world.py             always-running worlds: look-ahead screening, checkpoint/resume, persistent ledger
  views.py             role-based views: what each operator, the federation server and the observer may see
  processes.py         runs a world as separate processes: sky here, one twin per operator, the federation
  twin_node.py         one operator's digital twin as its own process (own key, own screening, own decisions)
  federation_node.py   the federation server as its own process (HTTP), no orbit code loaded
  federation_service.py  the federation's ledger, pinned keys, attestations, reputation and alert tailoring
  node_http.py         the small JSON-over-HTTP layer those processes use (stdlib only)
  constants.py         shared constants with no heavy imports
app/backend/           FastAPI backend (logins, role views, WebSocket live updates, serves the frontend)
app/frontend/          React + TypeScript + three.js frontend (Vite)
Dockerfile, docker-compose.yml  one-container build of the app
  wire.py              length-prefixed TCP message framing for the distributed demo
  hub.py               federation hub process (dumb relay, real asyncio TCP server)
  operator_node.py       one operator's standalone process (legitimate or attacker mode)
  distributed_demo.py    orchestrates hub + 3 operators + 1 attacker as real OS processes
tests/                 pytest suite (269 tests) covering the above
scripts/run_demo.py    end-to-end walkthrough, writes docs/sample_run.md
scripts/make_figures.py  renders the evaluation figures into docs/figures/ (PNG + PDF + CSV)
scripts/run_real_evaluation.py  3-scenario comparison on real data, writes docs/real_evaluation.md
scripts/make_dashboard.py  builds docs/conjunction_watch.html from the measured run data
scripts/run_distributed_demo.py  thin CLI wrapper around distributed_demo.run_demo()
scripts/run_pc_model_comparison.py  circular vs Foster, four columns, writes docs/pc_model_comparison.md
scripts/run_historical_replay.py  replays a documented event, writes docs/historical_replay_<event>.md
scripts/run_environment_demo.py  console walkthrough of environment.py
scripts/run_worlds.py  starts every world in worlds/, restarts crashed ones, serves docs/live_worlds.html
scripts/run_world.py   runs one world (normally started by run_worlds.py)
worlds/*.json          the world definitions (sandbox, adversarial, real_leo, real_leo_history)
scripts/run_midsem_demo.py  the mid-semester walkthrough (real catalogue, ~20 s)
notebooks/satellite_environment.ipynb  environment.py in a notebook
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
| Starlink × Kuiper | 168h (1 week) | Closest approach 1.57 km, Pc ~5.7e-6 — below threshold, but a real distribution of close approaches between two genuine megaconstellations |
| ~~Starlink × Kuiper ("20 Sep catalogue")~~ | 168h (1 week) | ~~Closest approach 828 m, Pc 2.03e-4~~ **Withdrawn.** This run propagated 10 Sep element sets from 20 Sep onward (10-17 days stale); see "Start the clock at the catalogue epoch" below |
| Starlink × Kuiper, from the catalogue epoch (10 Sep 07:58 UTC) | 168h (1 week) | Closest approach 1,573 m, Pc 5.7e-6, 0 of 30 cross-operator close approaches above threshold — the same answer as the earlier fresh-data week-long run |

On fresh element sets, no genuine Starlink × Kuiper approach crossed the
threshold in any run. The 828 m "crossing" reported earlier came from
propagating element sets 10-17 days past their epoch and has been
withdrawn; it is kept, struck through, above because the correction is
part of the result.
`docs/real_evaluation.md` carries the distribution, built on a real
catalogue with an assigned ownership boundary. TLE files change daily and `data/` is
gitignored, so re-running these scripts will produce different numbers;
`docs/sample_run.md` is the reproducible synthetic-data walkthrough.

Note: the original brute-force screening implementation did not scale to a
real multi-thousand-object catalog (O(n^2) pairwise comparison per
timestep). `twin.py` now uses vectorized SGP4 propagation
(`sgp4.api.SatrecArray`) and a KD-tree broad-phase screening pass
(`scipy.spatial.cKDTree.query_pairs`) instead — a 24h/30s-step backtest
against ~8,000-10,000 objects now runs in well under a minute.

## The three-scenario comparison, on real data

`evaluate.run_three_scenario_comparison` answers the project's central
question on **one engineered** conjunction. `real_evaluation.py` runs the
same comparison across every threshold-crossing conjunction in a real
CelesTrak catalogue, so the headline becomes a distribution measured on
genuine orbits:

```bash
PYTHONPATH=src python3 scripts/run_real_evaluation.py \
    --catalog data/starlink.tle Starlink \
    --cross data/kuiper.tle Kuiper \
    --hours 168
PYTHONPATH=src python3 scripts/make_figures.py        # renders fig_real_evaluation
```

It writes `docs/real_evaluation.md` (a report-ready section) and
`docs/figures/real_evaluation.json` (the data behind it).

**The honest problem it has to solve.** Genuine cross-operator threshold
crossings are rare. Week-long Starlink x Kuiper runs on fresh element sets
measured closest approaches of 3.45 km and 1.57 km, with nothing above
threshold. Megaconstellations are deliberately shelled apart. (An earlier
run reported an 828 m crossing; it came from stale element sets and has been
withdrawn, see below.) So the script does two runs and the report should
carry both:

- **Split catalogue (headline).** One real catalogue, its satellites
  assigned to two operators by NORAD parity, keeping only conjunctions that
  cross the assigned boundary. Orbits, geometry, miss distance, Pc and
  timing are all real and unmodified; the *ownership boundary* is the one
  assigned element, which is stated in the generated document rather than
  buried. Parity is the right split because catalogue numbers follow launch
  order, so it is uncorrelated with orbital shell — splitting by altitude
  would have drawn the boundary along a line the orbits themselves respect,
  reproducing the empty result by construction.
- **Cross-constellation (the existence check).** Two genuinely different
  operators, unmodified, reported with whatever it finds intact. A zero is
  what justifies the split; a crossing, if a future catalogue produces one,
  is the motivating scenario caught in the real sky, and the generated
  document switches framing accordingly rather than calling a hit a null.

**A modelling bug this work surfaced.** `evaluate.sigma_no_cooperation`
scales degraded public-tracking precision as a *fraction* of each
encounter's own time-to-TCA. That is harmless for the demo's single
30-minute encounter and wrong across a week-long catalogue run: it makes
tracking quality depend on how far ahead the propagation window happened to
open, so an encounter six days out appears to be detected days in advance
and the "lead time" measured is really the window length. The real-data
path therefore uses `real_evaluation.sigma_public_tracking`, anchored to
*absolute* time before TCA, and caps lead time at a rolling 72-hour
screening horizon — the window inside which operators actually screen
conjunctions and decide on maneuvers. The demo path is unchanged, so
`docs/sample_run.md` still shows the original walkthrough.

**A second correction the real data forced.** An encounter whose closest
approach falls inside the first 72 hours of the propagation window cannot
be given a full horizon of warning by any scenario — the window did not
open early enough. On the first real run, 194 of 432 threshold crossings
fell in that burn-in period and scored an identical lead time under both
regimes (gain exactly zero), because both were clipped by the window rather
than by tracking quality. Including them mixes "no-cooperation was late"
with "we started watching late", so both runs now exclude them and report
how many were set aside.

One consequence worth stating in the report: because cooperation flags an
encounter as soon as it enters the screening horizon, the cooperative lead
time saturates at that horizon. The quantity that actually varies across
encounters — and the one `fig_real_evaluation` plots — is how much later
no-cooperation gets there.

**A third correction: start the clock at the catalogue epoch.** Every
real-data script used to propagate from the wall clock ("now"). That is
only right on the day the TLE file was downloaded. A TLE degrades quickly
with age, mostly along-track, and the published headline run had been made
on 20 Sep from element sets dated 9-10 Sep, so every orbit in it was 10-17
days stale. `real_evaluation.resolve_epoch` and `fleets.catalogue_epoch` now
start the window at the newest element-set epoch in the file (the scripts
default to `--epoch catalogue`; `--epoch now` or an ISO time are still
available), and each run records the median element-set age at its start.
The re-run moved the headline very little (237 scored encounters, median
gain 49.3 h instead of 49.5 h), but it removed the Starlink x Kuiper
"crossing", which turned out to be an artefact of the stale orbits. Re-runs
of the same file now also give the same numbers on any day.

## Refined Pc: what changes

`scripts/run_pc_model_comparison.py` screens the real catalogue once and
re-scores the same 237 encounters four ways (full output in
`docs/pc_model_comparison.md`):

| Column | Median Pc | Median gain | Both regimes flag | Only no-coop flags | Neither flags |
|---|---|---|---|---|---|
| `circular` (today's default) | 2.33e-4 | 49.28 h | 237 | 0 | 0 |
| `circular-in-plane` | 3.96e-4 | 47.60 h | 237 | 0 | 0 |
| `foster(k=1)` (isotropic) | 3.96e-4 | 47.60 h | 237 | 0 | 0 |
| `foster(k=8)` (along-track 8x) | 5.22e-6 | 0.00 h (19.13 h where both flag) | 81 | 73 | 83 |

`circular-in-plane` and `foster(k=1)` agree to 0.01%, so the decomposition
holds: projecting the miss onto the encounter plane costs about 1.7 h of
the gain, and the along-track anisotropy does the rest. Under the
anisotropic model the single median is not the number to quote, because it
mixes three outcomes: on the 81 encounters both regimes flag, cooperation
still buys a median 19.1 h; on 73, only degraded public tracking crosses the
threshold, which is a false alarm (an unnecessary manoeuvre) that precise,
cooperative tracking avoids; and 83 that the circular model flagged are
below threshold for both once the error ellipsoid has its real shape. So
the circular model overstated the warning-time benefit, and understated a
benefit it could not see at all: fewer false alarms.

`scripts/run_real_evaluation.py --pc-model foster` runs the full
three-scenario evaluation with the anisotropic model on its own admitted
population and writes `docs/real_evaluation_foster.md` beside the circular
one.

## The dashboard

`docs/conjunction_watch.html` is the presentation surface: a measured
section built from the project's own output, and below it the scripted
scenario that explains the mechanism.

```bash
PYTHONPATH=src python3 scripts/make_dashboard.py          # writes docs/conjunction_watch.html
PYTHONPATH=src python3 scripts/make_dashboard.py --check  # report what would be embedded
PYTHONPATH=src python3 scripts/make_dashboard.py --run circular  # embed the circular-model run instead
```

It was previously hand-scripted — every number on the page typed in by
whoever wrote the HTML, which is fine for explaining a mechanism and
worthless as evidence, since nothing stopped it drifting away from what the
code actually produced. `make_dashboard.py` now injects the real run data
(`docs/figures/real_evaluation.json`, the reputation trajectories and the
robustness sweep) into `docs/conjunction_watch_template.html`, so the
dashboard cannot disagree with the report. By default it embeds the Foster-model run (`real_evaluation_foster.json`), the report's headline model. Design lives in the template;
the script owns only the data. Re-run it after any run that changes the
numbers.

The page keeps the scenario animation, now explicitly labelled as the
mechanism rather than the evidence, with the measured figures above it —
including the genuine cross-operator encounter, when a run finds one.

## Always-running worlds

Everything above runs once and stops. `src/satcollision/world.py` asks the
other question a federation has to answer: what happens if it is left
running? A *world* is one self-contained simulated sky with its own
operators, a clock that keeps moving, and the whole pipeline on every step:
look-ahead screening, Pc, signal abstraction, Ed25519 signing into the
hash-chained ledger, deconfliction, periodic federation rounds and
reputation. Three ship in `worlds/`:

| World | Fleet | What it shows |
|---|---|---|
| `sandbox` | Alpha / Beta / Gamma + 25 debris (synthetic), 60x | The federated path working: a new satellite is launched onto a crossing path every 45 simulated minutes, both operators sign, a joint plan is agreed every time. |
| `adversarial` | Same fleets, Gamma is a spoofer, 60x | Gamma withholds most of its own encounters (its partner must move alone) and invents fake ones. Its reputation falls to the degraded/suspended tiers within a few rounds, and reputation-weighted aggregation stops listening to it while the plain mean is pulled off. |
| `real_leo` | Every active LEO satellite in CelesTrak's `active` catalogue (`data/active.tle`, 15,079 objects below 2,000 km), grouped by name into six members: Starlink, Kuiper, OneWeb, Planet (FLOCK, SKYSAT, PELICAN, TANAGER), Spire (LEMUR) and ICEYE. The other 2,646 objects belong to no member. Foster Pc, 30x | Nothing engineered. It starts at the catalogue epoch and replays its first 7 days on a loop, because TLEs go stale within days. Screening the first 24 hours found 3 real cross-operator approaches above the Pc threshold (all Planet x Starlink), plus 49 within one operator and 14 with non-members. The cross-operator ones go through the federation. With only Starlink, Kuiper and OneWeb (the old setup) there were none: their shells don't overlap. |

```bash
python scripts/run_worlds.py              # starts all three + the live dashboard at http://127.0.0.1:8765/
python scripts/run_worlds.py sandbox      # just one
python scripts/run_worlds.py --reset      # start every world from scratch
python scripts/run_world.py --world worlds/adversarial.json --ticks 720 --fast   # one simulated day, no pacing
```

How it stays running:

* **One OS process per world**, supervised by `run_worlds.py`, which
  restarts a world that exits (backing off 2 s, 4 s ... 60 s) and serves the
  live page (`docs/live_worlds.html`) on 127.0.0.1 only. "Always running"
  means for as long as the launcher is up; stop it and every world resumes
  from the same tick next time.
* **Look-ahead screening.** Each tick screens only the slice of time
  `lookahead_s` ahead (30 min for the synthetic worlds, 2 h for real LEO), so
  every encounter is found once, with a fixed warning time, and then counts
  down. Consecutive slices overlap by two steps and duplicates are dropped by
  pair and time.
* **Checkpoint every tick.** Ledger and events are appended first, then an
  atomic `checkpoint.json` records how much of them is valid, then
  `state.json` for the dashboard. All output lives in `runs/<world>/`
  (git-ignored).
* **Deterministic.** Every random choice is keyed by what it decides (world,
  seed, encounter id, round) rather than by call order, and ledger
  timestamps are simulated time. Running 200 ticks straight and running 100,
  stopping, and running 100 more produce the same ledger hash for hash
  (`tests/test_world.py`). A tick that wrote ledger lines but died before
  its checkpoint is rolled back and redone, never duplicated.
* **Tamper-evident on disk.** On resume the ledger is rebuilt from
  `ledger.jsonl`, every hash re-derived and every signature re-checked
  against the operators' public keys. Editing one entry flags the world as
  tampered on the dashboard.
* **Config changes are explicit.** A checkpoint records a fingerprint of the
  settings that define the world; changing the fleet or the rules refuses
  to resume until `--reset`. `speed` (pacing only) can be changed freely.

Measured, one simulated day per world (720 / 720 / 576 ticks):

| World | Shared encounters | Joint plans | Partner moved alone | Private / in-house | Final reputation | Time per tick |
|---|---|---|---|---|---|---|
| sandbox | 19 | 19 | 0 | 12 | Alpha 0.83, Beta 0.81, Gamma 0.74 | ~1 ms |
| adversarial | 42 | 24 | 18 | 6 | Alpha 0.85, Beta 0.86, **Gamma 0.27** | ~1 ms |
| real_leo | 0 | 0 | 0 | 445 | all 0.67 (nothing to judge) | ~0.35 s |

Simplifications, stated plainly: agreed manoeuvres are recorded but never
applied to the orbits; the miss distance is the closest 30 s grid sample,
as everywhere else in the project; simulated operators' signing keys are
derived from the world name and seed so a restarted world reproduces its
own signatures (a real operator generates its key locally); and the
engineered encounters use `fleets.inject_actor_encounter`, whose closed-form
inversion often misses at steep crossing angles, so each injection is
verified against SGP4 and retried at another angle or satellite until the
two objects really meet.

### `real_leo_history`: truth and prediction from a week of real element sets

`worlds/real_leo_history.json` replaces the single catalogue snapshot with a
Space-Track `gp_history` file: every element set published for every LEO
object over a week (`data/historical/gp_history_week.tle`: 315,014 sets for
17,781 objects, 6 to 12 Nov 2026, about 2.9 sets per object per day). Like the
rest of `data/historical/` it is not in git, and it is not built into the
Docker image either (it is mounted read-only, see `docker-compose.yml`),
because Space-Track data must not be redistributed. Download it from
Space-Track's `gp_history` class with your own account (one query per week,
3LE format) and save it under that name.

What changes compared with `real_leo` (`src/satcollision/tle_history.py`,
`history_world.py`):

* **Truth** is the element set whose epoch is nearest to the moment,
  including one published a little later, the way an orbit fit uses
  observations on both sides of its epoch. Across each switch between sets
  the two positions are blended linearly over 10 minutes, so nothing jumps.
* **Prediction**: every twin screens against the public catalogue as it
  stood at that moment, using only the newest set with epoch at or before
  now (the as-of rule from `historical_replay.py`). So every party finds
  the same encounters with the same ids. The screen is the refined one
  (`screening.screen_conjunctions_refined`). It finds the closest approach
  between the 30 s samples, so fast crossings are not skipped. Approaches
  predicted closer than `candidate_km` (5 km) are assessed. Each tick takes
  only the approaches whose TCA falls in its own slice of the look-ahead, so
  each one is found exactly once. An operator then assesses its own
  side: its own satellite from its own ephemeris (the truth, with
  `precise_sigma_km` = 0.354 km per object, 0.5 km combined), the other object
  from the public prediction. Each side decides alone whether to share, so
  the two operators of one encounter can disagree (the log says when one
  side's Pc was under the threshold).
* **Measured covariance.** The RTN error of a TLE as a function of its age is
  fitted from the file itself: each set is propagated to the epochs of later
  sets of the same object and compared there, using a robust sigma
  (1.4826 x MAD, then combined with the median offset in quadrature) per age
  bin, Starlink and everyone else separately (`research/tle_error_fit.py`).
  The Foster Pc uses the sigma for the age that the other object's set will
  have at TCA.

  | Age (h) | Starlink σR / σT / σC (km) | Starlink median T | Others σR / σT / σC (km) |
  |---|---|---|---|
  | 0-3 | 0.11 / 0.82 / 0.05 | -0.11 | 0.005 / 0.044 / 0.008 |
  | 3-6 | 0.15 / 0.99 / 0.09 | +0.05 | 0.010 / 0.066 / 0.011 |
  | 6-12 | 0.20 / 2.04 / 0.14 | +0.49 | 0.018 / 0.116 / 0.017 |
  | 12-24 | 0.28 / 7.88 / 0.19 | +3.73 | 0.034 / 0.208 / 0.027 |
  | 24-36 | 0.38 / 20.3 / 0.24 | +9.34 | 0.051 / 0.361 / 0.044 |
  | 36-48 | 0.61 / 41.2 / 0.33 | +21.8 | 0.064 / 0.578 / 0.058 |
  | 48-72 | 1.24 / 87.0 / 0.45 | +44.4 | 0.097 / 1.14 / 0.083 |

  Starlink matches the 2021 pilot (about 7 km at 12-24 h and 21 km at
  24-36 h). For other objects the TLE-vs-later-TLE spread understates the
  true error: consecutive fits share observations.
* **Real manoeuvres** show up as jumps between consecutive sets. Sets more
  than 10 robust sigma (and 1 km) off the previous one are summarised in the
  log once per simulated hour. About 4% of all sets qualify, mostly
  Starlink's.
* **Agreed avoidance manoeuvres are optimised and applied to the truth**
  (`src/satcollision/avoidance.py`). When both operators sign, the world
  picks *when* each satellite burns, *which way* and *how much*: the least
  weighted fuel (sum of |dv|^2 / capability) that brings Pc under
  `target_pc` (1e-6). The Pc is evaluated at the *precise* combined
  covariance (0.5 km), because at the public covariance (8 to 20 km
  along-track at a day old) the Pc is diluted and a tiny burn would "meet"
  the target. Burn effects come from the Clohessy-Wiltshire solution,
  projected into the encounter plane.
  * **No private orbits are exchanged.** The plan's direction and size
    depend only on orbit geometry, which public element sets give well
    enough, so both operators compute the same plan from public data. Each
    then delivers its own part from its own precise orbit.
  * **The secure check.** A yes/no check ("target met?") applies both burns
    to both precise orbits; if it fails, the plan is scaled up and checked
    again. In the federation this runs inside the secure step and returns
    only the yes/no. Here it is simulated, and callers use nothing but the
    answer.
  * **Also accounted for:** burn-execution error (5% of the planned shift),
    min/max dv, the burn-time window, low-thrust finite burns for Starlink
    (1.7e-4 m/s², a few minutes per burn), satellites without propulsion
    (`no_propulsion`: the other side does all the moving, or "cannot avoid"
    is logged), and fairness weights.
  * **Secondary conjunctions.** Each manoeuvred track is re-screened
    against the public catalogue until its return burn, and a burn time
    that creates an approach over Pc 1e-5 is swapped for the next cheapest.
  * **After the burn.** The displacement is mirrored back to zero after TCA
    (a return burn). The operator view shows its own burn (dv, RTN
    components, time, Pc before -> target, check result). The observer sees
    total dv and the Pc at the precise covariance before -> after.

  On the first 12.5 h of the week (`"avoidance": {"avoidance": "fixed"}`
  restores the earlier rule: 1 km along-track at detection):

  | | Fixed 1 km rule | Optimised |
  |---|---|---|
  | Joint plans | 4 | 4 |
  | Pc <= 1e-6 at the precise covariance after the burns | 1 of 4 | 4 of 4 |
  | True miss after | 0.77 - 2.09 km | 1.89 - 2.72 km |
  | Total dv (both satellites, all plans) | 0.43 m/s | 1.04 m/s |
  | Secondary approaches avoided | - | 5 |

  The fixed rule spends less fuel because it is not sized to any target:
  three of its four plans leave Pc between 4e-5 and 3e-4. Where both meet
  the target, the optimised plan uses less (synthetic fast crossing: 25 vs
  30 cm/s per satellite for the same 1.9 km).
* **Truth the twins missed.** The sky also screens the truth (refined
  screen, on the nearest set at the middle of each tick's window, because
  the SGP4 refinement needs one set per object). The miss is then
  re-measured on the blended truth, applied burns included. A real
  approach under 2 km that no twin flagged is shown to the observer only.
* **The clock** starts 24 h after the first epoch (so most objects have a
  published set) and runs through the file's real dates. When it reaches
  the last epoch it stops, logs that, and the dashboard shows a banner. With
  `"at_end": "loop"` it replays the file and logs a cycle event instead.

On the week above, the first 12.5 h (300 ticks from 7 Nov 00:00), refined
screen vs the grid screen it replaced:

| | Grid (30 s samples, 25 km) | Refined |
|---|---|---|
| Cross-operator, over the threshold | 4 | 19 |
| ... joint plans (burns applied) | 2 | 4 |
| ... one side's own Pc under the threshold, or silent | 2 | 15 |
| In-house (one operator, or a member and a non-member) | 61 | 246 |
| Private watch (under the threshold, < 2 km) | 217 | 957 |
| Real approaches < 2 km no twin flagged | 18 | 368 |
| Median relative speed of the recorded approaches | 9.8 km/s | 12.2 km/s |
| Time per tick, one process | ~1.4-2 s | ~0.8-1.8 s |
| Time per tick, separate processes (8) | ~3 s | ~1.7-2.1 s |

The added approaches are mostly fast crossings, as `research/RESULTS.md`
section 1 found for the snapshot catalogue. The refined screen is also the
cheaper of the two here, because only approaches predicted inside 5 km are
assessed, against every pair inside the grid's 25 km. Each twin process uses
about 225 MB (about 1.6 GB for all eight).
`tests/test_history_world.py` checks it on a synthetic history: the as-of
rule, blending, the error fit, the jump flag, the burn, a fast crossing
between samples that a grid screen misses, the stop and loop at the end,
restarts, and that separate processes reach the same result as one process.

Still simplified: the twins' own ephemeris does not include the simulated
avoidance burns, and the public catalogue never learns of them (the TLEs
are history); an element set counts as published at its epoch, which is
earlier than it really appears; and a public prediction more than 5 km
off is never assessed (those approaches end up in the observer's "no twin
flagged it" list, which is why that list is long: Starlink's sets are
kilometres off within hours).

## Full-stack app: what each party sees

`app/` turns the always-running worlds into a web application whose point is
the federation's privacy claim: **the same encounter looks different to each
party, and the central server never sees a single position.** Each login gets
a different view, filtered on the server (`src/satcollision/views.py`) before
anything is sent:

| Login | Sees | Never gets |
|---|---|---|
| Operator (`alpha`, `beta`, `gamma`; `starlink`, `kuiper`, `oneweb`, `planet`, `spire`, `iceye` in Real LEO) | Its own digital twin: its satellites' exact positions, speeds and orbits; its own close approaches with exact miss distance and Pc; its agreed manoeuvre; the alerts the federation forwarded, cut to its reputation tier (`reputation.tailor_alert`) | Anyone else's precise state (others appear as the public catalogue shows them, rounded to 0.5 deg, about 50 km), close approaches it isn't in, others' reputations |
| `federation` | Exactly what the central server holds: signed abstracted signals, count reports, the hash-chained ledger, reputation and the aggregate. Its globe is deliberately empty. | Any position, velocity, orbit, miss distance or exact Pc |
| `observer` | Ground truth, for the demo, plus **Compare**: one encounter side by side as each operator's twin and the server hold it, built from the same filters as their own logins | |

The filtering is enforced, not cosmetic: an operator token asking for another
operator's orbit, for the observer's comparison, or for a world it does not
operate in gets 403 (`tests/test_app.py`, including one that walks every key
of the federation view to check no position-like field appears). Every one of
those tests runs twice: once with one process playing every part, once with
the separate processes described next.

Stack: FastAPI backend (JWT logins, live updates over a WebSocket per world),
React + TypeScript + three.js frontend built with Vite, served by the backend
as static files.

### Every operator and the federation as separate processes

By default (`APP_PROCESSES=1`) the app does not compute everyone's view from
one shared simulation. Each world runs as separate OS processes
(`src/satcollision/processes.py`):

| Process | Module | Holds | Never has |
|---|---|---|---|
| One per operator (Alpha, Beta, Gamma; Starlink, Kuiper, OneWeb) | `twin_node.py` | Its own private Ed25519 key (generated there on first start, kept in `runs/<world>/nodes/<op>/identity.key`); its twin's screening of its own satellites against the public catalogue, with exact miss distance and Pc; its own decisions (share or not, what to report) | Any other operator's key, results or decisions |
| The federation | `federation_node.py` + `federation_service.py` | Pinned public keys, the signed hash-chained ledger, attestations, reputation, rounds | Any orbit code at all: it never imports sgp4, scipy, the fleets or the world, and `GET /health` lists what it has loaded (tests check it is nothing) |
| The app | `app/backend` + `World(mode="sky")` | The simulation's ground truth (only the observer login sees it) and the clock | Any operator's private key |

Each tick runs in lockstep. The sky ticks. Every twin ticks to the same
point, then signs what it decided to share and posts it to the federation.
The federation then asks every twin which of the new encounter ids involve
it, one yes or no per operator and encounter. When a reporting period ends,
each twin signs its count report and the federation scores the round. An
operator's dashboard is served by its own process, which fetches its alerts
from the federation with a request signed by its own key. The federation
cuts those alerts to that operator's tier before they leave its process.

The federation pins the first key each operator registers, and later refuses
a different key for that operator. A message that doesn't verify against the
pinned key gets a 403. All traffic stays on 127.0.0.1 and needs a per-start
internal token. Each process checkpoints into its own folder, so after a
restart a twin that fell behind catches up, and if a process dies the app
restarts the group.

With the same seed, both modes produce identical reputations, ledgers,
encounters and manoeuvre plans. `tests/test_nodes.py` (8 tests) checks this,
along with the key isolation, the forgery refusals and resume.
`APP_PROCESSES=0` runs everything in one process. It is faster, and
`scripts/run_worlds.py` still uses it.

```bash
# Option 1: Docker (nothing else to install)
docker compose up --build          # http://localhost:8000, demo password "demo"

# Option 2: plain Python (the frontend build in app/frontend/dist is served as-is)
pip install -r app/requirements.txt
python -m app.backend              # http://127.0.0.1:8000

# Frontend development (hot reload on :5173, API proxied to :8000)
cd app/frontend && npm install && npm run dev
```

The app and `scripts/run_worlds.py` both advance the worlds in `runs/`; run one
or the other, not both at once. Configuration is by environment variable
(`APP_HOST`, `APP_PORT`, `APP_SECRET`, `APP_USERS_FILE`, `APP_SPEED`,
`APP_WORLDS`, `APP_RESET`, `APP_PROCESSES`; see `app/backend/main.py`).

**Deploying later.** The Docker image is the deployable unit; any host that
runs a container (a small VPS, Render, Railway, Fly.io) works. Before
exposing it: set `APP_SECRET` to a long random string, and point
`APP_USERS_FILE` at a JSON file of real accounts
(`{"alpha": {"role": "operator", "operator": "Alpha", "password": "..."}}`),
which also switches off the one-click demo logins. Put it behind HTTPS (the
host's TLS or a reverse proxy); the WebSocket follows automatically. The Real
LEO world needs `data/active.tle` in the build context. Download it with
`curl -o data/active.tle "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"`
and refresh it now and then, because TLEs go stale. When the world's definition
changes, the app moves the old run aside to `runs/<world>.old-<time>` and starts fresh.

Simplifications: passwords in the users file are compared as plain text
(fine for a demo, hash them before real users); "public catalogue" positions
are the true positions rounded, standing in for the km-level error of real
public TLEs. All the processes run on one computer and talk over localhost,
where a real deployment would put each twin on its operator's own machine.
Each twin screens against the same public TLEs that everyone can download, so
its "private" knowledge is its results and decisions, not better orbits. The
key file stands in for a hardware key store.

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
4. Verifiable secret sharing / range proofs on top of `secure_aggregation.py`
   so secure aggregation and Byzantine-robust aggregation can run in the
   same round (see "Trust model" above for why they currently can't).
5. A total privacy-budget accountant across the whole federation's
   lifetime in `federated_learning.py`, rather than the current per-round
   epsilon — real deployments cap cumulative epsilon across many rounds,
   not just report each round's individually.
6. Run the distributed demo's operators and hub across genuinely separate
   machines on a real LAN (currently all on `127.0.0.1`) — the wire
   protocol and signature verification don't change, only `--host`/`--port`
   would need to point somewhere other than loopback.
7. Download the 2021 element-set histories from Space-Track and run
   `scripts/run_historical_replay.py` for both registered China Space
   Station events, with the exact times from A/AC.105/1262.
8. Decide which Pc model the final report's headline uses (the anisotropic
   one is the more defensible) and regenerate the figures and dashboard
   from that run.
