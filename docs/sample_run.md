# Sample run output

```
==============================================================================
FEDERATED DIGITAL TWINS FOR SATELLITE CONSTELLATION COLLISION AVOIDANCE
End-to-end demonstration run
==============================================================================

[1] Building synthetic fleets (3 operators + ambient debris)...
    population: {'Alpha': 24, 'Beta': 15, 'Gamma': 12, 'DEBRIS': 25}   (total 76 tracked objects)
    engineering one genuine close approach for the walkthrough...
    actor A: Alpha-90000   actor B: Beta-99000

[2] Propagating 40-minute window and screening for conjunctions...
    1 conjunction(s) found within the 25km screening volume:
      Alpha-90000 <-> Beta-99000 TCA= 30.0min  miss=     0.0m  Pc=8.00e-04  geometry=crossing  <== crosses Pc threshold

[2b] Kalman-filter refinement: re-deriving this encounter's combined sigma from an
     actual tracking history instead of the fixed DEFAULT_COMBINED_SIGMA_KM=0.5km...
    fixed-sigma Pc   (sigma=0.5km, constant)      : 8.000e-04
    Kalman-derived Pc (sigma=0.0338km at TCA, evolved): 1.750e-01
    (0s since the simulated last tracking update at TCA -- this is exactly what the fixed constant can't represent)

[3] Abstracting the flagged conjunction into shareable signals...
    signal from A: AbstractedSignal(encounter_id='ENC-90000-99000-1800', sender_operator='Alpha', local_object_label='Alpha-90000', tca_offset_s=1800.0, geometry_class='crossing', severity_tier='threshold', threshold_crossed=True)
    signal from B: AbstractedSignal(encounter_id='ENC-90000-99000-1800', sender_operator='Beta', local_object_label='Beta-99000', tca_offset_s=1800.0, geometry_class='crossing', severity_tier='threshold', threshold_crossed=True)
    (no position/velocity/orbital-element field exists on this object —
     see signal.assert_no_raw_state, exercised in tests/test_twin_and_signal.py)

[3b] Cryptographic signing (real Ed25519 keys) + hash-chained log...
    Alpha signs its signal; Beta verifies it: True
    tamper check: mutating the signed payload after signing...
      verify(tampered content, same signature) = False  (must be False)
    impersonation check: an attacker signs with their own key, claiming to be 'Alpha'...
      verify(forged signature, real trusted registry) = False  (must be False -- registry only trusts the real keypair on file)

[4] Federation aggregation: 5 honest operators + 1 adversary reporting
    period counts (threshold/elevated/critical encounters)...
    plain mean   : [11.66666667 10.83333333 11.66666667]   error vs honest truth = 13.725
    trimmed mean : [4.   3.25 4.5 ]   error = 0.783
    Krum         : [4. 4. 4.]   error = 1.265
    every report above also gets signed and appended to the same tamper-evident log
    (note: the adversary's SPOOFED CONTENT still verifies -- signing proves who sent
     a message and that it wasn't altered in transit, not that its contents are honest;
     that is what Byzantine-robust aggregation above, and secure aggregation, are for)...
    hash-chained log: 8 entries, tail_hash=4cd7c192d3838748..., verify_chain() = True
    now simulating a cover-up: someone edits an already-logged entry after the fact...
    verify_chain() after the edit = False  reason: entry 2: stored hash does not match recomputed hash -- entry content was modified after logging

[4b] Secure aggregation on the same 6 reports (Bonawitz-style pairwise-masked
     additive secret sharing over a 127-bit field, real X25519 key agreement)...
      Op0        plaintext=[4. 4. 4.]   masked share (first coord) = 98330982707702448472776076819229474871
      Op1        plaintext=[2. 2. 6.]   masked share (first coord) = 49770138296130590944190751840133331477
      Op2        plaintext=[3. 4. 2.]   masked share (first coord) = 73124072911393207890534644803836957974
      Op3        plaintext=[5. 3. 4.]   masked share (first coord) = 7928042352156291319985561949697383158
      Op4        plaintext=[4. 1. 4.]   masked share (first coord) = 95900344137616221295833351682610999732
      Adversary  plaintext=[52. 51. 50.]   masked share (first coord) = 15228786515939703540054220336330064242
    plain mean_aggregate()      : [11.66666667 10.83333333 11.66666667]
    secure_mean_aggregate()     : [11.66666667 10.83333333 11.66666667]
    (identical result -- the aggregator computed the same mean, but its own
     arithmetic never touched a single operator's plaintext count vector; the
     'masked share' values above are exactly what it saw instead. Note this loses
     the Byzantine-robustness demonstrated above -- the adversary's spoofed value
     is still baked into this mean, since masked shares can't be trimmed/Krummed.
     See README 'Trust model' for why these two protections don't compose for free.)

[5] Maneuver deconfliction negotiation...
    ManeuverPlan(encounter_id='ENC-90000-99000-1800', primary_operator='Beta', primary_object='Beta-99000', primary_direction='+', secondary_operator='Alpha', secondary_object='Alpha-90000', secondary_direction='-', rationale='Directions are assigned by a rule fixed in advance and keyed only on the public encounter_id, so both operators derive the same complementary plan independently — guaranteeing the two maneuvers add up instead of risking cancellation, with no additional private data exchanged.')
    uncoordinated independent guessing: safe 49.8% of the time
    negotiated protocol:                safe 100.0% of the time

[6] Three-scenario comparison on this same engineered encounter...
    No Cooperation         lead_time= 10.83 min   raw_data_exposed=  0.0%   cross_operator_signal=False
    Full Data Sharing      lead_time= 30.00 min   raw_data_exposed=100.0%   cross_operator_signal=True
    Federated (Proposed)   lead_time= 30.00 min   raw_data_exposed=  0.0%   cross_operator_signal=True

[7] Robustness sweep: aggregation error vs. fraction of dishonest operators...
    adv_frac   mean_err   trimmed_mean_err   krum_err
        0%       0.000        0.477              1.416
       10%       3.491        0.505              1.296
       20%       2.968        0.517              1.455
       30%       9.452        0.819              1.385
       40%       7.182        0.369              1.430

[8] Federated learning: FedAvg + DP-SGD (Opacus) training a real risk-classifier
    across Alpha/Beta/Gamma's own non-IID local encounter data, evaluated on a held-out
    set mixing every operator's regime (never any single operator's own distribution)...
    centralized (pools raw data, no privacy -- upper bound) : {'accuracy': 0.9883333333333333, 'precision': 0.9976635514018691, 'recall': 0.9861431870669746, 'f1': 0.9918699186991868}
    local-only  (each operator alone, mean across 3)        : {'accuracy': 0.9838888888888889, 'precision': 0.9877889067006042, 'recall': 0.9899923017705928, 'f1': 0.9888338018448222}
    federated, no DP  (FedAvg only)                         : {'accuracy': 0.9883333333333333, 'precision': 0.9840909090909091, 'recall': 1.0, 'f1': 0.9919816723940436}
    federated + DP-SGD (FedAvg + Opacus)                    : {'accuracy': 0.9616666666666667, 'precision': 0.9495614035087719, 'recall': 1.0, 'f1': 0.9741282339707538}
    final-round (epsilon, delta=1e-5) per operator under DP-SGD: {'Alpha': np.float64(1.4374878778525355), 'Beta': np.float64(1.4374878778525355), 'Gamma': np.float64(1.4374878778525355)}
    (federated should land close to centralized without ever pooling raw data; DP-SGD
     costs a little more accuracy in exchange for the formal epsilon above on every
     operator's shared update -- see README 'Trust model' for what that epsilon means.)

[9] Incentive/reputation layer: 20 reporting periods, 6 operators, 4 strategies.
    Reputation is built only from things the federation can already see -- whether a
    counterparty attests the encounter you reported (or the one you stayed silent about),
    how far your counts sit from the robust consensus, and whether you contributed at all...
    operator   strategy     score trajectory (every 4th round)            final   alert tier
    Alpha      honest      0.62  0.73  0.82  0.88  0.92      0.92    full
    Beta       honest      0.67  0.78  0.79  0.82  0.88      0.88    full
    Gamma      honest      0.65  0.77  0.82  0.88  0.84      0.84    full
    Delta      free_rider  0.40  0.31  0.30  0.33  0.36      0.36    degraded
    Epsilon    spoofer     0.41  0.30  0.31  0.40  0.41      0.41    degraded
    Zeta       on_off      0.67  0.77  0.84  0.67  0.47      0.47    standard
    (Zeta is the interesting one: it behaves exactly like an honest operator for 12 rounds,
     peaks at 0.84, then defects -- and because reputation falls ~3.5x faster
     than it rises, 8 rounds of lying cost more than 12 rounds of honesty bought. Banking
     good behaviour to spend on one big lie is a net loss, which is the point.)

    aggregation error vs. the honest operators' own truth, mean of the last 5 rounds:
      plain mean            : 0.508
      trimmed mean          : 0.417   (robust, but memoryless)
      reputation-weighted   : 0.221   (carries history across rounds)

    the same 6 reports from step [4], but aggregated with reputation weights
    (the adversary has spent five rounds being caught, and no longer has influence)...
      plain mean            : [11.66666667 10.83333333 11.66666667]   error = 13.725
      trimmed mean          : [4.   3.25 4.5 ]   error = 0.783
      reputation-weighted   : [3.6 2.8 4. ]   error = 0.000

    reciprocity -- what each tier actually receives when an alert is shared:
      full       ['counterparty_operator', 'deconfliction_eligible', 'encounter_id', 'geometry_class', 'involves_you', 'severity_tier', 'tca_offset_s', 'tier']
      standard   ['encounter_id', 'involves_you', 'severity_tier', 'tca_offset_s', 'tier']
      degraded   ['encounter_id', 'involves_you', 'tier']
      suspended  ['encounter_id', 'tier']
    ...and the hard safety floor: a CRITICAL alert is delivered in full to everyone,
    whatever their reputation --
      suspended operator, critical alert: ['counterparty_operator', 'deconfliction_eligible', 'encounter_id', 'geometry_class', 'involves_you', 'safety_override', 'severity_tier', 'tca_offset_s', 'tier']
    (reputation allocates influence and privilege; it must never be a mechanism for
     withholding a collision warning, because the debris harms third parties who had
     no part in the misbehaviour. See README 'Trust model'.)

[10] Plain-language summaries of everything above -- deterministic templates,
     no language model anywhere: a safety summary has to be reproducible, auditable,
     and incapable of inventing a number that was never computed...

     the flagged encounter, as its OWN operator sees it (full fidelity):
       Alpha-90000 and Beta-99000 are crossing paths at a wide angle. Their closest approach is in 30 minutes, passing through what the model puts at the same point, under 10 metres apart — about a 1 in 1,200 chance of a collision. Closing speed is 14.5 km/s; modelled Pc is 8.00e-04. That figure is a model estimate, not a measurement: it assumes the simplified circular uncertainty model, so treat it as an order of magnitude rather than an exact number.

     the same encounter for a non-technical reader (audience='executive'):
       Two satellites are crossing paths at a wide angle. Their closest approach is in 30 minutes, passing through what the model puts at the same point, under 10 metres apart — about a 1 in 1,200 chance of a collision. That figure is a model estimate, not a measurement: it assumes the simplified circular uncertainty model, so treat it as an order of magnitude rather than an exact number.

     the same encounter as the COUNTERPARTY sees it, at three reputation tiers --
     note the summary is rendered from the tailored alert dict, so a lower tier
     cannot leak through a template edit (summaries.assert_no_undisclosed_terms):
       [full] Encounter ENC-90000-99000-1800 involves one of your satellites. The federation rates it just over the level at which encounters are shared. Closest approach is in 30 minutes. The two objects are crossing paths at a wide angle. The other object is operated by Beta, who received the matching alert at the same moment. You are eligible to open a deconfliction plan for this encounter, which assigns both sides complementary escape directions without either of you sharing trajectory data.
       [standard] Encounter ENC-90000-99000-1800 involves one of your satellites. The federation rates it just over the level at which encounters are shared. Closest approach is in 30 minutes. Some detail is held back at your current access tier (standard); it is released again as your reporting record improves.
       [degraded] Encounter ENC-90000-99000-1800 involves one of your satellites. Some detail is held back at your current access tier (degraded); it is released again as your reporting record improves.

     the agreed maneuver, written as an instruction for one side:
       Agreed plan for encounter ENC-90000-99000-1800: Beta moves Beta-99000 away from the encounter along the agreed escape axis, while Alpha moves Alpha-90000 the opposite way along the same axis. Both sides worked this out independently from the encounter identifier alone, so neither had to send the other any trajectory data, and the two maneuvers are guaranteed to add up rather than cancel. Your action: move Alpha-90000 the opposite way along the same axis.

     what step [4]'s federation round actually concluded:
       6 operators reported this period. Taking every report at face value gives an average of 34.2 flagged encounters per operator; discounting reports that disagree with everyone else's gives 11.8. That gap is the signature of at least one operator reporting something the rest of the federation cannot corroborate — the robust figure is the one to act on. Adversary currently carries no weight in this result, having been quarantined on its own reporting record.

     what the quarantined operator is told about its own standing:
       Adversary has a reporting reputation of 0.12 (suspended tier). You carry no weight in federated results and receive identifiers only. Reputation rises by reporting the encounters your own satellites are party to, and falls faster than it rises when reports cannot be corroborated by the operator on the other side of the encounter. Critical collision warnings reach you in full at any reputation.

     the whole thing as one incident report -- the detecting operator's own copy:
       INCIDENT SUMMARY — ENC-90000-99000-1800
         What happened   : Alpha-90000 and Beta-99000 are crossing paths at a wide angle. Their closest approach is in 30 minutes, passing through what the model puts at the same point, under 10 metres apart — about a 1 in 1,200 chance of a collision. Closing speed is 14.5 km/s; modelled Pc is 8.00e-04. That figure is a model estimate, not a measurement: it assumes the simplified circular uncertainty model, so treat it as an order of magnitude rather than an exact number.
         What was shared : Encounter ENC-90000-99000-1800 involves one of your satellites. The federation rates it just over the level at which encounters are shared. Closest approach is in 30 minutes. The two objects are crossing paths at a wide angle. The other object is operated by Beta, who received the matching alert at the same moment. You are eligible to open a deconfliction plan for this encounter, which assigns both sides complementary escape directions without either of you sharing trajectory data.
         What happens now: Agreed plan for encounter ENC-90000-99000-1800: Beta moves Beta-99000 away from the encounter along the agreed escape axis, while Alpha moves Alpha-90000 the opposite way along the same axis. Both sides worked this out independently from the encounter identifier alone, so neither had to send the other any trajectory data, and the two maneuvers are guaranteed to add up rather than cancel. Your action: move Beta-99000 away from the encounter along the agreed escape axis.
         How to check    : Every signal and report behind this summary was signed by the operator that sent it and appended to a hash-chained log, so any later edit to the record is detectable. Current log tail: 4cd7c192d3838748...

     ...and the same incident as a DEGRADED counterparty receives it. The private
     view isn't hidden from this copy, it was never passed in (conjunction=None):
       INCIDENT SUMMARY — ENC-90000-99000-1800
         What happened   : The operator on the other side of this encounter detected it in its own digital twin and shared an abstracted signal — never a trajectory — with the federation. What reached you is set out below.
         What was shared : Encounter ENC-90000-99000-1800 involves one of your satellites. Some detail is held back at your current access tier (degraded); it is released again as your reporting record improves.
         What happens now: A deconfliction plan exists for this encounter, but your current access tier does not include it. It is released as your reporting record improves.
         How to check    : Every signal and report behind this summary was signed by the operator that sent it and appended to a hash-chained log, so any later edit to the record is detectable. Current log tail: 4cd7c192d3838748...

==============================================================================
Done. See README.md for what's built vs. future work, and
major_project_final_definition.md (project root) for the full writeup.
==============================================================================
```
