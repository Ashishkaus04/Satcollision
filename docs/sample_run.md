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
    hash-chained log: 8 entries, tail_hash=f1753d2fc4aa6667..., verify_chain() = True
    now simulating a cover-up: someone edits an already-logged entry after the fact...
    verify_chain() after the edit = False  reason: entry 2: stored hash does not match recomputed hash -- entry content was modified after logging

[4b] Secure aggregation on the same 6 reports (Bonawitz-style pairwise-masked
     additive secret sharing over a 127-bit field, real X25519 key agreement)...
      Op0        plaintext=[4. 4. 4.]   masked share (first coord) = 113409798746388985793897490058134892877
      Op1        plaintext=[2. 2. 6.]   masked share (first coord) = 59990202516354902005699701156239867529
      Op2        plaintext=[3. 4. 2.]   masked share (first coord) = 101566873922284443885915691379161016163
      Op3        plaintext=[5. 3. 4.]   masked share (first coord) = 14241080829909873155624220033598656599
      Op4        plaintext=[4. 1. 4.]   masked share (first coord) = 40249463959553921904088273951756348589
      Adversary  plaintext=[52. 51. 50.]   masked share (first coord) = 10824946946446336718149230852947429697
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
    final-round (epsilon, delta=1e-5) per operator under DP-SGD: {'Alpha': np.float64(1.437487877739899), 'Beta': np.float64(1.437487877739899), 'Gamma': np.float64(1.437487877739899)}
    (federated should land close to centralized without ever pooling raw data; DP-SGD
     costs a little more accuracy in exchange for the formal epsilon above on every
     operator's shared update -- see README 'Trust model' for what that epsilon means.)

==============================================================================
Done. See README.md for what's built vs. future work, and
major_project_final_definition.md (project root) for the full writeup.
==============================================================================
```
