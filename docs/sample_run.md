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

[3] Abstracting the flagged conjunction into shareable signals...
    signal from A: AbstractedSignal(encounter_id='ENC-90000-99000-1800', sender_operator='Alpha', local_object_label='Alpha-90000', tca_offset_s=1800.0, geometry_class='crossing', severity_tier='threshold', threshold_crossed=True)
    signal from B: AbstractedSignal(encounter_id='ENC-90000-99000-1800', sender_operator='Beta', local_object_label='Beta-99000', tca_offset_s=1800.0, geometry_class='crossing', severity_tier='threshold', threshold_crossed=True)
    (no position/velocity/orbital-element field exists on this object —
     see signal.assert_no_raw_state, exercised in tests/test_twin_and_signal.py)

[4] Federation aggregation: 5 honest operators + 1 adversary reporting
    period counts (threshold/elevated/critical encounters)...
    plain mean   : [11.66666667 10.83333333 11.66666667]   error vs honest truth = 13.725
    trimmed mean : [4.   3.25 4.5 ]   error = 0.783
    Krum         : [4. 4. 4.]   error = 1.265

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

==============================================================================
Done. See README.md for what's built vs. future work, and
major_project_final_definition.md (project root) for the full writeup.
==============================================================================
```
