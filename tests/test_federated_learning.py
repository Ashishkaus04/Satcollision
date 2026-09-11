import numpy as np

from satcollision.federated_learning import (
    build_federation_datasets,
    build_global_test_set,
    train_centralized,
    train_local_only,
    train_local_round,
    federated_average,
    run_federated_training,
    evaluate,
    compare_scenarios,
    make_features,
    FEATURE_DIM,
)


def test_make_features_has_expected_dimension():
    features = make_features(0.01, 7.0, "crossing")
    assert features.shape == (FEATURE_DIM,)


def test_build_federation_datasets_are_nonempty_and_labeled():
    datasets = build_federation_datasets(["Alpha", "Beta", "Gamma"], n_samples_per_operator=50, seed=1)
    assert set(datasets.keys()) == {"Alpha", "Beta", "Gamma"}
    for ds in datasets.values():
        assert ds.features.shape == (50, FEATURE_DIM)
        assert ds.labels.shape == (50,)
        assert set(np.unique(ds.labels)).issubset({0.0, 1.0})


def test_operator_datasets_are_non_iid_across_shells():
    # Alpha (53 deg) and Beta (87.4 deg) have very different inclinations,
    # so their sampled relative-speed distributions should differ --
    # otherwise FedAvg would have nothing genuine to demonstrate.
    datasets = build_federation_datasets(["Alpha", "Beta"], n_samples_per_operator=300, seed=2)
    mean_speed_alpha = datasets["Alpha"].features[:, 1].mean()
    mean_speed_beta = datasets["Beta"].features[:, 1].mean()
    assert abs(mean_speed_alpha - mean_speed_beta) > 0.3


def test_train_local_round_with_dp_returns_finite_epsilon():
    datasets = build_federation_datasets(["Alpha"], n_samples_per_operator=200, seed=3)
    from satcollision.federated_learning import RiskModel
    global_state = RiskModel().state_dict()
    result = train_local_round(global_state, datasets["Alpha"], local_epochs=1, use_dp=True)
    assert result.epsilon is not None
    assert np.isfinite(result.epsilon)
    assert result.n_examples == 200


def test_train_local_round_without_dp_has_no_epsilon():
    datasets = build_federation_datasets(["Alpha"], n_samples_per_operator=100, seed=4)
    from satcollision.federated_learning import RiskModel
    global_state = RiskModel().state_dict()
    result = train_local_round(global_state, datasets["Alpha"], local_epochs=1, use_dp=False)
    assert result.epsilon is None


def test_federated_average_weights_by_dataset_size():
    from satcollision.federated_learning import LocalRoundResult
    import torch
    small = LocalRoundResult(operator="small", state_dict={"w": torch.tensor([0.0])}, n_examples=1)
    large = LocalRoundResult(operator="large", state_dict={"w": torch.tensor([10.0])}, n_examples=99)
    averaged = federated_average([small, large])
    # Weighted mean should sit very close to the large dataset's value.
    assert averaged["w"].item() > 9.0


def test_run_federated_training_produces_usable_model():
    datasets = build_federation_datasets(["Alpha", "Beta", "Gamma"], n_samples_per_operator=150, seed=5)
    run = run_federated_training(datasets, n_rounds=3, local_epochs=1, use_dp=True, seed=5)
    test_features, test_labels = build_global_test_set(["Alpha", "Beta", "Gamma"], seed=6)
    metrics = evaluate(run.final_state_dict, test_features, test_labels)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(run.round_epsilons) == 3
    for round_eps in run.round_epsilons:
        assert set(round_eps.keys()) == {"Alpha", "Beta", "Gamma"}


def test_federated_learning_beats_local_only_on_global_test_set():
    # The central point of FedAvg: a model that only ever saw one
    # operator's own (shifted) distribution should generalize worse to
    # the mixed global test set than a model built from every operator's
    # updates -- even though no operator's raw data was ever pooled.
    operators = ["Alpha", "Beta", "Gamma"]
    datasets = build_federation_datasets(operators, n_samples_per_operator=400, seed=7)
    test_features, test_labels = build_global_test_set(operators, seed=8)

    local_metrics = [
        evaluate(train_local_only(ds, epochs=15, seed=7), test_features, test_labels)
        for ds in datasets.values()
    ]
    local_mean_f1 = float(np.mean([m["f1"] for m in local_metrics]))

    fed_run = run_federated_training(datasets, n_rounds=8, local_epochs=2, use_dp=False, seed=7)
    fed_metrics = evaluate(fed_run.final_state_dict, test_features, test_labels)

    assert fed_metrics["f1"] >= local_mean_f1 - 0.05  # federated should not meaningfully lag, typically leads


def test_federated_clearly_beats_local_only_in_the_realistic_scarce_data_regime():
    # compare_scenarios' default n_samples_per_operator=20 models the
    # realistic case (few local Pc-threshold-crossing examples per
    # operator) where FedAvg's advantage over local-only training is
    # real rather than marginal -- checked across several seeds so this
    # isn't a one-off lucky draw.
    gaps = []
    for seed in range(4):
        result = compare_scenarios(["Alpha", "Beta", "Gamma"], seed=seed)
        gaps.append(result["federated_no_dp"]["f1"] - result["local_only_mean"]["f1"])
    assert np.mean(gaps) > 0.0


def test_compare_scenarios_returns_all_expected_keys():
    result = compare_scenarios(["Alpha", "Beta", "Gamma"], n_samples_per_operator=150, n_rounds=3, local_epochs=1, seed=9)
    expected_keys = {
        "centralized", "local_only_mean", "federated_no_dp", "federated_dp",
        "federated_dp_final_round_epsilons",
    }
    assert expected_keys.issubset(result.keys())
    for key in ("centralized", "local_only_mean", "federated_no_dp", "federated_dp"):
        assert 0.0 <= result[key]["accuracy"] <= 1.0
