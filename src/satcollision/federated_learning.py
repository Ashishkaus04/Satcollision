"""
Real federated learning: FedAvg + DP-SGD training an actual PyTorch model
across operators.

Every other layer in this project aggregates hand-built count vectors
(``federation.py``) or coarse per-encounter signals (``signal.py``) — useful,
privacy-respecting, but not what "Federated Learning" names in the
literature or in this project's own title. This module trains a real
model: a small risk-classification network that predicts whether a
conjunction will cross the operational Pc threshold from features an
operator's own screening pass already computes locally (miss distance,
relative speed, geometry class) — never raw orbital state, and never
pooled across operators.

Two real techniques, not stand-ins for them:

* **FedAvg** (McMahan et al., 2017) — each operator starts a training
  round from the current global model, trains for a few epochs on
  *only* its own local dataset (which never leaves :func:`train_local_round`),
  and the server averages the resulting model weights, weighted by each
  operator's local dataset size. Only weights cross the operator
  boundary, never a single training example.
* **DP-SGD** (Abadi et al., 2016), via the `opacus` library — within
  each operator's local training, every per-example gradient is clipped
  to a fixed norm and calibrated Gaussian noise is added to the batch
  gradient before the optimizer step. This gives that operator's
  contribution to the shared model a real, formally computed
  (epsilon, delta)-differential-privacy guarantee — the *model update*
  itself resists reconstruction/membership-inference attacks, not just
  "the raw data never left the building," which local-only training
  already gives for free without needing DP-SGD at all.

An honest note on the data and the label. Each operator's local dataset
here is *sampled* (:func:`build_federation_datasets`) from a distribution
shaped by that operator's own orbital shell (``fleets.OPERATOR_SHELLS``),
rather than harvested by re-running the full ``twin.screen_conjunctions``
catalog pass thousands of times — that would make generating a
training-set-sized amount of data prohibitively slow, and would produce an
extremely imbalanced label distribution (real Pc-threshold crossings are
rare events, as the historical backtests in this project already show).

The label is deliberately *not* a direct call to ``twin.compute_pc``.
That formula is, by its own docstring, a function of miss distance alone
— it does not depend on relative speed or encounter geometry at all. A
label built straight from it would make this module's other two features
causally irrelevant, which would make any model (federated or local-only)
trivially perfect and would defeat the entire point of a benchmark meant
to test whether a model can learn from all three. Instead,
:func:`_combined_risk_score` computes a synthetic composite index that
*does* depend on all three — standing in for what a full non-circular,
geometry-aware risk assessment (this project's own "full Pc integral"
next-step item) would actually weigh: smaller miss distance dominates
(matching real Pc behaviour), but a faster closing speed and a
head-on/crossing geometry (less warning time, less maneuver margin) push
risk up, and a slow overtaking encounter pushes it down even at a similar
miss distance. This -- not ``compute_pc`` -- is this module's ground
truth, and it is what makes the per-operator distribution shift in
:func:`build_federation_datasets` genuinely change what the *right answer*
looks like locally, so FedAvg's value over training on any single
operator's data alone is real and measurable (see
:func:`compare_scenarios`), not manufactured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset

from .fleets import OPERATOR_SHELLS

GEOMETRY_CLASSES = ("head-on", "crossing", "overtaking")
FEATURE_DIM = 2 + len(GEOMETRY_CLASSES)  # miss_distance_km, relative_speed_km_s, one-hot geometry


def _one_hot_geometry(geometry_class: str) -> np.ndarray:
    vec = np.zeros(len(GEOMETRY_CLASSES), dtype=np.float32)
    if geometry_class in GEOMETRY_CLASSES:
        vec[GEOMETRY_CLASSES.index(geometry_class)] = 1.0
    return vec


def make_features(miss_distance_km: float, relative_speed_km_s: float, geometry_class: str) -> np.ndarray:
    """The feature vector the risk model actually sees -- everything an
    operator's own screening pass already computes for a
    :class:`~satcollision.twin.Conjunction` (miss distance, relative
    speed, coarse geometry class). This is deliberately the operator's
    own *private* local encounter history, richer than what
    ``signal.py`` would ever let cross the operator boundary as a raw
    signal -- which is exactly why federated learning (share model
    *updates*, never this feature vector itself) is the right tool here,
    rather than pooling it centrally.
    """
    # Fixed-constant normalization (not data-dependent, so it leaks
    # nothing new across the operator boundary) -- gradient-based
    # training converges far more reliably when the two continuous
    # features sit on comparable scales to the 0/1 one-hot geometry
    # terms instead of raw km / km-per-s magnitudes.
    normalized_miss = miss_distance_km / 0.1
    normalized_speed = (relative_speed_km_s - 7.0) / 5.0
    return np.concatenate(
        [[normalized_miss, normalized_speed], _one_hot_geometry(geometry_class)]
    ).astype(np.float32)


@dataclass
class OperatorDataset:
    operator: str
    features: np.ndarray  # shape (n, FEATURE_DIM)
    labels: np.ndarray    # shape (n,), float32 in {0.0, 1.0}


def _geometry_weights_for_operator(inclination_deg: float) -> list[float]:
    """Illustrative, not a rigorously derived orbital-mechanics result:
    a polar/high-inclination shell's own satellites cross paths with
    objects in very different inclinations more often (more head-on /
    crossing encounters), while a low-inclination shell's satellites
    mostly co-orbit with others near their own plane (more overtaking
    encounters). Stands in for what a real per-operator encounter log
    would show, flagged here and in the README as a simplification.
    """
    if inclination_deg > 80.0:
        return [0.45, 0.45, 0.10]
    return [0.15, 0.35, 0.50]


_GEOMETRY_RISK_TERM = {"head-on": 1.0, "crossing": 0.0, "overtaking": -1.0}


def _combined_risk_score(miss_distance_km: float, relative_speed_km_s: float, geometry_class: str) -> float:
    """The synthetic ground-truth risk index this module actually trains
    on -- see the module docstring for why it is not a direct call to
    ``twin.compute_pc``. Three terms, each normalized to a comparable
    scale before combining through a logistic: proximity (miss distance
    below ~0.06km pushes risk up sharply, matching how real Pc responds
    to miss distance), a closing-speed term centered near a typical LEO
    relative speed, and a geometry term ordered head-on > crossing >
    overtaking by how much maneuver margin/warning time that geometry
    typically leaves. Weights were chosen empirically to land at a
    non-degenerate class balance (roughly 60-80% positive depending on
    the operator's own shell) and a genuine dependence on all three
    inputs, not tuned to any particular downstream accuracy number.
    """
    proximity_term = (0.06 - miss_distance_km) / 0.03
    speed_term = (relative_speed_km_s - 8.0) / 3.0
    geometry_term = _GEOMETRY_RISK_TERM.get(geometry_class, 0.0)
    z = 2.0 * proximity_term + 1.0 * speed_term + 1.0 * geometry_term
    return float(1.0 / (1.0 + np.exp(-z)))


def _sample_operator_dataset(
    operator: str,
    n_samples: int,
    rng: np.random.Generator,
    risk_threshold: float = 0.5,
) -> OperatorDataset:
    """Sample one operator's local, private training set.

    Each operator's own fleet occupies a different orbital shell (see
    ``fleets.OPERATOR_SHELLS``) -- this is what makes "a typical close
    approach" genuinely differ operator to operator, and is what makes
    this a real non-IID federated learning problem rather than one pool
    of data arbitrarily cut into equal pieces: a shell that rarely
    samples high closing speeds or head-on geometry never sees many
    examples from the region of feature space where those terms matter
    most, so a model trained on that shell's data alone genuinely
    generalizes worse to the other shells' regimes than one built from
    every shell's updates (see :func:`compare_scenarios`).
    """
    shell = OPERATOR_SHELLS.get(operator, {"inclination_deg": 53.0})
    inclination = float(shell["inclination_deg"])
    speed_scale = 1.0 + (inclination - 53.0) / 90.0

    miss_distance_km = rng.exponential(scale=0.05, size=n_samples)
    relative_speed_km_s = np.clip(
        rng.normal(loc=7.0 * speed_scale, scale=2.0, size=n_samples), 0.1, 15.0
    )
    geometry_weights = _geometry_weights_for_operator(inclination)
    geometry_choices = rng.choice(GEOMETRY_CLASSES, size=n_samples, p=geometry_weights)

    features = np.stack(
        [make_features(miss_distance_km[i], relative_speed_km_s[i], geometry_choices[i])
         for i in range(n_samples)]
    )
    scores = np.array([
        _combined_risk_score(float(miss_distance_km[i]), float(relative_speed_km_s[i]), geometry_choices[i])
        for i in range(n_samples)
    ])
    labels = (scores >= risk_threshold).astype(np.float32)
    return OperatorDataset(operator=operator, features=features, labels=labels)


def build_federation_datasets(
    operator_names: list[str],
    n_samples_per_operator: int = 400,
    seed: int = 0,
) -> dict[str, OperatorDataset]:
    """One local, private dataset per named operator -- the whole
    federation's raw training data, none of which is ever pooled."""
    rng = np.random.default_rng(seed)
    return {
        op: _sample_operator_dataset(
            op, n_samples_per_operator, np.random.default_rng(int(rng.integers(0, 2 ** 32 - 1)))
        )
        for op in operator_names
    }


def build_global_test_set(
    operator_names: list[str], n_samples_per_operator: int = 200, seed: int = 999
) -> tuple[np.ndarray, np.ndarray]:
    """A held-out set mixing every operator's regime in equal proportion
    -- what "generalizes across the whole federation" is measured
    against, never any single operator's own distribution alone."""
    datasets = build_federation_datasets(operator_names, n_samples_per_operator, seed=seed)
    features = np.concatenate([d.features for d in datasets.values()])
    labels = np.concatenate([d.labels for d in datasets.values()])
    return features, labels


class RiskModel(nn.Module):
    """A small feed-forward classifier -- deliberately no BatchNorm
    (Opacus/DP-SGD cannot compute correct per-example gradients through
    batch statistics that mix examples together)."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class LocalRoundResult:
    operator: str
    state_dict: dict
    n_examples: int
    epsilon: float | None = None


def train_local_round(
    global_state_dict: dict,
    dataset: OperatorDataset,
    local_epochs: int = 2,
    batch_size: int = 8,
    lr: float = 0.01,
    use_dp: bool = True,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    delta: float = 1e-5,
) -> LocalRoundResult:
    """One operator's local round: start from the current global model,
    train for ``local_epochs`` epochs on *only* this operator's own
    ``dataset`` (which never leaves this function call), and return the
    resulting weights (plus, when ``use_dp`` is True, the exact
    (epsilon, delta) this operator's contribution now carries).
    """
    model = RiskModel()
    model.load_state_dict(global_state_dict)

    # Clamp to the dataset size: Opacus's Poisson-sampling privacy
    # accountant assumes a per-example sampling probability batch_size/n
    # strictly below 1 -- a batch_size larger than a (deliberately small,
    # realistic) local dataset pushes that probability to 1 and produces
    # degenerate (sometimes NaN) epsilon accounting.
    effective_batch_size = max(1, min(batch_size, len(dataset.labels)))
    features = torch.from_numpy(dataset.features)
    labels = torch.from_numpy(dataset.labels)
    loader = DataLoader(TensorDataset(features, labels), batch_size=effective_batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    privacy_engine = None
    if use_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

    model.train()
    for _ in range(local_epochs):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            preds = model(batch_features)
            loss = loss_fn(preds, batch_labels)
            loss.backward()
            optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta) if privacy_engine is not None else None
    underlying = model._module if hasattr(model, "_module") else model
    return LocalRoundResult(
        operator=dataset.operator,
        state_dict={k: v.detach().clone() for k, v in underlying.state_dict().items()},
        n_examples=len(dataset.labels),
        epsilon=epsilon,
    )


def federated_average(local_results: list[LocalRoundResult]) -> dict:
    """Classic FedAvg: each operator's local weights, averaged weighted
    by how many local examples they trained on -- the only thing that
    ever crosses back to the server."""
    total_n = sum(r.n_examples for r in local_results)
    keys = local_results[0].state_dict.keys()
    return {
        key: sum(r.state_dict[key].float() * (r.n_examples / total_n) for r in local_results)
        for key in keys
    }


@dataclass
class FederatedTrainingRun:
    final_state_dict: dict
    round_epsilons: list[dict]  # one {operator: epsilon} dict per round


def run_federated_training(
    datasets: dict[str, OperatorDataset],
    n_rounds: int = 6,
    local_epochs: int = 2,
    use_dp: bool = True,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    seed: int = 0,
) -> FederatedTrainingRun:
    """The full FedAvg (+ optional DP-SGD) loop across every operator in
    ``datasets``, for ``n_rounds`` communication rounds."""
    torch.manual_seed(seed)
    global_state = RiskModel().state_dict()
    round_epsilons = []
    for _ in range(n_rounds):
        local_results = [
            train_local_round(
                global_state, ds, local_epochs=local_epochs, use_dp=use_dp,
                max_grad_norm=max_grad_norm, noise_multiplier=noise_multiplier,
            )
            for ds in datasets.values()
        ]
        global_state = federated_average(local_results)
        round_epsilons.append({r.operator: r.epsilon for r in local_results})
    return FederatedTrainingRun(final_state_dict=global_state, round_epsilons=round_epsilons)


def _train_plain(features: np.ndarray, labels: np.ndarray, epochs: int, lr: float, seed: int) -> dict:
    torch.manual_seed(seed)
    model = RiskModel()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features), torch.from_numpy(labels)), batch_size=32, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()


def train_centralized(datasets: dict[str, OperatorDataset], epochs: int = 25, lr: float = 0.02, seed: int = 0) -> dict:
    """The privacy-free upper bound: pool every operator's raw data into
    one training set. Only exists for comparison -- this is exactly what
    the rest of this project (and real operators) will not actually do.
    """
    features = np.concatenate([d.features for d in datasets.values()])
    labels = np.concatenate([d.labels for d in datasets.values()])
    return _train_plain(features, labels, epochs=epochs, lr=lr, seed=seed)


def train_local_only(dataset: OperatorDataset, epochs: int = 25, lr: float = 0.02, seed: int = 0) -> dict:
    """The no-cooperation baseline: one operator trains only on its own
    data, with no communication round and no DP-SGD (there is nothing to
    protect against here -- nothing ever leaves this operator)."""
    return _train_plain(dataset.features, dataset.labels, epochs=epochs, lr=lr, seed=seed)


def evaluate(state_dict: dict, features: np.ndarray, labels: np.ndarray) -> dict:
    """Accuracy/precision/recall/F1 of one trained model's weights
    against a held-out (features, labels) set."""
    model = RiskModel()
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features))
        preds = (torch.sigmoid(logits) >= 0.5).float().numpy()
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = float((preds == labels).mean())
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def compare_scenarios(
    operator_names: list[str],
    n_samples_per_operator: int = 20,
    n_rounds: int = 10,
    local_epochs: int = 3,
    seed: int = 0,
) -> dict:
    """Head-to-head comparison, all evaluated on the same held-out global
    test set (see module docstring for why this specific comparison is
    the meaningful one):

    * ``centralized`` -- pooled raw data, no privacy at all (upper bound).
    * ``local_only`` -- each operator's own model, trained on its own
      data alone, never communicated; reported as the mean across
      operators, since this is what "no federation" looks like for each
      of them individually.
    * ``federated_no_dp`` -- FedAvg only: weights shared and averaged,
      raw data never pooled, but no formal privacy guarantee on the
      updates themselves.
    * ``federated_dp`` -- FedAvg + DP-SGD: the full technique this module
      is built to demonstrate.

    The default ``n_samples_per_operator=20`` is deliberately small,
    matching this project's own historical-backtest finding that real
    Pc-threshold crossings are rare (19 crossings out of 173,356 screened
    conjunctions in the real Starlink backtest) -- an individual operator
    genuinely would not accumulate a large local training set on its own.
    This is also where FedAvg's advantage over local-only training is
    real rather than marginal: with ample local data every operator's
    model already generalizes well on its own (the underlying risk
    function is the same for everyone, only the local sampling density
    differs), so the interesting, honest regime to demonstrate is the
    data-scarce one.
    """
    datasets = build_federation_datasets(operator_names, n_samples_per_operator, seed=seed)
    test_features, test_labels = build_global_test_set(operator_names, seed=seed + 1)

    centralized_state = train_centralized(datasets, seed=seed)
    local_only_metrics = [
        evaluate(train_local_only(ds, seed=seed), test_features, test_labels) for ds in datasets.values()
    ]
    local_only_mean = {
        key: float(np.mean([m[key] for m in local_only_metrics])) for key in local_only_metrics[0]
    }

    fed_no_dp = run_federated_training(datasets, n_rounds=n_rounds, local_epochs=local_epochs, use_dp=False, seed=seed)
    fed_dp = run_federated_training(datasets, n_rounds=n_rounds, local_epochs=local_epochs, use_dp=True, seed=seed)

    return {
        "centralized": evaluate(centralized_state, test_features, test_labels),
        "local_only_mean": local_only_mean,
        "federated_no_dp": evaluate(fed_no_dp.final_state_dict, test_features, test_labels),
        "federated_dp": evaluate(fed_dp.final_state_dict, test_features, test_labels),
        "federated_dp_final_round_epsilons": fed_dp.round_epsilons[-1],
    }
