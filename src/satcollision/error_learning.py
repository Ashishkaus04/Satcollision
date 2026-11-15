r"""
Learning the public catalogue's error, per operator and federated.

Every operator knows its own satellites far better than the public TLE
catalogue does (GPS, its own orbit determination). So every operator can
measure, for its own fleet, how wrong the public prediction is: propagate
the newest public element set, compare it with its own ephemeris, and
record the radial / along-track / cross-track (RTN) error together with
what anyone can read from the element set itself (age at the time of
interest, altitude, inclination, eccentricity, drag term, how often the
object's sets are published). That is private training data: the labels
expose the operator's precise orbits.

A model of this error is what a twin needs to screen *another* operator's
satellite honestly: the mean error is a correction to the public
prediction and the spread is the covariance for the Pc. One operator's
fleet says little about another's (different altitudes, drag, manoeuvre
habits), so the small operators gain most from learning together. This
module trains the same small network three ways:

* **local-only**: each operator alone on its own fleet;
* **federated** (FedAvg, McMahan et al. 2017): each round, every operator
  trains a few epochs from the current global model on its own data and
  sends only its model weights; the federation averages them, weighted by
  sample count. Optionally the weighted sum goes through
  ``secure_aggregation`` (pairwise masks, so the aggregator sees only the
  sum) and each update can be clipped and noised (DP-FedAvg);
* **central**: one model on everybody's data pooled (the privacy-violating
  upper bound).

The network predicts, per RTN axis, a location mu and a log-scale of a
Student-t with 3 degrees of freedom (heavy tails: manoeuvres between sets
are common and must not inflate the spread of every prediction). Targets
and outputs are normalised by a fixed reference scale per axis that grows
with age (not fitted from data, so no statistics need to be shared).
For the Pc the scale is converted to a robust-sigma equivalent
(``T3_ROBUST_SIGMA`` x scale, the 1.4826 x MAD of a t3), the same kind of
spread the public error table (``tle_history.ErrorModel``) uses.

In the hindcast, the truth an operator compares against is the blended
nearest-epoch TLE ephemeris (``tle_history``), so labels are TLE-vs-later
-TLE consistency, not a real GPS ephemeris; see the paper's limitations.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .hindcast import propagate_pairs, set_features, truth_pairs, truth_weights
from .tle_history import SECONDS_PER_DAY, TLEHistory, to_rtn

NU = 3.0
T3_ROBUST_SIGMA = 1.4826 * 0.7649      # robust sigma of a unit-scale t3
FEATURE_NAMES = ["log_age", "age", "alt", "incl", "ecc", "bstar", "ndot", "cadence", "perigee"]
FEATURE_DIM = len(FEATURE_NAMES)
MAX_AGE_H = 72.0


def reference_scale(age_h: np.ndarray) -> np.ndarray:
    """(n, 3) fixed per-axis normalisation (km): roughly the size of the
    error at that age, so the network works on O(1) numbers."""
    a = np.asarray(age_h, float)[:, None]
    return np.concatenate([0.10 + 0.02 * a, 0.5 + 0.8 * a, 0.05 + 0.006 * a], axis=1)


def cadence(hist: TLEHistory, sets: np.ndarray, window_days: float = 3.0) -> np.ndarray:
    """Element sets per day this object had in the ``window_days`` before
    each set's epoch (published information)."""
    sets = np.asarray(sets, np.int64)
    rows = hist.row_of_set[sets]
    ep = hist.epochs[sets]
    pos = np.searchsorted(hist._keys, rows * hist._span + (ep - window_days - hist.jd0), side="left")
    return (sets - np.maximum(pos, hist.start[rows])) / window_days


def features(hist: TLEHistory, sets: np.ndarray, age_h: np.ndarray, _cache: Optional[dict] = None) -> np.ndarray:
    """(n, FEATURE_DIM) model inputs, fixed scaling (no data statistics)."""
    sets = np.asarray(sets, np.int64)
    uniq, inv = np.unique(sets, return_inverse=True)
    f = set_features(hist, uniq)
    cad = cadence(hist, uniq)
    age = np.clip(np.asarray(age_h, float), 0.0, 2 * MAX_AGE_H)
    bs = f["bstar"][inv]
    X = np.stack([
        np.log(age + 0.1) / math.log(MAX_AGE_H),
        age / MAX_AGE_H,
        (f["alt_km"][inv] - 550.0) / 300.0,
        f["incl_deg"][inv] / 90.0 - 1.0,
        np.clip(f["ecc"][inv] * 100.0, 0, 5),
        np.sign(bs) * np.log10(1.0 + np.abs(bs) * 1e5) / 2.0,
        np.clip(f["ndot"][inv] * 1e3, -5, 5),
        np.clip(cad[inv] / 5.0, 0, 3),
        (f["perigee_km"][inv] - 550.0) / 300.0,
    ], axis=1)
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# Training data: an operator's own fleet, public prediction vs own ephemeris
# ---------------------------------------------------------------------------

@dataclass
class ErrorSamples:
    X: np.ndarray          # (n, FEATURE_DIM)
    y: np.ndarray          # (n, 3) RTN error, km (prediction - truth)
    age_h: np.ndarray
    row: np.ndarray        # object row (for per-object splits)

    def __len__(self):
        return len(self.y)

    def subset(self, m):
        return ErrorSamples(self.X[m], self.y[m], self.age_h[m], self.row[m])

    @staticmethod
    def concat(parts: Sequence["ErrorSamples"]) -> "ErrorSamples":
        return ErrorSamples(np.concatenate([p.X for p in parts]), np.concatenate([p.y for p in parts]),
                            np.concatenate([p.age_h for p in parts]), np.concatenate([p.row for p in parts]))


def build_samples(hist: TLEHistory, rows: Sequence[int], jd_lo: float, jd_hi: float,
                  ages_per_set: int = 6, seed: int = 0, blend_s: float = 600.0,
                  max_samples: Optional[int] = None) -> ErrorSamples:
    """Errors of the public prediction of ``rows`` at times in [jd_lo, jd_hi):
    for every set of those objects, ``ages_per_set`` random ages (log-uniform,
    6 min to 72 h), kept if the time falls in the window and the set is still
    the newest public one *or older* (any set is a valid public prediction of
    its age: an operator screening at an update uses whatever set is newest
    then, and that set ages until TCA)."""
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows, np.int64)
    sets = np.concatenate([np.arange(hist.start[r], hist.start[r + 1]) for r in rows]) if len(rows) else np.zeros(0, np.int64)
    if sets.size == 0:
        return ErrorSamples(np.zeros((0, FEATURE_DIM), np.float32), np.zeros((0, 3)), np.zeros(0), np.zeros(0, np.int64))
    s = np.repeat(sets, ages_per_set)
    age = np.exp(rng.uniform(math.log(0.1), math.log(MAX_AGE_H), len(s)))
    t = hist.epochs[s] + age / 24.0
    keep = (t >= jd_lo) & (t < jd_hi)
    s, age, t = s[keep], age[keep], t[keep]
    # The truth at t is built from sets published after t: keep only labels
    # whose truth sets were also published before jd_hi (no look-ahead).
    ta, tb, _ = truth_weights(hist, hist.row_of_set[s], t, blend_s)
    keep = (hist.epochs[ta] < jd_hi) & (hist.epochs[tb] < jd_hi) & (ta != s) & (tb != s)
    s, age, t = s[keep], age[keep], t[keep]
    if max_samples is not None and len(s) > max_samples:
        pick = np.sort(rng.choice(len(s), max_samples, replace=False))
        s, age, t = s[pick], age[pick], t[pick]
    r_row = hist.row_of_set[s]
    r_pred, _, ok1 = propagate_pairs(hist, s, t)
    r_true, v_true, ok2 = truth_pairs(hist, r_row, t, blend_s)
    ok = ok1 & ok2
    err = to_rtn(r_pred[ok] - r_true[ok], r_true[ok], v_true[ok])
    good = np.all(np.isfinite(err), axis=1) & (np.abs(err).max(axis=1) < 5000.0)
    s, age, r_row, err = s[ok][good], age[ok][good], r_row[ok][good], err[good]
    return ErrorSamples(features(hist, s, age), err, age, r_row)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class ErrorNet(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 32):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.mu = nn.Linear(hidden, 3)
        self.log_s = nn.Linear(hidden, 3)
        with torch.no_grad():
            self.mu.weight.mul_(0.1); self.mu.bias.zero_()
            self.log_s.weight.mul_(0.1); self.log_s.bias.fill_(-1.0)

    def forward(self, x):
        h = self.body(x)
        return self.mu(h), self.log_s(h).clamp(-8.0, 4.0)


def t_nll(mu, log_s, y):
    z = (y - mu) / torch.exp(log_s)
    return (log_s + 0.5 * (NU + 1.0) * torch.log1p(z * z / NU)).mean()


def _tensors(s: ErrorSamples):
    ref = reference_scale(s.age_h)
    return torch.from_numpy(s.X), torch.from_numpy((s.y / ref).astype(np.float32))


def train_epochs(model: ErrorNet, data: ErrorSamples, epochs: int, lr: float = 3e-3, batch: int = 1024,
                 seed: int = 0, max_steps: Optional[int] = None, opt=None) -> ErrorNet:
    if len(data) == 0 or epochs <= 0:
        return model
    X, Y = _tensors(data)
    g = torch.Generator().manual_seed(seed)
    if opt is None:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    steps = 0
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), batch):
            if max_steps is not None and steps >= max_steps:
                return model
            steps += 1
            idx = perm[i:i + batch]
            mu, ls = model(X[idx])
            loss = t_nll(mu, ls, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def flat(model: nn.Module) -> np.ndarray:
    return torch.cat([p.detach().reshape(-1) for p in model.state_dict().values()]).numpy().astype(np.float64)


def load_flat(model: nn.Module, vec: np.ndarray) -> nn.Module:
    sd = model.state_dict()
    i = 0
    new = {}
    for k, p in sd.items():
        n = p.numel()
        new[k] = torch.from_numpy(vec[i:i + n].astype(np.float32)).reshape(p.shape)
        i += n
    model.load_state_dict(new)
    return model


def new_model(seed: int = 0) -> ErrorNet:
    torch.manual_seed(seed)
    return ErrorNet()


def split_by_object(data: ErrorSamples, frac: float = 0.2, seed: int = 0):
    """(train, validation), whole objects to one side: several samples of
    one element set share most of their error, so a random split would let
    the model memorise sets and look better than it is."""
    if len(data) == 0:
        return data, data
    h = (data.row.astype(np.uint64) * np.uint64(2654435761) + np.uint64(seed * 97 + 13)) % np.uint64(1000)
    val = h < np.uint64(int(frac * 1000))
    if val.all() or not val.any():
        val = np.zeros(len(data), bool)
        val[::5] = True
    return data.subset(~val), data.subset(val)


@torch.no_grad()
def val_nll(model: ErrorNet, data: ErrorSamples) -> float:
    if len(data) == 0:
        return float("nan")
    X, Y = _tensors(data)
    model.eval()
    mu, ls = model(X)
    return float(t_nll(mu, ls, Y))


def train_local(data: ErrorSamples, epochs: int = 30, seed: int = 0, lr: float = 3e-3,
                patience: int = 4, val_frac: float = 0.2) -> ErrorNet:
    """Train on one dataset, keeping the epoch with the best NLL on a
    held-out 20% of its objects (early stopping)."""
    m = new_model(seed)
    tr, va = split_by_object(data, val_frac, seed)
    if len(va) == 0 or len(tr) == 0:
        return train_epochs(m, data, epochs, lr=lr, seed=seed)
    best, best_state, bad = val_nll(m, va), copy.deepcopy(m.state_dict()), 0
    for e in range(epochs):
        train_epochs(m, tr, 1, lr=lr, seed=seed * 7919 + e)
        v = val_nll(m, va)
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.best_val_nll = best
    return m


@dataclass
class FedAvgLog:
    rounds: int = 0
    bytes_per_update: int = 0
    secure: bool = False
    dp: Optional[dict] = None
    history: List[dict] = field(default_factory=list)


def fedavg(datasets: Dict[str, ErrorSamples], rounds: int = 30, local_epochs: int = 1, seed: int = 0,
           lr: float = 1e-3, secure: bool = False, clip: Optional[float] = None,
           noise_multiplier: float = 0.0, weighting: str = "samples",
           eval_fn=None, patience: int = 4, val_frac: float = 0.2,
           local_steps: Optional[int] = None, eval_every: int = 1) -> tuple:
    """FedAvg over the operators in ``datasets``. Only flattened model
    weights (or, with ``secure``, pairwise-masked shares of the weighted
    weights) leave an operator. ``clip``/``noise_multiplier``: clip each
    operator's update (delta from the global model) to L2 norm ``clip`` and
    add Gaussian noise of std ``noise_multiplier * clip`` to it before
    sending (local DP on the update, per round).

    ``weighting``: "samples" (classic FedAvg) or "sqrt" (sqrt of sample
    count, so one dominant fleet does not drown the others)."""
    ops = sorted(k for k, d in datasets.items() if len(d) > 0)
    # Each operator holds out 20% of its own objects; after every round it
    # reports one number, its validation NLL (and the count), and the
    # federation keeps the round with the best weighted mean (early stopping).
    splits = {op: split_by_object(datasets[op], val_frac, seed) for op in ops}
    datasets = {op: splits[op][0] for op in ops}
    vals = {op: splits[op][1] for op in ops}
    glob = new_model(seed)
    best_v, best_state, bad = float("inf"), copy.deepcopy(glob.state_dict()), 0
    locals_ = {op: copy.deepcopy(glob) for op in ops}
    opts = {op: torch.optim.AdamW(locals_[op].parameters(), lr=lr, weight_decay=1e-4) for op in ops}
    log = FedAvgLog(rounds=rounds, secure=secure,
                    dp=None if clip is None else {"clip": clip, "noise_multiplier": noise_multiplier})
    rng = np.random.default_rng(seed + 991)
    mpc = None
    if secure:
        from .secure_aggregation import build_mpc_states
        mpc = build_mpc_states(ops)
    for r in range(rounds):
        g_vec = flat(glob)
        n = {op: len(datasets[op]) for op in ops}
        w = {op: (float(n[op]) if weighting == "samples" else math.sqrt(n[op])) for op in ops}
        updates = {}
        for k, op in enumerate(ops):
            # Each operator keeps its own model object and optimizer state
            # (private, never sent) and starts every round from the global
            # weights; resetting Adam every round makes short rounds noisy.
            local = locals_[op]
            local.load_state_dict(glob.state_dict())
            train_epochs(local, datasets[op], local_epochs if local_steps is None else 10 ** 6,
                         lr=lr, seed=seed * 1000 + r * 31 + k, max_steps=local_steps, opt=opts[op])
            delta = flat(local) - g_vec
            if clip is not None:
                norm = float(np.linalg.norm(delta))
                delta = delta * min(1.0, clip / max(norm, 1e-12))
                if noise_multiplier > 0:
                    delta = delta + rng.normal(0.0, noise_multiplier * clip, delta.shape)
            updates[op] = delta
        log.bytes_per_update = int(g_vec.size * 4)
        wsum = sum(w.values())
        if secure:
            from .federation import OperatorReport
            from .secure_aggregation import secure_sum_aggregate
            reports = {op: OperatorReport(operator=op, counts=updates[op] * (w[op] / wsum)) for op in ops}
            agg = secure_sum_aggregate(mpc, reports)
        else:
            agg = sum(updates[op] * (w[op] / wsum) for op in ops)
        load_flat(glob, g_vec + agg)
        if (r + 1) % eval_every and r + 1 < rounds:
            continue
        vn = {op: val_nll(glob, vals[op]) for op in ops if len(vals[op])}
        v = (sum(vn[op] * len(vals[op]) for op in vn) / max(sum(len(vals[op]) for op in vn), 1)) if vn else float("nan")
        entry = {"round": r + 1, "val_nll": v}
        if eval_fn is not None:
            entry.update(eval_fn(glob))
        log.history.append(entry)
        if not math.isfinite(v) or v < best_v - 1e-4:
            best_v, best_state, bad = (v if math.isfinite(v) else best_v), copy.deepcopy(glob.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    glob.load_state_dict(best_state)
    log.best_round = int(log.history[int(np.argmin([h["val_nll"] for h in log.history]))]["round"]) if log.history else 0
    return glob, log


# ---------------------------------------------------------------------------
# Using a model
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model: ErrorNet, X: np.ndarray, age_h: np.ndarray) -> tuple:
    """(bias km (n,3), robust sigma km (n,3)) of the public prediction error."""
    model.eval()
    ref = reference_scale(age_h)
    out_mu, out_s = [], []
    for i in range(0, len(X), 65536):
        mu, ls = model(torch.from_numpy(np.ascontiguousarray(X[i:i + 65536])))
        out_mu.append(mu.numpy()); out_s.append(np.exp(ls.numpy()))
    mu = np.concatenate(out_mu) if out_mu else np.zeros((0, 3))
    s = np.concatenate(out_s) if out_s else np.zeros((0, 3))
    return mu * ref, s * ref * T3_ROBUST_SIGMA


@torch.no_grad()
def evaluate(model: ErrorNet, data: ErrorSamples) -> dict:
    """Held-out NLL (normalised units), coverage of the robust 1- and 2-sigma
    intervals per axis (a calibrated model: about 0.68 and 0.95 of the core,
    a little less overall because of the tails), and the median absolute
    error of the bias-corrected prediction along-track."""
    if len(data) == 0:
        return {"n": 0}
    X, Y = _tensors(data)
    model.eval()
    mu, ls = model(X)
    nll = float(t_nll(mu, ls, Y))
    b, s = predict(model, data.X, data.age_h)
    z = np.abs(data.y - b) / s
    return {"n": int(len(data)), "nll": nll,
            "cover1": [float((z[:, k] < 1).mean()) for k in range(3)],
            "cover2": [float((z[:, k] < 2).mean()) for k in range(3)],
            "mae_t_raw_km": float(np.median(np.abs(data.y[:, 1]))),
            "mae_t_corrected_km": float(np.median(np.abs(data.y[:, 1] - b[:, 1])))}


def table_nll(table_sigma: np.ndarray, table_bias: np.ndarray, data: ErrorSamples) -> dict:
    """The same metrics for a fixed table (sigma, bias per sample), so the
    public error table can be compared with the networks on equal terms."""
    ref = reference_scale(data.age_h)
    scale = table_sigma / T3_ROBUST_SIGMA / ref
    z = (data.y - table_bias) / ref / scale
    nll = float(np.mean(np.log(scale) + 0.5 * (NU + 1) * np.log1p(z * z / NU)))
    zz = np.abs(data.y - table_bias) / table_sigma
    return {"n": int(len(data)), "nll": nll,
            "cover1": [float((zz[:, k] < 1).mean()) for k in range(3)],
            "cover2": [float((zz[:, k] < 2).mean()) for k in range(3)],
            "mae_t_raw_km": float(np.median(np.abs(data.y[:, 1]))),
            "mae_t_corrected_km": float(np.median(np.abs(data.y[:, 1] - table_bias[:, 1])))}


def gaussian_rdp_epsilon(noise_multiplier: float, rounds: int, delta: float = 1e-5) -> float:
    """(epsilon, delta) of ``rounds`` Gaussian mechanisms with noise
    ``noise_multiplier`` x sensitivity, by Renyi-DP composition (Mironov
    2017): eps = min_a rounds * a / (2 s^2) + log(1/delta) / (a - 1).
    The unit protected is an operator's whole update (operator-level DP),
    not one satellite or one element set."""
    if noise_multiplier <= 0:
        return float("inf")
    a = np.linspace(1.01, 512, 50000)
    return float(np.min(rounds * a / (2 * noise_multiplier ** 2) + math.log(1 / delta) / (a - 1)))
