"""What does a shared owner model reveal? Attacks by a rival operator.

    PYTHONPATH=src python research/owner_model_privacy.py <gp_history files...>

Each member shares one model of its own fleet's public-catalogue error
(error_learning.OwnerModels). A rival holds that model and the public
catalogue. Three questions, each for the model trained without DP and with
DP-SGD at epsilon 8 and 2:

(a) Position leak. Does the model help the rival locate the owner's
    satellites better than public data does? Measured on held-out errors
    after the split: median along-track error of the public prediction,
    and after subtracting the owner model's predicted bias, against the same
    with the public model's bias (trained only on public data).
(b) Membership inference (loss threshold, Yeom et al. 2018). The owner trains
    on 80% of its satellites (by object); can the rival tell which satellites
    were in the private training set from the model's loss on their public
    errors? Reported as attack AUC, per sample and per satellite (mean loss
    over the satellite's samples); 0.5 = no better than guessing.
(c) Fleet habits. The model's predicted along-track spread at 48 h, per
    operator, against the manoeuvre rate (jumps per satellite per day) that
    anyone can read from the public catalogue: if the model's fleet-level
    signal tracks a public statistic, it reveals nothing new at that level.

Writes research/results/owner_model_privacy.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satcollision import error_learning as el  # noqa: E402
from satcollision import hindcast as hc  # noqa: E402
from satcollision.tle_history import detect_jumps, fit_tle_errors  # noqa: E402

import torch  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def auc(member_scores, nonmember_scores):
    """P(score of a random member > score of a random non-member)."""
    a = np.asarray(member_scores); b = np.asarray(nonmember_scores)
    allv = np.concatenate([a, b])
    ranks = np.argsort(np.argsort(allv)) + 1.0
    ra = ranks[: len(a)].sum()
    return float((ra - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


@torch.no_grad()
def per_sample_nll(model, d):
    X, Y = el._tensors(d)
    model.eval()
    mu, ls = model(X)
    z = (Y - mu) / torch.exp(ls)
    return (ls + 0.5 * (el.NU + 1) * torch.log1p(z * z / el.NU)).sum(1).numpy()


def main(files):
    cfg = json.loads((ROOT / "worlds" / "real_leo_history.json").read_text())["fleet"]["operators"]
    hist = hc.load_histories(files, cache_dir=str(ROOT / "cache"))
    owner, ops = hc.owners_of(hist.names, cfg)
    first, last = hist.first_epoch, hist.last_epoch
    split = max(first + 1.5, last - 8.0)
    lo = first + 0.5
    pub_train = el.build_samples(hist, np.arange(hist.n_objects), lo, split, seed=2, ages_per_set=2, max_samples=300_000)
    public = el.train_local(pub_train, epochs=30, seed=0)
    out = {"operators": {}}
    for k, op in enumerate(ops):
        rows = np.nonzero(owner == k)[0]
        data = el.build_samples(hist, rows, lo, split, seed=0, max_samples=300_000)
        later = el.build_samples(hist, rows, split, last - 1.0, seed=1, max_samples=50_000)
        members, nonmembers = el.split_by_object(data, 0.2, seed=11)    # 80% of satellites train, 20% held out
        # early stopping would use held-out satellites; train for a fixed budget instead, as an owner would
        m0 = el.new_model(0)
        el.train_epochs(m0, members, 15, seed=0)
        models = {"no_dp": m0,
                  "dp_eps8": el.train_owner_dp(members, target_epsilon=8.0, epochs=5, seed=0),
                  "dp_eps2": el.train_owner_dp(members, target_epsilon=2.0, epochs=5, seed=0)}
        r = {"n_members": len(members), "n_nonmembers": len(nonmembers), "n_later": len(later)}
        # (a) position leak on later data
        raw = float(np.median(np.abs(later.y[:, 1])))
        bp, _ = el.predict(public, later.X, later.age_h)
        r["position"] = {"median_along_track_km_public_prediction": raw,
                         "after_public_model_bias": float(np.median(np.abs(later.y[:, 1] - bp[:, 1])))}
        for name, m in models.items():
            b, _ = el.predict(m, later.X, later.age_h)
            r["position"][f"after_owner_model_bias_{name}"] = float(np.median(np.abs(later.y[:, 1] - b[:, 1])))
        # (b) membership inference
        r["membership_auc"] = {}
        for name, m in models.items():
            lm, ln = per_sample_nll(m, members), per_sample_nll(m, nonmembers)
            s_auc = auc(-lm, -ln)
            def per_sat(d, l):
                rows_u, inv = np.unique(d.row, return_inverse=True)
                return np.bincount(inv, weights=l) / np.bincount(inv)
            sat_auc = auc(-per_sat(members, lm), -per_sat(nonmembers, ln))
            r["membership_auc"][name] = {"per_sample": s_auc, "per_satellite": sat_auc}
        # (c) fleet habit: predicted along-track spread at 48 h (median over the fleet's later samples)
        X48 = later.X.copy()
        age48 = np.full(len(later), 48.0)
        import math
        X48[:, 0] = math.log(48.1) / math.log(el.MAX_AGE_H); X48[:, 1] = 48.0 / el.MAX_AGE_H
        r["sigma_t_48h_km"] = {name: float(np.median(el.predict(m, X48, age48)[1][:, 1])) for name, m in models.items()}
        r["sigma_t_48h_km"]["public_model"] = float(np.median(el.predict(public, X48, age48)[1][:, 1]))
        out["operators"][op] = r
        log(op, json.dumps({k2: (v if not isinstance(v, dict) else {a: round(b, 3) if isinstance(b, float) else b
                                                                     for a, b in v.items()}) for k2, v in r.items()}))
    # public manoeuvre rate per operator (jumps per satellite per day, before the split)
    table = fit_tle_errors(hc.subset_history(hist, split), max_objects_per_group=1500, seed=0)
    jumps = detect_jumps(hist, table, 10.0, rows=np.nonzero(owner >= 0)[0])
    days = split - lo
    for k, op in enumerate(ops):
        n_sat = int((owner == k).sum())
        n_j = sum(1 for j in jumps if owner[hist.row_of[j["norad_id"]]] == k and lo <= j["epoch_jd"] < split
                  and max(abs(x) for x in j["rtn_km"]) >= 1.0)
        out["operators"][op]["public_manoeuvre_rate_per_sat_day"] = n_j / max(n_sat, 1) / days
    rates = [out["operators"][op]["public_manoeuvre_rate_per_sat_day"] for op in ops]
    for name in ("no_dp", "dp_eps2", "public_model"):
        sig = [out["operators"][op]["sigma_t_48h_km"][name] for op in ops]
        rho = float(np.corrcoef(np.argsort(np.argsort(rates)), np.argsort(np.argsort(sig)))[0, 1])
        out[f"rank_corr_sigma48_vs_public_manoeuvre_rate_{name}"] = rho
        log(f"rank correlation, {name} 48 h spread vs public manoeuvre rate: {rho:.2f}")
    p = ROOT / "research" / "results" / "owner_model_privacy.json"
    p.write_text(json.dumps(out, indent=1))
    log("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
