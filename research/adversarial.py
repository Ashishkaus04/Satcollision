"""Dishonest operators against the two-stage scheme.

Safety and cost (same Monte Carlo as sweep.py):
  honest      both operators alert at tau; secure Pc for alerted pairs
  suppress    operator B never alerts (withholds), only A's alerts trigger
  flood       operator A alerts on every approach it sees (to burn secure
              compute and to probe B's orbits); secure Pc on everything
  flood+check secure step runs only if the *other* operator's own stage-1 Pc
              reaches a plausibility floor tau_floor, so a flooder can't
              trigger it alone
Reported: share of dangerous encounters warned >= 24 h, and MPC cluster-hours/day.
"""
import argparse, json, math, pathlib
import numpy as np
from two_stage import load, H_S, T, MPC_PAIRS_PER_HOUR
import aniso
from sweep import chol2

UPD = 8 * 3600.0


def run(enc, k=1.0, sp=0.5, far=3.0, tau=3e-5, floors=(3e-5, 1e-5, 3e-6, 1e-6, 1e-7), seed=0):
    rng = np.random.default_rng(seed)
    scor = enc["tca_s"] >= H_S
    m, Sa, Sb, ok = aniso.geometry(enc, k, scor)
    m, Sa, Sb = m[ok], Sa[ok], Sb[ok]
    max_lead = np.minimum(enc["tca_s"][scor][ok], H_S)
    n_days = (enc["meta"]["hours"] * 3600 - H_S) / 86400
    leads = np.arange(H_S, 0, -UPD); nl = len(leads)
    p = sp / math.sqrt(2)
    pub = p + np.clip(leads / H_S, 0, 1) * (far / math.sqrt(2) - p)
    live = leads[None, :] <= max_lead[:, None]

    def cov(sa, sb):
        sa = np.broadcast_to(np.asarray(sa, float), (nl,)); sb = np.broadcast_to(np.asarray(sb, float), (nl,))
        return Sa[:, None] * (sa ** 2)[None, :, None, None] + Sb[:, None] * (sb ** 2)[None, :, None, None]

    def est(C):
        l11, l21, l22 = chol2(C)
        z = rng.standard_normal((2,) + l11.shape)
        e = np.stack([l11 * z[0], l21 * z[0] + l22 * z[1]], -1)
        return np.where(live, aniso.pc_small(m[:, None, :] + e, C), 0.0)

    Cf = cov(p, p)
    danger = aniso.pc_small(m, Cf[:, 0]) >= T
    pc_full = est(Cf)
    pcA = est(cov(p, pub))       # A: own (a) precise, other public
    pcB = est(cov(pub, p))
    kk = np.arange(nl)[None, :]

    def first(mask):
        a = mask.any(1); return np.where(a, np.argmax(mask, 1), -1)

    def score(trig_mask, name):
        trig = first(trig_mask)
        after = (trig[:, None] >= 0) & (kk >= trig[:, None]) & live
        act = first(after & (pc_full >= T))
        lead = np.where(act >= 0, leads[np.maximum(act, 0)], 0.0)
        return {"scenario": name, "r24": float((lead[danger] >= 24 * 3600).mean()),
                "recall": float((act[danger] >= 0).mean()),
                "mpc_h_day": float(after.sum() / n_days / MPC_PAIRS_PER_HOUR)}

    out = [score((pcA >= tau) | (pcB >= tau), "honest"),
           score(pcA >= tau, "B suppresses"),
           score(live, "A floods")]
    for fl in floors:
        # A floods, but the secure step needs B's own stage-1 Pc >= floor; honest A alerts still need B >= floor
        out.append(score(live & (pcB >= fl), f"A floods, check B >= {fl:g}"))
        out.append(score(((pcA >= tau) | (pcB >= tau)) & (pcB >= fl) & (pcA >= fl), f"honest, mutual check >= {fl:g}"))
    return {"k": k, "sp": sp, "far": far, "tau": tau, "dangerous_per_day": float(danger.sum() / n_days),
            "rows": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache"); ap.add_argument("--k", type=float, default=1.0); ap.add_argument("--far", type=float, default=3.0)
    a = ap.parse_args()
    r = run(load(a.cache), k=a.k, far=a.far)
    p = pathlib.Path(__file__).resolve().parent / "results" / (pathlib.Path(a.cache).stem + f"__adversarial_k{a.k:g}_far{a.far:g}.json")
    p.write_text(json.dumps(r, indent=1))
    for row in r["rows"]:
        print(f'{row["scenario"]:34s} warned>=24h {100*row["r24"]:5.1f}%  recall {100*row["recall"]:5.1f}%  MPC {row["mpc_h_day"]:7.1f} h/day')
    print(p)
