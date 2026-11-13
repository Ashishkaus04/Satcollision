r"""What does the other operator learn about my satellite from the federation's messages?

Threat model. Operator A (the attacker) knows its own satellites precisely and
everyone else's only from the public catalogue. It wants the public-catalogue
error delta of a target satellite S owned by operator B, i.e. B's private
knowledge beyond the public TLE (a 3-vector in S's RTN frame). delta is held
constant over the week: a worst case for privacy, since a fixed bias is the
easiest thing to accumulate evidence about.

Prior: delta ~ N(0, Sigma_pub), RTN with along-track ratio k (default 8).

What A observes about S, for every approach between S and one of A's own
satellites X (under 10 km, one week), at every 8-hourly update from 72 h before
TCA, under the two-stage scheme (channels are cumulative, in this order):
  ALERT   B's stage-1 alert bit (B's Pc >= tau; B knows S precisely, X publicly)
  TIER    + the severity tier the signal carries (bucket of B's Pc)
  TCA     + the TCA in whole seconds the signal carries (B's estimate)
  SECBIT  + the secure step's output bit (full-precision Pc >= T), after an alert
  SECVAL  instead of SECBIT, the secure step returns the Pc value (3 significant figures)
  MINIMAL a hardened design: alerts, tiers and TCAs stay inside the secure computation and
          A only learns "act now" (B's alert and full-precision Pc >= T) at each update

Posterior by importance sampling in two passes (prior samples, then a Gaussian
proposal fitted to pass 1), so a sharp posterior is resolved rather than
collapsing onto a grid point. B's estimation noise is integrated with K fixed
Monte Carlo draws per observation. Reported per target: posterior standard
deviation per RTN axis over the prior's (1 = nothing learned), the error of the
posterior mean over the prior sigma, and the effective sample size.
"""
import argparse, json, math, pathlib
import numpy as np
from scipy.stats import norm
from two_stage import load, H_S, T
import aniso

UPD = 8 * 3600.0
TAU = 3e-5
CHANNELS = ["MINIMAL", "ALERT", "TIER", "TCA", "SECBIT", "SECVAL"]


def unit_rtn_basis(r, v):
    rh = r / np.linalg.norm(r); ch = np.cross(r, v); ch /= np.linalg.norm(ch)
    return np.stack([rh, np.cross(ch, rh), ch], axis=-1)


def tier_of(pc):
    return np.digitize(pc, [TAU, T, 10 * T, 100 * T])


def observations(enc, geo, sel, sid, delta_true, rng, k, p, far, K, window=None):
    """Simulate everything A sees about S (one realisation) plus what A needs to score candidates."""
    m3, Sa, Sb, B3 = geo
    na = enc["norad_a"].astype(int)
    obs = []
    for i in sel:
        s_is_a = na[i] == sid
        rS, vS = (enc["ra"][i], enc["va"][i]) if s_is_a else (enc["rb"][i], enc["vb"][i])
        rX, vX = (enc["rb"][i], enc["vb"][i]) if s_is_a else (enc["ra"][i], enc["va"][i])
        sign = 1.0 if s_is_a else -1.0
        P = sign * (B3[i].T @ unit_rtn_basis(rS, vS))          # delta_rtn -> shift of A's miss estimate
        S_S, S_X = (Sa[i], Sb[i]) if s_is_a else (Sb[i], Sa[i])
        m_true = sign * m3[i]
        m_att = m_true + P @ delta_true
        vrel = sign * (enc["va"][i] - enc["vb"][i]); speed = np.linalg.norm(vrel); vh = vrel / speed
        tshift = unit_rtn_basis(rS, vS).T @ vh / speed         # d(TCA_att)/d(delta_rtn)
        C_full = p * p * (S_S + S_X)
        pc_full_true = float(aniso.pc_small(m_true, C_full))
        alerted = False
        for L in np.arange(H_S, 0, -UPD):
            if L > min(enc["tca_s"][i], H_S):
                continue
            if window is not None and not (window[0] <= enc["tca_s"][i] - L < window[1]):
                continue
            pub = p + min(L / H_S, 1.0) * (far / math.sqrt(2) - p)
            C_B = p * p * S_S + pub * pub * S_X
            Lc = np.linalg.cholesky(pub * pub * S_X + 1e-12 * np.eye(2))
            z = rng.standard_normal((K, 2)) @ Lc.T
            pcB_true = float(aniso.pc_small(m_true + Lc @ rng.standard_normal(2), C_B))
            alert = pcB_true >= TAU
            covX = aniso.rtn_unit_cov(rX[None], vX[None], k)[0] * pub * pub
            sd_t = math.sqrt(vh @ covX @ vh) / speed
            t_att = enc["tca_s"][i] + tshift @ delta_true
            tB = enc["tca_s"][i] + rng.normal() * sd_t
            alerted = alerted or alert
            obs.append(dict(P=P, m_att=m_att, C_B=C_B, z=z, alert=alert, tier=tier_of(pcB_true),
                            t_att=t_att, tshift=tshift, sd_t=sd_t, t_obs=math.floor(tB),
                            secure=alerted, C_full=C_full, bit=pc_full_true >= T,
                            logpc=math.log(max(pc_full_true, 1e-300))))
    return obs


def loglik(G, obs):
    out = {c: np.zeros(len(G)) for c in CHANNELS}
    for o in obs:
        m_grid = o["m_att"][None, :] - G @ o["P"].T
        pcB = aniso.pc_small(m_grid[:, None, :] + o["z"][None], o["C_B"])
        pa = np.clip((pcB >= TAU).mean(1), 1e-4, 1 - 1e-4)
        out["ALERT"] += np.log(pa if o["alert"] else 1 - pa)
        out["TIER"] += np.log(np.clip((tier_of(pcB) == o["tier"]).mean(1), 1e-4, 1))
        if o["alert"]:
            t = o["t_att"] - G @ o["tshift"]
            pk = norm.cdf((o["t_obs"] + 1 - t) / o["sd_t"]) - norm.cdf((o["t_obs"] - t) / o["sd_t"])
            out["TCA"] += np.log(np.clip(pk, 1e-12, 1))
        pcf = aniso.pc_small(m_grid, o["C_full"])
        pact = np.clip(pa * (pcf >= T), 1e-4, 1 - 1e-4)
        act = o["alert"] and o["bit"]
        out["MINIMAL"] += np.log(pact if act else 1 - pact)
        if o["secure"]:
            pb = np.where(pcf >= T, 1 - 1e-4, 1e-4)
            out["SECBIT"] += np.log(pb if o["bit"] else 1 - pb)
            out["SECVAL"] += -0.5 * ((np.log(np.maximum(pcf, 1e-300)) - o["logpc"]) / 0.005) ** 2
    base = out["ALERT"] + out["TIER"] + out["TCA"]
    return {"MINIMAL": out["MINIMAL"], "ALERT": out["ALERT"], "TIER": out["ALERT"] + out["TIER"], "TCA": base,
            "SECBIT": base + out["SECBIT"], "SECVAL": base + out["SECVAL"]}


def posterior(G, logw):
    w = np.exp(logw - logw.max()); w /= w.sum()
    mean = w @ G
    cov = (G - mean).T @ ((G - mean) * w[:, None])
    return mean, cov, 1.0 / (w ** 2).sum()


def run(enc, n_targets=100, k=8.0, sp=0.5, far=3.0, K=16, N=8000, seed=0, window=None):
    rng = np.random.default_rng(seed)
    p = sp / math.sqrt(2)
    s = math.sqrt(3.0 / (2.0 + k * k))
    sig = (far / math.sqrt(2)) * np.array([s, k * s, s])
    na, nb = enc["norad_a"].astype(int), enc["norad_b"].astype(int)
    m3, Sa, Sb, ok = aniso.geometry(enc, k)
    geo = (m3, Sa, Sb, aniso.plane_basis(enc["va"] - enc["vb"]))
    base = (enc["tca_s"] >= H_S) & ok
    tgts = np.unique(np.where(na % 2 == 1, na, nb)[base]); rng.shuffle(tgts)
    res = {c: [] for c in CHANNELS}
    done = 0
    for sid in tgts:
        # only approaches with one of A's satellites (S is odd -> B, partner even -> A)
        sel = np.nonzero(base & ((na == sid) | (nb == sid)))[0]
        if len(sel) < 3:
            continue
        delta = rng.normal(size=3) * sig
        obs = observations(enc, geo, sel, sid, delta, rng, k, p, far, K, window)
        if len(obs) == 0:
            continue
        G1 = rng.normal(size=(N, 3)) * sig
        ll1 = loglik(G1, obs)
        for c in CHANNELS:
            mean, cov, _ = posterior(G1, ll1[c])
            # pass 2: Gaussian proposal around pass 1, widened; weights = prior * lik / proposal
            cov2 = 4 * cov + np.diag((0.02 * sig) ** 2)
            Lp = np.linalg.cholesky(cov2)
            G2 = mean + rng.normal(size=(N, 3)) @ Lp.T
            d = np.linalg.solve(Lp, (G2 - mean).T).T
            logq = -0.5 * (d ** 2).sum(1)
            logprior = -0.5 * ((G2 / sig) ** 2).sum(1)
            ll2 = loglik(G2, obs)[c]
            m2, c2, ess = posterior(G2, ll2 + logprior - logq)
            sd = np.sqrt(np.maximum(np.diag(c2), 0))
            res[c].append({"norad": int(sid), "n_approaches": int(len(sel)), "n_obs": len(obs),
                           "n_alerts": int(sum(o["alert"] for o in obs)),
                           "sd_ratio": (sd / sig).tolist(), "err_ratio": (np.abs(m2 - delta) / sig).tolist(),
                           "ess": float(ess)})
        done += 1
        if done >= n_targets:
            break
    return res, sig.tolist()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache"); ap.add_argument("--targets", type=int, default=100)
    ap.add_argument("--k", type=float, default=8.0); ap.add_argument("--far", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window-h", type=float, default=None, help="only observations made in [72 h, 72 h + window) of the run")
    a = ap.parse_args()
    enc = load(a.cache)
    win = None if a.window_h is None else (H_S, H_S + a.window_h * 3600)
    res, sig = run(enc, a.targets, k=a.k, far=a.far, seed=a.seed, window=win)
    p = pathlib.Path(__file__).resolve().parent / "results" / (pathlib.Path(a.cache).stem + f"__leakage_k{a.k:g}_far{a.far:g}_win{a.window_h if a.window_h else 'all'}_s{a.seed}.json")
    p.write_text(json.dumps({"prior_sigma_rtn_km": sig, "results": res}))
    print("prior sigma RTN (km):", np.round(sig, 3))
    for c, rows in res.items():
        sd = np.array([r["sd_ratio"] for r in rows]); er = np.array([r["err_ratio"] for r in rows])
        ess = np.array([r["ess"] for r in rows])
        print(f"{c:7s} n={len(rows)} sd/prior median R,T,N={np.round(np.median(sd,0),3)} "
              f"p10={np.round(np.percentile(sd,10,0),3)} |err|/prior median={np.round(np.median(er,0),3)} ESS median={np.median(ess):.0f}")
    al = np.array([r["n_alerts"] for r in res["ALERT"]]); ob = np.array([r["n_obs"] for r in res["ALERT"]])
    print("observations per target median", np.median(ob), "alerts median", np.median(al))
    print(p)
