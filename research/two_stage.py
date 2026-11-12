r"""Two-stage federated screening: abstracted alerts as a trigger for a secure Pc step.

Knowledge model (per-object 1-sigma position uncertainty, km):
  * own satellite, precise:                SIG_P = 0.5 / sqrt(2)
  * the other operator's satellite, public: SIG_PUB(L), falling linearly from
    3.0 / sqrt(2) at >= H hours before TCA to 0.5 / sqrt(2) at TCA.
These reproduce the combined sigmas the project already used (0.5 km for full
sharing, 3.0 -> 0.5 km for no cooperation), so only the *federated* regime is new.

Regimes, for one encounter with TCA geometry G and lead time L:
  * no cooperation : Pc(G; SIG_PUB(L), SIG_PUB(L)) >= T
  * full sharing   : Pc(G; SIG_P, SIG_P) >= T, from the moment the encounter
                     enters the screening horizon H (upper bound)
  * two-stage      : stage 1, each operator alone computes
                     Pc(G; SIG_P, SIG_PUB(L)) (its own precise, the other public)
                     and raises an abstracted alert when it reaches tau_pre;
                     stage 2, a secure computation (MPC/HE, not implemented
                     here) returns the full-sharing Pc for alerted pairs only.
                     Lead = time of the first alert, if the full Pc >= T.

"Dangerous" = full-sharing Pc >= T: the encounters an operator should act on.
"""
from __future__ import annotations
import argparse, json, math, pathlib
import numpy as np
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from satcollision.pc import FosterPcModel, EncounterGeometry  # noqa: E402

T = 1e-4
H_S = 72 * 3600.0
SIG_P = 0.5 / math.sqrt(2)
SIG_FAR = 3.0 / math.sqrt(2)
SIG_NEAR = 0.5 / math.sqrt(2)
HBR = 0.02
SECURE_UPDATE_S = 8 * 3600.0     # secure Pc re-run per alerted pair, like CDM updates
MPC_PAIRS_PER_HOUR = 18_000      # ShareSat throughput, Valdma & Veskus (SDC9, 2025)
TAUS = [1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8]


def sig_pub(lead_s, far=SIG_FAR, near=SIG_NEAR, h=H_S):
    return near + np.clip(lead_s / h, 0, 1) * (far - near)


def load(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


class PcTable:
    """Pc of every encounter as a function of the 'public' sigma, on a grid,
    for the three knowledge cases. Interpolated in log Pc."""

    def __init__(self, enc, model: str, k: float, sig_grid):
        self.grid = sig_grid
        n = len(enc["miss_km"])
        if model == "circular":
            d2 = enc["miss_km"][:, None] ** 2
            def pc(sa, sb):
                s2 = sa ** 2 + sb ** 2
                return np.minimum(HBR ** 2 / (2 * s2) * np.exp(-d2 / (2 * s2)), 1.0)
            s = sig_grid[None, :]
            self.a_own = pc(SIG_P, s)          # A precise, B public
            self.b_own = self.a_own            # symmetric for the circular model
            self.nocoop = pc(s, s)
            self.full = pc(SIG_P, np.array([[SIG_P]]))[:, 0]
        else:
            m = FosterPcModel(along_track_ratio=k)
            self.a_own = np.zeros((n, len(sig_grid)))
            self.b_own = np.zeros_like(self.a_own)
            self.nocoop = np.zeros_like(self.a_own)
            self.full = np.zeros(n)
            for i in range(n):
                g = EncounterGeometry(miss_distance_km=float(enc["miss_km"][i]),
                                      position_a_km=enc["ra"][i], position_b_km=enc["rb"][i],
                                      velocity_a_km_s=enc["va"][i], velocity_b_km_s=enc["vb"][i])
                self.full[i] = m.pc(g, 0, sigma_a_km=SIG_P, sigma_b_km=SIG_P)
                for j, s in enumerate(sig_grid):
                    self.a_own[i, j] = m.pc(g, 0, sigma_a_km=SIG_P, sigma_b_km=s)
                    self.b_own[i, j] = m.pc(g, 0, sigma_a_km=s, sigma_b_km=SIG_P)
                    self.nocoop[i, j] = m.pc(g, 0, sigma_a_km=s, sigma_b_km=s)

    def at(self, table, sig):
        """Pc(sig) for every encounter; sig shape (n_lead,) -> (n_enc, n_lead)."""
        lg = np.log(np.maximum(table, 1e-300))
        j = np.clip(np.searchsorted(self.grid, sig) - 1, 0, len(self.grid) - 2)
        w = (sig - self.grid[j]) / (self.grid[j + 1] - self.grid[j])
        return np.exp(lg[:, j] * (1 - w) + lg[:, j + 1] * w)


def first_crossing_lead(pc_mat, leads, thr, max_lead):
    """Largest lead L (scan from far to near) with pc >= thr and L <= max_lead; 0 if never."""
    ok = (pc_mat >= thr) & (leads[None, :] <= max_lead[:, None])
    any_ = ok.any(axis=1)
    first = np.argmax(ok, axis=1)          # leads sorted descending
    return np.where(any_, leads[first], 0.0)


def analyse(enc, model="circular", k=8.0, far=SIG_FAR, n_boot=2000, seed=0, step_s=300.0):
    hours = enc["meta"]["hours"]
    scorable = enc["tca_s"] >= H_S
    e = {key: (v[scorable] if isinstance(v, np.ndarray) else v) for key, v in enc.items()}
    sig_grid = np.linspace(SIG_NEAR, max(far, SIG_NEAR + 1e-6), 48)
    tab = PcTable(e, model, k, sig_grid)
    leads = np.arange(H_S, -1, -step_s)                  # descending
    sig = sig_pub(leads, far=far)
    max_lead = np.minimum(e["tca_s"], H_S)
    danger = tab.full >= T
    n_days = (hours * 3600.0 - H_S) / 86400.0
    n = len(danger)
    # Leads per encounter: when no-coop first reaches T, and when stage 1 first
    # reaches each tau. Computed in chunks (the full matrix would not fit), and
    # only for encounters whose Pc can reach the smallest tau at all.
    reach = (np.maximum(tab.a_own, tab.b_own).max(axis=1) >= min(TAUS)) | (tab.nocoop.max(axis=1) >= T)
    idx_all = np.nonzero(reach)[0]
    lead_nc = np.zeros(n)
    lead_flag_tau = {tau: np.zeros(n) for tau in TAUS}
    # Ablation: the same two-stage scheme, but the trigger uses public data only
    # (no operator knowledge). Isolates what the operators' private alerts add.
    lead_pub_tau = {tau: np.zeros(n) for tau in TAUS}
    for c0 in range(0, len(idx_all), 20000):
        ix = idx_all[c0:c0 + 20000]
        sub = lambda t: t[ix]
        s1 = np.maximum(PcTable.at(tab, sub(tab.a_own), sig), PcTable.at(tab, sub(tab.b_own), sig))
        nc = PcTable.at(tab, sub(tab.nocoop), sig)
        lead_nc[ix] = first_crossing_lead(nc, leads, T, max_lead[ix])
        for tau in TAUS:
            lead_flag_tau[tau][ix] = first_crossing_lead(s1, leads, tau, max_lead[ix])
            lead_pub_tau[tau][ix] = first_crossing_lead(nc, leads, tau, max_lead[ix])
    out = {"model": model if model == "circular" else f"foster(k={k:g})", "sig_far_combined_km": far * math.sqrt(2),
           "n_scored": int(scorable.sum()), "n_dangerous": int(danger.sum()), "days_scored": n_days,
           "dangerous_per_day": danger.sum() / n_days,
           "nocoop": {"recall": float((lead_nc[danger] > 0).mean()) if danger.any() else None,
                      "median_lead_h": float(np.median(lead_nc[danger]) / 3600) if danger.any() else None,
                      "false_alarms_per_day": float(((lead_nc > 0) & ~danger).sum() / n_days)},
           "full_sharing": {"median_lead_h": H_S / 3600, "pairs_exposed": "all"},
           "two_stage": []}
    out["public_prefilter"] = []
    for tau in TAUS:
        lf = lead_pub_tau[tau]; fl = lf > 0; lh = np.where(danger, lf, 0.0)
        out["public_prefilter"].append({
            "tau_pre": tau, "recall": float(fl[danger].mean()) if danger.any() else None,
            "median_lead_h": float(np.median(lh[danger]) / 3600) if danger.any() else None,
            "alerts_per_day": float(fl.sum() / n_days),
            "secure_evals_per_day": float(np.ceil(lf[fl] / SECURE_UPDATE_S).sum() / n_days)})
    rng = np.random.default_rng(seed)
    di = np.nonzero(danger)[0]
    boots = rng.integers(0, len(di), size=(n_boot, len(di))) if len(di) else None
    for tau in TAUS:
        lead_flag = lead_flag_tau[tau]
        flagged = lead_flag > 0
        lead_hyb = np.where(danger, lead_flag, 0.0)
        secure_evals = np.ceil(lead_flag[flagged] / SECURE_UPDATE_S).sum()
        row = {"tau_pre": tau,
               "recall": float(flagged[danger].mean()) if danger.any() else None,
               "recall_24h": float((lead_hyb[danger] >= 24 * 3600).mean()) if danger.any() else None,
               "median_lead_h": float(np.median(lead_hyb[danger]) / 3600) if danger.any() else None,
               "alerts_per_day": float(flagged.sum() / n_days),
               "alert_precision": float(danger[flagged].mean()) if flagged.any() else None,
               "secure_evals_per_day": float(secure_evals / n_days),
               "mpc_hours_per_day": float(secure_evals / n_days / MPC_PAIRS_PER_HOUR)}
        if boots is not None:
            rec = flagged[di][boots].mean(axis=1)
            med = np.median(lead_hyb[di][boots], axis=1) / 3600
            row["recall_ci95"] = [float(np.percentile(rec, 2.5)), float(np.percentile(rec, 97.5))]
            row["median_lead_h_ci95"] = [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]
        out["two_stage"].append(row)
    if boots is not None:
        med = np.median(lead_nc[di][boots], axis=1) / 3600
        out["nocoop"]["median_lead_h_ci95"] = [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--model", choices=("circular", "foster"), default="circular")
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--far", type=float, default=3.0, help="combined public sigma far out, km")
    a = ap.parse_args()
    enc = load(a.cache)
    res = analyse(enc, a.model, a.k, far=a.far / math.sqrt(2))
    res["cache_meta"] = enc["meta"]
    name = pathlib.Path(a.cache).stem + f"__{res['model']}_far{a.far:g}.json"
    p = pathlib.Path("research/results") / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    print(p)
    print(json.dumps({k: v for k, v in res.items() if k not in ("two_stage", "public_prefilter", "cache_meta")}))
    for key in ("two_stage", "public_prefilter"):
        print(key)
        for r in res[key]:
            print({k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in r.items()})
