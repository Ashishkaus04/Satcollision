"""Figure: early-warning recall vs secure-computation load, private alerts vs public trigger."""
import json, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = pathlib.Path(sys.argv[1]); runs = json.loads(src.read_text())["seeds"]
def agg(key, metric):
    return np.mean([[row[metric] for row in r[key]] for r in runs], axis=0)
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#6b6b66"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for key, col, lab in (("two_stage", BLUE, "Operator alerts (own orbit precise)"),
                      ("public_trigger", ORANGE, "Public-catalogue trigger only")):
    x, y = agg(key, "mpc_hours_per_day"), 100 * agg(key, "recall_24h")
    ax.plot(x, y, color=col, lw=2, marker="o", ms=4.5, label=lab, zorder=3)
    for xi, yi, tau in zip(x, y, [r["tau_pre"] for r in runs[0][key]]):
        if key == "two_stage" and tau in (1e-4, 3e-5, 1e-5):
            ax.annotate(f"τ={tau:.0e}", (xi, yi), textcoords="offset points", xytext=(2, -16), fontsize=7.5, color=MUTED)
full = 100 * np.mean([r["full_sharing"]["recall_24h"] for r in runs])
nc = 100 * np.mean([r["no_cooperation"]["recall_24h"] for r in runs])
ax.axhline(full, color=MUTED, lw=1, ls="--"); ax.text(0.6, full + 1.5, f"Full sharing {full:.1f}% (all orbits exposed)", fontsize=7.5, color=MUTED)
ax.axhline(nc, color=MUTED, lw=1, ls=":"); ax.text(0.6, nc + 2, f"No cooperation {nc:.1f}%", fontsize=7.5, color=MUTED)
ax.set_xscale("log"); ax.set_ylim(0, 108)
ax.set_xlabel("Secure Pc computation needed (MPC cluster-hours per day, log scale)")
ax.set_ylabel("Dangerous encounters warned ≥24 h ahead (%)")
ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.legend(frameon=False, fontsize=8, loc="center right")
fig.tight_layout()
out = pathlib.Path("research/figures"); out.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out / f"fig_two_stage_tradeoff.{ext}", dpi=200)
print("ok")
