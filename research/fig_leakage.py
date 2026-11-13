"""Figure: what a rival operator learns about a satellite's private orbit offset, per channel."""
import json, glob, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sig = np.array([0.452, 3.618, 0.452])
order = ["MINIMAL", "ALERT", "TIER", "TCA", "SECBIT"]
labels = ["'Act now'\nonly", "Alerts", "+ tier", "+ TCA (1 s)", "+ secure bit"]
def med(win):
    rows = {}
    for f in glob.glob(f"research/results/*leakage*{win}_s*.json"):
        for c, r in json.load(open(f))["results"].items():
            rows.setdefault(c, []).extend(r)
    return {c: np.median([x["sd_ratio"] for x in rows[c]], axis=0) for c in order}
BLUE, AQUA, MUTED, INK = "#2a78d6", "#1baf7a", "#6b6b66", "#222222"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
for ax, axis, name in zip(axes, (1, 0), ("Along-track", "Radial (altitude)")):
    for off, win, col, lab in ((-0.2, "win24.0", BLUE, "1 day"), (0.2, "winall", AQUA, "1 week")):
        m = med(win)
        y = [100 * m[c][axis] for c in order]
        ax.bar(np.arange(len(order)) + off, y, width=0.38, color=col, label=lab)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_title(f"{name}: public 1σ = {sig[axis]*1000:.0f} m", fontsize=8.5, color=INK)
    ax.axhline(100, color=MUTED, lw=0.8, ls="--")
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
axes[0].set_ylabel("Attacker's remaining uncertainty\n(% of public catalogue)")
axes[0].legend(frameon=False, fontsize=7.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"research/figures/fig_leakage.{ext}", dpi=200)
print("ok")
