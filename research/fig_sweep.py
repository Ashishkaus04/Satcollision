"""Figure: secure compute needed for 99% of full sharing's 24 h warning, across assumptions."""
import json, sys, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
cases = json.loads(pathlib.Path(sys.argv[1]).read_text())["cases"]
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#6b6b66"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
ks = [1.0, 4.0, 8.0, 16.0]; fars = [1.5, 3.0, 6.0]
fig, axes = plt.subplots(1, 4, figsize=(8.2, 2.9), sharey=True)
for ax, k in zip(axes, ks):
    sel = [c for c in cases if c["k"] == k and c["sigma_precise_km"] == 0.5]
    sel.sort(key=lambda c: c["sigma_public_far_km"])
    x = [c["sigma_public_far_km"] for c in sel]
    ax.plot(x, [c["two_stage_mpc_h_for_99pct"] for c in sel], color=BLUE, lw=2, marker="o", ms=4, label="Operator alerts")
    ax.plot(x, [c["public_mpc_h_for_99pct"] for c in sel], color=ORANGE, lw=2, marker="o", ms=4, label="Public trigger")
    for c in sel:
        ax.annotate(f"{100*c['nocoop_r24']:.0f}%", (c["sigma_public_far_km"], 2.2), ha="center", fontsize=7, color=MUTED)
    ax.set_title("circular" if k == 1 else f"Foster, along-track ×{k:g}", fontsize=8.5, color=INK)
    ax.set_xticks(fars); ax.set_yscale("log"); ax.set_ylim(1.5, 100)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_xlabel("public σ far out (km)")
axes[0].set_ylabel("MPC cluster-hours/day\nfor 99% of full sharing")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.text(0.5, 0.005, "Grey: share of dangerous encounters warned ≥24 h with no cooperation. Precise σ 0.5 km combined; 10 Sep 2026 catalogue.",
         ha="center", fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(pathlib.Path("research/figures") / f"fig_sweep.{ext}", dpi=200)
print("ok")
