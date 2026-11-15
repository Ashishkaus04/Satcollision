"""Figures for the hindcast: caught vs false alarms (threshold sweep) and
error-model quality per operator.

    python research/fig_hindcast.py research/results/hindcast_<name>_s0.json [more seeds]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
SHOW = [("baseline", "Baseline (circular, 0.5 km)", "#888888", "--"),
        ("foster", "+ Foster (public error table)", "#1f77b4", "-"),
        ("foster_pub_local", "+ Foster + local-only models", "#ff7f0e", ":"),
        ("foster_pub_fl_sqrt", "+ Foster + FedAvg model", "#d62728", "-"),
        ("foster_pub_central", "+ Foster + pooled model", "#2ca02c", "-."),
        ("foster_pub_owner", "+ Foster + owners' models", "#9467bd", "-")]


def main(paths):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    stem = Path(paths[0]).stem.rsplit("_s", 1)[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (key_c, key_f, title) in zip(axes[:2], (("caught", "fa_day", "All assessments"),
                                                     ("mm_caught", "mm_fa_day", "Member vs member"))):
        for name, label, col, ls in SHOW:
            fa = np.mean([[p[key_f] for p in r["sweep"][name]] for r in runs], 0)
            c = np.mean([[p[key_c] for p in r["sweep"][name]] for r in runs], 0)
            ax.plot(np.maximum(fa, 1e-2), c, ls, color=col, label=label)
            T = [p["T"] for p in runs[0]["sweep"][name]]
            k = int(np.argmin(np.abs(np.log10(T) + 4)))
            ax.plot(max(fa[k], 1e-2), c[k], "o", color=col)
        ax.set_xscale("log")
        ax.set_xlabel("False alarms per day")
        ax.set_ylabel("Dangerous encounters caught")
        ax.set_title(title + " (dot: Pc threshold 1e-4)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    ax = axes[2]
    ops = list(runs[0]["model_quality"])
    keys = [("table", "Public table", "#1f77b4"), ("others_local", "Other operators' local models", "#ff7f0e"),
            ("fedavg_sqrt", "FedAvg", "#d62728"), ("central", "Pooled", "#2ca02c"), ("own_local", "Owner's own model", "#9467bd")]
    x = np.arange(len(ops))
    for i, (k, lab, colr) in enumerate(keys):
        vals = []
        for op in ops:
            q = [r["model_quality"][op] for r in runs]
            v = [qq["others_local_mean_nll"] if k == "others_local" else qq[k]["nll"] for qq in q]
            vals.append(np.mean([vv for vv in v if vv is not None]) if any(vv is not None for vv in v) else np.nan)
        ax.bar(x + (i - 2) * 0.16, vals, 0.16, label=lab, color=colr)
    ax.set_xticks(x, ops, rotation=20)
    ax.set_ylabel("Held-out NLL (lower is better)")
    ax.set_title("Error model on each fleet's satellites")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_{stem}.{ext}", dpi=150)
    print(OUT / f"fig_{stem}.png")


if __name__ == "__main__":
    main(sys.argv[1:])
