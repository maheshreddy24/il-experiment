"""Plot a bc_torch run: python agents/plot_bc.py checkpoints/bc_torch_<id>  ->  <run_dir>/loss_curves.png

Uses train.log (train loss + single-batch val loss every log_interval) and, if present,
eval.csv from eval_bc_torch.py (full-val-set loss and success rate per checkpoint).
"""
import csv
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"

run_dir = sys.argv[1]
pattern = re.compile(r"\[step (\d+)\] Train Loss: ([\d.]+), Val Loss: ([\d.]+)")
with open(os.path.join(run_dir, "train.log")) as f:
    rows = [tuple(map(float, m.groups())) for m in map(pattern.search, f) if m]
steps, train_loss, val_loss = np.array(rows).T

# Pool repeated evals of the same checkpoint: episode-weighted success, binomial s.e. over all episodes.
pooled = {}
eval_path = os.path.join(run_dir, "eval.csv")
if os.path.exists(eval_path):
    with open(eval_path) as f:
        for r in csv.DictReader(f):
            step = int(re.search(r"step_(\d+)", r["checkpoint"]).group(1))
            p = pooled.setdefault(step, dict(successes=0.0, episodes=0, val_loss=""))
            p["successes"] += float(r["success"]) * int(r["episodes"])
            p["episodes"] += int(r["episodes"])
            p["val_loss"] = r.get("val_loss") or p["val_loss"]
evals = []
for step, p in sorted(pooled.items()):
    rate = p["successes"] / p["episodes"]
    evals.append((step, rate, np.sqrt(rate * (1 - rate) / p["episodes"]), p["val_loss"], p["episodes"]))

plt.rcParams.update({
    "font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.edgecolor": GRID, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
})
fig, axes = plt.subplots(1, 2 if evals else 1, figsize=(13 if evals else 7, 4.6), squeeze=False)
axes = axes[0]


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=TEXT)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))


# Loss curves (log scale: train loss spans ~25x).
ax = axes[0]
ax.plot(steps, train_loss, color=BLUE, linewidth=2, label="train loss (avg over 1k steps)")
ax.plot(steps, val_loss, color=ORANGE, linewidth=2, alpha=0.55, label="val loss (single batch, noisy)")
full_val = [(s, float(v)) for s, _, _, v, _ in evals if v != ""]
if full_val:
    ax.plot(*zip(*full_val), "o", color=ORANGE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
            label="val loss (full val set, per checkpoint)")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
ax.annotate(f"{train_loss[-1]:.4f}", (steps[-1], train_loss[-1]), xytext=(6, 0), textcoords="offset points",
            va="center", color=TEXT_2)
ax.annotate(f"{val_loss[-1]:.4f}", (steps[-1], val_loss[-1]), xytext=(6, 0), textcoords="offset points",
            va="center", color=TEXT_2)
ax.legend(frameon=False, fontsize=9, loc="lower left")
style(ax, "BC action-chunk MSE (normalized actions)", "MSE (log scale)")

# Success rate per checkpoint.
if evals:
    ax = axes[1]
    s, succ, err, _, n = zip(*evals)
    ax.errorbar(s, succ, yerr=err, color=BLUE, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2, capsize=4, elinewidth=1.2)
    for x, y in zip(s, succ):
        ax.annotate(f"{y:.0%}", (x, y), xytext=(0, 10), textcoords="offset points", ha="center", color=TEXT_2)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlim(axes[0].get_xlim())
    style(ax, f"Success rate ({min(n)}+ episodes per checkpoint, ±1 s.e.)", "success")

plt.tight_layout()
out = os.path.join(run_dir, "loss_curves.png")
plt.savefig(out, dpi=130)
print(f"saved {out}")
