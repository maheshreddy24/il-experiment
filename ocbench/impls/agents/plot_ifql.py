"""Plot an ifql_torch run -> <run_dir>/analysis.png

python agents/plot_ifql.py checkpoints/ifql_torch_<id> [--bc checkpoints/bc_torch_<id>]

Reads metrics.csv (training curves), eval.csv (in-training evals), eval_ckpt.csv (eval_ifql_torch.py, pooled
with the in-training evals) and eval/*.npz (per-episode Q). --bc overlays a bc_torch run's success rate.
"""
import argparse
import csv
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.edgecolor": GRID, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "lines.linewidth": 2,
    "legend.frameon": False, "legend.fontsize": 8.5,
})


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def column(rows, key):
    return np.array([float(r[key]) for r in rows])


def style(ax, title, ylabel=None, xlabel="training step", kfmt=True):
    ax.set_title(title, loc="left", fontsize=11.5, color=TEXT)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if kfmt:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    if ax.get_legend_handles_labels()[0]:
        ax.legend()


def train_val(ax, metric, color, label):
    """Train solid, val dashed + lighter, same hue (val is a single batch per log step, so it is noisy)."""
    ax.plot(steps, column(m, f"train/{metric}"), color=color, label=f"{label} train")
    ax.plot(steps, column(m, f"val/{metric}"), color=color, linestyle="--", linewidth=1.4, alpha=0.6, label=f"{label} val")


def pooled_success(rows, mode_of):
    """{mode: [(step, rate, stderr, episodes)]}, pooling repeated evals of the same (mode, step)."""
    acc = {}
    for r in rows:
        mode = mode_of(r)
        if mode is None:
            continue
        step = int(float(r["step"])) if "step" in r else int(re.search(r"step_(\d+)", r["checkpoint"]).group(1))
        episodes = int(float(r.get("episodes", 100)))
        s = acc.setdefault((mode, step), [0.0, 0])
        s[0] += float(r["success"]) * episodes
        s[1] += episodes
    out = {}
    for (mode, step), (succ, n) in sorted(acc.items()):
        p = succ / n
        out.setdefault(mode, []).append((step, p, np.sqrt(p * (1 - p) / n), n))
    return out


parser = argparse.ArgumentParser()
parser.add_argument("run_dir")
parser.add_argument("--bc", help="bc_torch run dir to overlay its success rate")
args = parser.parse_args()
run_dir = args.run_dir

m = read_csv(os.path.join(run_dir, "metrics.csv"))
steps = column(m, "step")

fig, axes = plt.subplots(3, 3, figsize=(18, 13.5))
axes = axes.ravel()

# Row 1: losses.
train_val(axes[0], "loss/critic", BLUE, "critic")
style(axes[0], "Critic loss (HL-Gauss CE vs n-step TD target)", "cross-entropy")
train_val(axes[1], "loss/value", ORANGE, "value")
style(axes[1], "Value loss (expectile-weighted CE vs target Q)", "cross-entropy")
train_val(axes[2], "loss/actor", AQUA, "actor")
style(axes[2], "Actor flow-matching loss", "MSE")

# Row 2: what the critic learned.
ax = axes[3]
train_val(ax, "critic/q_success_eps", BLUE, "Q | success eps")
train_val(ax, "critic/q_failure_eps", ORANGE, "Q | failure eps")
style(ax, "Q on successful vs failed episodes (gap = critic signal)", "Q")

ax = axes[4]
ax.plot(steps, column(m, "train/critic/q_mean"), color=BLUE, label="Q mean")
ax.plot(steps, column(m, "train/value/v_mean"), color=ORANGE, label="V mean")
ax.plot(steps, column(m, "train/critic/td_target_mean"), color=AQUA, linestyle=":", label="TD target mean")
ax.plot(steps, column(m, "train/critic/q_min"), color=YELLOW, label="Q min (batch)")
style(ax, "Q / V / TD target (train batches)", "value")

ax = axes[5]
train_val(ax, "critic/td_error_abs", BLUE, "|Q - TD target|")
ax.plot(steps, column(m, "train/critic/head_disagreement"), color=ORANGE, label="|Q1 - Q2| train")
ax.plot(steps, column(m, "train/adv/std"), color=AQUA, label="std(Q - V) train")
style(ax, "Critic error, head disagreement, advantage spread", "value")

# Row 3: evaluation.
ax = axes[6]
in_training = pooled_success(read_csv(os.path.join(run_dir, "eval.csv")), lambda r: r["mode"])
from_ckpt = pooled_success(
    read_csv(os.path.join(run_dir, "eval_ckpt.csv")),
    lambda r: {"1": "flow_bc"}.get(r["num_samples"], "ifql" if r["num_samples"] == "32" else None),
)
series = {}
for source in (in_training, from_ckpt):
    for mode, pts in source.items():
        series.setdefault(mode, []).extend(pts)
series = {mode: pooled_success([dict(step=s, success=p, episodes=n) for s, p, _, n in pts], lambda r: mode)[mode]
          for mode, pts in series.items()}
labels = {"ifql": ("IFQL (best of 32 by Q)", BLUE), "flow_bc": ("flow BC (actor alone)", ORANGE)}
if args.bc:
    bc = pooled_success(read_csv(os.path.join(args.bc, "eval.csv")), lambda r: "bc")
    if bc:
        series["bc"] = bc["bc"]
        labels["bc"] = ("MSE BC (bc_torch)", AQUA)
for mode, (label, color) in labels.items():
    if mode in series:
        s, p, err, _ = zip(*series[mode])
        ax.errorbar(s, p, yerr=err, color=color, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    capsize=3, elinewidth=1.2, label=label)
        ax.annotate(f"{p[-1]:.0%}", (s[-1], p[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=TEXT_2)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
style(ax, "Success rate (±1 s.e., repeated evals pooled)", "success")

# Latest IFQL eval npz: Q along the episode and calibration.
npz = sorted(
    glob.glob(os.path.join(run_dir, "eval", "ifql_step_*.npz")) + glob.glob(os.path.join(run_dir, "eval", "ckpt_*_n32.npz")),
    key=lambda p: int(re.search(r"step_(\d+)", p).group(1)),
)
if npz:
    d = np.load(npz[-1])
    q, succ, lengths = d["q_chosen"], d["episode_success"], d["episode_length"]
    chunk_steps = np.arange(len(q)) * 25
    valid = chunk_steps[:, None] < lengths[None]
    ax = axes[7]
    for label, mask, color in [("successful episodes", succ == 1, BLUE), ("failed episodes", succ == 0, ORANGE)]:
        if mask.any():
            ax.plot(chunk_steps, np.nanmean(np.where(valid[:, mask], q[:, mask], np.nan), axis=1), color=color,
                    label=f"{label} (n={mask.sum()})")
    style(ax, f"Q of executed chunk along the episode ({os.path.basename(npz[-1])})", "Q", xlabel="env step", kfmt=False)

    ax = axes[8]
    q_init = q[0]
    edges = np.quantile(q_init, np.linspace(0, 1, 6))
    which = np.clip(np.digitize(q_init, edges[1:-1]), 0, 4)
    xs = [q_init[which == b].mean() for b in range(5) if (which == b).any()]
    ys = [succ[which == b].mean() for b in range(5) if (which == b).any()]
    ax.plot(xs, ys, "o-", color=BLUE, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label="episodes binned by Q at start (quintiles)")
    lo, hi = min(xs) - 0.01, max(xs) + 0.01
    ax.axhline(succ.mean(), color=TEXT_2, linewidth=1, linestyle="--", label=f"overall success {succ.mean():.0%}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    style(ax, "Does Q at the first chunk predict success?", "actual success rate", xlabel="predicted Q at start",
          kfmt=False)

plt.tight_layout()
out = os.path.join(run_dir, "analysis.png")
plt.savefig(out, dpi=110)
print(f"saved {out}")
