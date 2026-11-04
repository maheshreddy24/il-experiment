"""Compare a bc_torch run and an ifql_torch run up to a given training step.

python agents/plot_compare.py --bc checkpoints/bc_torch_<id> --ifql checkpoints/ifql_torch_<id> --max_step 30000
-> <ifql run dir>/compare_bc_ifql_<max_step>.png
"""
import argparse
import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.edgecolor": GRID, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "lines.linewidth": 2,
    "legend.frameon": False, "legend.fontsize": 8.5,
})

parser = argparse.ArgumentParser()
parser.add_argument("--bc", required=True)
parser.add_argument("--ifql", required=True)
parser.add_argument("--max_step", type=int, default=30000)
args = parser.parse_args()
max_step = args.max_step


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def smooth(y, k=5):
    """Centered moving average (the logged val losses come from a single batch, so they are noisy)."""
    if len(y) < k:
        return y
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def style(ax, title, ylabel, log=False):
    ax.set_title(title, loc="left", fontsize=11.5, color=TEXT)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: ""))
    ax.set_xlim(0, max_step * 1.02)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend()


def train_val(ax, steps, train, val, color, label):
    ax.plot(steps, train, color=color, label=f"{label} train")
    ax.plot(steps, val, color=color, linewidth=1, alpha=0.3)
    ax.plot(steps, smooth(val), color=color, linestyle="--", label=f"{label} val (smoothed)")


# BC: train.log lines "[step N] Train Loss: x, Val Loss: y".
pattern = re.compile(r"\[step (\d+)\] Train Loss: ([\d.]+), Val Loss: ([\d.]+)")
with open(os.path.join(args.bc, "train.log")) as f:
    bc = np.array([tuple(map(float, m.groups())) for m in map(pattern.search, f) if m])
bc = bc[bc[:, 0] <= max_step]
bc_steps, bc_train, bc_val = bc.T

# IFQL: metrics.csv.
m = [r for r in read_csv(os.path.join(args.ifql, "metrics.csv")) if float(r["step"]) <= max_step]
col = lambda key: np.array([float(r[key]) for r in m])
ifql_steps = col("step")


# Success: pool repeated evals per (method, step).
def pool(rows, method_of, step_of):
    acc = {}
    for r in rows:
        method = method_of(r)
        step = step_of(r)
        if method is None or step > max_step:
            continue
        n = int(float(r.get("episodes", 100)))
        s = acc.setdefault((method, step), [0.0, 0])
        s[0] += float(r["success"]) * n
        s[1] += n
    out = {}
    for (method, step), (succ, n) in sorted(acc.items()):
        p = succ / n
        out.setdefault(method, []).append((step, p, np.sqrt(p * (1 - p) / n)))
    return out


ckpt_step = lambda r: int(re.search(r"step_(\d+)", r["checkpoint"]).group(1))
success = pool(read_csv(os.path.join(args.bc, "eval.csv")), lambda r: "bc", ckpt_step)
for source, method_of in (
    (read_csv(os.path.join(args.ifql, "eval.csv")), lambda r: r["mode"]),
    (read_csv(os.path.join(args.ifql, "eval_ckpt.csv")), lambda r: {"32": "ifql", "1": "flow_bc"}.get(r["num_samples"])),
):
    for method, pts in pool(source, method_of, lambda r: int(float(r["step"]))).items():
        success.setdefault(method, []).extend(pts)
# Re-pool IFQL points that came from two sources (equal episode counts, so a plain mean per step is fine).
for method in ("ifql", "flow_bc"):
    if method in success:
        by_step = {}
        for s, p, e in success[method]:
            by_step.setdefault(s, []).append((p, e))
        success[method] = [(s, np.mean([p for p, _ in v]), np.sqrt(np.sum([e**2 for _, e in v])) / len(v))
                           for s, v in sorted(by_step.items())]

fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
axes = axes.ravel()

train_val(axes[0], bc_steps, bc_train, bc_val, BLUE, "BC")
style(axes[0], "BC: action-chunk MSE", "MSE (log scale)", log=True)

train_val(axes[1], ifql_steps, col("train/loss/actor"), col("val/loss/actor"), AQUA, "IFQL actor")
style(axes[1], "IFQL actor: flow-matching velocity MSE", "MSE (log scale)", log=True)

# Generalization gap, comparable across the two despite different loss units: val / train.
ax = axes[2]
ax.plot(bc_steps, smooth(bc_val) / bc_train, color=BLUE, label="BC")
ax.plot(ifql_steps, smooth(col("val/loss/actor")) / col("train/loss/actor"), color=AQUA, label="IFQL actor")
ax.axhline(1, color=TEXT_2, linewidth=1, linestyle=":")
style(ax, "Overfitting: val loss / train loss (1 = no gap)", "ratio")

train_val(axes[3], ifql_steps, col("train/loss/critic"), col("val/loss/critic"), BLUE, "critic")
ax = axes[3]
ax.axhline(1.20, color=TEXT_2, linewidth=1, linestyle=":", label="minimum possible (~1.20)")
style(ax, "IFQL critic Q(s,a): HL-Gauss cross-entropy", "cross-entropy")

train_val(axes[4], ifql_steps, col("train/loss/value"), col("val/loss/value"), ORANGE, "value")
ax = axes[4]
ax.axhline(0.60, color=TEXT_2, linewidth=1, linestyle=":", label="minimum possible (~0.60)")
style(ax, "IFQL value V(s): expectile-weighted cross-entropy", "cross-entropy")

ax = axes[5]
labels = {"bc": ("MSE BC", BLUE), "ifql": ("IFQL (best of 32 by Q)", AQUA), "flow_bc": ("flow BC (IFQL actor alone)", ORANGE)}
for method, (label, color) in labels.items():
    if method in success:
        s, p, e = zip(*success[method])
        ax.errorbar(s, p, yerr=e, color=color, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    capsize=3, elinewidth=1.2, label=label)
        ax.annotate(f"{p[-1]:.0%}", (s[-1], p[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=TEXT_2)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
style(ax, "Success rate (±1 s.e., evals pooled)", "success")

fig.suptitle(f"BC vs IFQL, first {max_step // 1000}k training steps  "
             f"(BC: batch 256, all 100 shards  |  IFQL: batch 128, 60 shards)",
             x=0.01, ha="left", fontsize=12.5, color=TEXT)
plt.tight_layout(rect=(0, 0, 1, 0.97))
out = os.path.join(args.ifql, f"compare_bc_ifql_{max_step // 1000}k.png")
plt.savefig(out, dpi=110)
print(f"saved {out}")

# Success-only figure: every checkpoint labeled.
fig, ax = plt.subplots(figsize=(11, 6))
offsets = {"bc": (-22, 0, "right"), "ifql": (22, 0, "left"), "flow_bc": (0, -16, "center")}
for method, (label, color) in labels.items():
    if method not in success:
        continue
    s, p, e = zip(*success[method])
    ax.errorbar(s, p, yerr=e, color=color, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                capsize=4, elinewidth=1.2, label=label)
    for x, y in zip(s, p):
        dx, dy, ha = offsets[method]
        ax.annotate(f"{y:.0%}", (x, y), xytext=(dx * 0.5, dy), textcoords="offset points", ha=ha, va="center",
                    color=TEXT_2, fontsize=9.5)
ax.set_ylim(0, 1)
xs = sorted({s for pts in success.values() for s, _, _ in pts})
ax.set_xticks(xs)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
style(ax, f"Success rate per checkpoint (±1 s.e.; 200 episodes per point)", "success")
ax.set_xlim(xs[0] - 6000, xs[-1] + 6000)
ax.legend(loc="lower right")
plt.tight_layout()
out = os.path.join(args.ifql, f"compare_success_{max_step // 1000}k.png")
plt.savefig(out, dpi=120)
print(f"saved {out}")
