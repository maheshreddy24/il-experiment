"""Extra BC vs IFQL analysis: data used, sample efficiency, #candidates sweep, speed of success, critic contribution.

python agents/plot_analysis.py --bc checkpoints/bc_torch_<id> --ifql checkpoints/ifql_torch_<id> \
    --bc_ckpt model_step_30000 --ifql_ckpt model_step_50000      ->  checkpoints/analysis/analysis_bc_vs_ifql.png
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
TEXT, TEXT_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.edgecolor": GRID, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "lines.linewidth": 2,
    "legend.frameon": False, "legend.fontsize": 8.5,
})
DATA_DIR = os.path.expanduser("~/.cache/ocbench/datasets/visual/visual-block-single-task1-v0")

parser = argparse.ArgumentParser()
parser.add_argument("--bc", required=True)
parser.add_argument("--ifql", required=True)
parser.add_argument("--bc_ckpt", default="model_step_30000")
parser.add_argument("--ifql_ckpt", default="model_step_50000")
parser.add_argument("--bc_batch", type=int, default=256)
parser.add_argument("--ifql_batch", type=int, default=128)
parser.add_argument("--ifql_shards", type=int, default=60)
parser.add_argument("--out_dir", default="checkpoints/analysis")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def style(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, loc="left", fontsize=11.5, color=TEXT)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend()


def pooled(pairs):
    """[(key, success, episodes)] -> {key: (rate, stderr, episodes)}"""
    acc = {}
    for k, p, n in pairs:
        a = acc.setdefault(k, [0.0, 0])
        a[0] += p * n
        a[1] += n
    return {k: (s / n, np.sqrt((s / n) * (1 - s / n) / n), n) for k, (s, n) in sorted(acc.items())}


# ---- Dataset episodes (cached; only rewards/terminals are read) ----
cache = os.path.join(args.out_dir, "dataset_episodes.npz")
if not os.path.exists(cache):
    lens, succ, shard = [], [], []
    train_paths = sorted(p for p in glob.glob(f"{DATA_DIR}/*.npz") if not p.endswith("-val.npz"))
    for i, p in enumerate(train_paths):
        with np.load(p) as d:
            t, r = d["terminals"], d["rewards"]
        off = np.concatenate([[0], np.flatnonzero(t) + 1])
        for s, e in zip(off[:-1], off[1:]):
            lens.append(e - s)
            succ.append(bool(np.any(r[s:e] == 1)))
            shard.append(i)
    np.savez(cache, L=np.array(lens), S=np.array(succ), shard=np.array(shard))
d = np.load(cache)
L, S, shard = d["L"], d["S"], d["shard"]
chunks = (L + 24) // 25  # training samples per episode (one per stored image)
bc_samples = int(chunks[S].sum())
ifql_mask = shard < args.ifql_shards
ifql_samples_succ = int(chunks[ifql_mask & S].sum())
ifql_samples_fail = int(chunks[ifql_mask & ~S].sum())

# ---- Success per checkpoint ----
step_of = lambda r: int(re.search(r"step_(\d+)", r["checkpoint"]).group(1))
bc_succ = pooled([(step_of(r), float(r["success"]), int(float(r["episodes"])))
                  for r in read_csv(os.path.join(args.bc, "eval.csv"))])
ifql_rows = [(int(r["step"]), r["mode"], float(r["success"]), 100) for r in read_csv(os.path.join(args.ifql, "eval.csv"))]
ifql_rows += [(int(r["step"]), int(r["num_samples"]), float(r["success"]), int(r["episodes"]))
              for r in read_csv(os.path.join(args.ifql, "eval_ckpt.csv"))]
mode_n = {"ifql": 32, "flow_bc": 1}
ifql_rows = [(s, mode_n.get(m, m), p, n) for s, m, p, n in ifql_rows]
ifql_succ = pooled([(s, p, n) for s, m, p, n in ifql_rows if m == 32])
flow_succ = pooled([(s, p, n) for s, m, p, n in ifql_rows if m == 1])
ifql_step = int(re.search(r"step_(\d+)", args.ifql_ckpt).group(1))
sweep = pooled([(m, p, n) for s, m, p, n in ifql_rows if s == ifql_step])

fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
axes = axes.ravel()

# A. Demonstrator dataset: episode lengths.
ax = axes[0]
bins = np.arange(0, 1300, 50)
ax.hist(L[S], bins=bins, color=BLUE, label=f"successful ({S.sum():,} eps, {S.mean():.1%})")
ax.hist(L[~S], bins=bins, color=ORANGE, label=f"failed ({(~S).sum():,} eps; time limit 1250)")
ax.axvline(np.median(L[S]), color=TEXT_2, linewidth=1, linestyle="--")
ax.annotate(f"median success length {np.median(L[S]):.0f}", (np.median(L[S]), ax.get_ylim()[1] * 0.92),
            xytext=(6, 0), textcoords="offset points", color=TEXT_2, fontsize=9)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
style(ax, "Demonstration data: episode length (all 100 shards)", "episode length (env steps)", "episodes")

# B. Training data used by each method.
ax = axes[1]
names = ["MSE BC", "IFQL critic (V, Q)", "IFQL actor"]
succ_vals = [bc_samples, ifql_samples_succ, ifql_samples_succ]
fail_vals = [0, ifql_samples_fail, 0]
y = np.arange(len(names))[::-1]
ax.barh(y, succ_vals, color=BLUE, height=0.55, label="from successful episodes")
ax.barh(y, fail_vals, left=succ_vals, color=ORANGE, height=0.55, label="from failed episodes")
for yi, s_, f_ in zip(y, succ_vals, fail_vals):
    ax.annotate(f"{s_ + f_:,}", (s_ + f_, yi), xytext=(6, 0), textcoords="offset points", va="center", color=TEXT_2)
ax.set_yticks(y)
ax.set_yticklabels(names)
ax.set_xlim(0, max(s + f for s, f in zip(succ_vals, fail_vals)) * 1.2)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
style(ax, "Training samples (1 sample = 3-camera image + 25-action chunk)", "samples", "")
ax.legend(loc="lower right")

# C. Success vs samples seen (fair x-axis: BC batch 256, IFQL batch 128).
ax = axes[2]
for succ, batch, label, color in [(bc_succ, args.bc_batch, "MSE BC", BLUE), (ifql_succ, args.ifql_batch, "IFQL (best of 32)", AQUA),
                                  (flow_succ, args.ifql_batch, "flow BC (actor alone)", ORANGE)]:
    s = np.array(list(succ.keys()))
    p, e, _ = zip(*succ.values())
    ax.errorbar(s * batch / 1e6, p, yerr=e, color=color, marker="o", markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5, capsize=3, elinewidth=1.2, label=label)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
style(ax, "Success vs training samples seen", "samples seen (millions) = steps x batch size", "success")

# D. Number of candidates the critic picks from.
ax = axes[3]
ns = np.array(list(sweep.keys()))
p, e, n = zip(*sweep.values())
ax.errorbar(ns, p, yerr=e, color=AQUA, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
            capsize=3, elinewidth=1.2, label=f"IFQL {args.ifql_ckpt.replace('model_step_', '')} steps")
for x, yv in zip(ns, p):
    ax.annotate(f"{yv:.0%}", (x, yv), xytext=(0, 10), textcoords="offset points", ha="center", color=TEXT_2, fontsize=9)
best_bc = max(v[0] for v in bc_succ.values())
ax.axhline(best_bc, color=BLUE, linewidth=1.2, linestyle="--", label=f"best MSE BC ({best_bc:.0%})")
ax.axhline(S.mean(), color=MUTED, linewidth=1.2, linestyle=":", label=f"demonstrator ({S.mean():.0%})")
ax.set_xscale("log", base=2)
ax.set_xticks(ns)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
style(ax, "More candidates for Q to choose from (N=1: actor alone)", "candidates per decision N (log scale)", "success")
ax.legend(loc="lower right")

# E. How fast do episodes succeed? Fraction of episodes solved by step t.
ax = axes[4]
t = np.arange(0, 1251, 5)
curves = [("demonstrator (dataset)", L, S, MUTED, ":")]
bc_npz = sorted(glob.glob(os.path.join(args.bc, "eval", f"{args.bc_ckpt}_seed*.npz")))
if bc_npz:
    eps = [np.load(p) for p in bc_npz]
    curves.append((f"MSE BC {args.bc_ckpt.replace('model_step_', '')}", np.concatenate([x["episode_length"] for x in eps]),
                   np.concatenate([x["episode_success"] for x in eps]).astype(bool), BLUE, "-"))
for n_, label, color in [(32, "IFQL (best of 32)", AQUA), (1, "flow BC", ORANGE)]:
    path = os.path.join(args.ifql, "eval", f"ckpt_{args.ifql_ckpt}_n{n_}.npz")
    if os.path.exists(path):
        x = np.load(path)
        curves.append((f"{label} {args.ifql_ckpt.replace('model_step_', '')}", x["episode_length"],
                       x["episode_success"].astype(bool), color, "-"))
for label, lens, succ, color, ls in curves:
    frac = [(succ & (lens <= ti)).mean() for ti in t]
    ax.plot(t, frac, color=color, linestyle=ls, label=f"{label} (n={len(lens)})")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
style(ax, "Fraction of episodes solved by step t", "env step t", "solved")
ax.legend(loc="upper left")

# F. What the critic adds: IFQL minus flow BC at each checkpoint.
ax = axes[5]
common = sorted(set(ifql_succ) & set(flow_succ))
gain = [ifql_succ[s][0] - flow_succ[s][0] for s in common]
err = [np.hypot(ifql_succ[s][1], flow_succ[s][1]) for s in common]
ax.bar([s / 1000 for s in common], gain, width=6, color=AQUA, yerr=err, capsize=3, ecolor=TEXT_2,
       error_kw=dict(elinewidth=1))
for s, g in zip(common, gain):
    ax.annotate(f"+{g * 100:.0f}" if g >= 0 else f"{g * 100:.0f}", (s / 1000, g), xytext=(0, 8),
                textcoords="offset points", ha="center", color=TEXT_2, fontsize=9)
ax.axhline(0, color=TEXT_2, linewidth=1)
ax.set_xticks([s / 1000 for s in common])
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}k"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:+.0f}"))
style(ax, "Critic's contribution: IFQL success minus flow BC (points)", "training step", "success points", legend=False)

plt.tight_layout()
out = os.path.join(args.out_dir, "analysis_bc_vs_ifql.png")
plt.savefig(out, dpi=110)
print(f"saved {out}")
