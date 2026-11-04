"""Grasp-failure analysis of traced rollouts (eval_grasp.py) vs the demonstrations (dataset_grasp_stats.py).

python agents/plot_grasp.py --seed 2 --ifql_run checkpoints/ifql_torch_<id>
-> checkpoints/analysis/grasp/grasp_analysis_seed<s>.png  + printed table
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
TEXT, TEXT_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.edgecolor": GRID, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "lines.linewidth": 2,
    "legend.frameon": False, "legend.fontsize": 8.5,
})
REACH, CLOSE_ON, MISSED, LIFT = 0.06, 0.2, 0.7, 0.02  # m, gripper closure, gripper closure, m
STEP_SECONDS = 0.02
CATEGORIES = [  # (key, label, color)
    ("success", "success", BLUE),
    ("lifted", "grasped & lifted, not placed", AQUA),
    ("attempted", "tried to grasp, never lifted", YELLOW),
    ("no_attempt", "reached, never tried to grasp", ORANGE),
    ("not_reached", "never reached the cube", MAGENTA),
]

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--settings", nargs="+", default=["bc", "ifql", "flow_bc"])
parser.add_argument("--ifql_run", default=None, help="ifql_torch run dir, for the Q-vs-success-rate panel")
parser.add_argument("--dir", default="checkpoints/analysis/grasp")
args = parser.parse_args()
LABELS = {"bc": "MSE BC 30k", "ifql": "IFQL 50k (best of 32)", "flow_bc": "flow BC 50k (actor alone)"}


def close_events(g, d=None):
    """Gripper closes (closure rising past CLOSE_ON): (count, count near the cube, count ending on nothing)."""
    rising = np.flatnonzero((g[1:] > CLOSE_ON) & (g[:-1] <= CLOSE_ON)) + 1
    ends = np.append(np.flatnonzero((g[1:] <= CLOSE_ON) & (g[:-1] > CLOSE_ON)) + 1, len(g))
    near = missed = 0
    for r in rising:
        e = ends[ends > r][0]
        is_near = d is None or d[r] < REACH
        near += int(is_near)
        missed += int(is_near and g[r:e].max() > MISSED)  # closed near the cube but on nothing: a missed grasp
    return len(rising), near, missed


def analyze(path):
    z = np.load(path, allow_pickle=True)
    succ, length = z["episode_success"].astype(bool), z["episode_length"]
    ee, blk = z["proprio__effector_pos"], z["privileged__block_0_pos"]
    g, out = z["proprio__gripper_opening"][..., 0], dict(success=succ, length=length, q=z["q_chosen"])
    cat, n_near, n_missed, n_all = [], [], [], []
    for i in range(len(succ)):
        T = length[i]
        d = np.linalg.norm(ee[i, :T] - blk[i, :T], axis=-1)
        n_c, n_n, n_m = close_events(g[i, :T], d)
        lifted = blk[i, :T, 2].max() - blk[i, 0, 2] > LIFT
        reached = d.min() < REACH
        n_all.append(n_c); n_near.append(n_n); n_missed.append(n_m)
        if succ[i]:
            cat.append("success")
        elif lifted:
            cat.append("lifted")
        elif n_n > 0:
            cat.append("attempted")
        elif reached:
            cat.append("no_attempt")
        else:
            cat.append("not_reached")
    out.update(category=np.array(cat), closes=np.array(n_all), near_closes=np.array(n_near), missed=np.array(n_missed))
    return out


data = {name: analyze(os.path.join(args.dir, f"{name}_seed{args.seed}.npz")) for name in args.settings}
ds = np.load(os.path.join(args.dir, "dataset_episodes.npz"))

# ---------------- Printed table ----------------
print(f"\n=== Outcomes (seed {args.seed}) ===")
hdr = f"{'setting':28s} {'episodes':>8s} " + " ".join(f"{k:>12s}" for k, _, _ in CATEGORIES)
print(hdr)
for name, d in data.items():
    n = len(d["category"])
    print(f"{LABELS[name]:28s} {n:8d} " + " ".join(f"{(d['category'] == k).sum():5d} ({(d['category'] == k).mean():4.0%})"
                                                    for k, _, _ in CATEGORIES))
for split in ("train", "val"):
    s, lifted, closes = ds[f"{split}_success"], ds[f"{split}_lifted"], ds[f"{split}_closes"]
    n = len(s)
    print(f"{'demonstrator ' + split:28s} {n:8d} {s.sum():5d} ({s.mean():4.0%}) {(~s & lifted).sum():5d} "
          f"({(~s & lifted).mean():4.0%}) {(~s & ~lifted & (closes > 0)).sum():5d} ({(~s & ~lifted & (closes > 0)).mean():4.0%})"
          f"   (no gripper/cube distance in the data: remaining failures {(~s & ~lifted & (closes == 0)).sum()})")

print("\n=== Grasp attempts per episode (gripper closes near the cube; 'missed' = closed on nothing) ===")
for name, d in data.items():
    for tag, m in (("success", d["success"]), ("failure", ~d["success"])):
        if m.any():
            print(f"{LABELS[name]:28s} {tag:8s} n={m.sum():3d}  attempts {d['near_closes'][m].mean():.2f}  "
                  f"missed {d['missed'][m].mean():.2f}  >1 attempt in {(d['near_closes'][m] > 1).mean():.0%} of episodes")
for split in ("train", "val"):
    s = ds[f"{split}_success"]
    for tag, m in (("success", s), ("failure", ~s)):
        print(f"{'demonstrator ' + split:28s} {tag:8s} n={m.sum():4d} closes {ds[f'{split}_closes'][m].mean():.2f}  "
              f"missed {ds[f'{split}_missed'][m].mean():.2f}  >1 close in {(ds[f'{split}_closes'][m] > 1).mean():.0%} of episodes")

print("\n=== Episode duration (steps; seconds = steps x 0.02) ===")
rows = [(f"demonstrator {sp}", ds[f"{sp}_length"], ds[f"{sp}_success"]) for sp in ("train", "val")]
rows += [(LABELS[n], d["length"], d["success"]) for n, d in data.items()]
for label, L, s in rows:
    fs = f"{L[~s].mean():.0f} ({L[~s].mean() * STEP_SECONDS:.1f}s, n={(~s).sum()})" if (~s).any() else "-"
    print(f"{label:28s} success {L[s].mean():6.0f} ± {L[s].std() / np.sqrt(s.sum()):.0f} ({L[s].mean() * STEP_SECONDS:.1f}s, "
          f"n={s.sum()})   failure {fs}")

# ---------------- Figure ----------------
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
axes = axes.ravel()


def style(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, loc="left", fontsize=11.5, color=TEXT)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color=GRID, linewidth=0.8, axis="both")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend()


# A. Outcome breakdown.
ax = axes[0]
names = [LABELS[n] for n in data] + ["demonstrator (train)"]
fracs = [[(d["category"] == k).mean() for k, _, _ in CATEGORIES] for d in data.values()]
s, lifted, closes = ds["train_success"], ds["train_lifted"], ds["train_closes"]
fracs.append([s.mean(), (~s & lifted).mean(), (~s & ~lifted & (closes > 0)).mean(), (~s & ~lifted & (closes == 0)).mean(), 0])
y = np.arange(len(names))[::-1]
left = np.zeros(len(names))
for j, (k, label, color) in enumerate(CATEGORIES):
    vals = np.array([f[j] for f in fracs])
    ax.barh(y, vals, left=left, color=color, height=0.6, label=label, edgecolor=SURFACE, linewidth=2)
    for yi, v, l0 in zip(y, vals, left):
        if v >= 0.04:
            ax.annotate(f"{v:.0%}", (l0 + v / 2, yi), ha="center", va="center", fontsize=9,
                        color="#0b0b0b" if color == YELLOW else "#ffffff")
    left += vals
ax.set_yticks(y)
ax.set_yticklabels([f"{LABELS[n]}\n(n={len(d['category'])})" for n, d in data.items()]
                   + [f"demonstrator (train)\n(n={len(s)})"])
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
style(ax, "Episode outcomes", "share of episodes", "", legend=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

# B. Grasp attempts per episode.
ax = axes[1]
groups = [(LABELS[n], d["near_closes"], d["missed"], d["success"]) for n, d in data.items()]
groups.append(("demonstrator (train)", ds["train_closes"], ds["train_missed"], ds["train_success"]))
x = np.arange(len(groups))
w = 0.38
for k, (tag, color) in enumerate((("successful episodes", BLUE), ("failed episodes", ORANGE))):
    means, errs = [], []
    for _, c, _, sc in groups:
        m = sc if k == 0 else ~sc
        means.append(c[m].mean() if m.any() else np.nan)
        errs.append(c[m].std() / np.sqrt(m.sum()) if m.sum() > 1 else 0)
    bars = ax.bar(x + (k - 0.5) * w, means, width=w, color=color, yerr=errs, capsize=3, ecolor=TEXT_2, label=tag)
    for xi, mval in zip(x + (k - 0.5) * w, means):
        if np.isfinite(mval):
            ax.annotate(f"{mval:.1f}", (xi, mval), xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=9, color=TEXT_2)
ax.set_xticks(x)
ax.set_xticklabels([g[0].replace(" (", "\n(") for g in groups], fontsize=9)
style(ax, "Grasp attempts per episode (gripper closes near the cube)", "", "attempts per episode")

# C. Episode duration.
ax = axes[2]
rows_c = [("demo train", ds["train_length"], ds["train_success"]), ("demo val", ds["val_length"], ds["val_success"])]
rows_c += [(LABELS[n].replace(" (", "\n("), d["length"], d["success"]) for n, d in data.items()]
x = np.arange(len(rows_c))
for k, (tag, color) in enumerate((("successful episodes", BLUE), ("failed episodes", ORANGE))):
    means, errs = [], []
    for _, L, sc in rows_c:
        m = sc if k == 0 else ~sc
        means.append(L[m].mean() * STEP_SECONDS if m.any() else np.nan)
        errs.append(L[m].std() / np.sqrt(m.sum()) * STEP_SECONDS if m.sum() > 1 else 0)
    ax.bar(x + (k - 0.5) * w, means, width=w, color=color, yerr=errs, capsize=3, ecolor=TEXT_2, label=tag)
    for xi, mval in zip(x + (k - 0.5) * w, means):
        if np.isfinite(mval):
            ax.annotate(f"{mval:.1f}s", (xi, mval), xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=9, color=TEXT_2)
ax.set_xticks(x)
ax.set_xticklabels([r[0] for r in rows_c], fontsize=9)
ax.set_ylim(0, 29)
style(ax, "Mean episode duration (failures always hit the 25 s limit)", "", "seconds (1 step = 0.02 s)")

# D. Q at the first chunk vs actual success (calibration), IFQL settings of this run.
ax = axes[3]
for name, color in (("ifql", AQUA), ("flow_bc", ORANGE)):
    if name not in data or len(data[name]["q"]) == 0:
        continue
    q0, sc = data[name]["q"][0], data[name]["success"]
    edges = np.quantile(q0, np.linspace(0, 1, 6))
    which = np.clip(np.digitize(q0, edges[1:-1]), 0, 4)
    xs = [q0[which == b].mean() for b in range(5) if (which == b).any()]
    ys = [sc[which == b].mean() for b in range(5) if (which == b).any()]
    ax.plot(xs, ys, "o-", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label=f"{LABELS[name]}  (Q-success corr r={np.corrcoef(q0, sc)[0, 1]:.2f})")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
style(ax, "Q at the first decision vs actual success (episodes in 5 Q-bins)", "mean predicted Q in bin", "success rate in bin")

# E. Q along the episode by outcome (IFQL).
ax = axes[4]
if "ifql" in data and len(data["ifql"]["q"]):
    d = data["ifql"]
    q = d["q"]
    t = np.arange(len(q)) * 25 * STEP_SECONDS
    valid = (np.arange(len(q))[:, None] * 25) < d["length"][None]
    for k, label, color in CATEGORIES[:3]:
        m = d["category"] == k
        if m.sum() >= 3:
            ax.plot(t, np.nanmean(np.where(valid[:, m], q[:, m], np.nan), axis=1), color=color,
                    label=f"{label} (n={m.sum()})")
style(ax, f"IFQL: Q of the executed chunk along the episode, by outcome", "time in episode (s)", "Q")

# F. Mean Q vs success rate across every evaluated IFQL (checkpoint, N).
ax = axes[5]
if args.ifql_run:
    with open(os.path.join(args.ifql_run, "eval_ckpt.csv")) as f:
        rows_f = list(csv.DictReader(f))
    for n_s, color, label in (("32", AQUA, "N=32"), ("1", ORANGE, "N=1 (actor alone)"), (None, MUTED, "other N")):
        sel = [r for r in rows_f if (r["num_samples"] == n_s if n_s else r["num_samples"] not in ("1", "32"))]
        if sel:
            ax.scatter([float(r["q_init_mean"]) for r in sel], [float(r["success"]) for r in sel], s=45, color=color,
                       edgecolor=SURFACE, linewidth=1.5, label=label, zorder=3)
    qs = np.array([float(r["q_init_mean"]) for r in rows_f])
    sr = np.array([float(r["success"]) for r in rows_f])
    ax.annotate(f"r = {np.corrcoef(qs, sr)[0, 1]:.2f} over {len(rows_f)} evals", (0.03, 0.93), xycoords="axes fraction",
                color=TEXT_2)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
style(ax, "Mean Q at start vs success rate, every IFQL eval (checkpoint x N)", "mean Q at first decision", "success rate")

plt.tight_layout()
out = os.path.join(args.dir, f"grasp_analysis_seed{args.seed}.png")
plt.savefig(out, dpi=105)
print(f"\nsaved {out}")
