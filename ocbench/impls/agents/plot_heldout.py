"""Figures + numbers for the held-out evaluation (eval_heldout.py): all policies start from the exact first states of
the 1000 validation episodes, so they can be compared episode-by-episode with the demonstrator.

python agents/plot_heldout.py   ->  <repo>/plots/heldout/*.png  (+ prints all numbers used in the write-up)
"""
import argparse
import glob
import os
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from grasp_utils import CATEGORIES, STEP_SECONDS
from matplotlib.ticker import FuncFormatter, PercentFormatter

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
TEXT, TEXT_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.edgecolor": GRID, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "lines.linewidth": 2,
    "legend.frameon": False, "legend.fontsize": 8.5,
})
CAT_COLORS = {"success": BLUE, "lifted": AQUA, "grasp_failed": YELLOW, "no_attempt": ORANGE, "not_reached": MAGENTA}
METHODS = {"bc": ("MSE BC", BLUE), "ifql": ("IFQL (best of 32 by Q)", AQUA), "flowbc": ("flow BC (IFQL actor alone)", ORANGE)}

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="checkpoints/analysis/heldout")
parser.add_argument("--dataset_stats", default="checkpoints/analysis/grasp/dataset_episodes.npz")
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "plots", "heldout"))
parser.add_argument("--main", nargs="+", default=["bc_30k", "ifql_50k", "flowbc_50k"],
                    help="runs for the per-model figures, or 'auto' = best checkpoint of each method")
parser.add_argument("--env_label", default="visual-block-single-task1-v0")
args = parser.parse_args()
OUT = os.path.abspath(args.out)
os.makedirs(OUT, exist_ok=True)

runs = {os.path.basename(p)[:-4]: dict(np.load(p, allow_pickle=True)) for p in sorted(glob.glob(f"{args.dir}/*.npz"))}
if args.main == ["auto"]:  # best checkpoint per method on the held-out starts
    args.main = [max((n for n in runs if n.startswith(k + "_")), key=lambda n: runs[n]["episode_success"].mean())
                 for k in ("bc", "ifql", "flowbc") if any(n.startswith(k + "_") for n in runs)]
    print("per-model figures for:", args.main)
ds = np.load(args.dataset_stats)
demo = dict(success=ds["val_success"], length=ds["val_length"], lifted=ds["val_lifted"], closes=ds["val_closes"])
N_DEMO = len(demo["success"])


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


def se(p, n):
    return np.sqrt(p * (1 - p) / n)


def demo_categories():
    """Demonstrator outcome per held-out episode. The data has no gripper position, so 'never reached' and 'reached,
    never tried' can't be separated; both go to no_attempt (gripper never closed)."""
    s, lifted, closes = demo["success"], demo["lifted"], demo["closes"]
    return np.where(s, "success", np.where(lifted, "lifted", np.where(closes > 0, "grasp_failed", "no_attempt")))


# ------------------------------------------------------------------ numbers
print(f"held-out episodes: {N_DEMO}; demonstrator success on them (recorded): {demo['success'].mean():.1%}\n")
print(f"{'run':14s} {'success':>16s} " + " ".join(f"{k:>14s}" for k, _ in CATEGORIES[1:]) +
      "  attempts/ep(succ)  >1 attempt(succ)  1st attempt ok  success time")
for name, r in runs.items():
    cat, s = r["stat_category"], r["episode_success"].astype(bool)
    n = len(s)
    att = r["stat_attempts"]
    first_ok = r["stat_first_attempt_ok"][r["stat_attempts"] > 0].mean()
    print(f"{name:14s} {s.mean():6.1%} ± {se(s.mean(), n):.1%} ({s.sum():4d}) " +
          " ".join(f"{(cat == k).sum():5d} ({(cat == k).mean():4.1%})" for k, _ in CATEGORIES[1:]) +
          f"  {att[s].mean():5.2f}             {(att[s] > 1).mean():5.1%}            {first_ok:5.1%}"
          f"        {r['episode_length'][s].mean() * STEP_SECONDS:5.1f}s")
dc = demo_categories()
print(f"{'demonstrator':14s} {demo['success'].mean():6.1%}         ({demo['success'].sum():4d}) " +
      " ".join(f"{(dc == k).sum():5d} ({(dc == k).mean():4.1%})" for k, _ in CATEGORIES[1:]) +
      f"  {demo['closes'][demo['success']].mean():5.2f} (all closes)  {(demo['closes'][demo['success']] > 1).mean():5.1%}"
      f"                     {demo['length'][demo['success']].mean() * STEP_SECONDS:5.1f}s")

if any((r.get("stat_cubes_placed", np.array([-1])) >= 0).all() and r["stat_cubes_placed"].max() > 1 for r in runs.values()):
    print("\nCubes at their goal at the end of the episode (multi-cube tasks):")
    for name, r in runs.items():
        cp = r["stat_cubes_placed"]
        print(f"  {name:12s} " + "  ".join(f"{k} cubes: {(cp == k).mean():5.1%}" for k in range(cp.max() + 1)))

print("\nPaired with the demonstrator on the same starts (main runs):")
for name in args.main:
    s = runs[name]["episode_success"].astype(bool)
    ds_ = demo["success"][: len(s)]
    print(f"  {name:12s} both succeed {(s & ds_).mean():.1%} | demo only {(~s & ds_).mean():.1%} | "
          f"policy only {(s & ~ds_).mean():.1%} | both fail {(~s & ~ds_).mean():.1%}")

# ------------------------------------------------------------------ 00: success at every checkpoint
fig, ax = plt.subplots(figsize=(11, 6))
for key, (label, color) in METHODS.items():
    pts = sorted((int(re.search(r"_(\d+)k$", n).group(1)) * 1000, r["episode_success"].mean(), len(r["episode_success"]))
                 for n, r in runs.items() if n.startswith(key + "_"))
    if not pts:
        continue
    s, p, n = map(np.array, zip(*pts))
    ax.errorbar(s, p, yerr=se(p, n), color=color, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                capsize=4, elinewidth=1.2, label=label)
    dx, ha = {"bc": (-10, "right"), "ifql": (10, "left"), "flowbc": (0, "center")}[key]
    for x, y in zip(s, p):
        ax.annotate(f"{y:.0%}", (x, y), xytext=(dx, -14 if key == "flowbc" else 0), textcoords="offset points",
                    ha=ha, va="center", fontsize=9, color=TEXT_2)
ax.axhline(demo["success"].mean(), color=MUTED, linestyle=":", linewidth=1.5,
           label=f"demonstrator on the same starts ({demo['success'].mean():.0%})")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
style(ax, f"{args.env_label}: success on the {N_DEMO} held-out starts, every checkpoint (±1 s.e.)", "training step",
      "success")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "00_success_all_checkpoints.png"), dpi=120)
plt.close()


# ------------------------------------------------------------------ per-model figures
def outcome_panel(ax, r, label):
    cat = r["stat_category"]
    keys = [k for k, _ in CATEGORIES]
    x = np.arange(len(keys))
    w = 0.38
    model = np.array([(cat == k).sum() for k in keys])
    demo_c = np.array([(dc[: len(cat)] == k).sum() for k in keys])
    ax.bar(x - w / 2, model, width=w, color=[CAT_COLORS[k] for k in keys], label=label)
    ax.bar(x + w / 2, demo_c, width=w, color=MUTED, alpha=0.55, label="demonstrator (same starts)")
    for xi, v in zip(x - w / 2, model):
        ax.annotate(f"{v}", (xi, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=TEXT_2)
    for xi, v in zip(x + w / 2, demo_c):
        ax.annotate(f"{v}", (xi, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(lbl, 14) for _, lbl in CATEGORIES], fontsize=8.5)
    style(ax, f"How the {len(cat)} episodes ended", "", "episodes")


def attempts_panel(ax, r):
    att, s = r["stat_attempts"], r["episode_success"].astype(bool)
    bins = np.arange(0, 6)
    labels = ["0", "1", "2", "3", "4", "5+"]
    w = 0.38
    for k, (m, tag, color) in enumerate(((s, "successful episodes", BLUE), (~s, "failed episodes", ORANGE))):
        counts = np.array([(np.minimum(att[m], 5) == b).sum() for b in bins])
        ax.bar(bins + (k - 0.5) * w, counts, width=w, color=color, label=f"{tag} (n={m.sum()})")
    ax.set_xticks(bins)
    ax.set_xticklabels(labels)
    style(ax, "Grasp attempts per episode (gripper closes near the cube)", "attempts", "episodes")


def solved_panel(ax, r, label, color):
    t = np.arange(0, 1251, 5)
    for lbl, L, s, c, ls in ((label, r["episode_length"], r["episode_success"].astype(bool), color, "-"),
                             ("demonstrator (same starts)", demo["length"], demo["success"], MUTED, ":")):
        ax.plot(t * STEP_SECONDS, [(s & (L <= ti)).mean() for ti in t], color=c, linestyle=ls,
                label=f"{lbl}: {s.mean():.0%} by 25 s")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    style(ax, "Fraction of episodes solved by time t", "time (s)", "solved")
    ax.legend(loc="upper left")


def q_panels(axes, r):
    q, L, cat = r["q_chosen"], r["episode_length"], r["stat_category"]
    valid = (np.arange(len(q))[:, None] * 25) < L[None]
    qv = np.where(valid, q, np.nan)

    # Q along the episode, by outcome.
    ax = axes[0]
    t = np.arange(len(q)) * 25 * STEP_SECONDS
    for k, lbl in CATEGORIES[:3]:
        m = cat == k
        if m.sum() >= 5:
            mean = np.nanmean(qv[:, m], axis=1)
            count = (~np.isnan(qv[:, m])).sum(axis=1)
            ax.plot(t[count >= 5], mean[count >= 5], color=CAT_COLORS[k], label=f"{lbl} (n={m.sum()})")
    style(ax, "Q of the executed chunk along the episode", "time in episode (s)", "Q")

    # Q aligned to the first grasp attempt.
    ax = axes[1]
    first, ok = r["stat_first_attempt_step"], r["stat_first_attempt_ok"].astype(bool)
    rel = np.arange(-4, 9)  # chunks relative to the chunk containing the first attempt
    for m, lbl, color in ((ok & (first >= 0), "first grasp attempt lifted the cube", BLUE),
                          (~ok & (first >= 0), "first grasp attempt failed", YELLOW)):
        idx = np.flatnonzero(m)
        if len(idx) < 3:
            continue
        rows = []
        for i in idx:
            c0 = first[i] // 25
            rows.append([qv[c0 + d, i] if 0 <= c0 + d < len(q) else np.nan for d in rel])
        rows = np.array(rows)
        mean = np.nanmean(rows, axis=0)
        err = np.nanstd(rows, axis=0) / np.sqrt(np.maximum((~np.isnan(rows)).sum(axis=0), 1))
        ax.errorbar(rel * 25 * STEP_SECONDS, mean, yerr=err, color=color, marker="o", markersize=5, capsize=2,
                    elinewidth=1, label=f"{lbl} (n={len(idx)})")
    ax.axvline(0, color=TEXT_2, linewidth=1, linestyle="--")
    ax.annotate("first grasp attempt", (0, ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points",
                color=TEXT_2, fontsize=9)
    style(ax, "Q around the first grasp attempt", "time relative to first grasp attempt (s)", "Q")

    # Q at the first decision vs success.
    ax = axes[2]
    q0, s = q[0], r["episode_success"].astype(bool)
    edges = np.quantile(q0, np.linspace(0, 1, 11))
    which = np.clip(np.digitize(q0, edges[1:-1]), 0, 9)
    bins = [b for b in range(10) if (which == b).any()]
    xs = np.array([q0[which == b].mean() for b in bins])
    ys = np.array([s[which == b].mean() for b in bins])
    ns = np.array([(which == b).sum() for b in bins])
    ax.errorbar(xs, ys, yerr=se(ys, ns), color=AQUA, marker="o", markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5, capsize=3, elinewidth=1.2,
                label=f"episodes in 10 bins of starting Q (corr r={np.corrcoef(q0, s)[0, 1]:.2f})")
    ax.axhline(s.mean(), color=MUTED, linestyle=":", linewidth=1.2, label=f"overall success {s.mean():.0%}")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    style(ax, "Does Q at the start predict success?", "Q of the first executed chunk (bin mean)", "success rate in bin")


for i, name in enumerate(args.main, start=1):
    key = name.split("_")[0]
    label, color = METHODS[key]
    r = runs[name]
    has_q = key == "ifql"
    fig, axes = plt.subplots(2 if has_q else 1, 3, figsize=(19, 10.5 if has_q else 5.4), squeeze=False)
    axes = axes.ravel()
    outcome_panel(axes[0], r, f"{label} {name.split('_')[1]}")
    attempts_panel(axes[1], r)
    solved_panel(axes[2], r, f"{label} {name.split('_')[1]}", color)
    if has_q:
        q_panels(axes[3:], r)
    fig.suptitle(f"{args.env_label} | {label}, checkpoint {name.split('_')[1]}: {r['episode_success'].mean():.1%} success on "
                 f"{len(r['episode_success'])} held-out starts", x=0.01, ha="left", fontsize=13, color=TEXT)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(os.path.join(OUT, f"0{i}_{name}.png"), dpi=110)
    plt.close()


# ------------------------------------------------------------------ 04: success, action error, grasp failures
def per_checkpoint(key, value_fn):
    pts = sorted((int(re.search(r"_(\d+)k$", n).group(1)) * 1000, *value_fn(r)) for n, r in runs.items()
                 if n.startswith(key + "_"))
    return [np.array(v) for v in zip(*pts)] if pts else None


action_csv = os.path.join(args.dir, "action_error.csv")
action_rows = []
if os.path.exists(action_csv):
    import csv

    with open(action_csv) as f:
        action_rows = list(csv.DictReader(f))
panels = 3 if action_rows else 2
fig, axes_ = plt.subplots(1, panels, figsize=(21 if panels == 3 else 15, 6.5))
ax_succ, ax_err, ax_grasp = axes_[0], (axes_[1] if panels == 3 else None), axes_[-1]
LABEL_POS = {"bc": (-8, 0, "right"), "ifql": (8, 0, "left"), "flowbc": (0, -13, "center")}


def label_points(ax, key, s, p):
    dx, dy, ha = LABEL_POS[key]
    for x, y in zip(s, p):
        ax.annotate(f"{y:.0%}", (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=8, color=TEXT_2)


for key, (label, color) in METHODS.items():
    for ax, fn in ((ax_succ, lambda r: (r["episode_success"].mean(), len(r["episode_success"]))),
                   (ax_grasp, lambda r: ((r["stat_category"] == "grasp_failed").mean(), len(r["stat_category"])))):
        res = per_checkpoint(key, fn)
        if res is not None:
            s, p, n = res
            ax.errorbar(s, p, yerr=se(p, n), color=color, marker="o", markersize=7, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, capsize=3, elinewidth=1.2, label=label)
            label_points(ax, key, s, p)
    rows = sorted((int(r["step"]), float(r["mse"]), float(r["mse_se"]), float(r["mse_grasp"]),
                   float(r["mse_grasp_se"])) for r in action_rows if r["method"] == key)
    if rows and ax_err is not None:
        s, m, m_se, g, g_se = map(np.array, zip(*rows))
        ax_err.errorbar(s, m, yerr=m_se, color=color, marker="o", markersize=7, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, capsize=3, label=f"{label}: all chunks")
        ax_err.errorbar(s, g, yerr=g_se, color=color, linestyle="--", marker="s", markersize=5, capsize=3,
                        alpha=0.8, label=f"{label}: grasp chunks")
dc_fail = (dc == "grasp_failed").mean()
ax_succ.axhline(demo["success"].mean(), color=MUTED, linestyle=":", linewidth=1.5,
                label=f"demonstrator ({demo['success'].mean():.0%})")
ax_grasp.axhline(dc_fail, color=MUTED, linestyle=":", linewidth=1.5, label=f"demonstrator ({dc_fail:.1%})")
for ax, title, ylabel, ylim in (
    (ax_succ, f"Success rate ({N_DEMO} held-out starts)", "success", (0, 1)),
    (ax_err, "Action error vs the demonstrator (held-out demo states)", "MSE per action dim (BC-normalized)", None),
    (ax_grasp, f"Tried to grasp, never lifted ({N_DEMO} held-out starts)", "share of episodes", (0, 0.6)),
):
    if ax is None:
        continue
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    if ylim:
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    else:
        ax.set_ylim(bottom=0)
    style(ax, title, "training step", ylabel)
ax_succ.legend(loc="lower right")
ax_grasp.legend(loc="upper right")
if ax_err is not None:
    ax_err.legend(fontsize=8)
fig.suptitle(args.env_label, x=0.01, ha="left", fontsize=13, color=TEXT)
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig(os.path.join(OUT, "04_success_action_error_grasp_failures.png"), dpi=115)
plt.close()

print(f"\nsaved figures to {OUT}")
