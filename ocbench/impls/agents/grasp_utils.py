"""Per-episode grasp analysis from rollout traces (gripper position/closure and cube positions at every step).
Works for any number of cubes: "near the cube" = near the closest cube, "lifted" = any cube lifted.

Gripper closure ("proprio/gripper_opening"): 0 = open, ~0.3 = closed on a cube, ~0.8-1.0 = closed on nothing.
"""
import re

import numpy as np

REACH = 0.06      # m, gripper pinch point to the nearest cube center: "near a cube"
CLOSE_ON = 0.2    # closure above this = gripper closed / closing
MISSED = 0.7      # a close that reaches this closed on nothing
LIFT = 0.02       # m, a cube raised above its start height = lifted
PLACED = 0.04     # m, cube within this distance of its goal at the end = placed
STEP_SECONDS = 0.02
CATEGORIES = [  # failure categories are mutually exclusive, in this order of precedence
    ("success", "success"),
    ("lifted", "lifted a cube, task not completed"),
    ("grasp_failed", "tried to grasp, never lifted"),
    ("no_attempt", "reached, never tried to grasp"),
    ("not_reached", "never reached a cube"),
]


def analyze_episode(ee, blocks, g, T, success, goals=None):
    """ee: (T_max, 3); blocks: (T_max, n_cubes, 3); g: (T_max,) closure; T: episode length; goals: (n_cubes, 3)."""
    ee, blocks, g = ee[:T], blocks[:T], g[:T]
    d = np.linalg.norm(ee[:, None] - blocks, axis=-1).min(axis=1)  # distance to the nearest cube
    z_rise = blocks[:, :, 2] - blocks[0, :, 2]                      # (T, n_cubes)
    any_rise = z_rise.max(axis=1)
    rising = np.flatnonzero((g[1:] > CLOSE_ON) & (g[:-1] <= CLOSE_ON)) + 1
    ends = np.append(np.flatnonzero((g[1:] <= CLOSE_ON) & (g[:-1] > CLOSE_ON)) + 1, T)
    attempts = []  # (start step, lifted a cube during this close, closed on nothing)
    for r in rising:
        if d[r] >= REACH:
            continue
        e = ends[ends > r][0]
        attempts.append((r, bool(any_rise[r:e].max() > LIFT), bool(g[r:e].max() > MISSED)))
    cubes_lifted = int((z_rise.max(axis=0) > LIFT).sum())
    reached = bool(d.min() < REACH)
    if success:
        category = "success"
    elif cubes_lifted:
        category = "lifted"
    elif attempts:
        category = "grasp_failed"
    elif reached:
        category = "no_attempt"
    else:
        category = "not_reached"
    cubes_placed = -1
    if goals is not None and len(goals) == blocks.shape[1]:
        cubes_placed = int((np.linalg.norm(blocks[-1] - goals, axis=-1) <= PLACED).sum())
    return dict(
        category=category,
        attempts=len(attempts),
        failed_attempts=sum(not ok for _, ok, _ in attempts),
        missed_closes=sum(m for _, _, m in attempts),
        first_attempt_step=attempts[0][0] if attempts else -1,
        first_attempt_ok=attempts[0][1] if attempts else False,
        first_lift_step=int(np.argmax(any_rise > LIFT)) if cubes_lifted else -1,
        reached=reached,
        lifted=cubes_lifted > 0,
        cubes_lifted=cubes_lifted,
        cubes_placed=cubes_placed,
    )


def cube_keys(traces):
    return sorted((k for k in traces if re.fullmatch(r'privileged/block_\d+_pos', k)),
                  key=lambda k: int(re.search(r'\d+', k).group()))


def analyze_traces(traces, lengths, successes, goals=None):
    """traces: dict from bc_utils.evaluate(record_traces=True). Returns dict of per-episode arrays."""
    ee = traces['proprio/effector_pos']
    blocks = np.stack([traces[k] for k in cube_keys(traces)], axis=2)  # (episodes, T, n_cubes, 3)
    g = traces['proprio/gripper_opening'][..., 0]
    rows = [analyze_episode(ee[i], blocks[i], g[i], int(lengths[i]), bool(successes[i]), goals)
            for i in range(len(lengths))]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}
