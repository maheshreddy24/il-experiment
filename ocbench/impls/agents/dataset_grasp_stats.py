"""Per-episode stats of the demonstrations from qpos: length, success, gripper closes, missed closes, cubes lifted.

python agents/dataset_grasp_stats.py --env_name visual-block-double-task1-v0 --splits val --num_val_shards 100 \
    --out checkpoints/analysis_double/grasp/dataset_episodes.npz
"""
import argparse
import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import ocbench
from bc_utils import dataset_paths
from grasp_utils import CLOSE_ON, LIFT, MISSED


def closes(g):
    """Gripper closes (closure rising past CLOSE_ON) and how many of those end on nothing."""
    rising = np.flatnonzero((g[1:] > CLOSE_ON) & (g[:-1] <= CLOSE_ON)) + 1
    ends = np.append(np.flatnonzero((g[1:] <= CLOSE_ON) & (g[:-1] > CLOSE_ON)) + 1, len(g))
    missed = sum(int(g[r : ends[ends > r][0]].max() > MISSED) for r in rising)
    return len(rising), missed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='visual-block-single-task1-v0')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--num_val_shards', type=int, default=None)
    parser.add_argument('--out', default='checkpoints/analysis/grasp/dataset_episodes.npz')
    args = parser.parse_args()

    # qpos addresses of the gripper joint and each cube's z, from the CPU version of the env.
    state_env = args.env_name.replace('visual-', '', 1).replace('block-', 'block-cpu-', 1)
    u = ocbench.make(state_env).unwrapped
    u.reset(seed=0)
    gripper_q = int(u._model.jnt_qposadr[u._gripper_opening_joint_id])
    cube_z = [int(u._model.joint(f'object_joint_{i}').qposadr[0]) + 2 for i in range(u._num_cubes)]

    train_paths, val_paths = dataset_paths(args.env_name)
    rows = {}
    for split, paths in (('train', train_paths), ('val', val_paths[: args.num_val_shards])):
        if split not in args.splits:
            continue
        L, S, C, M, LIFTED = [], [], [], [], []
        for p in paths:
            with np.load(p) as d:
                t, r, q = d['terminals'], d['rewards'], d['qpos']
            off = np.concatenate([[0], np.flatnonzero(t) + 1])
            for s, e in zip(off[:-1], off[1:]):
                n_close, n_missed = closes(np.clip(q[s:e, gripper_q] / 0.8, 0, 1))
                z = q[s:e][:, cube_z]
                L.append(e - s); S.append(bool(np.any(r[s:e] == 1))); C.append(n_close); M.append(n_missed)
                LIFTED.append(int(((z.max(axis=0) - z[0]) > LIFT).sum()))
        rows[split] = dict(length=np.array(L), success=np.array(S), closes=np.array(C), missed=np.array(M),
                           cubes_lifted=np.array(LIFTED), lifted=np.array(LIFTED) > 0)
        d = rows[split]
        s = d['success']
        print(f"{split}: {len(s)} episodes, success {s.mean():.1%}, success length {d['length'][s].mean():.0f}, "
              f"failures that lifted a cube {d['lifted'][~s].mean() if (~s).any() else float('nan'):.1%}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **{f'{split}_{k}': v for split, d in rows.items() for k, v in d.items()})
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
