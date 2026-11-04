"""Action error on the held-out demonstrations: how far each policy's executed action chunk is from the
demonstrator's, at every held-out state (successful validation episodes).

All errors are in one common unit: actions normalized with the BC checkpoint's normalizer (the BC training loss scale),
so BC, flow BC and IFQL are directly comparable. Reported overall and on "grasp chunks" (chunks during which the
demonstrator closes its gripper).

python agents/eval_action_error.py --bc_run checkpoints/bc_torch_<id> --ifql_run checkpoints/ifql_torch_<id>
-> checkpoints/analysis/heldout/action_error.csv
"""
import argparse
import csv
import os

import numpy as np
import ocbench
import torch
from bc_utils import ChunkDataset, dataset_paths, load_npz, set_seed
from eval_bc_torch import load_policy
from grasp_utils import CLOSE_ON
from ifql_torch import IFQL

GRIPPER_QPOS = 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bc_run', default='checkpoints/bc_torch_1793611331')
    parser.add_argument('--ifql_run', default='checkpoints/ifql_torch_1793629332')
    parser.add_argument('--steps', type=int, nargs='+', default=[10000, 20000, 30000, 40000, 50000, 60000, 70000])
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--env_name', default='visual-block-single-task1-v0')
    parser.add_argument('--num_val_shards', type=int, default=100, help='held-out demos = the first N val shards')
    parser.add_argument('--out', default='checkpoints/analysis/heldout/action_error.csv')
    args = parser.parse_args()
    device = torch.device('cuda')

    # Held-out demos, normalized with the BC normalizer (the common unit).
    _, bc_norm, _ = load_policy(os.path.join(args.bc_run, f'model_step_{args.steps[0]}.pt'), device)
    _, val_paths = dataset_paths(args.env_name)
    val_paths = val_paths[: args.num_val_shards]
    data = load_npz(val_paths, success_only=True)
    ds = ChunkDataset(data, bc_norm, chunk_length=25)
    # Gripper closure per step for the same (successful) episodes; load_dataset doesn't keep qpos.
    gripper = []
    for p in val_paths:
        with np.load(p) as d:
            off = np.concatenate([[0], np.flatnonzero(d['terminals']) + 1])
            for s, e in zip(off[:-1], off[1:]):
                if np.any(d['rewards'][s:e] == 1):
                    gripper.append(d['qpos'][s:e, GRIPPER_QPOS])
    gripper = np.concatenate(gripper)
    assert len(gripper) == len(data['actions']), (len(gripper), len(data['actions']))
    closure = np.clip(gripper / 0.8, 0, 1)[ds.act_idxs]  # (samples, 25)
    grasp_chunk = ((closure[:, 1:] > CLOSE_ON) & (closure[:, :-1] <= CLOSE_ON)).any(axis=1)
    targets = ds.actions[ds.act_idxs].reshape(len(ds), -1)  # BC-normalized demo chunks
    n = len(ds)
    print(f'held-out demo samples: {n} ({grasp_chunk.sum()} grasp chunks)', flush=True)

    def errors(predict):
        """predict(uint8 obs batch on device) -> raw-action-space chunks (B, 25, 7) as numpy."""
        sq = np.empty(n, dtype=np.float64)
        for i in range(0, n, args.batch):
            sl = slice(i, min(i + args.batch, n))
            obs = torch.from_numpy(ds.images[ds.obs_idxs[sl]]).to(device)
            raw = predict(obs)
            pred = bc_norm.normalize(raw.reshape(-1, raw.shape[-1])).reshape(raw.shape[0], -1)
            sq[sl] = ((pred - targets[sl]) ** 2).mean(axis=1)
        return sq

    rows = []
    for step in args.steps:
        set_seed(0)
        model, norm, cfg = load_policy(os.path.join(args.bc_run, f'model_step_{step}.pt'), device)

        @torch.no_grad()
        def bc_predict(obs):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['bf16']):
                out = model(obs).float().cpu().numpy()
            return norm.unnormalize(out.reshape(len(out), 25, -1))

        runs = [('bc', bc_predict)]
        del_model = model
        agent = IFQL.load_for_eval(os.path.join(args.ifql_run, f'model_step_{step}.pt'))
        for name, k in (('ifql', 32), ('flowbc', 1)):
            def ifql_predict(obs, k=k):
                a, _ = agent.sample_actions(obs, k)
                a = a.float().cpu().numpy()
                return agent.normalizer.unnormalize(a.reshape(len(a), 25, -1))
            runs.append((name, ifql_predict))

        for name, predict in runs:
            sq = errors(predict)
            row = dict(run=f'{name}_{step // 1000}k' if step % 1000 == 0 else f'{name}_{step}', method=name, step=step, samples=n, mse=sq.mean(),
                       mse_se=sq.std() / np.sqrt(n), mse_grasp=sq[grasp_chunk].mean(),
                       mse_grasp_se=sq[grasp_chunk].std() / np.sqrt(grasp_chunk.sum()),
                       mse_other=sq[~grasp_chunk].mean(), grasp_samples=int(grasp_chunk.sum()))
            rows.append(row)
            print(f"{row['run']:12s} action MSE {row['mse']:.4f} (grasp chunks {row['mse_grasp']:.4f}, "
                  f"other {row['mse_other']:.4f})", flush=True)
        del agent, del_model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
