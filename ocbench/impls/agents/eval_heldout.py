"""Evaluate policies from the exact first states of the 1000 held-out (validation) episodes.

python agents/eval_heldout.py --run bc_30k:checkpoints/bc_torch_<id>/model_step_30000.pt \
                              --run ifql_50k:checkpoints/ifql_torch_<id>/model_step_50000.pt:32 [--traces]
-> checkpoints/analysis/heldout/<name>.npz with per-episode success, length, grasp stats and Q (IFQL);
   full per-step traces too with --traces.
"""
import argparse
import os
import time

import numpy as np
import ocbench
import torch
from bc_utils import evaluate, load_heldout_starts, set_seed
from eval_bc_torch import load_policy
from grasp_utils import analyze_traces
from ifql_torch import IFQL


def load(spec, device):
    """Returns (policy factory taking a Q-records list, config, normalizer)."""
    parts = spec.split(':')
    path = parts[1]
    if len(parts) == 3:  # IFQL checkpoint + number of candidates
        agent = IFQL.load_for_eval(path)
        return (lambda records: agent.make_policy(int(parts[2]), records)), agent.config, agent.normalizer
    model, normalizer, config = load_policy(path, device)

    @torch.no_grad()
    def policy(obs_np):
        preds = []
        for i in range(0, len(obs_np), 50):
            x = torch.from_numpy(obs_np[i : i + 50]).to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config['bf16']):
                preds.append(model(x).float().cpu().numpy())
        return np.concatenate(preds)

    return (lambda records: policy), config, normalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', required=True, help='name:checkpoint[:num_samples]')
    parser.add_argument('--traces', action='store_true', help='also save full per-step gripper/cube traces')
    parser.add_argument('--batch', type=int, default=250, help='parallel worlds per batch')
    parser.add_argument('--limit', type=int, default=None, help='only the first N held-out starts (for testing)')
    parser.add_argument('--env_name', default='visual-block-single-task1-v0')
    parser.add_argument('--num_val_shards', type=int, default=100, help='held-out set = episodes of the first N val shards')
    parser.add_argument('--out_dir', default='checkpoints/analysis/heldout')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda')
    qpos, qvel, _ = load_heldout_starts(args.env_name, args.num_val_shards)
    qpos, qvel = qpos[: args.limit], qvel[: args.limit]
    n = len(qpos)

    for spec in args.run:
        name = spec.split(':')[0]
        out = os.path.join(args.out_dir, f'{name}.npz')
        if os.path.exists(out):
            print(f'{name}: exists, skipping', flush=True)
            continue
        start = time.time()
        make_policy, config, normalizer = load(spec, device)
        success, length, q, stats, traces = [], [], [], [], []
        for b in range(0, n, args.batch):
            sl = slice(b, min(b + args.batch, n))
            records = []
            set_seed(0)
            st = evaluate(make_policy(records), args.env_name, normalizer, sl.stop - sl.start, config['chunk_length'],
                          record_traces=True, init_states=(qpos[sl], qvel[sl]))
            success.append(st['episode_success'])
            length.append(st['episode_length'])
            max_chunks = -(-ocbench.parse_env_spec(args.env_name)[3] // config['chunk_length'])
            qb = np.full((max_chunks, sl.stop - sl.start), np.nan, dtype=np.float32)  # Q of each executed chunk
            if records:
                qc = np.stack([r['q_chosen'] for r in records])
                qb[: len(qc)] = qc
            q.append(qb)
            stats.append(analyze_traces(st['traces'], st['episode_length'], st['episode_success'], st['goals']))
            if args.traces:
                traces.append(st['traces'])
        del make_policy
        torch.cuda.empty_cache()
        result = dict(checkpoint=spec, episode_success=np.concatenate(success), episode_length=np.concatenate(length),
                      q_chosen=np.concatenate(q, axis=1))
        result.update({f'stat_{k}': np.concatenate([s[k] for s in stats]) for k in stats[0]})
        if args.traces:
            result.update({f"trace_{k.replace('/', '__')}": np.concatenate([t[k] for t in traces]) for k in traces[0]})
        np.savez_compressed(out, **result)
        print(f"{name}: success {result['episode_success'].mean():.3f} over {n} held-out starts "
              f"({time.time() - start:.0f}s) -> {out}", flush=True)


if __name__ == '__main__':
    main()
