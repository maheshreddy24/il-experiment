"""Roll out policies with per-step gripper/cube traces, for grasp-failure analysis (see plot_grasp.py).

python agents/eval_grasp.py --episodes 200 --seed 2 \
    --setting bc:checkpoints/bc_torch_<id>/model_step_30000.pt \
    --setting ifql:checkpoints/ifql_torch_<id>/model_step_50000.pt:32 \
    --setting flow_bc:checkpoints/ifql_torch_<id>/model_step_50000.pt:1
-> checkpoints/analysis/grasp/<name>_seed<s>.npz
"""
import argparse
import os

import numpy as np
import torch
from bc_utils import evaluate, set_seed
from eval_bc_torch import load_policy
from ifql_torch import IFQL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', action='append', required=True,
                        help='name:checkpoint (BC) or name:checkpoint:num_samples (IFQL)')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--out_dir', default='checkpoints/analysis/grasp')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda')

    for setting in args.setting:
        parts = setting.split(':')
        name, path = parts[0], parts[1]
        records, q_chosen = [], None
        if len(parts) == 3:  # IFQL checkpoint with num_samples
            agent = IFQL.load_for_eval(path)
            config, normalizer = agent.config, agent.normalizer
            policy = agent.make_policy(int(parts[2]), records)
        else:
            model, normalizer, config = load_policy(path, device)

            @torch.no_grad()
            def policy(obs_np, model=model, config=config):
                preds = []
                for i in range(0, len(obs_np), 50):
                    x = torch.from_numpy(obs_np[i : i + 50]).to(device)
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config['bf16']):
                        preds.append(model(x).float().cpu().numpy())
                return np.concatenate(preds)

        set_seed(args.seed)
        stats = evaluate(policy, config['env_name'], normalizer, args.episodes, config['chunk_length'], record_traces=True)
        if records:
            q_chosen = np.stack([r['q_chosen'] for r in records])  # (num_chunks, episodes)
        out = os.path.join(args.out_dir, f'{name}_seed{args.seed}.npz')
        np.savez_compressed(
            out, checkpoint=path, episode_success=stats['episode_success'], episode_length=stats['episode_length'],
            q_chosen=q_chosen if q_chosen is not None else np.zeros((0, args.episodes)),
            **{k.replace('/', '__'): v for k, v in stats['traces'].items()},
        )
        print(f"{name}: success {stats['success']:.3f}, length {stats['length']:.0f} -> {out}", flush=True)


if __name__ == '__main__':
    main()
