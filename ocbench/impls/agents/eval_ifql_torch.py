"""Evaluate ifql_torch checkpoints: success rate with Q-based selection vs. the actor alone, Q statistics, videos.

python agents/eval_ifql_torch.py checkpoints/ifql_torch_<id>/model_step_*.pt --episodes 100 --videos 4
python agents/eval_ifql_torch.py <ckpt> --num_samples 1 8 32 64     # sweep how many candidates the critic picks from

Writes next to the checkpoints:
  eval_ckpt.csv                                one row per (checkpoint, num_samples)
  eval/ckpt_<ckpt>_n<N>.npz                    per-episode success/length and Q of every chosen chunk
  videos/<ckpt>_n<N>_ep<i>_<ok|fail>.mp4
Plot with: python agents/plot_ifql.py checkpoints/ifql_torch_<id>
"""
import argparse
import csv
import os
import re
import time

import imageio
import numpy as np
from bc_utils import evaluate, set_seed
from ifql_torch import IFQL


def step_of(path):
    match = re.search(r'step_(\d+)', path)
    return int(match.group(1)) if match else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', nargs='+')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--num_samples', type=int, nargs='+', default=[32, 1],
                        help='candidates per state; 1 = actor alone (flow BC), >1 = pick the highest-Q candidate')
    parser.add_argument('--videos', type=int, default=0, help='episodes to record per (checkpoint, num_samples)')
    parser.add_argument('--video_frame_skip', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    for path in sorted(args.checkpoints, key=step_of):
        agent = IFQL.load_for_eval(path)
        config = agent.config
        name = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.dirname(path)
        os.makedirs(os.path.join(out_dir, 'eval'), exist_ok=True)

        for num_samples in args.num_samples:
            set_seed(args.seed)  # same initial states for every checkpoint and num_samples
            records = []
            start = time.time()
            stats = evaluate(
                agent.make_policy(num_samples, records), config['env_name'], agent.normalizer, args.episodes,
                config['chunk_length'], video_episodes=args.videos, video_frame_skip=args.video_frame_skip,
            )
            success = stats['episode_success'].astype(np.float32)
            q_chosen = np.stack([r['q_chosen'] for r in records])  # (num_chunks, episodes)
            q_samples_mean = np.stack([r['q_samples_mean'] for r in records])
            np.savez(
                os.path.join(out_dir, 'eval', f'ckpt_{name}_n{num_samples}.npz'),
                episode_success=success, episode_length=stats['episode_length'], q_chosen=q_chosen,
                q_samples_mean=q_samples_mean, q_samples_std=np.stack([r['q_samples_std'] for r in records]),
            )
            row = dict(
                checkpoint=os.path.basename(path),
                step=step_of(path),
                num_samples=num_samples,
                episodes=args.episodes,
                seed=args.seed,
                success=float(success.mean()),
                success_stderr=float(success.std() / np.sqrt(args.episodes)),
                length=float(stats['length']),
                q_init_mean=float(q_chosen[0].mean()),
                q_init_success_eps=float(q_chosen[0][success == 1].mean()) if success.any() else np.nan,
                q_init_failure_eps=float(q_chosen[0][success == 0].mean()) if (success == 0).any() else np.nan,
                q_selection_gain=float((q_chosen - q_samples_mean).mean()),
                eval_time_s=time.time() - start,
            )

            if args.videos:
                os.makedirs(os.path.join(out_dir, 'videos'), exist_ok=True)
                for i, frames in enumerate(stats['videos']):
                    tag = 'ok' if success[i] else 'fail'
                    fps = max(1, round(50 / args.video_frame_skip))  # 50 env steps per sim second -> real speed
                    imageio.mimsave(os.path.join(out_dir, 'videos', f'{name}_n{num_samples}_ep{i}_{tag}.mp4'), frames, fps=fps)

            print(
                f"{row['checkpoint']} N={num_samples}: success {row['success']:.3f} ± {row['success_stderr']:.3f}, "
                f"length {row['length']:.0f}, Q_init {row['q_init_mean']:.3f} "
                f"(succ {row['q_init_success_eps']:.3f} / fail {row['q_init_failure_eps']:.3f}), "
                f"selection gain {row['q_selection_gain']:.4f} ({row['eval_time_s']:.0f}s)",
                flush=True,
            )
            csv_path = os.path.join(out_dir, 'eval_ckpt.csv')
            new_file = not os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if new_file:
                    writer.writeheader()
                writer.writerow(row)


if __name__ == '__main__':
    main()
