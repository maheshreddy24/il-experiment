"""Videos of a policy on the first N held-out starting states (front camera), named by outcome.

python agents/heldout_videos.py --run bc_60k:checkpoints/bc_double/model_step_60000.pt \
    --env_name visual-block-double-task1-v0 --episodes 8 --out_dir checkpoints/bc_double/videos_heldout
IFQL: --run ifql_70k:checkpoints/ifql_double/model_step_70000.pt:32  (":1" = actor alone)
Small footprint (a few worlds, ~2 GB GPU), so it can run next to a training job.
"""
import argparse
import os

import imageio
import numpy as np
import torch
from bc_utils import evaluate, load_heldout_starts, set_seed
from eval_heldout import load
from grasp_utils import analyze_traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='name:checkpoint[:num_samples]')
    parser.add_argument('--env_name', default='visual-block-double-task1-v0')
    parser.add_argument('--episodes', type=int, default=8, help='first N held-out episodes')
    parser.add_argument('--frame_skip', type=int, default=5, help='record every k-th env step (5 = real time at 10 fps)')
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    qpos, qvel, _ = load_heldout_starts(args.env_name, num_val_shards=-(-args.episodes // 10))
    qpos, qvel = qpos[: args.episodes], qvel[: args.episodes]
    make_policy, config, normalizer = load(args.run, torch.device('cuda'))
    set_seed(0)
    st = evaluate(make_policy([]), args.env_name, normalizer, args.episodes, config['chunk_length'],
                  video_episodes=args.episodes, video_frame_skip=args.frame_skip, record_traces=True,
                  init_states=(qpos, qvel))
    stats = analyze_traces(st['traces'], st['episode_length'], st['episode_success'], st['goals'])
    name = args.run.split(':')[0]
    for i, frames in enumerate(st['videos']):
        tag = f"{stats['category'][i]}_placed{stats['cubes_placed'][i]}_{st['episode_length'][i] * 0.02:.0f}s"
        path = os.path.join(args.out_dir, f'{name}_heldout{i:03d}_{tag}.mp4')
        imageio.mimsave(path, frames, fps=max(1, round(50 / args.frame_skip)))
        print(f'{path}  (grasp attempts {stats["attempts"][i]}, cubes lifted {stats["cubes_lifted"][i]})', flush=True)
    print(f"success {np.mean(st['episode_success']):.0%} on {args.episodes} held-out starts")


if __name__ == '__main__':
    main()
