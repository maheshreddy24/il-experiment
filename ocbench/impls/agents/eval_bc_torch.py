"""Evaluate bc_torch checkpoints: success rate, train/val loss, and videos.

python agents/eval_bc_torch.py checkpoints/bc_torch_<id>/model_step_*.pt --episodes 100 --videos 4 --loss

Writes next to the checkpoints:
  eval.csv                        one row per checkpoint (success, loss, ...)
  eval/<ckpt>_seed<s>.npz         per-episode success and length (appended across repeated evals)
  videos/<ckpt>_ep<i>_<ok|fail>.mp4
  eval_summary.png                success and loss vs. training step
"""
import argparse
import csv
import os
import re
import time

import bc_utils  # sets MUJOCO_GL=egl before mujoco is imported
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from bc_torch import Policy
from bc_utils import ActionNormalizer, evaluate, make_datasets, set_seed
from torch.utils.data import DataLoader


def load_policy(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt['config']
    normalizer = ActionNormalizer(**ckpt['action_normalizer'])
    state = ckpt['model_state_dict']
    num_cameras = state['mlp.inp.0.weight'].shape[1] // 512
    action_dim = len(normalizer.center)
    model = Policy(num_cameras, action_dim, config['chunk_length'], config['hidden_dim'], config['num_blocks']).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, normalizer, config


@torch.no_grad()
def dataset_loss(model, dataset, batch_size, bf16, device):
    """Mean action-chunk MSE over a whole dataset (normalized action space, same as the training loss)."""
    total, n = 0.0, 0
    for obs, act in DataLoader(dataset, batch_size=batch_size, num_workers=4):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            pred = model(obs.to(device)).float()
        total += F.mse_loss(pred, act.to(device), reduction='sum').item() / act.shape[1]
        n += len(act)
    return total / n


def step_of(name):
    match = re.search(r'step_(\d+)', name)
    return int(match.group(1)) if match else -1


def plot_summary(csv_path):
    with open(csv_path) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: step_of(r['checkpoint']))
    steps = np.array([step_of(r['checkpoint']) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    success = np.array([float(r['success']) for r in rows])
    stderr = np.array([float(r['success_stderr']) for r in rows])
    axes[0].errorbar(steps, success, yerr=stderr, fmt='o-', capsize=3)
    axes[0].set(title='Success rate', xlabel='training step', ylim=(0, 1))
    for key in ('train_loss', 'val_loss'):
        vals = [(s, float(r[key])) for s, r in zip(steps, rows) if r.get(key) not in (None, '')]
        if vals:
            axes[1].plot(*zip(*vals), 'o-', label=key)
    axes[1].set(title='Action-chunk MSE (normalized actions)', xlabel='training step')
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(csv_path), 'eval_summary.png')
    plt.savefig(out, dpi=110)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', nargs='+')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--videos', type=int, default=0, help='number of episodes to record per checkpoint')
    parser.add_argument('--video_frame_skip', type=int, default=5, help='record every k-th env step')
    parser.add_argument('--loss', action='store_true', help='also compute train/val action MSE per checkpoint')
    parser.add_argument('--loss_train_shards', type=int, default=5, help='train shards to measure train loss on')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    device = torch.device('cuda')
    checkpoints = sorted(args.checkpoints, key=step_of)

    loss_sets = None
    if args.loss:
        _, normalizer, config = load_policy(checkpoints[0], device)
        train_set, val_set, _ = make_datasets(
            config['env_name'], normalizer=normalizer, num_shards=args.loss_train_shards,
            num_val_shards=config['num_val_shards'], chunk_length=config['chunk_length'],
        )
        loss_sets = dict(train_loss=train_set, val_loss=val_set)

    for path in checkpoints:
        set_seed(args.seed)  # same initial states for every checkpoint
        model, normalizer, config = load_policy(path, device)
        name = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.dirname(path)

        @torch.no_grad()
        def policy(obs_np):
            preds = []
            for i in range(0, len(obs_np), args.batch_size):
                x = torch.from_numpy(obs_np[i : i + args.batch_size]).to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config['bf16']):
                    preds.append(model(x).float().cpu().numpy())
            return np.concatenate(preds)

        start = time.time()
        stats = evaluate(
            policy, config['env_name'], normalizer, args.episodes, config['chunk_length'],
            video_episodes=args.videos, video_frame_skip=args.video_frame_skip,
        )
        row = dict(
            checkpoint=os.path.basename(path),
            episodes=args.episodes,
            seed=args.seed,
            success=float(stats['success']),
            success_stderr=float(np.std(stats['episode_success']) / np.sqrt(args.episodes)),
            length=float(stats['length']),
            eval_time_s=time.time() - start,
            train_loss='',
            val_loss='',
        )
        os.makedirs(os.path.join(out_dir, 'eval'), exist_ok=True)
        npz_path = os.path.join(out_dir, 'eval', f'{name}_seed{args.seed}.npz')
        if os.path.exists(npz_path):  # repeated evals of the same checkpoint: append episodes
            prev = np.load(npz_path)
            ep_s = np.concatenate([prev['episode_success'], stats['episode_success']])
            ep_l = np.concatenate([prev['episode_length'], stats['episode_length']])
        else:
            ep_s, ep_l = stats['episode_success'], stats['episode_length']
        np.savez(npz_path, episode_success=ep_s, episode_length=ep_l)
        if loss_sets is not None:
            for key, dataset in loss_sets.items():
                row[key] = dataset_loss(model, dataset, args.batch_size, config['bf16'], device)

        if args.videos:
            os.makedirs(os.path.join(out_dir, 'videos'), exist_ok=True)
            for i, frames in enumerate(stats['videos']):
                tag = 'ok' if stats['episode_success'][i] else 'fail'
                fps = max(1, round(50 / args.video_frame_skip))  # 50 env steps per second of sim time -> ~real speed
                imageio.mimsave(os.path.join(out_dir, 'videos', f'{name}_ep{i}_{tag}.mp4'), frames, fps=fps)

        loss_str = f", train loss {row['train_loss']:.4f}, val loss {row['val_loss']:.4f}" if loss_sets else ''
        print(
            f"{row['checkpoint']}: success {row['success']:.3f} ± {row['success_stderr']:.3f}, "
            f"length {row['length']:.0f}{loss_str} ({row['eval_time_s']:.0f}s)",
            flush=True,
        )

        csv_path = os.path.join(out_dir, 'eval.csv')
        new_file = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row), extrasaction='ignore')
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    plot_summary(csv_path)


if __name__ == '__main__':
    main()
