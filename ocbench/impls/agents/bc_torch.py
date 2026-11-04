import ast
import csv
import gc
import os
import sys
import time

import bc_utils  # sets MUJOCO_GL=egl before mujoco is imported
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bc_utils import ActionNormalizer, ShardGroups, evaluate, fit_normalizer, get_logger, make_datasets, set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm


class ResnetStack(nn.Module):
    """conv -> maxpool(stride 2) -> residual blocks (IMPALA)."""

    def __init__(self, in_ch, out_ch, num_blocks):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.ReLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1))
            for _ in range(num_blocks)
        )

    def forward(self, x):
        x = self.pool(self.conv(x))
        for block in self.blocks:
            x = x + block(x)
        return x


class ImpalaEncoder(nn.Module):
    """IMPALA-small encoder: (N, 3, 224, 224) -> (N, out_dim)."""

    def __init__(self, image_size=224, channels=(16, 32, 32), num_blocks=1, out_dim=512):
        super().__init__()
        stacks, in_ch = [], 3
        for ch in channels:
            stacks.append(ResnetStack(in_ch, ch, num_blocks))
            in_ch = ch
            image_size = (image_size + 1) // 2
        self.stacks = nn.Sequential(*stacks)
        self.norm = nn.LayerNorm(in_ch)
        self.fc = nn.Linear(in_ch * image_size * image_size, out_dim)
        self.fc_norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = F.relu(self.stacks(x))
        x = self.norm(x.permute(0, 2, 3, 1))  # LayerNorm over channels
        return self.fc_norm(F.gelu(self.fc(x.flatten(1))))


class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_blocks):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_blocks)
        )
        self.block_norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_blocks))
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.inp(x)
        for block, norm in zip(self.blocks, self.block_norms):
            x = norm(F.gelu(x + block(x)))
        return self.out(x)


class Policy(nn.Module):
    """Multi-camera images -> flattened action chunk (chunk_length * action_dim)."""

    def __init__(self, num_cameras, action_dim, chunk_length, hidden_dim, num_blocks, image_size=224):
        super().__init__()
        self.encoder = ImpalaEncoder(image_size)  # shared across cameras
        self.mlp = ResidualMLP(num_cameras * 512, hidden_dim, chunk_length * action_dim, num_blocks)

    def forward(self, obs):
        # obs: (B, num_cameras, H, W, 3) uint8
        B, V, H, W, C = obs.shape
        x = obs.reshape(B * V, H, W, C).permute(0, 3, 1, 2).float() / 255.0
        feats = self.encoder(x).reshape(B, -1)  # concat per-camera features
        return self.mlp(feats)


def infinite(loader):
    while True:
        yield from loader


def parse_overrides(config, argv):
    """`--set key=value [key=value ...]` on the command line overrides config entries (values parsed as Python)."""
    if '--set' in argv:
        for item in argv[argv.index('--set') + 1 :]:
            key, value = item.split('=', 1)
            if key not in config:
                raise KeyError(f'unknown config key: {key}')
            try:
                config[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                config[key] = value
    return config


class Optimisation:
    def __init__(self, config, model, train_set, val_set, normalizer, resume_ckpt=None, load_group=None):
        """load_group(i) -> train set of shard group i, for shard rotation (None = everything stays in RAM)."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.normalizer = normalizer
        self.load_group = load_group
        self.group_idx = 0
        self.loader_kwargs = dict(batch_size=config['batch_size'], shuffle=True, drop_last=True, pin_memory=True)
        self.set_train_set(train_set)
        self.val_iter = infinite(DataLoader(val_set, num_workers=2, persistent_workers=True, **self.loader_kwargs))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.global_step = 0
        if resume_ckpt is None:
            self.checkpoint_dir = f"checkpoints/{config.get('run_name') or f'bc_torch_{int(time.time())}'}"
        else:
            # Continue in the same run dir: train.log / eval.csv / model_step_*.pt keep accumulating there.
            self.checkpoint_dir = os.path.dirname(config['resume'])
            self.optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
            self.global_step = resume_ckpt['global_step']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = get_logger(os.path.join(self.checkpoint_dir, "train.log"))
        if resume_ckpt is not None:
            self.logger.info(f"Resumed from {config['resume']} at step {self.global_step}")

    def set_train_set(self, train_set):
        self.train_iter = infinite(DataLoader(train_set, num_workers=self.config['num_workers'], persistent_workers=True,
                                              **self.loader_kwargs))
        self.steps_per_epoch = len(train_set) // self.config['batch_size']

    def rotate(self):
        """Swap in the next shard group: free the current one first (RAM), then load the next."""
        self.train_iter = None
        gc.collect()
        self.group_idx += 1
        start = time.time()
        self.set_train_set(self.load_group(self.group_idx))
        self.logger.info(f"[step {self.global_step}] Rotated to shard group {self.group_idx} ({time.time() - start:.0f}s)")

    def loss(self, obs, act):
        obs = obs.to(self.device, non_blocking=True)
        act = act.to(self.device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.config['bf16']):
            pred = self.model(obs)
        return F.mse_loss(pred.float(), act)

    @torch.no_grad()
    def predict(self, obs):
        """numpy uint8 images -> numpy normalized action chunks (used by the env rollout)."""
        self.model.eval()
        preds = []
        for i in range(0, len(obs), self.config['batch_size']):
            x = torch.from_numpy(obs[i : i + self.config['batch_size']]).to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.config['bf16']):
                preds.append(self.model(x).float().cpu().numpy())
        self.model.train()
        return np.concatenate(preds)

    @torch.no_grad()
    def val_loss(self):
        self.model.eval()
        loss = self.loss(*next(self.val_iter)).item()
        self.model.train()
        return loss

    def rollout(self):
        start = time.time()
        stats = evaluate(self.predict, self.config['env_name'], self.normalizer, self.config['eval_episodes'], self.config['chunk_length'])
        self.logger.info(
            f"[step {self.global_step}] Eval success: {stats['success']:.3f}, length: {stats['length']:.0f} ({time.time() - start:.0f}s)"
        )
        # Same eval.csv format as eval_bc_torch.py, so plot_bc.py / plot_compare.py pick these up.
        episodes = self.config['eval_episodes']
        row = dict(checkpoint=f"model_step_{self.global_step}.pt", episodes=episodes, seed='', success=float(stats['success']),
                   success_stderr=float(np.std(stats['episode_success']) / np.sqrt(episodes)), length=float(stats['length']),
                   eval_time_s=time.time() - start, train_loss='', val_loss='')
        path = os.path.join(self.checkpoint_dir, "eval.csv")
        new_file = not os.path.exists(path)
        if not new_file:
            with open(path) as f:
                fieldnames = next(csv.reader(f))
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row) if new_file else fieldnames, extrasaction='ignore')
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def save_checkpoint(self):
        # Per-step files hold weights only (~0.6GB); last.pt also holds optimizer state for resuming and is overwritten.
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "action_normalizer": self.normalizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_step_{self.global_step}.pt")
        torch.save(ckpt, checkpoint_path)
        torch.save({**ckpt, "optimizer_state_dict": self.optimizer.state_dict()}, os.path.join(self.checkpoint_dir, "last.pt"))
        self.logger.info(f"Checkpoint saved at {checkpoint_path}")

    def train(self):
        self.model.train()
        total_loss, last_time = 0.0, time.time()
        pbar = tqdm(total=self.config['num_steps'], initial=self.global_step, dynamic_ncols=True)
        while self.global_step < self.config['num_steps']:
            loss = self.loss(*next(self.train_iter))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            self.global_step += 1
            pbar.update(1)
            if self.global_step % 50 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{self.global_step / self.steps_per_epoch:.2f}")

            if self.global_step % self.config['log_interval'] == 0:
                steps_per_sec = self.config['log_interval'] / (time.time() - last_time)
                self.logger.info(
                    f"[step {self.global_step}] Train Loss: {total_loss / self.config['log_interval']:.4f}, "
                    f"Val Loss: {self.val_loss():.4f}, {steps_per_sec:.2f} it/s"
                )
                total_loss, last_time = 0.0, time.time()
            if self.global_step % self.config['eval_interval'] == 0:
                self.rollout()
            if self.global_step % self.config['save_interval'] == 0:
                self.save_checkpoint()
            if (self.load_group is not None and self.global_step % self.config['rotate_every'] == 0
                    and self.global_step < self.config['num_steps']):
                self.rotate()

        pbar.close()
        self.save_checkpoint()


if __name__ == "__main__":
    config = {
        'env_name': 'visual-block-single-task1-v0',
        'num_shards': None,       # None = all 100 shards (~105GB of images in RAM after success filtering); use e.g. 10 for a quick test
        'num_val_shards': 10,
        'chunk_length': 25,       # must equal the dataset's observation_interval (25)
        'hidden_dim': 4096,
        'num_blocks': 4,          # residual blocks of 2 layers each -> 8 hidden layers like the JAX config
        'learning_rate': 1e-4,
        'batch_size': 256,
        'num_steps': 70_000,      # total steps; when resuming, training continues from the checkpoint's step up to this
        'resume': None,           # None = train from scratch; else a last.pt to continue from
        'run_name': None,         # checkpoint dir name under checkpoints/ (None = bc_torch_<time>)
        'shard_group_size': None, # None = all shards in RAM; else load this many shards at a time and rotate
        'rotate_every': 5000,     # steps between shard-group swaps (when shard_group_size is set)
        'log_interval': 1000,
        'eval_interval': 10_000,
        'eval_episodes': 100,
        'save_interval': 10_000,
        'num_workers': 8,
        'bf16': True,
        'seed': 0,
    }

    config = parse_overrides(config, sys.argv)
    set_seed(config['seed'])

    resume_ckpt, normalizer = None, None
    if config['resume'] is not None:
        resume_ckpt = torch.load(config['resume'], map_location="cpu", weights_only=False)
        normalizer = ActionNormalizer(**resume_ckpt['action_normalizer'])  # must match the loaded weights
        for key in ('hidden_dim', 'num_blocks', 'chunk_length', 'batch_size', 'learning_rate', 'num_shards'):
            if resume_ckpt['config'][key] != config[key]:
                print(f"WARNING: {key} differs from the resumed run: {resume_ckpt['config'][key]} -> {config[key]}")

    groups, load_group = None, None
    if config['shard_group_size']:
        groups = ShardGroups(config['env_name'], config['num_shards'], config['shard_group_size'])
        if normalizer is None:
            normalizer = fit_normalizer(groups.paths)  # one normalizer for all shards
        print(f"shard rotation: {len(groups.paths)} shards in {len(groups)} groups of {config['shard_group_size']}, "
              f"swapped every {config['rotate_every']} steps")

        def load_group(i):
            return make_datasets(config['env_name'], normalizer=normalizer, num_val_shards=1,
                                 chunk_length=config['chunk_length'], train_paths=groups[i])[0]

    train_set, val_set, normalizer = make_datasets(
        config['env_name'], normalizer=normalizer, num_shards=config['num_shards'], num_val_shards=config['num_val_shards'],
        chunk_length=config['chunk_length'], train_paths=groups[0] if groups else None,
    )
    num_cameras = train_set.images.shape[1]
    action_dim = train_set.actions.shape[-1]
    print(f"train samples: {len(train_set)}, val samples: {len(val_set)}, cameras: {num_cameras}, action_dim: {action_dim}")

    model = Policy(num_cameras, action_dim, config['chunk_length'], config['hidden_dim'], config['num_blocks'])
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt['model_state_dict'])
    trainer = Optimisation(config, model, train_set, val_set, normalizer, resume_ckpt, load_group)
    del train_set
    del resume_ckpt
    trainer.train()
