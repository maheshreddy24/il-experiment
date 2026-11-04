"""Implicit Flow Q-Learning (IFQL) in PyTorch, for multi-camera visual OCBench tasks with action chunks.

Port of agents/ifql.py:
  - actor: flow-matching behavior policy over normalized action chunks
  - value V(s) and critic Q(s, a) (2-head ensemble): IQL with HL-Gauss categorical outputs on [0, 1]
  - acting: sample `num_samples` chunks from the flow, execute the one with the highest Q
"""
import copy
import csv
import gc
import math
import os
import sys
import time

import bc_utils  # sets MUJOCO_GL=egl before mujoco is imported
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bc_torch import ImpalaEncoder, ResidualMLP, infinite, parse_overrides
from bc_utils import ShardGroups, evaluate, fit_normalizer, get_logger, make_datasets, set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

NUM_BINS = 101
BIN_EDGES = torch.linspace(0.0, 1.0, NUM_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
SIGMA = 0.75 / NUM_BINS  # 0.75 bin widths


def hl_gauss_cross_entropy(logits, targets):
    """Cross entropy between predicted bin logits and a Gaussian-smoothed histogram of scalar targets in [0, 1]."""
    edges = BIN_EDGES.to(logits.device)
    cdf = torch.erf((edges - targets.float().clamp(0, 1)[..., None]) / (math.sqrt(2) * SIGMA))
    probs = (cdf[..., 1:] - cdf[..., :-1]) / (cdf[..., -1:] - cdf[..., :1]).clamp_min(1e-8)
    return -(probs * F.log_softmax(logits.float(), dim=-1)).sum(-1)


def logits_to_value(logits):
    return (F.softmax(logits.float(), dim=-1) * BIN_CENTERS.to(logits.device)).sum(-1)


class MultiCameraEncoder(ImpalaEncoder):
    """(B, num_cameras, H, W, 3) uint8 -> (B, num_cameras * 512): shared IMPALA CNN per camera, features concatenated."""

    def forward(self, obs):
        B, V, H, W, C = obs.shape
        x = obs.reshape(B * V, H, W, C).permute(0, 3, 1, 2).float() / 255.0
        return super().forward(x).reshape(B, -1)


class CategoricalValue(nn.Module):
    """Image encoder + `num_heads` ResMLP heads with HL-Gauss bins. V(s) if action_dim == 0, else Q(s, a)."""

    def __init__(self, num_cameras, action_dim, hidden_dim, num_blocks, num_heads):
        super().__init__()
        self.encoder = MultiCameraEncoder()
        self.heads = nn.ModuleList(
            ResidualMLP(num_cameras * 512 + action_dim, hidden_dim, NUM_BINS, num_blocks) for _ in range(num_heads)
        )

    def forward_feats(self, feats, actions=None):
        x = feats if actions is None else torch.cat([feats, actions.to(feats.dtype)], dim=-1)
        return torch.stack([head(x) for head in self.heads])  # (num_heads, B, NUM_BINS) logits

    def forward(self, obs, actions=None):
        return self.forward_feats(self.encoder(obs), actions)


class ActorFlow(nn.Module):
    """Velocity field v(s, x_t, t) for flow matching from N(0, I) to the action-chunk distribution."""

    def __init__(self, num_cameras, action_dim, hidden_dim, num_blocks):
        super().__init__()
        self.encoder = MultiCameraEncoder()
        self.mlp = ResidualMLP(num_cameras * 512 + action_dim + 65, hidden_dim, action_dim, num_blocks)

    def forward_feats(self, feats, x_t, t):
        angles = 2 * math.pi * t * torch.arange(1, 33, device=t.device, dtype=t.dtype)
        time_emb = torch.cat([t, angles.cos(), angles.sin()], dim=-1)
        return self.mlp(torch.cat([feats, x_t.to(feats.dtype), time_emb.to(feats.dtype)], dim=-1))

    def forward(self, obs, x_t, t):
        return self.forward_feats(self.encoder(obs), x_t, t)


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp_min(1)


class IFQL:
    def __init__(self, config, train_set, val_set, normalizer, load_group=None):
        """load_group(i) -> train set of shard group i, for shard rotation (None = everything stays in RAM)."""
        self.load_group = load_group
        self.group_idx = 0
        self.config = c = config
        self.device = torch.device("cuda")
        self.normalizer = normalizer
        num_cameras = train_set.images.shape[1]
        action_dim = train_set.actions.shape[-1] * c['chunk_length']

        self.value = CategoricalValue(num_cameras, 0, c['value_hidden_dim'], c['value_num_blocks'], 1).to(self.device)
        self.critic = CategoricalValue(num_cameras, action_dim, c['value_hidden_dim'], c['value_num_blocks'], 2).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor = ActorFlow(num_cameras, action_dim, c['actor_hidden_dim'], c['actor_num_blocks']).to(self.device)
        self.params = [*self.value.parameters(), *self.critic.parameters(), *self.actor.parameters()]
        self.optimizer = torch.optim.Adam(self.params, lr=c['learning_rate'])
        for name, net in [('value', self.value), ('critic', self.critic), ('actor', self.actor)]:
            print(f"{name} params: {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M")

        self.loader_kwargs = dict(batch_size=c['batch_size'], shuffle=True, drop_last=True, pin_memory=True,
                                  persistent_workers=True)
        self.set_train_set(train_set)
        self.val_iter = infinite(DataLoader(val_set, num_workers=2, **self.loader_kwargs))

        self.run_dir = f"checkpoints/{c.get('run_name') or f'ifql_torch_{int(time.time())}'}"
        os.makedirs(os.path.join(self.run_dir, "eval"), exist_ok=True)
        self.logger = get_logger(os.path.join(self.run_dir, "train.log"))
        self.global_step = 0
        self.csv_writers = {}

    def set_train_set(self, train_set):
        self.train_iter = infinite(DataLoader(train_set, num_workers=self.config['num_workers'], **self.loader_kwargs))
        self.steps_per_epoch = len(train_set) // self.config['batch_size']

    def rotate(self):
        """Swap in the next shard group: free the current one first (RAM), then load the next."""
        self.train_iter = None
        gc.collect()
        self.group_idx += 1
        start = time.time()
        self.set_train_set(self.load_group(self.group_idx))
        self.logger.info(f"[step {self.global_step}] Rotated to shard group {self.group_idx} ({time.time() - start:.0f}s)")

    @classmethod
    def load_for_eval(cls, path):
        """Rebuild actor/critic/value from a model_step_*.pt (or last.pt) checkpoint, for evaluation only."""
        self = cls.__new__(cls)
        self.device = torch.device("cuda")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.config = c = ckpt['config']
        self.normalizer = bc_utils.ActionNormalizer(**ckpt['action_normalizer'])
        num_cameras = ckpt['value']['heads.0.inp.0.weight'].shape[1] // 512
        action_dim = ckpt['critic']['heads.0.inp.0.weight'].shape[1] - num_cameras * 512
        self.value = CategoricalValue(num_cameras, 0, c['value_hidden_dim'], c['value_num_blocks'], 1).to(self.device)
        self.critic = CategoricalValue(num_cameras, action_dim, c['value_hidden_dim'], c['value_num_blocks'], 2).to(self.device)
        self.actor = ActorFlow(num_cameras, action_dim, c['actor_hidden_dim'], c['actor_num_blocks']).to(self.device)
        for name in ('value', 'critic', 'actor'):
            getattr(self, name).load_state_dict(ckpt[name])
            getattr(self, name).eval()
        self.global_step = ckpt['global_step']
        return self

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.config['bf16'])

    def agg(self, qs):
        """Combine critic heads: (2, B) -> (B,)."""
        return qs.min(0).values if self.config['q_agg'] == 'min' else qs.mean(0)

    def compute_losses(self, batch):
        c = self.config
        obs, actions, next_obs, rewards, discounts, success = [x.to(self.device, non_blocking=True) for x in batch]
        rewards, discounts, success = rewards.float(), discounts.float(), success.float()
        with self.autocast():
            with torch.no_grad():
                target_q = self.agg(logits_to_value(self.target_critic(obs, actions)))
                next_v = logits_to_value(self.value(next_obs))[0]

            # Value loss (IQL expectile regression): fit V(s) to an expectile of Q(s, a) over dataset actions.
            v_logits = self.value(obs)[0]
            v = logits_to_value(v_logits)
            weight = torch.where(target_q >= v.detach(), c['expectile'], 1 - c['expectile'])
            value_loss = (weight * hl_gauss_cross_entropy(v_logits, target_q)).mean()

            # Critic loss (n-step TD): Q(s, a) <- R_{t:t+n} + gamma^n * V(s_{t+n}).
            td_target = rewards + discounts * next_v
            q_logits = self.critic(obs, actions)
            critic_loss = hl_gauss_cross_entropy(q_logits, td_target[None]).mean()

            # Actor loss (flow matching): predict the velocity x1 - x0 at x_t = (1 - t) x0 + t x1.
            x0 = torch.randn_like(actions)
            u = torch.rand(actions.shape[0], 1, device=self.device)
            t = 1 - (1 - u) ** (1 / c['beta_b'])  # t ~ Beta(1, beta_b)
            x_t = (1 - t) * x0 + t * actions
            pred = self.actor(obs, x_t, t).float()
            per_sample = ((pred - (actions - x0)) ** 2).mean(-1)
            actor_loss = masked_mean(per_sample, success) if c['bc_success_only'] else per_sample.mean()

        total_loss = value_loss + critic_loss + actor_loss
        with torch.no_grad():
            qs = logits_to_value(q_logits)
            q = self.agg(qs)
            adv = q - v
            metrics = {
                'loss/total': total_loss,
                'loss/value': value_loss,
                'loss/critic': critic_loss,
                'loss/actor': actor_loss,
                'value/v_mean': v.mean(),
                'value/v_min': v.min(),
                'value/v_max': v.max(),
                'critic/q_mean': q.mean(),
                'critic/q_min': q.min(),
                'critic/q_max': q.max(),
                'critic/q_success_eps': masked_mean(q, success),  # Q on transitions from episodes that succeeded...
                'critic/q_failure_eps': masked_mean(q, 1 - success),  # ...vs. that failed: should separate
                'critic/td_target_mean': td_target.mean(),
                'critic/td_error_abs': (q - td_target).abs().mean(),
                'critic/head_disagreement': (qs[0] - qs[1]).abs().mean(),
                'critic/target_q_mean': target_q.mean(),
                'adv/mean': adv.mean(),
                'adv/std': adv.std(),
                'adv/frac_positive': (adv > 0).float().mean(),
                'data/reward_mean': rewards.mean(),
                'data/discount_mean': discounts.mean(),
                'data/success_frac': success.mean(),
            }
        return total_loss, metrics

    @torch.no_grad()
    def sample_actions(self, obs, num_samples):
        """obs: (B, ...) uint8 on device -> (actions (B, A), Q of every sample (B, N))."""
        c = self.config
        B = obs.shape[0]
        with self.autocast():
            feats = self.actor.encoder(obs).repeat_interleave(num_samples, dim=0)
            x = torch.randn(B * num_samples, self.actor.mlp.out.out_features, device=self.device)
            for i in range(c['flow_steps']):
                t = torch.full((B * num_samples, 1), i / c['flow_steps'], device=self.device)
                x = x + self.actor.forward_feats(feats, x, t).float() / c['flow_steps']
            x = x.clamp(-1, 1)
            critic_feats = self.critic.encoder(obs).repeat_interleave(num_samples, dim=0)
            q = self.agg(logits_to_value(self.critic.forward_feats(critic_feats, x))).view(B, num_samples)
        best = q.argmax(dim=1)
        actions = x.view(B, num_samples, -1)[torch.arange(B, device=self.device), best]
        return actions, q

    def make_policy(self, num_samples, records):
        """numpy uint8 images -> numpy normalized action chunks; appends per-call Q statistics to `records`."""

        def policy(obs_np):
            actions, qs = [], []
            for i in range(0, len(obs_np), self.config['eval_batch_size']):
                obs = torch.from_numpy(obs_np[i : i + self.config['eval_batch_size']]).to(self.device)
                a, q = self.sample_actions(obs, num_samples)
                actions.append(a.cpu().numpy())
                qs.append(q.float().cpu().numpy())
            qs = np.concatenate(qs)
            records.append(dict(q_chosen=qs.max(1), q_samples_mean=qs.mean(1), q_samples_std=qs.std(1)))
            return np.concatenate(actions)

        return policy

    def log_csv(self, name, row):
        path = os.path.join(self.run_dir, f"{name}.csv")
        if name not in self.csv_writers:
            f = open(path, "a", newline="")
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if f.tell() == 0:
                writer.writeheader()
            self.csv_writers[name] = (f, writer)
        f, writer = self.csv_writers[name]
        writer.writerow(row)
        f.flush()

    def rollout(self):
        c = self.config
        modes = {'ifql': c['num_samples']}
        if c['eval_flow_bc']:
            modes['flow_bc'] = 1  # same actor, no Q selection: shows how much the critic helps
        for mode, num_samples in modes.items():
            start, records = time.time(), []
            stats = evaluate(self.make_policy(num_samples, records), c['env_name'], self.normalizer, c['eval_episodes'], c['chunk_length'])
            episode_success = stats['episode_success'].astype(np.float32)
            q_chosen = np.stack([r['q_chosen'] for r in records])  # (num_chunks, num_episodes)
            q_init = q_chosen[0]  # Q of the first chunk: the critic's prediction of success from the start state
            np.savez(
                os.path.join(self.run_dir, "eval", f"{mode}_step_{self.global_step}.npz"),
                episode_success=episode_success,
                episode_length=stats['episode_length'],
                q_chosen=q_chosen,
                q_samples_mean=np.stack([r['q_samples_mean'] for r in records]),
                q_samples_std=np.stack([r['q_samples_std'] for r in records]),
            )
            row = {
                'step': self.global_step,
                'mode': mode,
                'success': stats['success'],
                'length': stats['length'],
                'q_init_mean': q_init.mean(),
                'q_init_success_eps': q_init[episode_success == 1].mean() if episode_success.any() else np.nan,
                'q_init_failure_eps': q_init[episode_success == 0].mean() if (episode_success == 0).any() else np.nan,
                'q_selection_gain': (q_chosen - np.stack([r['q_samples_mean'] for r in records])).mean(),
                'time_s': time.time() - start,
            }
            self.log_csv("eval", row)
            self.logger.info(
                f"[step {self.global_step}] Eval {mode}: success {row['success']:.3f}, length {row['length']:.0f}, "
                f"Q_init {row['q_init_mean']:.3f} (succ eps {row['q_init_success_eps']:.3f} / fail eps "
                f"{row['q_init_failure_eps']:.3f}), selection gain {row['q_selection_gain']:.4f} ({row['time_s']:.0f}s)"
            )

    def save_checkpoint(self):
        # Per-step files hold what's needed to act/analyze; last.pt also holds target net + optimizer for resuming.
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "value": self.value.state_dict(),
            "action_normalizer": self.normalizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }
        path = os.path.join(self.run_dir, f"model_step_{self.global_step}.pt")
        torch.save(ckpt, path)
        ckpt.update(target_critic=self.target_critic.state_dict(), optimizer=self.optimizer.state_dict())
        torch.save(ckpt, os.path.join(self.run_dir, "last.pt"))
        self.logger.info(f"Checkpoint saved at {path}")

    def train(self):
        c = self.config
        critic_params = list(self.critic.parameters())
        target_params = list(self.target_critic.parameters())
        sums, count, last_time = {}, 0, time.time()
        pbar = tqdm(total=c['num_steps'], initial=self.global_step, dynamic_ncols=True)
        while self.global_step < c['num_steps']:
            loss, metrics = self.compute_losses(next(self.train_iter))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            metrics['grad_norm'] = torch.nn.utils.clip_grad_norm_(self.params, float('inf'))
            self.optimizer.step()
            with torch.no_grad():
                torch._foreach_lerp_(target_params, critic_params, c['tau'])  # target += tau * (critic - target)
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0) + v.detach()
            count += 1
            self.global_step += 1
            pbar.update(1)

            if self.global_step % 50 == 0:
                pbar.set_postfix(
                    critic=f"{metrics['loss/critic'].item():.3f}",
                    actor=f"{metrics['loss/actor'].item():.3f}",
                    q=f"{metrics['critic/q_mean'].item():.3f}",
                    epoch=f"{self.global_step / self.steps_per_epoch:.2f}",
                )

            if self.global_step % c['log_interval'] == 0:
                row = {'step': self.global_step, 'it_per_s': count / (time.time() - last_time)}
                row.update({f"train/{k}": (v / count).item() for k, v in sums.items()})
                with torch.no_grad():
                    _, val_metrics = self.compute_losses(next(self.val_iter))
                row.update({f"val/{k}": v.item() for k, v in val_metrics.items()})
                self.log_csv("metrics", row)
                self.logger.info(
                    f"[step {self.global_step}] loss value {row['train/loss/value']:.4f} critic {row['train/loss/critic']:.4f} "
                    f"actor {row['train/loss/actor']:.4f} (val actor {row['val/loss/actor']:.4f}) | "
                    f"V {row['train/value/v_mean']:.3f} Q {row['train/critic/q_mean']:.3f} "
                    f"Q succ/fail {row['train/critic/q_success_eps']:.3f}/{row['train/critic/q_failure_eps']:.3f} | "
                    f"{row['it_per_s']:.2f} it/s"
                )
                sums, count, last_time = {}, 0, time.time()
            if self.global_step % c['eval_interval'] == 0:
                self.rollout()
            if self.global_step % c['save_interval'] == 0:
                self.save_checkpoint()
            if self.load_group is not None and self.global_step % c['rotate_every'] == 0 and self.global_step < c['num_steps']:
                self.rotate()

        pbar.close()
        if self.global_step % c['save_interval'] != 0:
            self.save_checkpoint()


if __name__ == "__main__":
    config = {
        'env_name': 'visual-block-single-task1-v0',
        'num_shards': 60,         # each shard is ~1.3GB of images in RAM; all 100 (~128GB with failed episodes) don't fit in 125GB
        'num_val_shards': 10,
        'chunk_length': 25,       # must equal the dataset's observation_interval (25)
        'n_step': 200,            # n-step TD horizon (multiple of 25)
        'discount': 0.9999,
        'actor_hidden_dim': 4096,
        'actor_num_blocks': 4,    # same actor size as BC
        'value_hidden_dim': 2048, # JAX uses 4096; smaller here so V, Q (2 heads) and target Q fit and run on one L4
        'value_num_blocks': 4,
        'learning_rate': 1e-4,
        'batch_size': 128,
        'tau': 0.001,             # target critic Polyak rate
        'expectile': 0.5,         # 0.5 = V is the mean of Q over dataset actions (JAX default); 0.7-0.9 = more optimistic
        'q_agg': 'mean',          # 'mean' or 'min' over the 2 critic heads
        'num_samples': 32,        # candidate action chunks per state at test time
        'flow_steps': 10,
        'beta_b': 1.5,            # flow-matching time t ~ Beta(1, beta_b)
        'bc_success_only': True,  # actor imitates successful episodes only; V/Q learn from all episodes
        'num_steps': 70_000,
        'log_interval': 500,
        'eval_interval': 10_000,
        'eval_episodes': 100,
        'eval_flow_bc': True,     # also evaluate the actor alone (1 sample, no Q selection)
        'eval_batch_size': 25,
        'save_interval': 10_000,
        'num_workers': 8,
        'bf16': True,
        'seed': 0,
        'run_name': None,         # checkpoint dir name under checkpoints/ (None = ifql_torch_<time>)
        'shard_group_size': None, # None = all shards in RAM; else load this many shards at a time and rotate
        'rotate_every': 5000,     # steps between shard-group swaps (when shard_group_size is set)
    }
    config = parse_overrides(config, sys.argv)

    set_seed(config['seed'])
    data_kwargs = dict(
        num_shards=config['num_shards'],
        chunk_length=config['chunk_length'],
        success_only=False,  # the critic needs failed episodes to learn what doesn't work
        n_step=config['n_step'],
        discount=config['discount'],
    )
    groups, load_group, normalizer = None, None, None
    if config['shard_group_size']:
        groups = ShardGroups(config['env_name'], config['num_shards'], config['shard_group_size'])
        normalizer = fit_normalizer(groups.paths)  # one normalizer for all shards
        print(f"shard rotation: {len(groups.paths)} shards in {len(groups)} groups of {config['shard_group_size']}, "
              f"swapped every {config['rotate_every']} steps")

        def load_group(i):
            return make_datasets(config['env_name'], normalizer=normalizer, num_val_shards=1, train_paths=groups[i],
                                 **data_kwargs)[0]

    train_set, val_set, normalizer = make_datasets(
        config['env_name'], normalizer=normalizer, num_val_shards=config['num_val_shards'],
        train_paths=groups[0] if groups else None, **data_kwargs)
    print(
        f"train samples: {len(train_set)} (success frac {train_set.success.mean():.2f}), val samples: {len(val_set)}, "
        f"nonzero n-step rewards: {(train_set.rewards > 0).mean():.3f}"
    )
    agent = IFQL(config, train_set, val_set, normalizer, load_group)
    del train_set
    agent.train()
