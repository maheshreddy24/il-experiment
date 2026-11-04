"""PPO + BC (actor-critic with a behavior-cloning term) for multi-camera visual OCBench tasks.

    loss = L_PPO-clip  +  vf_coef * L_value  +  lambda * L_BC,     L_BC = -log pi(a_demo | s_demo)

- One decision = one 25-step action chunk (the dataset has an image every 25 steps), as in bc_torch / ifql_torch.
- Actor: the bc_torch Policy (gives the Gaussian mean, in normalized action space) + a learned log-std per dim,
  initialized from a bc_torch checkpoint. Critic: its own image encoder + small ResMLP -> V(s).
- Rollouts: `num_envs` parallel MJWarp episodes per iteration, sparse reward (1 on success).
- lambda decays by `bc_decay` every iteration (DAPG-style); the first `critic_warmup_iters` iterations train only V.
"""
import csv
import os
import time

import bc_utils  # sets MUJOCO_GL=egl before mujoco is imported
import numpy as np
import ocbench
import torch
import torch.nn as nn
from bc_torch import Policy, ResidualMLP, infinite
from bc_utils import ActionNormalizer, evaluate, get_logger, make_datasets, set_seed
from ifql_torch import MultiCameraEncoder
from ocbench.mjwarp import make_env
from torch.distributions import Normal
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Critic(nn.Module):
    def __init__(self, num_cameras, hidden_dim, num_blocks):
        super().__init__()
        self.encoder = MultiCameraEncoder()
        self.mlp = ResidualMLP(num_cameras * 512, hidden_dim, 1, num_blocks)

    def forward(self, obs):
        return self.mlp(self.encoder(obs)).squeeze(-1)


class Agent:
    def __init__(self, config):
        self.config = c = config
        self.device = torch.device("cuda")

        # Actor: initialized from a bc_torch checkpoint (weights + its action normalizer).
        ckpt = torch.load(c['bc_checkpoint'], map_location=self.device, weights_only=False)
        bc_config = ckpt['config']
        self.normalizer = ActionNormalizer(**ckpt['action_normalizer'])
        num_cameras = ckpt['model_state_dict']['mlp.inp.0.weight'].shape[1] // 512
        self.action_dim = len(self.normalizer.center) * c['chunk_length']
        self.actor = Policy(num_cameras, len(self.normalizer.center), c['chunk_length'],
                            bc_config['hidden_dim'], bc_config['num_blocks']).to(self.device)
        self.actor.load_state_dict(ckpt['model_state_dict'])
        self.log_std = nn.Parameter(torch.full((self.action_dim,), c['init_log_std'], device=self.device))
        self.critic = Critic(num_cameras, c['critic_hidden_dim'], c['critic_num_blocks']).to(self.device)

        self.actor_optimizer = torch.optim.Adam([
            dict(params=self.actor.parameters(), lr=c['actor_lr']),
            dict(params=[self.log_std], lr=c['log_std_lr']),
        ])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c['critic_lr'])
        self.bc_coef = c['bc_coef']

        # Demonstrations for the BC term (successful episodes, same normalizer as the actor).
        demo_set, _, _ = make_datasets(c['env_name'], normalizer=self.normalizer, num_shards=c['num_demo_shards'],
                                       num_val_shards=1, chunk_length=c['chunk_length'], success_only=True)
        self.demo_iter = infinite(DataLoader(demo_set, batch_size=c['bc_batch_size'], shuffle=True, drop_last=True,
                                             num_workers=c['num_workers'], pin_memory=True, persistent_workers=True))
        print(f"demo samples: {len(demo_set)}, actor params: "
              f"{sum(p.numel() for p in self.actor.parameters()) / 1e6:.1f}M, critic params: "
              f"{sum(p.numel() for p in self.critic.parameters()) / 1e6:.1f}M")

        self.run_dir = f"checkpoints/ppo_bc_torch_{int(time.time())}"
        os.makedirs(self.run_dir, exist_ok=True)
        self.logger = get_logger(os.path.join(self.run_dir, "train.log"))
        self.iteration = 0
        self.csv_files = {}

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.config['bf16'])

    def dist(self, obs):
        # The actor runs in fp32 (TF32 matmuls) by default: with bf16 weights, even tiny updates flip weights to the
        # neighboring bf16 value, which for this sharp 175-dim policy alone costs ~0.05 KL per update.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.config['actor_bf16']):
            mean = self.actor(obs).float()
        log_std = self.log_std.clamp(self.config['min_log_std'], self.config['max_log_std'])
        return Normal(mean, log_std.exp())

    def value(self, obs):
        with self.autocast():
            return self.critic(obs).float()

    @torch.no_grad()
    def act(self, obs_np, deterministic=False):
        """numpy uint8 images -> (normalized action chunks, joint log-probs, values), in batches."""
        actions, logps, values = [], [], []
        for i in range(0, len(obs_np), self.config['policy_batch_size']):
            obs = torch.from_numpy(obs_np[i : i + self.config['policy_batch_size']]).to(self.device)
            dist = self.dist(obs)
            a = dist.mean if deterministic else dist.sample()
            actions.append(a.cpu().numpy())
            logps.append(dist.log_prob(a).sum(-1).cpu().numpy())
            values.append(self.value(obs).cpu().numpy())
        return np.concatenate(actions), np.concatenate(logps), np.concatenate(values)

    def rollout(self):
        """Run `num_envs` full episodes with the stochastic policy; returns per-decision data with GAE advantages."""
        c = self.config
        n, H = c['num_envs'], c['chunk_length']
        family, _, env_kwargs, max_steps = ocbench.parse_env_spec(c['env_name'])
        env = make_env(family, dict(env_kwargs, ob_type='states'), n)
        env.reset(seeds=np.random.randint(0, 2**32, n, dtype=np.uint32))
        obs = env.get_pixel_observation()
        low, high = env.single_action_space.low, env.single_action_space.high

        done = np.zeros(n, dtype=bool)
        success = np.zeros(n, dtype=bool)
        lengths = np.zeros(n, dtype=np.int32)
        steps = 0
        buf = {k: [] for k in ('obs', 'actions', 'logp', 'values', 'rewards', 'ended', 'world')}
        while not done.all():
            active = np.flatnonzero(~done)
            a, logp, v = self.act(obs[active])
            chunks = np.clip(self.normalizer.unnormalize(a.reshape(len(active), H, -1)), low, high)
            env_actions = np.zeros((n, H, chunks.shape[-1]), dtype=np.float32)
            env_actions[active] = chunks
            chunk_reward = np.zeros(n, dtype=np.float32)
            for t in range(H):
                action = env_actions[:, t].copy()
                action[done] = 0.0
                _, reward, terminated, _, info = env.step(action)
                steps += 1
                chunk_reward[~done] += reward[~done]
                new_done = ~done & (terminated | (steps >= max_steps))
                success[new_done] = info['success'][new_done]
                lengths[new_done] = steps
                done |= new_done
                if done.all():
                    break
            for k, val in (('obs', obs[active].copy()), ('actions', a), ('logp', logp), ('values', v),
                           ('rewards', chunk_reward[active]), ('ended', done[active].copy()), ('world', active)):
                buf[k].append(val)
            still = np.flatnonzero(~done)
            if len(still):
                obs[still] = env.get_pixel_observation(still)
        env.close()

        data = {k: np.concatenate(v) for k, v in buf.items()}
        # GAE per world. Every episode ends inside the rollout (success, failure, or the time limit, after which
        # the return is 0), so the value after the last decision is 0.
        adv = np.zeros(len(data['rewards']), dtype=np.float32)
        for w in range(n):
            idx = np.flatnonzero(data['world'] == w)
            gae, next_value = 0.0, 0.0
            for i in idx[::-1]:
                not_end = 1.0 - data['ended'][i]
                delta = data['rewards'][i] + c['gamma'] * next_value * not_end - data['values'][i]
                gae = delta + c['gamma'] * c['gae_lambda'] * not_end * gae
                adv[i] = gae
                next_value = data['values'][i]
        data['advantages'] = adv
        data['returns'] = adv + data['values']
        stats = dict(rollout_success=success.mean(), rollout_length=lengths.mean(), rollout_decisions=len(adv))
        return data, stats

    def bc_loss(self):
        """-log pi(a_demo | s_demo): joint log-likelihood over all action dims, same scale as the PPO log-ratio."""
        obs, act = next(self.demo_iter)
        dist = self.dist(obs.to(self.device, non_blocking=True))
        return -dist.log_prob(act.to(self.device, non_blocking=True)).sum(-1).mean()

    def learn(self, data):
        c = self.config
        warmup = self.iteration <= c['critic_warmup_iters']
        adv = data['advantages']
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        to = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=self.device)
        sums, count, stopped_early = {}, 0, False

        # Recompute old log-probs with the same batch shapes as the update: bf16 numerics differ with batch size, and
        # summed over 175 dims at a small std that alone shifts log pi by ~0.5, which would look like a large ratio.
        rollout_logp = data['logp']
        if not warmup:
            old = []
            with torch.no_grad():
                for start in range(0, len(adv), c['minibatch_size']):
                    sl = slice(start, start + c['minibatch_size'])
                    obs = torch.from_numpy(data['obs'][sl]).to(self.device)
                    old.append(self.dist(obs).log_prob(to(data['actions'][sl])).sum(-1).cpu().numpy())
            data['logp'] = np.concatenate(old)
        for epoch in range(c['n_epochs']):
            perm = np.random.permutation(len(adv))
            for start in range(0, len(perm) - c['minibatch_size'] + 1, c['minibatch_size']):
                idx = perm[start : start + c['minibatch_size']]
                obs = torch.from_numpy(data['obs'][idx]).to(self.device, non_blocking=True)

                value_loss = ((self.value(obs) - to(data['returns'][idx])) ** 2).mean()
                info = {'value_loss': value_loss}
                if warmup:
                    loss = c['vf_coef'] * value_loss
                else:
                    dist = self.dist(obs)
                    logp = dist.log_prob(to(data['actions'][idx])).sum(-1)
                    log_ratio = logp - to(data['logp'][idx])
                    ratio = log_ratio.exp()
                    a = to(adv[idx])
                    policy_loss = -torch.min(ratio * a, ratio.clamp(1 - c['policy_clip'], 1 + c['policy_clip']) * a).mean()
                    bc_loss = self.bc_loss() if self.bc_coef > 0 else torch.zeros((), device=self.device)
                    entropy = dist.entropy().sum(-1).mean()
                    loss = policy_loss + c['vf_coef'] * value_loss + self.bc_coef * bc_loss - c['ent_coef'] * entropy
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                    info.update(policy_loss=policy_loss, bc_loss=bc_loss, bc_loss_per_dim=bc_loss / self.action_dim,
                                entropy=entropy, approx_kl=approx_kl,
                                clip_frac=((ratio - 1).abs() > c['policy_clip']).float().mean())

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([*self.actor.parameters(), self.log_std], c['max_grad_norm'])
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), c['max_grad_norm'])
                if not warmup:
                    self.actor_optimizer.step()
                self.critic_optimizer.step()

                for k, v in info.items():
                    sums[k] = sums.get(k, 0.0) + v.detach()
                count += 1
                if not warmup and info['approx_kl'] > 1.5 * c['target_kl']:
                    stopped_early = True
                    break
            if stopped_early:
                break

        stats = {k: (v / max(count, 1)).item() for k, v in sums.items()}
        returns, values = data['returns'], data['values']
        stats.update(
            epochs_done=epoch + 1,
            kl_early_stop=int(stopped_early),
            explained_variance=float(1 - np.var(returns - values) / (np.var(returns) + 1e-8)),
            logp_recompute_gap=float(np.abs(data['logp'] - rollout_logp).mean()) if not warmup else float('nan'),
            mean_std=self.log_std.detach().clamp(c['min_log_std'], c['max_log_std']).exp().mean().item(),
            bc_coef=self.bc_coef,
            critic_warmup=int(warmup),
        )
        if not warmup:
            self.bc_coef *= c['bc_decay']
        return stats

    def evaluate(self):
        """Deterministic (mean-action) eval with the same protocol as bc_torch / ifql_torch."""
        start = time.time()
        stats = evaluate(lambda obs: self.act(obs, deterministic=True)[0], self.config['env_name'], self.normalizer,
                         self.config['eval_episodes'], self.config['chunk_length'])
        row = dict(iteration=self.iteration, success=float(stats['success']),
                   success_stderr=float(np.std(stats['episode_success']) / np.sqrt(self.config['eval_episodes'])),
                   length=float(stats['length']), time_s=time.time() - start)
        self.log_csv("eval", row)
        self.logger.info(f"[iter {self.iteration}] Eval (mean action): success {row['success']:.3f} ± "
                         f"{row['success_stderr']:.3f}, length {row['length']:.0f} ({row['time_s']:.0f}s)")

    def log_csv(self, name, row):
        if name not in self.csv_files:
            f = open(os.path.join(self.run_dir, f"{name}.csv"), "a", newline="")
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if f.tell() == 0:
                writer.writeheader()
            self.csv_files[name] = (f, writer)
        f, writer = self.csv_files[name]
        writer.writerow({k: row.get(k, '') for k in writer.fieldnames})
        f.flush()

    def save_checkpoint(self):
        path = os.path.join(self.run_dir, f"model_iter_{self.iteration}.pt")
        torch.save({
            "actor": self.actor.state_dict(),
            "log_std": self.log_std.detach(),
            "critic": self.critic.state_dict(),
            "action_normalizer": self.normalizer.state_dict(),
            "iteration": self.iteration,
            "bc_coef": self.bc_coef,
            "config": self.config,
        }, path)
        self.logger.info(f"Checkpoint saved at {path}")

    def train(self):
        c = self.config
        self.evaluate()  # iteration 0 = the BC checkpoint before any RL
        for self.iteration in tqdm(range(1, c['num_iterations'] + 1), dynamic_ncols=True):
            t0 = time.time()
            data, rollout_stats = self.rollout()
            t1 = time.time()
            learn_stats = self.learn(data)
            row = dict(iteration=self.iteration, **rollout_stats, **learn_stats,
                       rollout_time_s=t1 - t0, learn_time_s=time.time() - t1)
            self.log_csv("metrics", row)
            self.logger.info(
                f"[iter {self.iteration}] rollout success {row['rollout_success']:.3f} ({row['rollout_decisions']} "
                f"decisions) | policy {row.get('policy_loss', float('nan')):.4f} value {row['value_loss']:.4f} "
                f"bc/dim {row.get('bc_loss_per_dim', float('nan')):.3f} (lambda {row['bc_coef']:.3f}) | "
                f"kl {row.get('approx_kl', float('nan')):.4f} clip {row.get('clip_frac', float('nan')):.3f} "
                f"std {row['mean_std']:.3f} EV {row['explained_variance']:.2f}"
                f"{' [critic warmup]' if row['critic_warmup'] else ''} | "
                f"{row['rollout_time_s']:.0f}s + {row['learn_time_s']:.0f}s"
            )
            if self.iteration % c['eval_interval'] == 0:
                self.evaluate()
            if self.iteration % c['save_interval'] == 0:
                self.save_checkpoint()
        if c['num_iterations'] % c['save_interval'] != 0:
            self.save_checkpoint()


if __name__ == "__main__":
    config = {
        'env_name': 'visual-block-single-task1-v0',
        'bc_checkpoint': 'checkpoints/bc_torch_1793611331/model_step_30000.pt',  # actor init + action normalizer
        'num_demo_shards': 20,     # demos for the BC term (~1.3GB RAM per shard)
        'chunk_length': 25,        # one decision = one 25-step chunk (dataset observation interval)

        # PPO (gamma / lambda are per decision; an episode is <= 50 decisions)
        'num_iterations': 100,
        'num_envs': 100,           # parallel episodes per iteration -> ~3-5k decisions per rollout
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'policy_clip': 0.2,
        'n_epochs': 2,
        'minibatch_size': 128,
        'target_kl': 0.05,         # stop the epoch loop early if approx KL > 1.5x this
        'vf_coef': 0.5,
        'ent_coef': 0.0,           # exploration comes from the learned std
        'max_grad_norm': 1.0,
        'critic_warmup_iters': 3,  # train only V first, so early advantages don't wreck the BC actor

        # actor / critic
        'actor_lr': 1e-7,          # tiny on purpose: the policy is sharp (175 dims, std ~0.135); 3e-7 already gives ~0.03 KL in 20 steps
        'log_std_lr': 1e-3,        # separate, so the exploration std can still adapt
        'actor_bf16': False,       # fp32/TF32 actor: bf16 weight rounding alone adds ~0.05 KL per update
        'critic_lr': 1e-4,
        'init_log_std': -2.0,      # std ~0.135 in normalized action units
        'min_log_std': -5.0,
        'max_log_std': 0.0,
        'critic_hidden_dim': 1024,
        'critic_num_blocks': 2,

        # BC term: lambda * -log pi(a_demo | s_demo)
        'bc_coef': 1.0,            # lambda
        'bc_decay': 0.97,          # lambda *= this each iteration (1.0 = constant): 1.0 -> 0.22 @ 50 -> 0.05 @ 100
        'bc_batch_size': 128,

        'eval_interval': 10,
        'eval_episodes': 100,
        'save_interval': 10,
        'policy_batch_size': 50,
        'num_workers': 6,
        'bf16': True,
        'seed': 0,
    }

    set_seed(config['seed'])
    agent = Agent(config)
    agent.train()
