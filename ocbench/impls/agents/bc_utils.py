import os

os.environ.setdefault('MUJOCO_GL', 'egl')  # headless rendering; must be set before mujoco is imported
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import glob
import logging
import random
import re

import numpy as np
import ocbench
import torch
from ocbench.mjwarp import make_env


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(f"bc_torch.{log_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(fmt)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


class ActionNormalizer:
    """Maps each action dim's [p, 100-p] percentile range to [-1, 1]; the gripper (last dim) is already in [-1, 1]."""

    def __init__(self, center, half_range, clip=5.0):
        self.center = np.asarray(center, dtype=np.float32)
        self.half_range = np.asarray(half_range, dtype=np.float32)
        self.clip = float(clip)
        self.low = np.full_like(self.center, -self.clip)
        self.high = np.full_like(self.center, self.clip)
        self.low[-1], self.high[-1] = -1.0, 1.0

    @classmethod
    def fit(cls, actions, percentile=1.0, clip=5.0):
        low = np.percentile(actions, percentile, axis=0)
        high = np.percentile(actions, 100 - percentile, axis=0)
        low[-1], high[-1] = -1.0, 1.0
        return cls((low + high) / 2, np.maximum((high - low) / 2, 1e-6), clip)

    def normalize(self, actions):
        return np.clip((actions - self.center) / self.half_range, self.low, self.high).astype(np.float32)

    def unnormalize(self, actions):
        return (np.clip(actions, self.low, self.high) * self.half_range + self.center).astype(np.float32)

    def state_dict(self):
        return dict(center=self.center, half_range=self.half_range, clip=self.clip)


def load_npz(paths, success_only=True):
    """Load dataset shards into one dict, optionally keeping only successful episodes."""
    episode_mask = None
    if success_only:
        episode_mask = []
        for path in paths:
            with np.load(path) as data:
                offsets = np.concatenate([[0], np.flatnonzero(data['terminals']) + 1])
                rewards = data['rewards']
            episode_mask += [bool(np.any(rewards[s:e] == 1)) for s, e in zip(offsets[:-1], offsets[1:])]
    return ocbench.load_dataset(paths, episode_mask=episode_mask)


class ChunkDataset(torch.utils.data.Dataset):
    """(image at step t, normalized actions t..t+chunk-1) pairs.

    Visual datasets only store an image every `observation_interval` steps, so t is always a multiple of it.
    """

    def __init__(self, data, normalizer, chunk_length):
        interval = int(data['observation_interval'])
        assert chunk_length == interval, f'chunk_length must equal the dataset observation_interval ({interval})'
        self.interval = interval
        self.images = data['observations']  # (num_frames, num_cameras, H, W, 3) uint8
        self.actions = normalizer.normalize(data['actions'])

        # One sample per stored image: episode `ep`, step t = episode start + k * interval.
        self.offsets, self.obs_offsets = ocbench.get_episode_offsets(data)
        samples_per_ep = (np.diff(self.offsets) + interval - 1) // interval
        self.ep = np.repeat(np.arange(len(samples_per_ep)), samples_per_ep)
        k = np.arange(len(self.ep)) - np.repeat(np.cumsum(samples_per_ep) - samples_per_ep, samples_per_ep)
        self.t = self.offsets[self.ep] + k * interval
        self.ep_end = self.offsets[self.ep + 1]
        self.obs_idxs = self.obs_offsets[self.ep] + k
        # Repeat the last action when the chunk runs past the end of the episode.
        self.act_idxs = np.minimum(self.t[:, None] + np.arange(chunk_length), self.ep_end[:, None] - 1)

    def __len__(self):
        return len(self.obs_idxs)

    def __getitem__(self, i):
        obs = torch.from_numpy(self.images[self.obs_idxs[i]])
        act = torch.from_numpy(self.actions[self.act_idxs[i]].reshape(-1))
        return obs, act


class RLChunkDataset(ChunkDataset):
    """ChunkDataset plus n-step RL targets: (obs, action chunk, next_obs, n-step reward, discount, episode success).

    next_obs is the image n_step steps later (or the episode's final image if it ends sooner);
    discount = gamma^steps, or 0 if the episode terminated (succeeded) within those steps.
    """

    def __init__(self, data, normalizer, chunk_length, n_step, discount):
        super().__init__(data, normalizer, chunk_length)
        assert n_step % self.interval == 0, f'n_step must be a multiple of the observation interval ({self.interval})'
        rewards, masks = data['rewards'], data['masks']
        steps = np.minimum(n_step, self.ep_end - self.t)
        offsets = np.arange(n_step)
        self.rewards = np.zeros(len(self.t), dtype=np.float32)
        bootstrap_masks = np.ones(len(self.t), dtype=np.float32)
        for i in range(0, len(self.t), 50_000):  # in blocks to bound memory
            sl = slice(i, i + 50_000)
            idxs = np.minimum(self.t[sl, None] + offsets, self.ep_end[sl, None] - 1)
            valid = offsets < steps[sl, None]
            self.rewards[sl] = (rewards[idxs] * valid * discount**offsets).sum(1)
            bootstrap_masks[sl] = np.where(valid, masks[idxs], 1.0).prod(1)
        self.discounts = (discount**steps * bootstrap_masks).astype(np.float32)
        local_next = self.t - self.offsets[self.ep] + steps
        self.next_obs_idxs = self.obs_offsets[self.ep] + (local_next + self.interval - 1) // self.interval
        ep_success = np.array([np.any(rewards[s:e] == 1) for s, e in zip(self.offsets[:-1], self.offsets[1:])])
        self.success = ep_success[self.ep].astype(np.float32)

    def __getitem__(self, i):
        obs, act = super().__getitem__(i)
        next_obs = torch.from_numpy(self.images[self.next_obs_idxs[i]])
        return obs, act, next_obs, self.rewards[i], self.discounts[i], self.success[i]


def dataset_paths(env_name, num_shards=None):
    """(train shard paths, val shard paths) already on disk; downloads via ocbench only if nothing is there.
    Val paths are the -val.npz files that exist (setup.sh downloads only the held-out ones)."""
    category = 'visual' if env_name.startswith('visual-') else ('lite' if '-lite-' in env_name else 'full')
    root = os.path.join(os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'), 'ocbench', 'datasets')
    local = os.path.join(root, category, env_name.replace('-cpu-', '-', 1))
    train = sorted(p for p in glob.glob(os.path.join(local, '*.npz')) if not p.endswith('-val.npz'))
    if not train:
        train, _ = ocbench.download_datasets(env_name, num_shards=num_shards)
    val = sorted(glob.glob(os.path.join(local, '*-val.npz')))
    return train[:num_shards], val


def fit_normalizer(paths, percentile=1.0, clip=5.0):
    """Fit the action normalizer on the actions of ALL given shards (only the small 'actions' arrays are read)."""
    actions = []
    for p in paths:
        with np.load(p) as d:
            actions.append(d['actions'])
    return ActionNormalizer.fit(np.concatenate(actions), percentile, clip)


def make_datasets(
    env_name, normalizer=None, num_shards=None, num_val_shards=10, chunk_length=25, success_only=True, n_step=None,
    discount=None, train_paths=None,
):
    """Returns train/val ChunkDatasets (BC), or RLChunkDatasets if n_step is given.
    train_paths: load only these training shards (shard rotation); default = the first num_shards shards."""
    all_train, val_paths = dataset_paths(env_name, num_shards)
    train_data = load_npz(train_paths or all_train, success_only=success_only)
    val_data = load_npz(val_paths[:num_val_shards], success_only=success_only)
    if normalizer is None:
        normalizer = ActionNormalizer.fit(train_data['actions'])
    if n_step is None:
        make = lambda data: ChunkDataset(data, normalizer, chunk_length)
    else:
        make = lambda data: RLChunkDataset(data, normalizer, chunk_length, n_step, discount)
    return make(train_data), make(val_data), normalizer


class ShardGroups:
    """Splits the training shards into groups that fit in RAM; training cycles through them (shard rotation)."""

    def __init__(self, env_name, num_shards=None, group_size=None):
        self.paths, _ = dataset_paths(env_name, num_shards)
        self.group_size = group_size or len(self.paths)
        self.groups = [self.paths[i : i + self.group_size] for i in range(0, len(self.paths), self.group_size)]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        return self.groups[i % len(self.groups)]


def set_initial_states(env, qpos, qvel):
    """Overwrite every world's state with given (nworld, nq) / (nworld, nv) arrays right after env.reset(), e.g. the
    first states of held-out dataset episodes. Mirrors the env's own partial-reset path: re-run kinematics and point the
    arm controller's IK targets at the new effector pose / gripper opening."""
    wp, data = env._wp, env._data
    data.qpos.assign(np.asarray(qpos, dtype=np.float32))
    data.qvel.assign(np.asarray(qvel, dtype=np.float32))
    env._mjw.kinematics(env._model, data)
    env._mjw.com_pos(env._model, data)
    wp.launch(
        env._ik_kernels.hold_current_attach_targets,
        dim=env._nworld,
        inputs=[data.qpos, data.site_xpos, data.site_xmat, env._ik_target_pos, env._ik_target_xmat,
                env._ik_target_gripper, env._attach_site_id, env._gripper_opening_joint_id],
        device=data.qpos.device,
    )
    env._mjw.forward(env._model, data)
    return env._finish_reset()


def load_heldout_starts(env_name, num_val_shards=None):
    """First qpos/qvel and first images of every held-out (validation) episode."""
    _, val_paths = dataset_paths(env_name)
    val_paths = val_paths[:num_val_shards]
    qpos, qvel, images = [], [], []
    for path in val_paths:
        with np.load(path) as d:
            data = {k: d[k] for k in ('terminals', 'actions', 'observations', 'observation_interval', 'qpos', 'qvel')}
        offsets, obs_offsets = ocbench.get_episode_offsets(data)
        qpos.append(data['qpos'][offsets[:-1]])
        qvel.append(data['qvel'][offsets[:-1]])
        images.append(data['observations'][obs_offsets[:-1]])
    return np.concatenate(qpos), np.concatenate(qvel), np.concatenate(images)


TRACE_KEYS = ('proprio/effector_pos', 'proprio/gripper_opening', 'proprio/gripper_contact')  # + every cube position


def evaluate(policy_fn, env_name, normalizer, num_episodes, chunk_length, video_episodes=0, video_frame_skip=5,
             record_traces=False, init_states=None):
    """Roll out `num_episodes` parallel MJWarp envs; the policy sees new images every `chunk_length` steps.

    The first `video_episodes` episodes are also rendered every `video_frame_skip` steps and returned as `videos`.
    record_traces: also return per-step gripper/cube state (TRACE_KEYS) as `traces`, (num_episodes, max_steps, ...),
    NaN after an episode ends.
    """
    family, _, env_kwargs, max_steps = ocbench.parse_env_spec(env_name)
    # Physics on GPU with state observations; camera images are rendered on demand below.
    env = make_env(family, dict(env_kwargs, ob_type='states'), num_episodes)
    env.reset(seeds=np.random.randint(0, 2**32, num_episodes, dtype=np.uint32))
    if init_states is not None:  # (qpos, qvel) per episode, e.g. the held-out set's first states
        set_initial_states(env, *init_states)
    obs = env.get_pixel_observation()
    action_low, action_high = env.single_action_space.low, env.single_action_space.high

    done = np.zeros(num_episodes, dtype=bool)
    success = np.zeros(num_episodes, dtype=bool)
    lengths = np.zeros(num_episodes, dtype=np.int32)
    videos = [[env.render_world(i).copy()] for i in range(video_episodes)]
    traces = None
    step = 0
    while not done.all():
        actions = normalizer.unnormalize(policy_fn(obs).reshape(num_episodes, chunk_length, -1))
        actions = np.clip(actions, action_low, action_high)
        for c in range(chunk_length):
            action = actions[:, c].copy()
            action[done] = 0.0
            _, _, terminated, _, info = env.step(action)
            step += 1
            if record_traces:
                if traces is None:
                    trace_keys = list(TRACE_KEYS) + sorted(k for k in info if re.fullmatch(r'privileged/block_\d+_pos', k))
                    traces = {k: np.full((num_episodes, max_steps, *np.shape(info[k])[1:]), np.nan, dtype=np.float32)
                              for k in trace_keys}
                for k in traces:
                    traces[k][~done, step - 1] = info[k][~done]
            new_done = ~done & (terminated | (step >= max_steps))
            success[new_done] = info['success'][new_done]
            lengths[new_done] = step
            for i in range(video_episodes):
                if not done[i] and (step % video_frame_skip == 0 or new_done[i]):
                    videos[i].append(env.render_world(i).copy())
            done |= new_done
            if done.all():
                break
        active = np.flatnonzero(~done)
        if len(active):
            obs[active] = env.get_pixel_observation(active)
    goals = np.asarray(env._cpu_env.cur_task_info.get('goal_xyzs', np.zeros((0, 3))), dtype=np.float32)
    env.close()
    return dict(
        success=success.mean(),
        length=lengths.mean(),
        episode_success=success,
        episode_length=lengths,
        videos=[np.stack(frames) for frames in videos],
        traces=traces,
        goals=goals,
    )
