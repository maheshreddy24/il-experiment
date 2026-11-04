import collections
import re
import time

import gymnasium
import numpy as np
import ocbench
from gymnasium.spaces import Box


class ActionNormalizer:
    """Percentile-based action normalizer."""

    def __init__(self, q_low, q_high, clip, gripper_dims=()):
        self.q_low = np.asarray(q_low, dtype=np.float32)
        self.q_high = np.asarray(q_high, dtype=np.float32)
        self.clip = float(clip)
        self.gripper_dims = np.zeros_like(self.q_low, dtype=bool)
        self.gripper_dims[np.asarray(gripper_dims, dtype=np.int64)] = True
        self.q_low[self.gripper_dims] = -1.0
        self.q_high[self.gripper_dims] = 1.0
        self.center = (self.q_low + self.q_high) / 2
        self.half_range = (self.q_high - self.q_low) / 2
        self.valid = self.half_range > 0
        self.action_low = np.full_like(self.q_low, -self.clip)
        self.action_high = np.full_like(self.q_high, self.clip)
        self.action_low[self.gripper_dims] = -1.0
        self.action_high[self.gripper_dims] = 1.0

    @classmethod
    def fit(cls, actions, percentile, clip):
        q_low = np.percentile(actions, percentile, axis=0).astype(np.float32)
        q_high = np.percentile(actions, 100 - percentile, axis=0).astype(np.float32)
        return cls(q_low, q_high, clip, gripper_dims=(-1,))

    def normalize(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        normalized = np.zeros_like(actions, dtype=np.float32)
        normalized[..., self.valid] = (actions[..., self.valid] - self.center[self.valid]) / self.half_range[self.valid]
        return np.clip(normalized, self.action_low, self.action_high)

    def unnormalize(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        actions = np.clip(actions, self.action_low, self.action_high)
        return actions * self.half_range + self.center


class ActionNormWrapper(gymnasium.ActionWrapper):
    """Environment wrapper that maps normalized policy actions back to env actions."""

    def __init__(self, env, action_normalizer):
        super().__init__(env)
        self.action_normalizer = action_normalizer
        self.action_space = Box(
            low=action_normalizer.action_low.astype(np.float32),
            high=action_normalizer.action_high.astype(np.float32),
            dtype=np.float32,
        )

    def action(self, action):
        action = self.action_normalizer.unnormalize(action)
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def wrap_env(env, frame_stack=None, action_normalizer=None):
    env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
    if action_normalizer is not None:
        env = ActionNormWrapper(env, action_normalizer)
    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
    env.reset()
    return env


def make_eval_env(env_name, frame_stack=None, action_normalizer=None):
    env = gymnasium.make(env_name)
    return wrap_env(env, frame_stack=frame_stack, action_normalizer=action_normalizer)


def postprocess_datasets(train_dataset, val_dataset, action_normalizer, action_clip_eps=1e-5):
    train_dataset = dict(train_dataset)
    val_dataset = None if val_dataset is None else dict(val_dataset)
    for dataset in (train_dataset, val_dataset):
        if dataset is None:
            continue
        if action_normalizer is not None:
            dataset['actions'] = action_normalizer.normalize(dataset['actions'])
        elif action_clip_eps is not None:
            dataset['actions'] = np.clip(dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps)
    return train_dataset, val_dataset


def load_npz_datasets(
    train_paths,
    val_paths,
    success_only=False,
    action_normalizer=None,
    action_norm_percentile=None,
    action_norm_clip=5.0,
    action_clip_eps=None,
    return_action_normalizer=False,
):
    """Load datasets with optional success filtering and action normalization or clipping."""

    def load(paths):
        if not paths:
            return None
        episode_mask = None
        if success_only:
            episode_mask = []
            for path in paths:
                with np.load(path) as data:
                    offsets = np.concatenate([[0], np.flatnonzero(data['terminals']) + 1])
                    rewards = data['rewards']
                keep = [np.any(rewards[start:end] == 1) for start, end in zip(offsets[:-1], offsets[1:])]
                if not any(keep):
                    raise ValueError(f'No successful trajectories found in {path}.')
                episode_mask.extend(keep)
        return ocbench.load_dataset(paths, episode_mask=episode_mask)

    if action_norm_percentile is not None and action_norm_percentile < 0:
        action_norm_percentile = None
    if action_norm_percentile is not None and action_normalizer is None:
        # Fit on all training actions, before filtering successful episodes.
        sizes = []
        for path in train_paths:
            with np.load(path) as data:
                sizes.append(len(data['terminals']))
        actions = None
        offset = 0
        for path, size in zip(train_paths, sizes):
            with np.load(path) as data:
                values = data['actions']
            if actions is None:
                actions = np.empty((sum(sizes), *values.shape[1:]), dtype=np.float32)
            actions[offset : offset + size] = values
            offset += size
            del values
        action_normalizer = ActionNormalizer.fit(actions, action_norm_percentile, action_norm_clip)
        del actions
    train_dataset = load(train_paths)
    val_dataset = load(val_paths)
    train_dataset, val_dataset = postprocess_datasets(
        train_dataset, val_dataset, action_normalizer, action_clip_eps=action_clip_eps
    )

    if return_action_normalizer:
        return train_dataset, val_dataset, action_normalizer
    return train_dataset, val_dataset
