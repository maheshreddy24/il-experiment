from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from ocbench import get_episode_offsets


def get_success_masks(dataset):
    """Mark all transitions in trajectories that contain reward 1."""
    terminals = np.asarray(dataset['terminals']) > 0
    rewards = np.asarray(dataset['rewards'])
    terminal_idxs = np.flatnonzero(terminals)
    if len(terminal_idxs) == 0 or terminal_idxs[-1] != len(rewards) - 1:
        terminal_idxs = np.concatenate([terminal_idxs, [len(rewards) - 1]])
    initial_idxs = np.concatenate([[0], terminal_idxs[:-1] + 1])
    success_masks = np.zeros(len(rewards), dtype=np.float32)
    for initial_idx, terminal_idx in zip(initial_idxs, terminal_idxs):
        if np.any(rewards[initial_idx : terminal_idx + 1] == 1):
            success_masks[initial_idx : terminal_idx + 1] = 1
    return success_masks


def filter_success_trajectories(dataset):
    """Keep trajectories that contain reward 1."""
    offsets, _ = get_episode_offsets(dataset)
    keep = get_success_masks(dataset)[offsets[:-1]] > 0
    if not np.any(keep):
        raise ValueError('No successful trajectories found in dataset.')
    return select_episodes(dataset, np.flatnonzero(keep))


def select_episodes(dataset, episode_idxs):
    offsets, observation_offsets = get_episode_offsets(dataset)
    selected = {}
    for key, value in dataset.items():
        if key == 'observation_interval':
            selected[key] = value
            continue
        boundaries = observation_offsets if key == 'observations' else offsets
        parts = [value[boundaries[i] : boundaries[i + 1]] for i in episode_idxs]
        selected[key] = np.concatenate(parts) if parts else value[:0]
    return selected


def concat_datasets(datasets):
    if any(int(d['observation_interval']) != int(datasets[0]['observation_interval']) for d in datasets):
        raise ValueError('Cannot combine datasets with different observation intervals.')
    if len(datasets) == 1:
        return datasets[0]
    return {
        key: datasets[0][key] if key == 'observation_interval' else np.concatenate([d[key] for d in datasets])
        for key in datasets[0]
    }


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Episode observations with dense control transitions."""

    @classmethod
    def create(cls, freeze=True, **fields):
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), fields)
        return cls(fields)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = len(self['actions'])
        self.observation_interval = int(self['observation_interval'])
        self.frame_stack = None
        self.p_aug = None
        self.action_chunk_length = 1
        self.n_step = 1
        self.discount = 1.0
        self.update_episode_locs()

    def update_episode_locs(self):
        self.episode_offsets, self.observation_offsets = get_episode_offsets(self)
        self.initial_locs = self.episode_offsets[:-1]
        self.terminal_locs = self.episode_offsets[1:] - 1
        self.valid_idxs = None
        if self.size and self.observation_interval > 1:
            self.valid_idxs = np.concatenate(
                [
                    np.arange(start, end, self.observation_interval)
                    for start, end in zip(self.episode_offsets[:-1], self.episode_offsets[1:])
                ]
            )

    def get_random_idxs(self, num_idxs):
        if self.observation_interval == 1:
            return np.random.randint(self.size, size=num_idxs)
        return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]

    def _get_data_idxs(self, idxs):
        return idxs

    def _get_field(self, key, idxs):
        if key == 'observations':
            return self.get_observations(idxs)
        return self._dict[key][self._get_data_idxs(idxs)]

    def get_observations(self, idxs, steps=0, stack=1):
        episode_idxs = np.searchsorted(self.terminal_locs, idxs)
        local_steps = idxs - self.episode_offsets[episode_idxs] + steps
        initial_frames = self.observation_offsets[episode_idxs]
        frames = initial_frames + (local_steps + self.observation_interval - 1) // self.observation_interval
        if stack == 1:
            return self['observations'][frames]
        return np.concatenate(
            [self['observations'][np.maximum(frames - i, initial_frames)] for i in reversed(range(stack))], axis=-1
        )

    def sample(self, batch_size: int, idxs=None, keys=None):
        """Sample transitions, optionally selecting minibatch fields."""
        idxs = self.get_random_idxs(batch_size) if idxs is None else np.asarray(idxs)
        interval = self.observation_interval
        if interval > 1:
            if np.any((idxs - self.get_initial_idxs(idxs)) % interval):
                raise ValueError('Samples must start at captured observation timestamps.')
            if self.action_chunk_length != interval:
                raise ValueError('Sparse observations require action_chunk_length == observation_interval.')
            if self.n_step != 1 and self.n_step % interval:
                raise ValueError('Sparse n_step must be a multiple of observation_interval.')

        requested = (
            tuple(key for key in self if key != 'observation_interval') + ('next_observations',)
            if keys is None
            else keys
        )
        batch = {
            key: self._get_field(key, idxs)
            for key in requested
            if key in self and key not in ('observations', 'observation_interval')
        }
        steps = 1
        horizon = interval if self.n_step == 1 else self.n_step
        target_keys = ('rewards', 'discounts', 'steps', 'next_observations')
        if interval > 1:
            target_keys += ('masks', 'terminals')
        if horizon > 1 and any(key in requested for key in target_keys):
            rewards, discounts, steps, next_idxs = self.get_n_step_returns(idxs, n_step=horizon)
            batch.update(rewards=rewards, discounts=discounts, steps=steps)
            if interval > 1:
                batch['masks'] = self._get_field('masks', next_idxs)
                batch['terminals'] = self._get_field('terminals', next_idxs)
        if self.action_chunk_length > 1 and 'actions' in batch:
            batch['actions'] = self.get_action_chunks(idxs)
        stack = 1 if self.frame_stack is None else self.frame_stack
        if 'observations' in requested:
            batch['observations'] = self.get_observations(idxs, stack=stack)
        if 'next_observations' in requested:
            batch['next_observations'] = self.get_observations(idxs, steps=steps, stack=stack)
        if keys is not None:
            batch = {key: batch[key] for key in keys}
        image_keys = [key for key in ('observations', 'next_observations') if key in batch]
        if image_keys and self.p_aug is not None and np.random.rand() < self.p_aug:
            self.augment(batch, image_keys)
        return batch

    def get_terminal_idxs(self, idxs):
        """Return the terminal index of each sampled transition."""
        terminal_locs = self.terminal_locs[self.terminal_locs < self.size]
        if len(terminal_locs) == 0:
            return np.full_like(idxs, self.size - 1)
        terminal_pos = np.searchsorted(terminal_locs, idxs, side='left')
        terminal_pos = np.minimum(terminal_pos, len(terminal_locs) - 1)
        return terminal_locs[terminal_pos]

    def get_initial_idxs(self, idxs):
        """Return the initial index of each sampled transition."""
        initial_pos = np.searchsorted(self.initial_locs, idxs, side='right') - 1
        return self.initial_locs[initial_pos]

    def get_action_chunks(self, idxs):
        """Return action chunks, repeating terminal actions when needed."""
        terminal_idxs = self.get_terminal_idxs(idxs)
        offsets = np.arange(self.action_chunk_length)
        chunk_idxs = np.minimum(idxs[:, None] + offsets[None, :], terminal_idxs[:, None])
        action_chunks = self._get_field('actions', chunk_idxs)
        return action_chunks.reshape(len(idxs), -1)

    def get_n_step_returns(self, idxs, n_step=None):
        """Return n-step rewards, discounts, and bootstrap transition indices."""
        n_step = self.n_step if n_step is None else n_step
        terminal_idxs = self.get_terminal_idxs(idxs)
        steps = np.minimum(n_step, terminal_idxs - idxs + 1).astype(np.int32)
        offsets = np.arange(n_step)
        reward_idxs = np.minimum(idxs[:, None] + offsets[None, :], terminal_idxs[:, None])
        valid = offsets[None, :] < steps[:, None]

        rewards = self._get_field('rewards', reward_idxs)
        masks = self._get_field('masks', reward_idxs)
        reward_discounts = (self.discount**offsets).astype(np.float32)
        n_step_rewards = (rewards * valid * reward_discounts).sum(axis=1).astype(np.float32)
        bootstrap_masks = np.where(valid, masks, 1.0).prod(axis=1).astype(np.float32)
        discounts = ((self.discount**steps) * bootstrap_masks).astype(np.float32)
        next_idxs = idxs + steps - 1
        return n_step_rewards, discounts, steps, next_idxs

    def augment(self, batch, keys):
        """Apply image augmentation, keeping crops on device."""
        padding = 3
        batch_size = len(batch[keys[0]])
        num_views = batch[keys[0]].shape[1] if batch[keys[0]].ndim == 5 else 1
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size * num_views, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size * num_views, 1), dtype=np.int64)], axis=1)

        def crop(arr):
            if arr.ndim not in (4, 5):
                return arr
            cropped = batched_random_crop(arr.reshape((-1, *arr.shape[-3:])), crop_froms, padding)
            return cropped.reshape(arr.shape)

        for key in keys:
            batch[key] = jax.tree_util.tree_map(crop, batch[key])


class HistoryDataset(Dataset):
    """Dataset with history-conditioned sampling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_interval = None  # Interval between past states; set outside the class.
        self.history_length = None  # Number of past states; set outside the class.

    def sample(self, batch_size: int, idxs=None, keys=None):
        """Sample a batch of transitions with history fields."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        idxs = np.asarray(idxs)
        base_keys = None if keys is None else tuple(key for key in keys if key != 'histories')
        batch = super().sample(batch_size, idxs, keys=base_keys)
        if keys is None or 'histories' in keys:
            batch['histories'] = self.get_histories(idxs)
        return batch

    def get_histories(self, idxs):
        """Return past observations, repeating initial states when needed."""
        if self.history_length == 0:
            return np.zeros((len(idxs), 0), dtype=self._dict['observations'].dtype)
        initial_idxs = self.get_initial_idxs(idxs)
        offsets = np.arange(1, self.history_length + 1) * self.history_interval
        history_idxs = np.maximum(idxs[:, None] - offsets[None, :], initial_idxs[:, None])
        histories = self._get_field('observations', history_idxs)
        return histories.reshape(len(idxs), -1)


class ReplayBuffer(Dataset):
    """Control-step ring buffer with separate episode endpoints."""

    @classmethod
    def create(cls, transition, size):
        fields = {
            key: np.zeros((size, *np.asarray(value).shape), dtype=np.asarray(value).dtype)
            for key, value in transition.items()
            if key != 'next_observations'
        }
        fields['observation_interval'] = np.asarray(1, dtype=np.int32)
        return cls(fields)

    def __init__(self, *args, **kwargs):
        self.pointer = 0
        self.endpoints = {}
        self.last_observation = None
        super().__init__(*args, **kwargs)
        self.max_size = self.size
        self.size = 0
        self.update_episode_locs()

    def _get_data_idxs(self, idxs):
        idxs = np.asarray(idxs)
        return (self.pointer + idxs) % len(self['actions']) if self.size == len(self['actions']) else idxs

    def get_observations(self, idxs, steps=0, stack=1):
        initial_idxs = self.get_initial_idxs(idxs)
        terminal_idxs = self.get_terminal_idxs(idxs)
        frames = []
        for i in reversed(range(stack)):
            target_idxs = np.maximum(idxs + steps - i, initial_idxs)
            values = self['observations'][self._get_data_idxs(np.minimum(target_idxs, self.size - 1))].copy()
            endpoint = ((target_idxs == terminal_idxs + 1) | (target_idxs == self.size)) & (target_idxs > idxs)
            flat_values = values.reshape((-1, *self['observations'].shape[1:]))
            for row in np.flatnonzero(endpoint):
                target = target_idxs.reshape(-1)[row]
                flat_values[row] = (
                    self.last_observation
                    if target == self.size
                    else self.endpoints[int(self._get_data_idxs(target - 1))]
                )
            frames.append(values)
        return frames[0] if stack == 1 else np.concatenate(frames, axis=-1)

    def update_episode_locs(self):
        terminals = self._get_field('terminals', np.arange(self.size))
        self.terminal_locs = np.flatnonzero(terminals)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[self.terminal_locs + 1 < self.size] + 1])

    def add_transition(self, transition):
        for key, values in self.items():
            if key != 'observation_interval':
                values[self.pointer] = transition[key]
        self.endpoints.pop(self.pointer, None)
        self.last_observation = np.array(transition['next_observations'], copy=True)
        if transition['terminals']:
            self.endpoints[self.pointer] = self.last_observation
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def clear(self):
        self.size = self.pointer = 0
        self.endpoints.clear()
        self.last_observation = None
        self.update_episode_locs()
