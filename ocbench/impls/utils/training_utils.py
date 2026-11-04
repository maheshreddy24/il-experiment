from concurrent.futures import ThreadPoolExecutor

import numpy as np

from utils.datasets import (
    Dataset,
    HistoryDataset,
    ReplayBuffer,
    filter_success_trajectories,
    get_success_masks,
)


class DatasetPrefetcher:
    def __init__(self, num_groups, load_fn, initial_group, cache_all):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.load_fn = load_fn
        self.groups = {0: initial_group} if cache_all else None
        if cache_all:
            self.futures = {idx: self.executor.submit(load_fn, idx) for idx in range(1, num_groups)}
        else:
            self.future = self.executor.submit(load_fn, 1)

    def get(self, idx):
        if self.groups is None:
            group = self.future.result()
            self.future = None
            return group
        if idx not in self.groups:
            self.groups[idx] = self.futures.pop(idx).result()
        return self.groups[idx]

    def prefetch(self, idx):
        if self.groups is None:
            self.future = self.executor.submit(self.load_fn, idx)

    def close(self):
        self.executor.shutdown(cancel_futures=True)


def setup_agent_config(
    *,
    config,
    env_name,
    frame_stack,
    online_steps,
    action_chunk_length,
    train_dataset,
):
    if online_steps > 0:
        assert 'visual' not in env_name, 'Online fine-tuning is currently not supported for visual environments.'
        assert frame_stack is None, 'Online RL is currently only supported with flat, unstacked observations.'
    if action_chunk_length < 1:
        raise ValueError(f'Invalid action_chunk_length: {action_chunk_length}')

    if config['agent_name'] == 'history_fbc':
        if env_name.startswith('visual-'):
            raise ValueError('History agents currently require state observations.')
        assert frame_stack is None, 'History agents are currently only supported with flat, unstacked observations.'
        assert online_steps == 0, 'History agents are currently only supported for offline training.'

    n_step = config.get('n_step', 1)
    if int(train_dataset['observation_interval']) > 1:
        interval = int(train_dataset['observation_interval'])
        if not env_name.startswith('visual-') or '-cpu-' in env_name or online_steps > 0:
            raise ValueError('Sparse image datasets require offline visual MJWarp training/evaluation.')
        if action_chunk_length != interval:
            raise ValueError(f'Sparse images require action_chunk_length={interval}.')
        if config['agent_name'] in ('bc', 'fbc'):
            n_step = interval
        elif n_step % interval:
            raise ValueError(f'Sparse value targets require n_step to be a multiple of {interval}.')

    return n_step


def setup_datasets(
    train_dataset,
    val_dataset,
    FLAGS,
    config,
    n_step,
    replay_buffer=None,
    filter_success=True,
):
    if FLAGS.success_only and filter_success:
        train_dataset = filter_success_trajectories(train_dataset)
        if val_dataset is not None:
            val_dataset = filter_success_trajectories(val_dataset)

    if config.get('bc_success_only', False):
        train_dataset = dict(train_dataset, bc_masks=get_success_masks(train_dataset))
        if val_dataset is not None:
            val_dataset = dict(val_dataset, bc_masks=get_success_masks(val_dataset))

    dataset_cls = HistoryDataset if config['agent_name'] == 'history_fbc' else Dataset
    train_dataset = dataset_cls.create(**train_dataset)
    if val_dataset is not None:
        val_dataset = dataset_cls.create(**val_dataset)

    if FLAGS.online_steps == 0:
        replay_buffer = train_dataset
    elif replay_buffer is None:
        example_transition = {k: v[0] for k, v in train_dataset.sample(1, idxs=np.asarray([0])).items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.online_buffer_size)

    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            if int(dataset['observation_interval']) > 1:
                interval = int(dataset['observation_interval'])
                if FLAGS.action_chunk_length != interval or n_step % interval:
                    raise ValueError('Sparse dataset interval must match action chunks and divide n_step.')
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            dataset.action_chunk_length = FLAGS.action_chunk_length
            dataset.n_step = n_step
            dataset.discount = config.get('discount', 1.0)
            if config['agent_name'] == 'history_fbc':
                dataset.history_interval = config['history_interval']
                dataset.history_length = config['history_length']

    return train_dataset, val_dataset, replay_buffer
