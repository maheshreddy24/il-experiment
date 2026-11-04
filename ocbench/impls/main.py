import glob
import json
import os
import random
import time

import jax
import numpy as np
import ocbench
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import ActionNormalizer, load_npz_datasets, make_eval_env, postprocess_datasets
from envs.streaming_data import generate_mjwarp_datasets
from envs.online import collect_online_episodes, collect_online_mjwarp_episodes, sample_online_batch
from utils.evaluation import evaluate
from utils.evaluation_mjwarp import evaluate_mjwarp
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from utils.training_utils import (
    DatasetPrefetcher,
    setup_agent_config,
    setup_datasets,
)
from utils.viz_utils import get_action_histogram, get_traj_observations

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'block-double-task1-v0', 'Environment name.')
flags.DEFINE_string('dataset_path', None, 'Dataset file or directory.')
flags.DEFINE_string('dataset_root', None, 'Root directory for downloaded datasets.')
flags.DEFINE_integer('dataset_replace_interval', 10000, 'Dataset replace interval.')
flags.DEFINE_integer('num_shards', None, 'Number of training shards to use.')
flags.DEFINE_integer('dataset_group_size', 1, 'Number of consecutive training shards to merge before cycling.')
flags.DEFINE_enum('dataset_prefetch_mode', 'none', ['none', 'next', 'all'], 'Dataset prefetch mode. `next` keeps one group ahead; `all` caches every group.')
flags.DEFINE_integer('streaming_data', 0, 'Collect new datasets during training.')
flags.DEFINE_integer('collection_interval', 100000, 'Streaming-data collection interval.')
flags.DEFINE_integer('collection_episodes', 25000, 'Number of streaming-data training episodes per collection.')
flags.DEFINE_integer('collection_val_episodes', 100, 'Number of streaming-data validation episodes per collection.')
flags.DEFINE_integer('collection_max_parallel_episodes', 2500, 'Max parallel MJWarp collection episodes.')
flags.DEFINE_integer('collection_observation_interval', 1, 'Image interval for streaming visual collection; >1 uses sparse storage.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 2000000, 'Number of offline steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval. Set to 0 to disable saving.')

flags.DEFINE_integer('eval_episodes', 500, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 25, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('action_chunk_length', 25, 'Action chunk length.')
flags.DEFINE_integer('success_only', 1, 'Whether to keep only successful trajectories.')
flags.DEFINE_float('action_norm_percentile', 1.0, 'Action normalization lower percentile. Set to -1 to disable it.')
flags.DEFINE_float('action_norm_clip', 5.0, 'Action normalization clip value.')

flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('online_buffer_size', 2000000, 'Online replay buffer size.')
flags.DEFINE_float('online_replay_ratio', 0.5, 'Fraction of each online batch sampled from the online buffer.')
flags.DEFINE_integer('online_collection_interval', 100, 'Online update interval between data collection.')
flags.DEFINE_integer('online_collection_episodes', 50, 'Number of episodes to collect per online collection.')

config_flags.DEFINE_config_file('agent', os.path.join(os.path.dirname(__file__), 'agents/fbc.py'), lock_config=False)


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='ocbench', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Make environment and datasets.
    config = FLAGS.agent
    use_history = config['agent_name'] == 'history_fbc'
    dataset_idx = 0
    dataset_groups = []
    collection_idx = 0
    streaming_data = FLAGS.streaming_data
    _, backend, _, max_episode_steps = ocbench.parse_env_spec(FLAGS.env_name)
    use_mjwarp = backend == 'mjwarp'
    datasets_success_filtered = False
    if streaming_data and not use_mjwarp:
        raise ValueError('Streaming data collection requires an MJWarp environment; omit `-cpu-`.')
    collection_kwargs = dict(
        collection_episodes=FLAGS.collection_episodes,
        collection_val_episodes=FLAGS.collection_val_episodes,
        root_seed=FLAGS.seed,
        max_parallel_episodes=FLAGS.collection_max_parallel_episodes,
        observation_interval=FLAGS.collection_observation_interval,
    )

    env = None
    eval_env = None
    if streaming_data:
        raw_train_dataset, raw_val_dataset, metrics = generate_mjwarp_datasets(
            FLAGS.env_name,
            collection_idx=collection_idx,
            **collection_kwargs,
        )
        print(
            f'Collected MJWarp data {collection_idx}: '
            f'{metrics["train/transitions"]} train transitions in {metrics["collection_time"]:.2f}s.'
        )
        action_norm_percentile = FLAGS.action_norm_percentile
        if action_norm_percentile is not None and action_norm_percentile < 0:
            action_norm_percentile = None
        action_normalizer = None
        if action_norm_percentile is not None:
            action_normalizer = ActionNormalizer.fit(
                raw_train_dataset['actions'],
                action_norm_percentile,
                FLAGS.action_norm_clip,
            )
        train_dataset, val_dataset = postprocess_datasets(
            raw_train_dataset,
            raw_val_dataset,
            action_normalizer,
        )
    else:
        if FLAGS.dataset_path is None:
            train_paths, val_paths = ocbench.download_datasets(FLAGS.env_name, FLAGS.dataset_root, FLAGS.num_shards)
        else:
            dataset_path = os.path.expanduser(FLAGS.dataset_path)
            if os.path.isdir(dataset_path):
                train_paths = [
                    file for file in sorted(glob.glob(os.path.join(dataset_path, '*.npz'))) if not file.endswith('-val.npz')
                ]
            else:
                train_paths = [dataset_path]
            if FLAGS.num_shards is not None:
                train_paths = train_paths[: FLAGS.num_shards]
            val_paths = [path.removesuffix('.npz') + '-val.npz' for path in train_paths]
        if not train_paths:
            raise FileNotFoundError(f'No training datasets found at {FLAGS.dataset_path}')
        if FLAGS.dataset_group_size < 1:
            raise ValueError(f'Invalid dataset_group_size: {FLAGS.dataset_group_size}')
        for i in range(0, len(train_paths), FLAGS.dataset_group_size):
            group_train_paths = train_paths[i : i + FLAGS.dataset_group_size]
            group_val_paths = [path for path in val_paths[i : i + FLAGS.dataset_group_size] if os.path.exists(path)]
            dataset_groups.append((group_train_paths, group_val_paths))
        datasets_success_filtered = use_mjwarp and bool(FLAGS.success_only)
        train_paths, val_paths = dataset_groups[dataset_idx]
        train_dataset, val_dataset, action_normalizer = load_npz_datasets(
            train_paths,
            val_paths,
            success_only=datasets_success_filtered,
            action_norm_percentile=FLAGS.action_norm_percentile,
            action_norm_clip=FLAGS.action_norm_clip,
            action_clip_eps=1e-5,
            return_action_normalizer=True,
        )
        if not use_mjwarp:
            env = make_eval_env(
                FLAGS.env_name,
                frame_stack=FLAGS.frame_stack,
                action_normalizer=action_normalizer,
            )
            eval_env = make_eval_env(
                FLAGS.env_name,
                frame_stack=FLAGS.frame_stack,
                action_normalizer=action_normalizer,
            )
    if action_normalizer is not None:
        action_normalizer_path = os.path.join(FLAGS.save_dir, 'action_normalizer.npz')
        np.savez(
            action_normalizer_path,
            q_low=action_normalizer.q_low,
            q_high=action_normalizer.q_high,
            clip=action_normalizer.clip,
            gripper_dims=np.flatnonzero(action_normalizer.gripper_dims),
        )
        print(f'Saved action normalizer to {action_normalizer_path}')
    n_step = setup_agent_config(
        config=config,
        env_name=FLAGS.env_name,
        frame_stack=FLAGS.frame_stack,
        online_steps=FLAGS.online_steps,
        action_chunk_length=FLAGS.action_chunk_length,
        train_dataset=train_dataset,
    )
    num_eval_envs = FLAGS.eval_episodes + FLAGS.video_episodes
    eval_envs = []
    if not use_mjwarp:
        eval_envs.append(eval_env)
        for _ in range(num_eval_envs - 1):
            eval_envs.append(
                make_eval_env(FLAGS.env_name, frame_stack=FLAGS.frame_stack, action_normalizer=action_normalizer)
            )
    online_envs = None
    online_mjwarp_env = None
    if FLAGS.online_steps > 0 and FLAGS.online_replay_ratio > 0:
        if use_mjwarp:
            online_mjwarp_env = ocbench.make(FLAGS.env_name, nworld=FLAGS.online_collection_episodes)
        else:
            online_envs = [env]
            for _ in range(FLAGS.online_collection_episodes - 1):
                online_envs.append(
                    make_eval_env(FLAGS.env_name, frame_stack=FLAGS.frame_stack, action_normalizer=action_normalizer)
                )

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    train_dataset, val_dataset, replay_buffer = setup_datasets(
        train_dataset,
        val_dataset,
        FLAGS,
        config,
        n_step,
        filter_success=not datasets_success_filtered,
    )

    # Create agent.
    agent_class = agents[config['agent_name']]
    sample_kwargs = getattr(agent_class, 'sample_kwargs', {})
    example_batch = train_dataset.sample(1, **sample_kwargs)

    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    dataset_prefetcher = None
    total_steps = FLAGS.offline_steps + FLAGS.online_steps

    def load_dataset_group(group_idx):
        train_paths, val_paths = dataset_groups[group_idx]
        return load_npz_datasets(
            train_paths,
            val_paths,
            success_only=datasets_success_filtered,
            action_normalizer=action_normalizer,
            action_clip_eps=1e-5,
        )

    if (
        FLAGS.dataset_prefetch_mode != 'none'
        and FLAGS.dataset_replace_interval != 0
        and total_steps >= FLAGS.dataset_replace_interval
        and len(dataset_groups) > 1
    ):
        dataset_prefetcher = DatasetPrefetcher(
            num_groups=len(dataset_groups),
            load_fn=load_dataset_group,
            initial_group=(train_dataset, val_dataset),
            cache_all=FLAGS.dataset_prefetch_mode == 'all',
        )

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    metric_rng = jax.random.PRNGKey(FLAGS.seed + 1)
    for i in tqdm.tqdm(
        range(1, total_steps + 1), position=0, smoothing=0.1, dynamic_ncols=True
    ):
        if i <= FLAGS.offline_steps:
            # Offline RL.
            batch = train_dataset.sample(config['batch_size'], **sample_kwargs)
            agent, update_info = agent.update(batch)
        else:
            # Online RL.
            online_step = i - FLAGS.offline_steps
            should_collect = online_step == 1 or (
                FLAGS.online_collection_interval != 0 and (online_step - 1) % FLAGS.online_collection_interval == 0
            )
            if FLAGS.online_replay_ratio > 0 and should_collect:
                if use_mjwarp:
                    online_rng, expl_metrics = collect_online_mjwarp_episodes(
                        agent,
                        online_mjwarp_env,
                        max_episode_steps,
                        replay_buffer,
                        online_rng,
                        config,
                        action_normalizer,
                        FLAGS.action_chunk_length,
                    )
                else:
                    online_rng, expl_metrics = collect_online_episodes(
                        agent,
                        online_envs,
                        replay_buffer,
                        online_rng,
                        config,
                        FLAGS.action_chunk_length,
                    )

            batch = sample_online_batch(
                train_dataset,
                replay_buffer,
                config['batch_size'],
                FLAGS.online_replay_ratio,
            )

            agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_info = dict(update_info)
            if hasattr(agent, 'compute_metrics'):
                metric_rng, cur_metric_rng = jax.random.split(metric_rng)
                train_info.update(agent.compute_metrics(batch, rng=cur_metric_rng))
            train_metrics = {f'training/{k}': v for k, v in train_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'], **sample_kwargs)
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                if hasattr(agent, 'compute_metrics'):
                    metric_rng, cur_metric_rng = jax.random.split(metric_rng)
                    val_info.update(agent.compute_metrics(val_batch, rng=cur_metric_rng))
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            eval_metrics = {}
            if use_mjwarp:
                eval_info, trajs, renders = evaluate_mjwarp(
                    agent=agent,
                    env_name=FLAGS.env_name,
                    config=config,
                    action_normalizer=action_normalizer,
                    frame_stack=FLAGS.frame_stack,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    action_chunk_length=FLAGS.action_chunk_length,
                    observation_interval=train_dataset.observation_interval,
                )
            else:
                eval_info, trajs, renders = evaluate(
                    agent=agent,
                    env=eval_envs,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    action_chunk_length=FLAGS.action_chunk_length,
                )
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            metric_rng, cur_hist_rng = jax.random.split(metric_rng)
            if not use_history:
                action_hist_batches = 8
                action_dim = train_dataset['actions'].shape[-1]
                hist_observations = [
                    train_dataset.sample(config['batch_size'], **sample_kwargs)['observations']
                    for _ in range(action_hist_batches)
                ]
                eval_observations = []
                for _ in range(action_hist_batches):
                    observations = get_traj_observations(trajs, config['batch_size'])
                    if observations is not None:
                        eval_observations.append(observations)
                eval_metrics['action_hist/policy'] = get_action_histogram(
                    agent,
                    hist_observations,
                    eval_observations,
                    cur_hist_rng,
                    action_dim,
                    FLAGS.action_chunk_length,
                )

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if FLAGS.save_interval != 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

        if (
            streaming_data
            and FLAGS.collection_interval != 0
            and i % FLAGS.collection_interval == 0
        ):
            collection_idx += 1
            raw_train_dataset, raw_val_dataset, metrics = generate_mjwarp_datasets(
                FLAGS.env_name,
                collection_idx=collection_idx,
                **collection_kwargs,
            )
            print(
                f'Collected MJWarp data {collection_idx}: '
                f'{metrics["train/transitions"]} train transitions in {metrics["collection_time"]:.2f}s.'
            )
            train_dataset, val_dataset = postprocess_datasets(
                raw_train_dataset,
                raw_val_dataset,
                action_normalizer,
            )
            train_dataset, val_dataset, replay_buffer = setup_datasets(
                train_dataset,
                val_dataset,
                FLAGS,
                config,
                n_step,
                replay_buffer=replay_buffer,
            )
            wandb.log({f'collection/{k}': v for k, v in metrics.items()}, step=i)

        if (
            not streaming_data
            and FLAGS.dataset_replace_interval != 0
            and i % FLAGS.dataset_replace_interval == 0
            and len(dataset_groups) > 1
        ):
            dataset_idx = (dataset_idx + 1) % len(dataset_groups)
            if dataset_prefetcher is None:
                train_dataset, val_dataset = load_dataset_group(dataset_idx)
            else:
                train_dataset, val_dataset = dataset_prefetcher.get(dataset_idx)
            train_dataset, val_dataset, replay_buffer = setup_datasets(
                train_dataset,
                val_dataset,
                FLAGS,
                config,
                n_step,
                replay_buffer=replay_buffer,
                filter_success=not datasets_success_filtered,
            )
            if dataset_prefetcher is not None and i + FLAGS.dataset_replace_interval <= total_steps:
                dataset_prefetcher.prefetch((dataset_idx + 1) % len(dataset_groups))

    if dataset_prefetcher is not None:
        dataset_prefetcher.close()
    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
