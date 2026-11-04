import pathlib
import pickle
import sys
from collections import defaultdict

import numpy as np
from absl import app, flags
from tqdm import tqdm

import ocbench
from ocbench.oracles.controllers.block import BlockController
from ocbench.oracles.controllers.chamber import ChamberController
from ocbench.oracles.controllers.hanoi import HanoiController
from ocbench.oracles.controllers.switch import SwitchController

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'block-double-task1-v0', 'Environment name.')
flags.DEFINE_string('save_path', 'data/block-double-task1-v0.npz', 'Save path.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of training episodes.')
flags.DEFINE_integer('max_parallel_episodes', None, 'Maximum parallel MJWarp episodes.')
flags.DEFINE_integer('observation_interval', 1, 'Visual capture interval. Values >1 save sparse images with dense controls.',)


def make_controller(env, env_name):
    env_name = env_name.removeprefix('visual-')
    if env_name.startswith('block-'):
        return BlockController(env=env)
    if env_name.startswith('chamber-'):
        return ChamberController(env=env)
    if env_name.startswith('hanoi-'):
        return HanoiController(env=env)
    if env_name.startswith('switch-'):
        return SwitchController(env=env)
    raise ValueError(f'Unsupported OCBench environment: {env_name}')


def generate_mjwarp_dataset():
    impls_path = pathlib.Path(__file__).resolve().parents[1] / 'impls'
    if str(impls_path) not in sys.path:
        sys.path.insert(0, str(impls_path))
    from envs.streaming_data import generate_mjwarp_datasets

    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10
    max_parallel_episodes = FLAGS.max_parallel_episodes
    if max_parallel_episodes is None:
        max_parallel_episodes = FLAGS.num_episodes
    train_dataset, val_dataset, metrics, train_metadata, val_metadata = generate_mjwarp_datasets(
        FLAGS.env_name,
        collection_episodes=num_train_episodes,
        collection_val_episodes=num_val_episodes,
        root_seed=FLAGS.seed,
        collection_idx=0,
        max_parallel_episodes=max_parallel_episodes,
        return_metadata=True,
        observation_interval=FLAGS.observation_interval,
    )

    train_path = FLAGS.save_path
    val_path = FLAGS.save_path.replace('.npz', '-val.npz')
    train_metadata_path = FLAGS.save_path.replace('.npz', '-metadata.pkl')
    val_metadata_path = FLAGS.save_path.replace('.npz', '-val-metadata.pkl')
    pathlib.Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(train_path, **train_dataset)
    if val_dataset is not None:
        np.savez_compressed(val_path, **val_dataset)
    for path, metadata in [(train_metadata_path, train_metadata), (val_metadata_path, val_metadata)]:
        with open(path, 'wb') as f:
            pickle.dump(metadata, f)

    print('Train steps:', metrics['train/transitions'])
    print('Train success rate:', metrics['train/success_rate'])
    print('Train average episode length:', metrics['train/avg_episode_length'])
    if val_dataset is not None:
        print('Validation steps:', metrics['val/transitions'])
        print('Validation success rate:', metrics['val/success_rate'])
        print('Validation average episode length:', metrics['val/avg_episode_length'])


def main(_):
    np.random.seed(FLAGS.seed)
    if FLAGS.observation_interval > 1 and not FLAGS.env_name.startswith('visual-'):
        raise ValueError('Sparse image collection requires a visual environment.')

    _, backend, _, _ = ocbench.parse_env_spec(FLAGS.env_name)
    if backend == 'mjwarp':
        generate_mjwarp_dataset()
        return
    if FLAGS.observation_interval > 1:
        raise ValueError('Sparse image collection requires an MJWarp environment; omit `-cpu-`.')

    env = ocbench.make(FLAGS.env_name)

    dataset = defaultdict(list)
    total_steps = 0
    total_train_steps = 0
    total_val_steps = 0
    successes = []
    episode_lengths = []
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10
    train_metadata = {'episodes': []}
    val_metadata = {'episodes': []}
    discarded_episodes = 0

    ep_idx = 0
    with tqdm(total=num_train_episodes + num_val_episodes) as pbar:
        while ep_idx < num_train_episodes + num_val_episodes:
            is_train = ep_idx < num_train_episodes
            split = 'train' if is_train else 'val'
            split_ep_idx = ep_idx if is_train else ep_idx - num_train_episodes
            episode_start = total_train_steps if is_train else total_val_steps

            ob, info = env.reset()
            agent = make_controller(env, FLAGS.env_name)
            agent.reset(ob, info, transition_idx=episode_start)
            episode_data = defaultdict(list)
            episode_success = False
            episode_failure = False
            episode_healthy = True

            done = False
            while not done:
                transition_idx = episode_start + len(episode_data['actions'])
                action = np.asarray(agent.select_action(ob, info, transition_idx=transition_idx))
                action = np.clip(action, -1, 1)
                next_ob, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated
                episode_success = episode_success or bool(info['success'])
                episode_failure = episode_failure or bool(info['failure'])
                episode_healthy = episode_healthy and bool(info['healthy'])

                episode_data['observations'].append(ob)
                episode_data['actions'].append(action)
                episode_data['rewards'].append(float(reward))
                episode_data['masks'].append(1.0 - float(terminated))
                episode_data['terminals'].append(done)
                episode_data['qpos'].append(info['prev_qpos'])
                episode_data['qvel'].append(info['prev_qvel'])
                if 'prev_dynamics_info' in info:
                    episode_data['dynamics_info'].append(info['prev_dynamics_info'])
                if 'prev_button_states' in info:
                    episode_data['button_states'].append(info['prev_button_states'])

                ob = next_ob

            if not episode_healthy:
                discarded_episodes += 1
                continue

            episode_data['observations'].append(next_ob)
            for key, value in episode_data.items():
                dataset[key].extend(value)

            episode_length = len(episode_data['actions'])
            episode_end = episode_start + episode_length - 1
            agent.finish_episode(episode_end, info)
            episode_metadata = dict(
                episode_idx=split_ep_idx,
                split=split,
                env_name=FLAGS.env_name,
                task_name=env.unwrapped.cur_task_info['task_name'],
                task_id=env.unwrapped.cur_task_id,
                start=episode_start,
                end=episode_end,
                length=episode_length,
                success=episode_success,
                failure=episode_failure,
                healthy=episode_healthy,
                speed=agent.speed,
                p_mistake=agent.p_mistake,
                segments=agent.segments,
            )
            if is_train:
                train_metadata['episodes'].append(episode_metadata)
            else:
                val_metadata['episodes'].append(episode_metadata)

            total_steps += episode_length
            successes.append(episode_success)
            episode_lengths.append(episode_length)
            if is_train:
                total_train_steps += episode_length
            else:
                total_val_steps += episode_length

            ep_idx += 1
            pbar.update(1)

    print('Discarded unhealthy episodes:', discarded_episodes)

    print('Total steps:', total_steps)
    print('Success rate:', np.mean(successes))
    print('Average episode length:', np.mean(episode_lengths))

    train_path = FLAGS.save_path
    val_path = FLAGS.save_path.replace('.npz', '-val.npz')
    train_metadata_path = FLAGS.save_path.replace('.npz', '-metadata.pkl')
    val_metadata_path = FLAGS.save_path.replace('.npz', '-val-metadata.pkl')
    pathlib.Path(train_path).parent.mkdir(parents=True, exist_ok=True)

    train_dataset = {}
    val_dataset = {}
    for k, v in dataset.items():
        if k == 'observations' and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k == 'terminals':
            dtype = bool
        else:
            dtype = np.float32
        split = total_train_steps + num_train_episodes if k == 'observations' else total_train_steps
        train_dataset[k] = np.array(v[:split], dtype=dtype)
        val_dataset[k] = np.array(v[split:], dtype=dtype)

    train_dataset['observation_interval'] = np.asarray(1, dtype=np.int32)
    val_dataset['observation_interval'] = np.asarray(1, dtype=np.int32)
    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dataset)
    for path, metadata in [(train_metadata_path, train_metadata), (val_metadata_path, val_metadata)]:
        with open(path, 'wb') as f:
            pickle.dump(metadata, f)


if __name__ == '__main__':
    app.run(main)
