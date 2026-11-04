from collections import defaultdict
import inspect
import time

import jax
import numpy as np
import ocbench
from ocbench.mjwarp import make_env
from tqdm import tqdm

from envs.streaming_data import make_collection_seeds
from utils.evaluation import add_eval_metrics, add_to, summarize_eval_metrics, supply_rng


def _stacked_observation(frames):
    return np.concatenate([frames[:, i] for i in range(frames.shape[1])], axis=-1)


def _history_observations(observations, steps, config):
    history_length = int(config['history_length'])
    if history_length == 0:
        return np.zeros((observations.shape[0], 0), dtype=observations.dtype)
    offsets = np.arange(1, history_length + 1) * int(config['history_interval'])
    history_idxs = np.maximum(steps[:, None] - offsets[None, :], 0)
    histories = observations[np.arange(observations.shape[0])[:, None], history_idxs]
    return histories.reshape(observations.shape[0], -1)


def evaluate_mjwarp(
    agent,
    env_name,
    config=None,
    action_normalizer=None,
    frame_stack=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=25,
    eval_temperature=0,
    action_chunk_length=1,
    observation_interval=1,
):
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    use_temperature = 'temperature' in inspect.signature(agent.sample_actions).parameters
    num_episodes = num_eval_episodes + num_video_episodes
    if num_episodes == 0:
        return {}, [], []
    use_history = config is not None and config.get('agent_name') == 'history_fbc'
    if use_history and frame_stack is not None:
        raise ValueError('MJWarp history eval only supports flat, unstacked observations.')

    family, backend, env_kwargs, max_episode_steps = ocbench.parse_env_spec(env_name)
    if backend != 'mjwarp':
        raise ValueError('MJWarp evaluation requires an environment name without `-cpu-`.')
    sparse = observation_interval > 1
    if sparse:
        if env_kwargs.get('ob_type') != 'pixels' or action_chunk_length != observation_interval:
            raise ValueError(
                'Sparse evaluation requires visual observations and matching action chunks/observation interval.'
            )
        env_kwargs = dict(env_kwargs, ob_type='states', visualize_info=False)
    env = make_env(family, env_kwargs, num_episodes)
    seeds = make_collection_seeds(np.random.randint(0, 2**32), 0, 10, num_episodes)
    raw_observations, _ = env.reset(seeds=seeds)
    if sparse:
        raw_observations = env.get_pixel_observation()
    stack = 1 if frame_stack is None else int(frame_stack)
    frames = np.repeat(raw_observations[:, None], stack, axis=1)
    observations = raw_observations if stack == 1 else _stacked_observation(frames)
    history_observations = None
    if use_history:
        history_observations = np.empty(
            (num_episodes, max_episode_steps + 1, raw_observations.shape[-1]),
            dtype=raw_observations.dtype,
        )
        history_observations[:, 0] = raw_observations

    action_dim = env.single_action_space.shape[-1]
    raw_action_low = env.single_action_space.low
    raw_action_high = env.single_action_space.high
    if action_normalizer is None:
        policy_action_low = raw_action_low
        policy_action_high = raw_action_high
    else:
        policy_action_low = action_normalizer.action_low
        policy_action_high = action_normalizer.action_high
    if action_chunk_length > 1:
        policy_action_low = np.tile(policy_action_low, action_chunk_length)
        policy_action_high = np.tile(policy_action_high, action_chunk_length)

    done = np.zeros(num_episodes, dtype=bool)
    steps = np.zeros(num_episodes, dtype=np.int32)
    returns = np.zeros(num_episodes, dtype=np.float32)
    start_times = np.full(num_episodes, time.time(), dtype=np.float64)
    stats = defaultdict(list)
    trajs = [defaultdict(list) for _ in range(num_eval_episodes)]
    renders = [[] for _ in range(num_video_episodes)]
    total_steps = 0

    with tqdm(total=num_episodes, desc='eval', position=1, leave=False, dynamic_ncols=True) as pbar:
        while not np.all(done):
            action_kwargs = dict(observations=observations)
            if use_history:
                action_kwargs['histories'] = _history_observations(history_observations, steps, config)
            if use_temperature:
                action_kwargs['temperature'] = eval_temperature
            actions = actor_fn(**action_kwargs)
            actions = np.asarray(actions, dtype=np.float32)
            actions = np.clip(actions, policy_action_low, policy_action_high)
            action_chunks = actions.reshape(num_episodes, action_chunk_length, action_dim)
            if action_normalizer is not None:
                flat_actions = action_normalizer.unnormalize(action_chunks.reshape(-1, action_dim))
                action_chunks = flat_actions.reshape(num_episodes, action_chunk_length, action_dim)
            action_chunks = np.clip(action_chunks, raw_action_low, raw_action_high)

            for chunk_idx in range(action_chunk_length):
                active = ~done
                if not np.any(active):
                    break
                action = action_chunks[:, chunk_idx].copy()
                action[done] = 0.0
                prev_observations = observations if sparse else observations.copy()
                raw_next_observations, reward, terminated, _, info = env.step(action)
                steps[active] += 1
                returns[active] += reward[active]
                total_steps += int(active.sum())
                if use_history:
                    history_observations[active, steps[active]] = raw_next_observations[active]

                time_limit = steps >= max_episode_steps
                new_done = active & (terminated | time_limit)
                if sparse:
                    capture = np.flatnonzero(active & ((steps % observation_interval == 0) | new_done))
                    next_observations = observations
                    if len(capture):
                        next_observations = observations.copy()
                        images = env.get_pixel_observation(capture)
                        frames[capture] = np.roll(frames[capture], shift=-1, axis=1)
                        frames[capture, -1] = images
                        next_observations[capture] = _stacked_observation(frames[capture])
                elif stack > 1:
                    frames = np.roll(frames, shift=-1, axis=1)
                    frames[:, -1, :] = raw_next_observations
                    next_observations = _stacked_observation(frames)
                else:
                    next_observations = raw_next_observations

                for world_id in np.where(active)[0]:
                    world_done = bool(new_done[world_id])
                    if world_id < num_eval_episodes:
                        transition_info = {
                            'success': bool(info['success'][world_id]),
                            'failure': bool(info['failure'][world_id]),
                            'healthy': bool(info['healthy'][world_id]),
                        }
                        transition = dict(
                            action=action[world_id],
                            reward=reward[world_id],
                            done=world_done,
                            info=transition_info,
                        )
                        if not sparse or chunk_idx == 0:
                            transition['observation'] = prev_observations[world_id]
                        if not sparse or world_id in capture:
                            transition['next_observation'] = next_observations[world_id]
                        add_to(trajs[world_id], transition)
                    else:
                        video_idx = world_id - num_eval_episodes
                        if steps[world_id] % video_frame_skip == 0 or world_done:
                            renders[video_idx].append(env.render_world(world_id).copy())

                for world_id in np.where(new_done)[0]:
                    final_info = {
                        'success': bool(info['success'][world_id]),
                        'failure': bool(info['failure'][world_id]),
                        'healthy': bool(info['healthy'][world_id]),
                        'episode': {
                            'final_reward': float(reward[world_id]),
                            'return': float(returns[world_id]),
                            'length': int(steps[world_id]),
                            'duration': time.time() - start_times[world_id],
                        },
                    }
                    if world_id < num_eval_episodes:
                        add_eval_metrics(stats, final_info)
                    pbar.update(1)

                done |= new_done
                observations = next_observations
                if np.any(new_done) or total_steps % max(num_episodes * 25, 1) == 0:
                    pbar.set_postfix(active=int((~done).sum()), steps=total_steps)

    summarize_eval_metrics(stats, max_episode_steps)
    for k, v in stats.items():
        stats[k] = np.mean(v)

    trajs = [dict(traj) for traj in trajs]
    renders = [np.asarray(render) for render in renders]
    env.close()
    return stats, trajs, renders
