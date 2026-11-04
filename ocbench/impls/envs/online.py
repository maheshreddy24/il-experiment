import collections
import time

import jax
import numpy as np

from utils.evaluation import clip_actions, flatten


def _all_finite(tree):
    return all(np.all(np.isfinite(np.asarray(value))) for value in jax.tree_util.tree_leaves(tree))


def _add_trajectory(replay_buffer, trajectory, bc_success_only):
    if bc_success_only:
        bc_mask = float(any(transition['rewards'] == 1 for transition in trajectory))
    for transition in trajectory:
        if bc_success_only:
            transition['bc_masks'] = bc_mask
        replay_buffer.add_transition(transition)


def collect_online_episodes(agent, envs, replay_buffer, rng, config, action_chunk_length):
    stats = collections.defaultdict(list)
    num_transitions = 0
    states = []
    for env in envs:
        ob, _ = env.reset()
        states.append(dict(env=env, ob=ob, traj=[]))

    active = list(range(len(envs)))
    while active:
        observations = [states[idx]['ob'] for idx in active]
        observations += [observations[-1]] * (len(envs) - len(observations))
        observations = jax.tree_util.tree_map(lambda *obs: np.stack(obs), *observations)

        rng, key = jax.random.split(rng)
        actions = agent.sample_actions(observations=observations, seed=key)
        actions = np.array(actions)
        actions = clip_actions(actions, envs[0], action_chunk_length)
        action_dim = envs[0].action_space.shape[-1]
        action_chunks = actions.reshape(len(envs), action_chunk_length, action_dim)

        new_active = []
        for idx, action_chunk in zip(active, action_chunks[: len(active)]):
            state = states[idx]
            done = False
            info = {}
            for action in action_chunk:
                ob = state['ob']
                next_ob, reward, terminated, truncated, info = state['env'].step(action.copy())
                transition_state = {key: info[key] for key in ('qpos', 'qvel', 'control') if key in info}
                numerical_failure = _all_finite(action) and not _all_finite((next_ob, reward, transition_state))
                info = dict(info)
                info['numerical_failure'] = numerical_failure
                if numerical_failure:
                    next_ob = jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), ob)
                    reward = 0.0
                    terminated = True
                    truncated = False
                    info['success'] = False
                    info['failure'] = True
                    info['healthy'] = False
                done = terminated or truncated

                transition = dict(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_ob,
                )
                state['traj'].append(transition)
                state['ob'] = next_ob
                num_transitions += 1
                if done:
                    break

            if done:
                _add_trajectory(replay_buffer, state['traj'], config.get('bc_success_only', False))
                for k, v in flatten(info).items():
                    stats[k].append(v)
            else:
                new_active.append(idx)
        active = new_active

    replay_buffer.update_episode_locs()
    metrics = {f'exploration/{k}': np.mean(v) for k, v in stats.items()}
    metrics['exploration/transitions'] = num_transitions
    metrics['exploration/online_buffer_size'] = replay_buffer.size
    return rng, metrics


def collect_online_mjwarp_episodes(
    agent,
    env,
    max_episode_steps,
    replay_buffer,
    rng,
    config,
    action_normalizer,
    action_chunk_length,
):
    num_episodes = env.nworld
    observations, _ = env.reset()
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
    trajectories = [[] for _ in range(num_episodes)]
    stats = collections.defaultdict(list)
    num_transitions = 0

    while not np.all(done):
        rng, key = jax.random.split(rng)
        policy_actions = agent.sample_actions(observations=observations, seed=key)
        policy_actions = np.asarray(policy_actions, dtype=np.float32)
        policy_actions = np.clip(policy_actions, policy_action_low, policy_action_high)
        policy_action_chunks = policy_actions.reshape(num_episodes, action_chunk_length, action_dim)
        if action_normalizer is None:
            raw_action_chunks = policy_action_chunks
        else:
            raw_actions = action_normalizer.unnormalize(policy_action_chunks.reshape(-1, action_dim))
            raw_action_chunks = raw_actions.reshape(num_episodes, action_chunk_length, action_dim)
        raw_action_chunks = np.clip(raw_action_chunks, raw_action_low, raw_action_high)

        for chunk_idx in range(action_chunk_length):
            active = ~done
            if not np.any(active):
                break
            action = raw_action_chunks[:, chunk_idx].copy()
            action[done] = 0.0
            next_observations, rewards, terminated, truncated, info = env.step(action)
            steps[active] += 1
            returns[active] += rewards[active]
            time_limit = steps >= max_episode_steps
            new_done = active & (terminated | truncated | time_limit)
            numerical_failure = active & info['numerical_failure']

            for world_id in np.where(active)[0]:
                failed_numerically = numerical_failure[world_id]
                transition = dict(
                    observations=observations[world_id].copy(),
                    actions=policy_action_chunks[world_id, chunk_idx].copy(),
                    rewards=0.0 if failed_numerically else float(rewards[world_id]),
                    terminals=float(new_done[world_id]),
                    masks=0.0 if failed_numerically else 1.0 - float(terminated[world_id]),
                    next_observations=(
                        observations[world_id].copy()
                        if failed_numerically
                        else next_observations[world_id].copy()
                    ),
                )
                trajectories[world_id].append(transition)
                num_transitions += 1

            for world_id in np.where(new_done)[0]:
                _add_trajectory(replay_buffer, trajectories[world_id], config.get('bc_success_only', False))
                final_info = {
                    'success': bool(info['success'][world_id]),
                    'failure': bool(info['failure'][world_id]),
                    'healthy': bool(info['healthy'][world_id]),
                    'numerical_failure': bool(numerical_failure[world_id]),
                    'episode': {
                        'final_reward': float(rewards[world_id]),
                        'return': float(returns[world_id]),
                        'length': int(steps[world_id]),
                        'duration': time.time() - start_times[world_id],
                    },
                }
                for k, v in flatten(final_info).items():
                    stats[k].append(v)

            done |= new_done
            if np.any(numerical_failure):
                next_observations = next_observations.copy()
                next_observations[numerical_failure] = observations[numerical_failure]
            observations = next_observations

    replay_buffer.update_episode_locs()
    metrics = {f'exploration/{k}': np.mean(v) for k, v in stats.items()}
    metrics['exploration/transitions'] = num_transitions
    metrics['exploration/online_buffer_size'] = replay_buffer.size
    return rng, metrics


def sample_online_batch(train_dataset, replay_buffer, batch_size, online_replay_ratio):
    online_batch_size = int(batch_size * online_replay_ratio)
    offline_batch_size = batch_size - online_batch_size
    batches = []
    if offline_batch_size > 0:
        batches.append(train_dataset.sample(offline_batch_size))
    if online_batch_size > 0:
        if replay_buffer.size == 0:
            raise ValueError('Online replay buffer is empty.')
        batches.append(replay_buffer.sample(online_batch_size))
    if len(batches) == 1:
        return batches[0]
    return {k: np.concatenate([batch[k] for batch in batches], axis=0) for k in batches[0]}
