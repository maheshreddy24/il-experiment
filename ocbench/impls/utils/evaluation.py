from collections import defaultdict
import inspect

import jax
import numpy as np
from tqdm import tqdm


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def clip_actions(actions, env, action_chunk_length):
    """Clip flat action chunks to the environment action space."""
    action_low = np.asarray(env.action_space.low)
    action_high = np.asarray(env.action_space.high)
    if action_chunk_length > 1:
        action_low = np.tile(action_low, action_chunk_length)
        action_high = np.tile(action_high, action_chunk_length)
    return np.clip(actions, action_low, action_high)


EVAL_METRIC_KEYS = frozenset(
    [
        'success',
        'failure',
        'healthy',
        'episode.final_reward',
        'episode.return',
        'episode.length',
        'episode.duration',
    ]
)


def add_eval_metrics(stats, info):
    """Append whitelisted evaluation metrics from an info dictionary."""
    add_to(stats, {k: v for k, v in flatten(info).items() if k in EVAL_METRIC_KEYS})


def summarize_eval_metrics(stats, max_episode_steps):
    """Add derived evaluation metrics from per-episode stats."""
    if 'success' not in stats or 'episode.length' not in stats:
        return

    successes = np.asarray(stats['success'], dtype=np.float32)
    steps = np.asarray(stats['episode.length'], dtype=np.float32)
    mean_steps = steps.mean()
    success_steps = steps[successes > 0]
    mean_success_steps = success_steps.mean() if len(success_steps) else np.nan
    success_rate = successes.mean()

    stats['mean_steps'].append(mean_steps)
    stats['mean_success_steps'].append(mean_success_steps)
    stats['throughput'].append(success_rate * max_episode_steps / mean_steps)
    stats['success_weighted_speed'].append(
        success_rate * max_episode_steps / mean_success_steps if len(success_steps) else 0.0
    )


def evaluate(
    agent,
    env,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=25,
    eval_temperature=0,
    action_chunk_length=1,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        config: Agent configuration, required for history-conditioned agents.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        action_chunk_length: Number of open-loop actions per policy call.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    use_temperature = 'temperature' in inspect.signature(agent.sample_actions).parameters
    use_history = config is not None and config.get('agent_name') == 'history_fbc'
    if use_history:
        history_offsets = np.arange(1, config['history_length'] + 1) * config['history_interval']
    envs = list(env) if isinstance(env, (list, tuple)) else [env]
    num_episodes = num_eval_episodes + num_video_episodes
    if num_episodes == 0:
        return {}, [], []
    if len(envs) < num_episodes:
        raise ValueError(f'Expected {num_episodes} evaluation environments, got {len(envs)}.')

    envs = envs[:num_episodes]
    max_episode_steps = envs[0].spec.max_episode_steps
    action_dim = envs[0].action_space.shape[-1]
    trajs = [None] * num_eval_episodes
    stats = defaultdict(list)
    renders = [None] * num_video_episodes
    states = []
    for episode_idx, env in enumerate(envs):
        observation, info = env.reset()
        states.append(
            dict(
                env=env,
                episode_idx=episode_idx,
                observation=observation,
                observations=[observation] if use_history else None,
                info=info,
                traj=defaultdict(list),
                render=[],
                step=0,
            )
        )

    active = list(range(num_episodes))
    total_steps = 0
    with tqdm(total=num_episodes, desc='eval', position=1, leave=False, dynamic_ncols=True) as pbar:
        while active:
            observations = [states[idx]['observation'] for idx in active]
            observations += [observations[-1]] * (num_episodes - len(observations))
            observations = jax.tree_util.tree_map(lambda *obs: np.stack(obs), *observations)
            action_kwargs = dict(observations=observations)
            if use_history:
                histories = []
                for idx in active:
                    state = states[idx]
                    if config['history_length'] == 0:
                        history = np.zeros((0,), dtype=state['observation'].dtype)
                    else:
                        history_idxs = np.maximum(state['step'] - history_offsets, 0)
                        history = np.concatenate([state['observations'][i] for i in history_idxs], axis=-1)
                    histories.append(history)
                histories += [histories[-1]] * (num_episodes - len(histories))
                action_kwargs['histories'] = np.stack(histories)
            if use_temperature:
                action_kwargs['temperature'] = eval_temperature
            actions = actor_fn(**action_kwargs)
            actions = np.array(actions)
            actions = clip_actions(actions, envs[0], action_chunk_length)
            action_chunks = actions.reshape(num_episodes, action_chunk_length, action_dim)

            new_active = []
            for idx, action_chunk in zip(active, action_chunks[: len(active)]):
                state = states[idx]
                done = False
                should_render = state['episode_idx'] >= num_eval_episodes
                for action in action_chunk:
                    observation = state['observation']
                    next_observation, reward, terminated, truncated, info = state['env'].step(action)
                    done = terminated or truncated
                    state['step'] += 1
                    total_steps += 1

                    if should_render and (state['step'] % video_frame_skip == 0 or done):
                        state['render'].append(state['env'].render().copy())

                    transition = dict(
                        observation=observation,
                        next_observation=next_observation,
                        action=action,
                        reward=reward,
                        done=done,
                        info=info,
                    )
                    add_to(state['traj'], transition)
                    state['observation'] = next_observation
                    if use_history:
                        state['observations'].append(next_observation)
                    state['info'] = info
                    if done:
                        break

                if done:
                    if state['episode_idx'] < num_eval_episodes:
                        add_eval_metrics(stats, state['info'])
                        trajs[state['episode_idx']] = state['traj']
                    else:
                        renders[state['episode_idx'] - num_eval_episodes] = np.array(state['render'])
                    pbar.update(1)
                else:
                    new_active.append(idx)
            active = new_active
            pbar.set_postfix(active=len(active), steps=total_steps)

    summarize_eval_metrics(stats, max_episode_steps)
    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
