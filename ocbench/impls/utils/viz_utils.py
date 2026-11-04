import jax
import numpy as np
import wandb


def get_traj_observations(trajs, max_observations):
    obs_seqs = [traj['observation'] for traj in trajs if traj is not None and 'observation' in traj]
    lengths = [len(obs_seq) for obs_seq in obs_seqs]
    total = sum(lengths)
    if total == 0:
        return None

    sample_size = min(max_observations, total)
    flat_idxs = np.sort(np.random.choice(total, size=sample_size, replace=False))
    observations = []
    offset = 0
    pos = 0
    for obs_seq, length in zip(obs_seqs, lengths):
        while pos < sample_size and flat_idxs[pos] < offset + length:
            observations.append(obs_seq[flat_idxs[pos] - offset])
            pos += 1
        offset += length
    return jax.tree_util.tree_map(lambda *obs: np.stack(obs), *observations)


def get_action_histogram(
    agent,
    batch_observation_batches,
    eval_observation_batches,
    rng,
    action_dim,
    action_chunk_length,
):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sources = [('batch states', batch_observation_batches)]
    if len(eval_observation_batches) > 0:
        sources.append(('eval states', eval_observation_batches))

    rngs = jax.random.split(rng, len(sources))
    source_actions = []
    for (_, observation_batches), source_rng in zip(sources, rngs):
        sample_rngs = jax.random.split(source_rng, len(observation_batches))
        action_batches = []
        for observations, sample_rng in zip(observation_batches, sample_rngs):
            actions = agent.sample_actions(observations=observations, seed=sample_rng)
            actions = np.asarray(actions, dtype=np.float32)
            actions = actions.reshape(-1, action_chunk_length, action_dim).reshape(-1, action_dim)
            action_batches.append(actions)
        source_actions.append(np.concatenate(action_batches, axis=0))

    fig, axes = plt.subplots(len(sources), action_dim, figsize=(2.2 * action_dim, 2.4 * len(sources)), squeeze=False)
    for row, ((label, _), actions) in enumerate(zip(sources, source_actions)):
        for dim in range(action_dim):
            ax = axes[row, dim]
            ax.hist(actions[:, dim], bins=50, color='#4c78a8', alpha=0.85)
            if row == 0:
                ax.set_title(f'a{dim}', fontsize=9)
            if dim == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.tick_params(labelsize=7)

    fig.tight_layout()
    image = wandb.Image(fig)
    plt.close(fig)
    return image
