import time

import numpy as np
import ocbench
from ocbench.mjwarp import make_env
from ocbench.dataset_utils import DATASET_KEYS

from utils.datasets import concat_datasets, get_episode_offsets, select_episodes


SPLIT_IDS = {
    'train_reset': 0,
    'val_reset': 1,
    'train_oracle': 2,
    'val_oracle': 3,
}
EPISODE_INDEX_BITS = 18
COLLECTION_INDEX_BITS = 12
MAX_EPISODES_PER_SPLIT = 1 << EPISODE_INDEX_BITS
MAX_COLLECTIONS = 1 << COLLECTION_INDEX_BITS


def _mix_uint32(value):
    value = int(value) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return np.uint32(value)


def make_collection_seeds(root_seed, collection_idx, split, num_episodes):
    split_id = SPLIT_IDS[split] if isinstance(split, str) else int(split)
    if num_episodes >= MAX_EPISODES_PER_SPLIT:
        raise ValueError(f'collection episodes per split must be < {MAX_EPISODES_PER_SPLIT}.')
    if collection_idx >= MAX_COLLECTIONS:
        raise ValueError(f'collection_idx must be < {MAX_COLLECTIONS}.')
    seed_sequence = np.random.SeedSequence([int(root_seed), int(collection_idx), int(split_id)])
    seeds = np.empty(num_episodes, dtype=np.uint32)
    root_mask = np.random.SeedSequence([int(root_seed)]).generate_state(1, dtype=np.uint32)[0]
    for idx, child in enumerate(seed_sequence.spawn(num_episodes)):
        spawn_idx = int(child.spawn_key[-1])
        packed = (int(collection_idx) << (EPISODE_INDEX_BITS + 2)) | (split_id << EPISODE_INDEX_BITS) | spawn_idx
        seeds[idx] = _mix_uint32(np.uint32(packed) ^ root_mask)
    if len(np.unique(seeds)) != num_episodes:
        raise AssertionError('Duplicate collection seeds.')
    return seeds


def _assert_disjoint(*seed_groups):
    seeds = np.concatenate([np.asarray(group, dtype=np.uint32) for group in seed_groups])
    if len(np.unique(seeds)) != len(seeds):
        raise AssertionError('Duplicate collection seeds across splits.')


def _dedupe_collection_seed_groups(root_seed, collection_idx, seed_groups):
    used = set()
    for split, seeds in seed_groups:
        split_id = SPLIT_IDS[split]
        for seed_idx in range(len(seeds)):
            seed = int(seeds[seed_idx])
            retry = 0
            while seed in used:
                retry += 1
                seed = int(
                    np.random.SeedSequence(
                        [int(root_seed), int(collection_idx), int(split_id), int(seed_idx), int(retry)]
                    ).generate_state(1, dtype=np.uint32)[0]
                )
            seeds[seed_idx] = np.uint32(seed)
            used.add(seed)


def _compact_dataset_gpu(
    wp,
    device,
    dense,
    lengths_gpu,
    lengths,
    world_start,
    world_count,
    nworld,
    max_episode_steps,
    action_dim,
):
    from ocbench.mjwarp.envs.manipulation_kernels import compact_field, compact_transitions

    split_lengths = lengths[world_start : world_start + world_count].astype(np.int32, copy=False)
    total = int(split_lengths.sum())
    offsets = np.concatenate([[0], np.cumsum(split_lengths[:-1])]).astype(np.int32)
    offsets_gpu = wp.array(offsets, dtype=wp.int32, device=device)
    actions = wp.empty((total, action_dim), dtype=wp.float32, device=device)
    rewards = wp.empty(total, dtype=wp.float32, device=device)
    masks = wp.empty(total, dtype=wp.float32, device=device)
    terminals = wp.empty(total, dtype=wp.int32, device=device)
    wp.launch(
        compact_transitions,
        dim=max_episode_steps * world_count,
        inputs=[
            dense['actions'],
            dense['rewards'],
            dense['masks'],
            dense['terminals'],
            lengths_gpu,
            offsets_gpu,
            actions,
            rewards,
            masks,
            terminals,
            world_start,
            world_count,
            nworld,
            action_dim,
        ],
        device=device,
    )
    dataset = {
        'actions': actions.numpy(),
        'rewards': rewards.numpy(),
        'masks': masks.numpy(),
        'terminals': terminals.numpy().astype(bool),
    }
    for key, value in dense.items():
        if key in dataset:
            continue
        count = total
        field_lengths, field_offsets = lengths_gpu, offsets_gpu
        max_steps = max_episode_steps
        if key == 'observations':
            count += world_count
            max_steps += 1
            field_lengths = wp.array(lengths + 1, dtype=wp.int32, device=device)
            field_offsets = wp.array(offsets + np.arange(world_count), dtype=wp.int32, device=device)
        compact = wp.empty((count, value.shape[1]), dtype=wp.float32, device=device)
        wp.launch(
            compact_field,
            dim=(max_steps * world_count, value.shape[1]),
            inputs=[value, field_lengths, field_offsets, compact, world_start, world_count, nworld],
            device=device,
        )
        dataset[key] = compact.numpy()
    return dataset


def _episode_returns(rewards, lengths):
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    return np.asarray([rewards[offsets[idx] : offsets[idx + 1]].sum() for idx in range(len(lengths))], dtype=np.float32)


def _make_episode_metadata(env_name, split, lengths, successes, speed_dt, episode_info):
    episodes = []
    start = 0
    for episode_idx, (length, success, cur_speed_dt, info) in enumerate(
        zip(lengths, successes, speed_dt, episode_info)
    ):
        episode = dict(
            episode_idx=episode_idx,
            split=split,
            env_name=env_name,
            **info,
            start=int(start),
            end=int(start + length - 1),
            length=int(length),
            success=bool(success),
            speed=float(1.0 / cur_speed_dt),
        )
        episode['segments'] = [
            dict(segment, start=int(start + segment['start']), end=int(start + segment['end']))
            for segment in info['segments']
        ]
        episodes.append(episode)
        start += length
    return {'episodes': episodes}


def _check_dataset(dataset, num_episodes):
    expected_keys = set(DATASET_KEYS) | {'qpos', 'qvel'}
    expected_keys.update(key for key in ('dynamics_info', 'button_states') if key in dataset)
    get_episode_offsets(dataset)
    if set(dataset.keys()) != expected_keys:
        raise AssertionError(f'Unexpected dataset keys: {sorted(dataset.keys())}.')
    if int(dataset['terminals'].sum()) != int(num_episodes):
        raise AssertionError(f'Expected {num_episodes} terminal transitions, got {dataset["terminals"].sum()}.')
    for key, value in dataset.items():
        if value.dtype != np.uint8 and not np.all(np.isfinite(value)):
            raise AssertionError(f'Non-finite values found in dataset field {key}.')


def _find_nonfinite_episodes(dataset):
    offsets, observation_offsets = get_episode_offsets(dataset)
    invalid = []
    for key, value in dataset.items():
        if key == 'observation_interval' or value.dtype == np.uint8:
            continue
        finite = np.all(np.isfinite(value).reshape(len(value), -1), axis=1)
        boundaries = observation_offsets if key == 'observations' else offsets
        invalid.extend(np.searchsorted(boundaries[1:], np.flatnonzero(~finite), side='right'))
    return np.unique(invalid).astype(np.int64)


def _make_mjwarp_controller(family, env, oracle_seeds, max_episode_steps):
    if family == 'block':
        from ocbench.mjwarp.controllers.block import BlockMjWarpController

        return BlockMjWarpController(env, seed=oracle_seeds, max_steps=max_episode_steps)
    if family == 'chamber':
        from ocbench.mjwarp.controllers.chamber import ChamberMjWarpController

        return ChamberMjWarpController(env, seed=oracle_seeds, max_steps=max_episode_steps)
    if family == 'switch':
        from ocbench.mjwarp.controllers.switch import SwitchMjWarpController

        return SwitchMjWarpController(env, seed=oracle_seeds, max_steps=max_episode_steps)
    if family == 'hanoi':
        from ocbench.mjwarp.controllers.hanoi import HanoiMjWarpController

        return HanoiMjWarpController(env, seed=oracle_seeds, max_steps=max_episode_steps)
    if family == 'bowling':
        from ocbench.mjwarp.controllers.bowling import BowlingMjWarpController

        return BowlingMjWarpController(env, seed=oracle_seeds, max_steps=max_episode_steps)
    raise ValueError(f'Streaming MJWarp collection is not wired for {family} yet.')


def _record_mjwarp_observations(wp, family, env, controller, output, step, nworld):
    if family == 'block':
        wp.launch(
            env._kernels.record_observations,
            dim=nworld,
            inputs=[
                env.data.qpos,
                env.data.qvel,
                env.data.site_xpos,
                env.data.site_xmat,
                env.data.cfrc_ext,
                controller.done,
                output,
                env._ik_arm_qpos_ids,
                env._gpu_object_qpos_addrs,
                step,
                nworld,
                env.cpu_env._num_cubes,
                env._pinch_site_id,
                env._gripper_opening_joint_id,
                env._right_pad_body_id,
            ],
            device=env.data.qpos.device,
        )
        return
    if family == 'chamber':
        wp.launch(
            env._kernels.record_observations,
            dim=nworld,
            inputs=[
                env.data.qpos,
                env.data.qvel,
                env.data.site_xpos,
                env.data.site_xmat,
                env.data.cfrc_ext,
                env.data.mocap_pos,
                env._gpu_button_states,
                env._gpu_mirrored,
                controller.done,
                output,
                env._ik_arm_qpos_ids,
                env._gpu_object_qpos_addrs,
                env._gpu_button_qpos_addrs,
                env._gpu_button_dof_addrs,
                env._gpu_button_base_mocap_ids,
                step,
                nworld,
                env.cpu_env._num_cubes,
                env.cpu_env._num_buttons,
                env.cpu_env._num_button_states,
                env._pinch_site_id,
                env._gripper_opening_joint_id,
                env._right_pad_body_id,
                env._drawer_qpos_addr,
                env._drawer_dof_addr,
                env._drawer_base_mocap_id,
                env._window_qpos_addr,
                env._window_dof_addr,
                env._window_base_mocap_id,
            ],
            device=env.data.qpos.device,
        )
        return
    if family == 'switch':
        wp.launch(
            env._kernels.record_observations,
            dim=nworld,
            inputs=[
                env.data.qpos,
                env.data.qvel,
                env.data.site_xpos,
                env.data.site_xmat,
                env.data.cfrc_ext,
                env._gpu_button_states,
                controller.done,
                output,
                env._ik_arm_qpos_ids,
                env._gpu_button_qpos_addrs,
                env._gpu_button_dof_addrs,
                step,
                nworld,
                env.cpu_env._num_buttons,
                env.cpu_env._num_button_states,
                env.cpu_env._dropped_button_state,
                env._pinch_site_id,
                env._gripper_opening_joint_id,
                env._right_pad_body_id,
            ],
            device=env.data.qpos.device,
        )
        return
    if family == 'hanoi':
        wp.launch(
            env._kernels.record_observations,
            dim=nworld,
            inputs=[
                env.data.qpos,
                env.data.qvel,
                env.data.site_xpos,
                env.data.site_xmat,
                env.data.cfrc_ext,
                controller.done,
                output,
                env._ik_arm_qpos_ids,
                env._gpu_disk_qpos_addrs,
                env._gpu_peg_pos,
                env._gpu_peg_yaw,
                step,
                nworld,
                env.cpu_env._num_disks,
                env.cpu_env._num_pegs,
                env._pinch_site_id,
                env._gripper_opening_joint_id,
                env._right_pad_body_id,
            ],
            device=env.data.qpos.device,
        )
        return
    if family == 'bowling':
        wp.launch(
            env._kernels.record_observations,
            dim=nworld,
            inputs=[
                env.data.qpos,
                env.data.qvel,
                env.data.site_xpos,
                env.data.site_xmat,
                env.data.cfrc_ext,
                controller.done,
                output,
                env._ik_arm_qpos_ids,
                env._ball_qpos_addr,
                env._ball_qvel_addr,
                env._gpu_pin_qpos_addrs,
                step,
                nworld,
                env.cpu_env._num_pins,
                env._pinch_site_id,
                env._gripper_opening_joint_id,
                env._right_pad_body_id,
            ],
            device=env.data.qpos.device,
        )
        return
    raise ValueError(f'Unsupported MJWarp observation recorder for {family}.')


def _slice_dataset(dataset, start, end):
    offsets, _ = get_episode_offsets(dataset)
    first = np.searchsorted(offsets, start)
    last = len(offsets) - 1 if end is None else np.searchsorted(offsets, end)
    return select_episodes(dataset, range(first, last))


def _generate_mjwarp_dataset_batch(
    family,
    env_kwargs,
    max_episode_steps,
    reset_seeds,
    oracle_seeds,
    return_metadata=False,
    observation_interval=1,
):
    from ocbench.mjwarp.envs.manipulation_kernels import record_transition
    from ocbench.mjwarp.metadata import MjWarpMetadata, record_episode_outcomes

    reset_seeds = np.asarray(reset_seeds, dtype=np.uint32).copy()
    oracle_seeds = np.asarray(oracle_seeds, dtype=np.uint32).copy()
    nworld = len(reset_seeds)

    pixels = env_kwargs.get('ob_type') == 'pixels'
    collection_kwargs = dict(env_kwargs, ob_type='states', visualize_info=False) if pixels else env_kwargs
    env = make_env(family, collection_kwargs, nworld)
    env.reset(seeds=reset_seeds)
    env._ensure_ik_buffers()
    controller = _make_mjwarp_controller(family, env, oracle_seeds, max_episode_steps)
    controller.reset()

    wp = env._wp
    for resample_attempt in range(100):
        wp.synchronize()
        reset_done_worlds = np.nonzero(controller.done.numpy().astype(bool))[0].astype(np.int32)
        if len(reset_done_worlds) == 0:
            break
        new_reset_seeds = np.empty(len(reset_done_worlds), dtype=np.uint32)
        new_oracle_seeds = np.empty(len(reset_done_worlds), dtype=np.uint32)
        for idx, world_id in enumerate(reset_done_worlds):
            seed_sequence = np.random.SeedSequence(
                [
                    int(reset_seeds[world_id]),
                    int(oracle_seeds[world_id]),
                    int(resample_attempt),
                    0xD0E,
                ]
            )
            new_reset_seeds[idx], new_oracle_seeds[idx] = seed_sequence.generate_state(2, dtype=np.uint32)
        env.reset_worlds(reset_done_worlds, seeds=new_reset_seeds)
        reset_seeds[reset_done_worlds] = new_reset_seeds
        oracle_seeds[reset_done_worlds] = new_oracle_seeds
        controller = _make_mjwarp_controller(family, env, oracle_seeds, max_episode_steps)
        controller.reset()
    else:
        raise AssertionError('Failed to resample reset-time terminal MJWarp worlds.')

    device = env.data.qpos.device
    ob_info = env.compute_ob_info()
    rows = max_episode_steps * nworld
    if not pixels:
        obs_dim = env.compute_state_observation(ob_info).shape[-1]
        observations = wp.empty(((max_episode_steps + 1) * nworld, obs_dim), dtype=wp.float32, device=device)
    actions = wp.empty((rows, env.single_action_space.shape[0]), dtype=wp.float32, device=device)
    rewards = wp.empty(rows, dtype=wp.float32, device=device)
    masks = wp.empty(rows, dtype=wp.float32, device=device)
    terminals = wp.empty(rows, dtype=wp.int32, device=device)
    state_arrays = dict(qpos=env.data.qpos, qvel=env.data.qvel)
    if family in ('chamber', 'switch'):
        state_arrays['button_states'] = env._gpu_button_states
    state_buffers = {
        key: wp.empty((rows, value.shape[1]), dtype=value.dtype, device=device) for key, value in state_arrays.items()
    }

    episode_healthy = wp.ones(nworld, dtype=wp.int32, device=device)
    episode_failure = wp.zeros(nworld, dtype=wp.int32, device=device)
    failure = env._gpu_failure if family == 'hanoi' else wp.zeros(nworld, dtype=wp.int32, device=device)
    metadata = MjWarpMetadata(family, env, controller, max_episode_steps, state_arrays) if return_metadata else None
    if pixels:
        pixel_frames = [[image] for image in env.get_pixel_observation()]

    if not pixels:
        _record_mjwarp_observations(wp, family, env, controller, observations, 0, nworld)
    for step in range(max_episode_steps):
        # Match the CPU collector's state before each action.
        for key, value in state_arrays.items():
            wp.copy(state_buffers[key], value, dest_offset=step * value.size, count=value.size)
        target_pos, target_xmat, target_gripper = controller.make_targets()
        if metadata is not None:
            metadata.record_targets(step)
        action = env.ee_target_arrays_to_joint_actions_gpu(target_pos, target_xmat, target_gripper)
        env.step_joint_actions_gpu(action, controller.done)
        if not pixels:
            _record_mjwarp_observations(wp, family, env, controller, observations, step + 1, nworld)
        if family == 'bowling':
            wp.launch(
                env._kernels.record_transition_bowling,
                dim=nworld,
                inputs=[
                    action,
                    env._gpu_healthy,
                    controller.done,
                    controller.episode_length,
                    env._gpu_prev_num_knocked_pins,
                    env._gpu_num_knocked_pins,
                    actions,
                    rewards,
                    masks,
                    terminals,
                    step,
                    nworld,
                    max_episode_steps,
                ],
                device=device,
            )
        else:
            healthy = env._gpu_alive if family == 'hanoi' else env._gpu_healthy
            wp.launch(
                record_transition,
                dim=nworld,
                inputs=[
                    action,
                    env._gpu_success,
                    healthy,
                    controller.done,
                    controller.episode_length,
                    actions,
                    rewards,
                    masks,
                    terminals,
                    step,
                    nworld,
                    max_episode_steps,
                    env.single_action_space.shape[0],
                ],
                device=device,
            )
        wp.launch(
            record_episode_outcomes,
            dim=nworld,
            inputs=[controller.done, env._gpu_healthy, failure, episode_healthy, episode_failure],
            device=device,
        )
        if metadata is not None:
            # Preserve terminal states before finished worlds are parked.
            metadata.record_state()
        if pixels:
            active_worlds = np.flatnonzero(controller.done.numpy() == 0)
            if observation_interval > 1 and (step + 1) % observation_interval:
                ended = terminals[step * nworld : (step + 1) * nworld].numpy().astype(bool)
                active_worlds = active_worlds[ended[active_worlds]]
            # Capture terminal states before update_done parks completed worlds.
            if len(active_worlds):
                images = env.get_pixel_observation(active_worlds)
                for world_id, image in zip(active_worlds, images):
                    pixel_frames[world_id].append(image)
        controller.update_done()
        if pixels and np.all(controller.done.numpy()):
            break

    wp.synchronize()
    lengths = controller.episode_length.numpy().copy()
    successes = controller.episode_success.numpy().copy().astype(bool)
    speed_dt = controller.speed_dt.numpy().copy()
    dense = {
        'actions': actions,
        'rewards': rewards,
        'masks': masks,
        'terminals': terminals,
        **state_buffers,
    }
    if not pixels:
        dense['observations'] = observations
    action_dim = env.single_action_space.shape[0]
    dataset = _compact_dataset_gpu(
        wp,
        device,
        dense,
        controller.episode_length,
        lengths,
        world_start=0,
        world_count=nworld,
        nworld=nworld,
        max_episode_steps=max_episode_steps,
        action_dim=action_dim,
    )
    if 'dynamics_info' in ob_info:
        # Layouts stay fixed within an episode; chamber button states do not.
        dataset['dynamics_info'] = np.repeat(ob_info['dynamics_info'].astype(np.float32), lengths, axis=0)
        if family == 'chamber':
            dataset['dynamics_info'][:, : env.cpu_env._num_buttons] = dataset['button_states']
    healthy = episode_healthy.numpy().astype(bool)
    failures = episode_failure.numpy().astype(bool)
    episode_info = metadata.finish(dataset, lengths, healthy) if metadata is not None else [{} for _ in lengths]
    for idx, info in enumerate(episode_info):
        info.update(healthy=bool(healthy[idx]), failure=bool(failures[idx]))
    if pixels:
        dataset['observations'] = np.concatenate([np.stack(frames) for frames in pixel_frames])
    dataset['observation_interval'] = np.asarray(observation_interval, dtype=np.int32)
    env.close()
    return dataset, lengths, successes, speed_dt, episode_info


def _generate_mjwarp_dataset_chunks(
    family,
    env_kwargs,
    max_episode_steps,
    reset_seeds,
    oracle_seeds,
    max_parallel_episodes,
    return_metadata=False,
    observation_interval=1,
):
    dataset_chunks = []
    lengths = []
    successes = []
    speed_dt = []
    episode_info = []
    for start in range(0, len(reset_seeds), max_parallel_episodes):
        end = min(start + max_parallel_episodes, len(reset_seeds))
        chunk_reset_seeds = reset_seeds[start:end]
        chunk_oracle_seeds = oracle_seeds[start:end]
        dataset, cur_lengths, cur_successes, cur_speed_dt, cur_info = _generate_mjwarp_dataset_batch(
            family,
            env_kwargs,
            max_episode_steps,
            chunk_reset_seeds,
            chunk_oracle_seeds,
            return_metadata=return_metadata,
            observation_interval=observation_interval,
        )
        original_lengths = cur_lengths.copy()

        pending = np.union1d(
            _find_nonfinite_episodes(dataset),
            np.flatnonzero(~np.isfinite(cur_speed_dt) | ~np.asarray([info['healthy'] for info in cur_info])),
        )
        replacements = {}
        for attempt in range(1, 101):
            if len(pending) == 0:
                break
            print(f'Resampling {len(pending)} unhealthy or non-finite MJWarp episodes (attempt {attempt}).')
            replacement_reset_seeds = np.empty(len(pending), dtype=np.uint32)
            replacement_oracle_seeds = np.empty(len(pending), dtype=np.uint32)
            for idx, episode_idx in enumerate(pending):
                seed_sequence = np.random.SeedSequence(
                    [
                        int(chunk_reset_seeds[episode_idx]),
                        int(chunk_oracle_seeds[episode_idx]),
                        attempt,
                        0xBAD,
                    ]
                )
                replacement_reset_seeds[idx], replacement_oracle_seeds[idx] = seed_sequence.generate_state(
                    2, dtype=np.uint32
                )
            replacement_dataset, replacement_lengths, replacement_successes, replacement_speed_dt, replacement_info = (
                _generate_mjwarp_dataset_batch(
                    family,
                    env_kwargs,
                    max_episode_steps,
                    replacement_reset_seeds,
                    replacement_oracle_seeds,
                    return_metadata=return_metadata,
                    observation_interval=observation_interval,
                )
            )
            invalid_replacements = set(_find_nonfinite_episodes(replacement_dataset))
            invalid_replacements.update(
                np.flatnonzero(
                    ~np.isfinite(replacement_speed_dt) | ~np.asarray([info['healthy'] for info in replacement_info])
                )
            )
            replacement_offsets = np.concatenate([[0], np.cumsum(replacement_lengths, dtype=np.int64)])
            next_pending = []
            for replacement_idx, episode_idx in enumerate(pending):
                if replacement_idx in invalid_replacements:
                    next_pending.append(episode_idx)
                    continue
                replacements[int(episode_idx)] = (
                    replacement_dataset,
                    int(replacement_offsets[replacement_idx]),
                    int(replacement_offsets[replacement_idx + 1]),
                )
                cur_lengths[episode_idx] = replacement_lengths[replacement_idx]
                cur_successes[episode_idx] = replacement_successes[replacement_idx]
                cur_speed_dt[episode_idx] = replacement_speed_dt[replacement_idx]
                cur_info[episode_idx] = replacement_info[replacement_idx]
            pending = np.asarray(next_pending, dtype=np.int64)
        if len(pending) != 0:
            raise AssertionError('Failed to resample unhealthy or non-finite MJWarp episodes.')

        if replacements:
            original_offsets = np.concatenate([[0], np.cumsum(original_lengths, dtype=np.int64)])
            parts = []
            for episode_idx in range(len(cur_lengths)):
                if episode_idx in replacements:
                    source, first, last = replacements[episode_idx]
                else:
                    source = dataset
                    first, last = original_offsets[episode_idx : episode_idx + 2]
                parts.append(_slice_dataset(source, first, last))
            dataset = concat_datasets(parts)

        _check_dataset(dataset, len(cur_lengths))
        dataset_chunks.append(dataset)
        lengths.append(cur_lengths)
        successes.append(cur_successes)
        speed_dt.append(cur_speed_dt)
        episode_info.extend(cur_info)
    return (
        concat_datasets(dataset_chunks),
        np.concatenate(lengths, axis=0),
        np.concatenate(successes, axis=0),
        np.concatenate(speed_dt, axis=0),
        episode_info,
    )


def generate_mjwarp_datasets(
    env_name,
    collection_episodes,
    collection_val_episodes,
    root_seed,
    collection_idx,
    max_parallel_episodes=None,
    return_metadata=False,
    observation_interval=1,
):
    family, backend, env_kwargs, max_episode_steps = ocbench.parse_env_spec(env_name)
    if backend != 'mjwarp':
        raise ValueError('Streaming data collection requires an MJWarp environment; omit `-cpu-`.')
    if observation_interval > 1 and env_kwargs.get('ob_type') != 'pixels':
        raise ValueError('Sparse image collection requires a visual environment.')
    train_count = int(collection_episodes)
    val_count = int(collection_val_episodes)
    nworld = train_count + val_count
    if train_count < 1:
        raise ValueError(f'Invalid collection_episodes: {collection_episodes}.')
    if val_count < 0:
        raise ValueError(f'Invalid collection_val_episodes: {collection_val_episodes}.')
    if nworld < 1:
        raise ValueError('At least one collection episode is required.')
    if max_parallel_episodes is None:
        max_parallel_episodes = nworld
    else:
        max_parallel_episodes = int(max_parallel_episodes)
        if max_parallel_episodes < 1:
            raise ValueError(f'Invalid max_parallel_episodes: {max_parallel_episodes}.')

    train_reset_seeds = make_collection_seeds(root_seed, collection_idx, 'train_reset', train_count)
    val_reset_seeds = make_collection_seeds(root_seed, collection_idx, 'val_reset', val_count)
    train_oracle_seeds = make_collection_seeds(root_seed, collection_idx, 'train_oracle', train_count)
    val_oracle_seeds = make_collection_seeds(root_seed, collection_idx, 'val_oracle', val_count)
    _dedupe_collection_seed_groups(
        root_seed,
        collection_idx,
        [
            ('train_reset', train_reset_seeds),
            ('val_reset', val_reset_seeds),
            ('train_oracle', train_oracle_seeds),
            ('val_oracle', val_oracle_seeds),
        ],
    )
    _assert_disjoint(train_reset_seeds, val_reset_seeds, train_oracle_seeds, val_oracle_seeds)

    reset_seeds = np.concatenate([train_reset_seeds, val_reset_seeds], axis=0)
    oracle_seeds = np.concatenate([train_oracle_seeds, val_oracle_seeds], axis=0)

    start_time = time.time()
    dataset, lengths, successes, speed_dt, episode_info = _generate_mjwarp_dataset_chunks(
        family,
        env_kwargs,
        max_episode_steps,
        reset_seeds,
        oracle_seeds,
        max_parallel_episodes,
        return_metadata=return_metadata,
        observation_interval=observation_interval,
    )
    train_transition_count = int(lengths[:train_count].sum())
    train_dataset = _slice_dataset(dataset, 0, train_transition_count)
    val_dataset = _slice_dataset(dataset, train_transition_count, None) if val_count > 0 else None

    _check_dataset(train_dataset, train_count)
    if val_dataset is not None:
        _check_dataset(val_dataset, val_count)

    train_returns = _episode_returns(train_dataset['rewards'], lengths[:train_count])
    metrics = {
        'collection_idx': int(collection_idx),
        'collection_time': time.time() - start_time,
        'train/transitions': int(sum(lengths[:train_count])),
        'train/success_rate': float(successes[:train_count].mean()),
        'train/avg_episode_length': float(lengths[:train_count].mean()),
        'train/avg_return': float(train_returns.mean()),
    }
    if val_count > 0:
        val_returns = _episode_returns(val_dataset['rewards'], lengths[train_count:])
        metrics.update(
            {
                'val/transitions': int(sum(lengths[train_count:])),
                'val/success_rate': float(successes[train_count:].mean()),
                'val/avg_episode_length': float(lengths[train_count:].mean()),
                'val/avg_return': float(val_returns.mean()),
            }
        )
    if return_metadata:
        train_metadata = _make_episode_metadata(
            env_name,
            'train',
            lengths[:train_count],
            successes[:train_count],
            speed_dt[:train_count],
            episode_info[:train_count],
        )
        val_metadata = _make_episode_metadata(
            env_name,
            'val',
            lengths[train_count:],
            successes[train_count:],
            speed_dt[train_count:],
            episode_info[train_count:],
        )
        return train_dataset, val_dataset, metrics, train_metadata, val_metadata
    return train_dataset, val_dataset, metrics
