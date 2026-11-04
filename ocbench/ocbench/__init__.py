"""OCBench: A Controllable Robotic Manipulation Benchmark"""

import gymnasium
from gymnasium.envs.registration import register, registry
from ocbench.dataset_utils import download_datasets, get_episode_offsets, load_dataset

__all__ = (
    'download_datasets',
    'get_episode_offsets',
    'load_dataset',
    'make',
    'make_env_and_datasets',
    'parse_env_spec',
)

_LITE_KWARGS = dict(
    action_delta_scale=5.0,
    lite=True,
    control_timestep=0.1,
)


def parse_env_spec(env_name):
    """Return the family, backend, constructor kwargs, and episode limit."""
    visual = env_name.startswith('visual-')
    family, task = env_name.removeprefix('visual-').split('-', 1)
    backend = 'cpu' if task.startswith('cpu-') else 'mjwarp'
    cpu_name = env_name if backend == 'cpu' else f'{family}-cpu-{task}'
    if visual and backend == 'mjwarp':
        cpu_name = f'visual-{cpu_name}'
    spec = gymnasium.spec(cpu_name)
    return family, backend, dict(spec.kwargs), int(spec.max_episode_steps)


def make(env_name, **kwargs):
    """Make a batched MJWarp environment, or a single CPU environment with a `-cpu-` ID.

    MJWarp uses 1024 worlds by default; pass `nworld` to choose the batch size.
    For MJWarp environments, stop each episode after the step limit returned by `parse_env_spec`.
    CPU environments truncate episodes automatically at their time limit.
    """
    family, backend, env_kwargs, _ = parse_env_spec(env_name)
    if backend == 'cpu':
        return gymnasium.make(env_name, **kwargs)
    from ocbench.mjwarp import make_env

    nworld = kwargs.pop('nworld', 1024)
    env_kwargs.update(kwargs)
    return make_env(family, env_kwargs, nworld)


def make_env_and_datasets(env_name, dataset_root=None, num_shards=None, **env_kwargs):
    """Make an OCBench environment and load its released datasets.

    Args:
        env_name: Environment name (e.g., 'block-lite-single-task1-v0').
        dataset_root: Root directory for downloaded datasets. Defaults to '$XDG_CACHE_HOME/ocbench/datasets' if
            XDG_CACHE_HOME is set, or '~/.cache/ocbench/datasets' otherwise.
        num_shards: Number of training shards to load, in filename order. If None, load all shards.
        **env_kwargs: Keyword arguments to pass to make (e.g., nworld=1024 for MJWarp).

    Returns:
        A tuple of the environment, training dataset, and validation dataset. Each dataset is a dictionary of
        NumPy arrays; see load_dataset for the dataset format.
    """
    train_paths, val_paths = download_datasets(env_name, dataset_root=dataset_root, num_shards=num_shards)
    train_dataset = load_dataset(train_paths)
    val_dataset = load_dataset(val_paths)
    env = make(env_name, **env_kwargs)
    return env, train_dataset, val_dataset


# Block environments.
register(
    id='block-cpu-single-task1-v0',
    entry_point='ocbench.envs.block_env:BlockEnv',
    max_episode_steps=1250,
    kwargs=dict(env_type='single', task_id=1),
)
register(
    id='block-cpu-double-task1-v0',
    entry_point='ocbench.envs.block_env:BlockEnv',
    max_episode_steps=2500,
    kwargs=dict(env_type='double', task_id=1),
)
register(
    id='block-cpu-double-task2-v0',
    entry_point='ocbench.envs.block_env:BlockEnv',
    max_episode_steps=2500,
    kwargs=dict(env_type='double', task_id=2),
)
register(
    id='block-cpu-triple-task1-v0',
    entry_point='ocbench.envs.block_env:BlockEnv',
    max_episode_steps=3750,
    kwargs=dict(env_type='triple', task_id=1),
)
register(
    id='block-cpu-triple-task2-v0',
    entry_point='ocbench.envs.block_env:BlockEnv',
    max_episode_steps=3750,
    kwargs=dict(env_type='triple', task_id=2),
)
register(
    id='block-cpu-quadruple-task1-v0',
    entry_point='ocbench.envs.block_env:BlockEnv',
    max_episode_steps=5000,
    kwargs=dict(env_type='quadruple', task_id=1),
)
register(
    id='block-cpu-quadruple-task2-v0',
    entry_point='ocbench.envs.block_env:BlockEnv',
    max_episode_steps=5000,
    kwargs=dict(env_type='quadruple', task_id=2),
)
register(
    id='block-cpu-quadruple-task3-v0',
    entry_point='ocbench.envs.block_env:BlockEnv',
    max_episode_steps=5000,
    kwargs=dict(env_type='quadruple', task_id=3),
)

# Chamber environments.
register(
    id='chamber-cpu-easy-task1-v0',
    entry_point='ocbench.envs.chamber_env:ChamberEnv',
    max_episode_steps=4000,
    kwargs=dict(env_type='easy', task_id=1),
)
register(
    id='chamber-cpu-easy-task2-v0',
    entry_point='ocbench.envs.chamber_env:ChamberEnv',
    max_episode_steps=4000,
    kwargs=dict(env_type='easy', task_id=2),
)
register(
    id='chamber-cpu-easy-task3-v0',
    entry_point='ocbench.envs.chamber_env:ChamberEnv',
    max_episode_steps=4000,
    kwargs=dict(env_type='easy', task_id=3),
)
register(
    id='chamber-cpu-medium-task1-v0',
    entry_point='ocbench.envs.chamber_env:ChamberEnv',
    max_episode_steps=5000,
    kwargs=dict(env_type='medium', task_id=1),
)
register(
    id='chamber-cpu-medium-task2-v0',
    entry_point='ocbench.envs.chamber_env:ChamberEnv',
    max_episode_steps=5000,
    kwargs=dict(env_type='medium', task_id=2),
)
register(
    id='chamber-cpu-hard-task1-v0',
    entry_point='ocbench.envs.chamber_env:ChamberEnv',
    max_episode_steps=5500,
    kwargs=dict(env_type='hard', task_id=1),
)
register(
    id='chamber-cpu-hard-task2-v0',
    entry_point='ocbench.envs.chamber_env:ChamberEnv',
    max_episode_steps=5500,
    kwargs=dict(env_type='hard', task_id=2),
)

# Switch environments.
register(
    id='switch-cpu-3x3-task1-v0',
    entry_point='ocbench.envs.switch_env:SwitchEnv',
    max_episode_steps=1500,
    kwargs=dict(env_type='3x3', task_id=1),
)
register(
    id='switch-cpu-3x3-task2-v0',
    entry_point='ocbench.envs.switch_env:SwitchEnv',
    max_episode_steps=1500,
    kwargs=dict(env_type='3x3', task_id=2),
)
register(
    id='switch-cpu-4x4-task1-v0',
    entry_point='ocbench.envs.switch_env:SwitchEnv',
    max_episode_steps=1500,
    kwargs=dict(env_type='4x4', task_id=1),
)
register(
    id='switch-cpu-4x4-task2-v0',
    entry_point='ocbench.envs.switch_env:SwitchEnv',
    max_episode_steps=1500,
    kwargs=dict(env_type='4x4', task_id=2),
)
register(
    id='switch-cpu-5x5-task1-v0',
    entry_point='ocbench.envs.switch_env:SwitchEnv',
    max_episode_steps=4000,
    kwargs=dict(env_type='5x5', task_id=1),
)
register(
    id='switch-cpu-5x5-task2-v0',
    entry_point='ocbench.envs.switch_env:SwitchEnv',
    max_episode_steps=4000,
    kwargs=dict(env_type='5x5', task_id=2),
)

# Hanoi environments.
register(
    id='hanoi-cpu-single-task1-v0',
    entry_point='ocbench.envs.hanoi_env:HanoiEnv',
    max_episode_steps=1000,
    kwargs=dict(env_type='single', task_id=1),
)
register(
    id='hanoi-cpu-single-task2-v0',
    entry_point='ocbench.envs.hanoi_env:HanoiEnv',
    max_episode_steps=1000,
    kwargs=dict(env_type='single', task_id=2),
)
register(
    id='hanoi-cpu-double-task1-v0',
    entry_point='ocbench.envs.hanoi_env:HanoiEnv',
    max_episode_steps=3000,
    kwargs=dict(env_type='double', task_id=1),
)
register(
    id='hanoi-cpu-double-task2-v0',
    entry_point='ocbench.envs.hanoi_env:HanoiEnv',
    max_episode_steps=3000,
    kwargs=dict(env_type='double', task_id=2),
)
register(
    id='hanoi-cpu-triple-task1-v0',
    entry_point='ocbench.envs.hanoi_env:HanoiEnv',
    max_episode_steps=7000,
    kwargs=dict(env_type='triple', task_id=1),
)
register(
    id='hanoi-cpu-triple-task2-v0',
    entry_point='ocbench.envs.hanoi_env:HanoiEnv',
    max_episode_steps=7000,
    kwargs=dict(env_type='triple', task_id=2),
)

# Bowling environment.
register(
    id='bowling-cpu-task1-v0',
    entry_point='ocbench.envs.bowling_env:BowlingEnv',
    max_episode_steps=500,
    kwargs=dict(task_id=1),
)

# Lite block environments.
register(
    id='block-cpu-lite-single-task1-v0',
    entry_point='ocbench.envs.block_env:BlockEnv',
    max_episode_steps=200,
    kwargs=dict(env_type='single', task_id=1, **_LITE_KWARGS),
)
register(
    id='block-cpu-lite-double-task1-v0',
    entry_point='ocbench.envs.block_env:BlockEnv',
    max_episode_steps=400,
    kwargs=dict(env_type='double', task_id=1, **_LITE_KWARGS),
)
register(
    id='block-cpu-lite-double-task2-v0',
    entry_point='ocbench.envs.block_env:BlockEnv',
    max_episode_steps=400,
    kwargs=dict(env_type='double', task_id=2, **_LITE_KWARGS),
)
register(
    id='block-cpu-lite-triple-task1-v0',
    entry_point='ocbench.envs.block_env:BlockEnv',
    max_episode_steps=600,
    kwargs=dict(env_type='triple', task_id=1, **_LITE_KWARGS),
)
register(
    id='block-cpu-lite-triple-task2-v0',
    entry_point='ocbench.envs.block_env:BlockEnv',
    max_episode_steps=600,
    kwargs=dict(env_type='triple', task_id=2, **_LITE_KWARGS),
)
register(
    id='block-cpu-lite-quadruple-task1-v0',
    entry_point='ocbench.envs.block_env:BlockEnv',
    max_episode_steps=800,
    kwargs=dict(env_type='quadruple', task_id=1, **_LITE_KWARGS),
)
register(
    id='block-cpu-lite-quadruple-task2-v0',
    entry_point='ocbench.envs.block_env:BlockEnv',
    max_episode_steps=800,
    kwargs=dict(env_type='quadruple', task_id=2, **_LITE_KWARGS),
)
register(
    id='block-cpu-lite-quadruple-task3-v0',
    entry_point='ocbench.envs.block_env:BlockEnv',
    max_episode_steps=800,
    kwargs=dict(env_type='quadruple', task_id=3, **_LITE_KWARGS),
)

# Lite chamber environments.
register(
    id='chamber-cpu-lite-easy-task1-v0',
    entry_point='ocbench.envs.chamber_env:ChamberEnv',
    max_episode_steps=500,
    kwargs=dict(env_type='easy', task_id=1, **_LITE_KWARGS),
)
register(
    id='chamber-cpu-lite-easy-task2-v0',
    entry_point='ocbench.envs.chamber_env:ChamberEnv',
    max_episode_steps=500,
    kwargs=dict(env_type='easy', task_id=2, **_LITE_KWARGS),
)
register(
    id='chamber-cpu-lite-easy-task3-v0',
    entry_point='ocbench.envs.chamber_env:ChamberEnv',
    max_episode_steps=500,
    kwargs=dict(env_type='easy', task_id=3, **_LITE_KWARGS),
)
register(
    id='chamber-cpu-lite-medium-task1-v0',
    entry_point='ocbench.envs.chamber_env:ChamberEnv',
    max_episode_steps=600,
    kwargs=dict(env_type='medium', task_id=1, **_LITE_KWARGS),
)
register(
    id='chamber-cpu-lite-medium-task2-v0',
    entry_point='ocbench.envs.chamber_env:ChamberEnv',
    max_episode_steps=600,
    kwargs=dict(env_type='medium', task_id=2, **_LITE_KWARGS),
)
register(
    id='chamber-cpu-lite-hard-task1-v0',
    entry_point='ocbench.envs.chamber_env:ChamberEnv',
    max_episode_steps=700,
    kwargs=dict(env_type='hard', task_id=1, **_LITE_KWARGS),
)
register(
    id='chamber-cpu-lite-hard-task2-v0',
    entry_point='ocbench.envs.chamber_env:ChamberEnv',
    max_episode_steps=700,
    kwargs=dict(env_type='hard', task_id=2, **_LITE_KWARGS),
)

# Lite switch environments.
register(
    id='switch-cpu-lite-3x3-task1-v0',
    entry_point='ocbench.envs.switch_env:SwitchEnv',
    max_episode_steps=300,
    kwargs=dict(env_type='3x3', task_id=1, **_LITE_KWARGS),
)
register(
    id='switch-cpu-lite-3x3-task2-v0',
    entry_point='ocbench.envs.switch_env:SwitchEnv',
    max_episode_steps=300,
    kwargs=dict(env_type='3x3', task_id=2, **_LITE_KWARGS),
)
register(
    id='switch-cpu-lite-4x4-task1-v0',
    entry_point='ocbench.envs.switch_env:SwitchEnv',
    max_episode_steps=300,
    kwargs=dict(env_type='4x4', task_id=1, **_LITE_KWARGS),
)
register(
    id='switch-cpu-lite-4x4-task2-v0',
    entry_point='ocbench.envs.switch_env:SwitchEnv',
    max_episode_steps=300,
    kwargs=dict(env_type='4x4', task_id=2, **_LITE_KWARGS),
)
register(
    id='switch-cpu-lite-5x5-task1-v0',
    entry_point='ocbench.envs.switch_env:SwitchEnv',
    max_episode_steps=800,
    kwargs=dict(env_type='5x5', task_id=1, **_LITE_KWARGS),
)
register(
    id='switch-cpu-lite-5x5-task2-v0',
    entry_point='ocbench.envs.switch_env:SwitchEnv',
    max_episode_steps=800,
    kwargs=dict(env_type='5x5', task_id=2, **_LITE_KWARGS),
)

# Lite Hanoi environments.
register(
    id='hanoi-cpu-lite-single-task1-v0',
    entry_point='ocbench.envs.hanoi_env:HanoiEnv',
    max_episode_steps=300,
    kwargs=dict(env_type='single', task_id=1, **_LITE_KWARGS),
)
register(
    id='hanoi-cpu-lite-single-task2-v0',
    entry_point='ocbench.envs.hanoi_env:HanoiEnv',
    max_episode_steps=300,
    kwargs=dict(env_type='single', task_id=2, **_LITE_KWARGS),
)
register(
    id='hanoi-cpu-lite-double-task1-v0',
    entry_point='ocbench.envs.hanoi_env:HanoiEnv',
    max_episode_steps=900,
    kwargs=dict(env_type='double', task_id=1, **_LITE_KWARGS),
)
register(
    id='hanoi-cpu-lite-double-task2-v0',
    entry_point='ocbench.envs.hanoi_env:HanoiEnv',
    max_episode_steps=900,
    kwargs=dict(env_type='double', task_id=2, **_LITE_KWARGS),
)
register(
    id='hanoi-cpu-lite-triple-task1-v0',
    entry_point='ocbench.envs.hanoi_env:HanoiEnv',
    max_episode_steps=2100,
    kwargs=dict(env_type='triple', task_id=1, **_LITE_KWARGS),
)
register(
    id='hanoi-cpu-lite-triple-task2-v0',
    entry_point='ocbench.envs.hanoi_env:HanoiEnv',
    max_episode_steps=2100,
    kwargs=dict(env_type='triple', task_id=2, **_LITE_KWARGS),
)

# Visual environments.
for _spec in list(registry.values()):
    if (
        isinstance(_spec.entry_point, str)
        and _spec.entry_point.startswith('ocbench.envs.')
        and not _spec.id.startswith('visual-')
    ):
        register(
            id=f'visual-{_spec.id}',
            entry_point=_spec.entry_point,
            max_episode_steps=_spec.max_episode_steps,
            kwargs=dict(
                **_spec.kwargs,
                ob_type='pixels',
                width=224,
                height=224,
                visualize_info=False,
                pixel_cameras=('front', 'side', 'ur5e/wrist'),
            ),
        )
