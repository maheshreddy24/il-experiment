"""MuJoCo Warp-backed OCBench utilities."""

IK_ITERS = 4
MAX_PLAN_KEYS = 16


def make_env(family, env_kwargs, nworld):
    if family == 'block':
        from ocbench.mjwarp.envs.block import BlockMjWarpEnv

        return BlockMjWarpEnv(
            nworld=nworld,
            terminate_at_success=True,
            physics_timestep=0.002,
            **env_kwargs,
        )
    if family == 'chamber':
        from ocbench.mjwarp.envs.chamber import ChamberMjWarpEnv

        return ChamberMjWarpEnv(
            nworld=nworld,
            terminate_at_success=True,
            physics_timestep=0.002,
            **env_kwargs,
        )
    if family == 'switch':
        from ocbench.mjwarp.envs.switch import SwitchMjWarpEnv

        return SwitchMjWarpEnv(
            nworld=nworld,
            terminate_at_success=True,
            physics_timestep=0.002,
            **env_kwargs,
        )
    if family == 'hanoi':
        from ocbench.mjwarp.envs.hanoi import HanoiMjWarpEnv

        return HanoiMjWarpEnv(
            nworld=nworld,
            terminate_at_success=True,
            physics_timestep=0.002,
            **env_kwargs,
        )
    if family == 'bowling':
        from ocbench.mjwarp.envs.bowling import BowlingMjWarpEnv

        return BowlingMjWarpEnv(
            nworld=nworld,
            terminate_at_success=True,
            physics_timestep=0.002,
            **env_kwargs,
        )
    raise ValueError(f'Unsupported MJWarp env family: {family}.')
