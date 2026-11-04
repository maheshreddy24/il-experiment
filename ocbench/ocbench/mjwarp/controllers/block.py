from __future__ import annotations

import numpy as np

from ocbench.mjwarp.controllers import block_kernels as kernels
from ocbench.mjwarp.controllers.controller_kernels import update_done
from ocbench.mjwarp.primitives.cube import CubeMjWarpPrimitive
from ocbench.mjwarp.primitives.primitive import MjWarpPrimitive


class BlockMjWarpController:
    def __init__(self, env, seed, max_steps: int):
        if env._data is None:
            raise ValueError('Call `env.reset` before creating BlockMjWarpController.')

        self.env = env
        self.max_steps = int(max_steps)
        self.wp = env._wp
        self.device = env.data.qpos.device
        self.nworld = env.nworld
        seeds = np.asarray(seed)
        if seeds.ndim == 0:
            seeds = np.random.SeedSequence(int(seeds)).generate_state(self.nworld, dtype=np.uint32)
        seeds = np.asarray(seeds, dtype=np.uint32)
        if seeds.shape != (self.nworld,):
            raise ValueError(f'Expected {self.nworld} controller seeds, got shape {seeds.shape}.')
        self.seeds = self.wp.array(seeds.astype(np.int32), dtype=self.wp.int32, device=self.device)

        env._ensure_gpu_step_buffers()
        self._allocate_state()
        self._allocate_constants()

    def _allocate_state(self):
        wp = self.wp
        device = self.device
        nworld = self.nworld
        max_cubes = kernels.MAX_CUBES

        self.done = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.episode_length = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.episode_success = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.controller_done = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.agent_done = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.active_steps = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.is_mistake = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.rng_counter = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.speed_dt = wp.zeros(nworld, dtype=wp.float32, device=device)
        self.goal_order = wp.zeros((nworld, max_cubes), dtype=wp.int32, device=device)
        self.stack_goal_xyz = wp.zeros((nworld, max_cubes), dtype=wp.vec3f, device=device)
        self.target_block = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.target_pos = wp.zeros(nworld, dtype=wp.vec3f, device=device)
        self.target_yaw = wp.zeros(nworld, dtype=wp.float32, device=device)

        self.primitive = MjWarpPrimitive(self.env)
        self.primitive.bind_to(self)
        self.cube_primitive = CubeMjWarpPrimitive(self.env)
        self.cube_primitive.bind_to(self)

    def _allocate_constants(self):
        env = self.env
        cpu_env = env.cpu_env

        target_lo = np.array([cpu_env._target_sampling_bounds[0, 0], cpu_env._target_sampling_bounds[0, 1], cpu_env._cube_size])
        target_hi = np.array([cpu_env._target_sampling_bounds[1, 0], cpu_env._target_sampling_bounds[1, 1], cpu_env._cube_size])
        self.target_lo = self._vec3(target_lo)
        self.target_hi = self._vec3(target_hi)
        self.task_mode, self.stack_anywhere = self._task_config()
        self.tilt_randomization = int(not cpu_env._lite)
        self.segment_dt_scale = 0.5 if cpu_env._lite else 1.0

    def _vec3(self, value):
        return self.wp.vec3(float(value[0]), float(value[1]), float(value[2]))

    def _task_config(self):
        task_name = self.env.cpu_env.cur_task_info['task_name']
        if task_name == 'move':
            return 0, 0
        if task_name in {'stack', 'grid', 'stack_anywhere'}:
            return 2, int(task_name == 'stack_anywhere')
        return 1, 0

    def reset(self):
        env = self.env
        wp = self.wp
        wp.launch(
            kernels.reset_oracle,
            dim=self.nworld,
            inputs=[
                env.data.qpos,
                env.data.site_xpos,
                env.data.site_xmat,
                env.data.mocap_pos,
                env.data.mocap_quat,
                env._gpu_object_qpos_addrs,
                env._gpu_cube_target_mocap_ids,
                self.rng_counter,
                self.is_mistake,
                self.done,
                self.episode_length,
                self.episode_success,
                self.controller_done,
                self.agent_done,
                self.active_steps,
                self.speed_dt,
                self.goal_order,
                self.stack_goal_xyz,
                self.active_block,
                self.target_block,
                self.target_pos,
                self.target_yaw,
                self.key_count,
                self.key_time,
                self.key_xyz,
                self.key_quat,
                self.key_grasp,
                self.key_stop,
                self.key_tangent,
                self.gate_time,
                self.gate_xyz,
                self.gate_passed,
                self.phase_start,
                self.phase_end,
                self.phase_c1,
                self.phase_c2,
                self.pick_check_time,
                self.pick_source_z,
                self.plan_time,
                self.last_time,
                self.pick_checked,
                self.num_pick_retries,
                self.seeds,
                env.cpu_env._num_cubes,
                self.task_mode,
                self.stack_anywhere,
                float(env.cpu_env._cube_size),
                float(env.cpu_env._min_object_init_dist),
                self.target_lo,
                self.target_hi,
                self.arm_lo,
                self.arm_hi,
                env._pinch_site_id,
                self.tilt_randomization,
                self.segment_dt_scale,
            ],
            device=self.device,
        )

    def make_targets(self):
        env = self.env
        wp = self.wp
        wp.launch(
            kernels.make_targets,
            dim=self.nworld,
            inputs=[
                env.data.qpos,
                env.data.site_xpos,
                env.data.site_xmat,
                env.data.mocap_pos,
                env.data.mocap_quat,
                env._gpu_object_qpos_addrs,
                env._gpu_cube_target_mocap_ids,
                self.rng_counter,
                self.is_mistake,
                self.done,
                self.controller_done,
                self.agent_done,
                self.active_steps,
                self.speed_dt,
                self.goal_order,
                self.stack_goal_xyz,
                self.active_block,
                self.target_block,
                self.target_pos,
                self.target_yaw,
                self.key_count,
                self.key_time,
                self.key_xyz,
                self.key_quat,
                self.key_grasp,
                self.key_stop,
                self.key_tangent,
                self.gate_time,
                self.gate_xyz,
                self.gate_passed,
                self.phase_start,
                self.phase_end,
                self.phase_c1,
                self.phase_c2,
                self.pick_check_time,
                self.pick_source_z,
                self.plan_time,
                self.last_time,
                self.pick_checked,
                self.num_pick_retries,
                self.target_attach_pos,
                self.target_attach_xmat,
                self.target_gripper,
                self.down_xmat,
                self.down_xmat_inv,
                self.t_pa_rot,
                self.t_pa_translation,
                self.seeds,
                env.cpu_env._num_cubes,
                self.task_mode,
                self.stack_anywhere,
                float(env.cpu_env._cube_size),
                float(env.cpu_env._min_object_init_dist),
                self.target_lo,
                self.target_hi,
                self.arm_lo,
                self.arm_hi,
                self.workspace_lo,
                self.workspace_hi,
                self.ee_low,
                self.ee_high,
                env._pinch_site_id,
                env._gripper_opening_joint_id,
                float(env.cpu_env._control_timestep),
                1250,
                self.tilt_randomization,
            ],
            device=self.device,
        )
        return self.target_attach_pos, self.target_attach_xmat, self.target_gripper

    def update_done(self):
        self.wp.launch(
            update_done,
            dim=self.nworld,
            inputs=[
                self.env._gpu_success,
                self.env._gpu_healthy,
                self.done,
                self.episode_length,
                self.episode_success,
                self.max_steps,
            ],
            device=self.device,
        )
        self.env.park_done_worlds_gpu(self.done)

    def stats(self):
        self.wp.synchronize()
        lengths = self.episode_length.numpy().copy()
        successes = self.episode_success.numpy().copy().astype(bool)
        done = self.done.numpy().copy().astype(bool)
        return dict(
            completed=int(done.sum()),
            transitions=int(lengths.sum()),
            success_rate=float(successes.mean()),
            avg_episode_length=float(lengths.mean()),
            min_episode_length=int(lengths.min()),
            max_episode_length=int(lengths.max()),
        )
