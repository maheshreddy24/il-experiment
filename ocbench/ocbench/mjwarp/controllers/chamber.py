from __future__ import annotations

import numpy as np

from ocbench.mjwarp.controllers import chamber_kernels as kernels
from ocbench.mjwarp.controllers.controller_kernels import update_done
from ocbench.mjwarp.primitives.button import ButtonMjWarpPrimitive
from ocbench.mjwarp.primitives.cube import CubeMjWarpPrimitive
from ocbench.mjwarp.primitives.drawer import DrawerMjWarpPrimitive
from ocbench.mjwarp.primitives.primitive import MjWarpPrimitive
from ocbench.mjwarp.primitives.window import WindowMjWarpPrimitive


class ChamberMjWarpController:
    def __init__(self, env, seed, max_steps: int, p_mistake: float = 0.2, cube_settle_steps: int = 25):
        if env._data is None:
            raise ValueError('Call `env.reset` before creating ChamberMjWarpController.')

        self.env = env
        self.max_steps = int(max_steps)
        self.wp = env._wp
        self.device = env.data.qpos.device
        self.nworld = env.nworld
        self.p_mistake = float(p_mistake)
        self.cube_settle_steps = int(cube_settle_steps)
        seeds = np.asarray(seed)
        if seeds.ndim == 0:
            seeds = np.random.SeedSequence(int(seeds)).generate_state(self.nworld, dtype=np.uint32)
        seeds = np.asarray(seeds, dtype=np.uint32)
        if seeds.shape != (self.nworld,):
            raise ValueError(f'Expected {self.nworld} controller seeds, got shape {seeds.shape}.')
        self.seeds = self.wp.array(seeds.astype(np.int32), dtype=self.wp.int32, device=self.device)

        env._ensure_gpu_step_buffers()
        self._allocate_state()
        self.tilt_randomization = int(not env.cpu_env._lite)
        self.segment_dt_scale = 0.5 if env.cpu_env._lite else 1.0
        task_name = env.cpu_env.cur_task_info['task_name']
        if task_name in {'put_in', 'put_all_in'}:
            self.task_mode = 0
        elif task_name == 'open_all':
            self.task_mode = 2
        else:
            self.task_mode = 1

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
        self.active_settle_steps = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.settle_after_cube = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.is_mistake = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.rng_counter = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.speed_dt = wp.zeros(nworld, dtype=wp.float32, device=device)

        self.active_task = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.target_button = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.target_slide_pos = wp.zeros(nworld, dtype=wp.float32, device=device)
        self.target_block = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.target_pos = wp.zeros(nworld, dtype=wp.vec3f, device=device)
        self.target_yaw = wp.zeros(nworld, dtype=wp.float32, device=device)
        self.goal_order = wp.zeros((nworld, max_cubes), dtype=wp.int32, device=device)
        self.stack_goal_xyz = wp.zeros((nworld, max_cubes), dtype=wp.vec3f, device=device)

        self.primitive = MjWarpPrimitive(self.env)
        self.primitive.bind_to(self)
        self.cube_primitive = CubeMjWarpPrimitive(self.env)
        self.cube_primitive.bind_to(self)
        self.button_primitive = ButtonMjWarpPrimitive(self.env)
        self.button_primitive.bind_to(self)
        self.drawer_primitive = DrawerMjWarpPrimitive(self.env)
        self.drawer_primitive.bind_to(self)
        self.window_primitive = WindowMjWarpPrimitive(self.env)
        self.window_primitive.bind_to(self)

    def reset(self):
        env = self.env
        wp = self.wp
        wp.launch(
            kernels.reset_controller,
            dim=self.nworld,
            inputs=[
                env.data.qpos,
                env.data.site_xpos,
                env.data.site_xmat,
                env.data.mocap_pos,
                env._gpu_object_qpos_addrs,
                env._gpu_button_states,
                env._gpu_target_button_states,
                env._gpu_mirrored,
                self.rng_counter,
                self.is_mistake,
                self.done,
                self.episode_length,
                self.episode_success,
                self.controller_done,
                self.agent_done,
                self.active_steps,
                self.active_settle_steps,
                self.settle_after_cube,
                self.speed_dt,
                self.goal_order,
                self.stack_goal_xyz,
                self.active_task,
                self.target_button,
                self.target_slide_pos,
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
                float(env.cpu_env._cube_size),
                env._drawer_qpos_addr,
                env._window_qpos_addr,
                env._drawer_site_id,
                env._window_site_id,
                self.button0_site_id,
                self.button1_site_id,
                env._drawer_base_mocap_id,
                self.drawer_open_threshold,
                self.drawer_slide_min,
                self.drawer_slide_max,
                self.window_slide_min,
                self.window_slide_max,
                self.p_mistake,
                self.cube_settle_steps,
                self.nominal_drawer_pos,
                self.wp.vec3(
                    float(env.cpu_env._target_sampling_bounds[0, 0]),
                    float(env.cpu_env._target_sampling_bounds[0, 1]),
                    float(env.cpu_env._cube_size),
                ),
                self.wp.vec3(
                    float(env.cpu_env._target_sampling_bounds[1, 0]),
                    float(env.cpu_env._target_sampling_bounds[1, 1]),
                    float(env.cpu_env._cube_size),
                ),
                env._pinch_site_id,
                self.arm_lo,
                self.arm_hi,
                self.segment_dt_scale,
            ],
            device=self.device,
        )
        self._start_plans()

    def _start_plans(self):
        env = self.env
        wp = self.wp
        wp.launch(
            kernels.start_chamber_plans,
            dim=self.nworld,
            inputs=[
                env.data.qpos,
                env.data.site_xpos,
                env.data.site_xmat,
                env.data.mocap_pos,
                env._gpu_object_qpos_addrs,
                env._gpu_button_states,
                env._gpu_target_button_states,
                env._gpu_mirrored,
                self.rng_counter,
                self.is_mistake,
                self.done,
                self.controller_done,
                self.agent_done,
                self.active_steps,
                self.active_settle_steps,
                self.settle_after_cube,
                self.active_task,
                self.goal_order,
                self.stack_goal_xyz,
                self.target_button,
                self.target_slide_pos,
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
                float(env.cpu_env._cube_size),
                env._drawer_qpos_addr,
                env._window_qpos_addr,
                env._drawer_site_id,
                env._window_site_id,
                self.button0_site_id,
                self.button1_site_id,
                env._drawer_base_mocap_id,
                self.drawer_open_threshold,
                self.drawer_slide_min,
                self.drawer_slide_max,
                self.window_slide_min,
                self.window_slide_max,
                self.p_mistake,
                self.cube_settle_steps,
                self.nominal_drawer_pos,
                self.wp.vec3(
                    float(env.cpu_env._target_sampling_bounds[0, 0]),
                    float(env.cpu_env._target_sampling_bounds[0, 1]),
                    float(env.cpu_env._cube_size),
                ),
                self.wp.vec3(
                    float(env.cpu_env._target_sampling_bounds[1, 0]),
                    float(env.cpu_env._target_sampling_bounds[1, 1]),
                    float(env.cpu_env._cube_size),
                ),
                env._pinch_site_id,
                self.speed_dt,
                self.arm_lo,
                self.arm_hi,
                1000,
                self.tilt_randomization,
            ],
            device=self.device,
        )

    def make_targets(self):
        env = self.env
        wp = self.wp
        self._start_plans()
        wp.launch(
            kernels.make_targets,
            dim=self.nworld,
            inputs=[
                env.data.qpos,
                env.data.site_xpos,
                env.data.site_xmat,
                env.data.mocap_pos,
                env._gpu_object_qpos_addrs,
                env._gpu_button_states,
                env._gpu_target_button_states,
                env._gpu_mirrored,
                self.rng_counter,
                self.is_mistake,
                self.done,
                self.controller_done,
                self.agent_done,
                self.active_steps,
                self.active_settle_steps,
                self.settle_after_cube,
                self.speed_dt,
                self.goal_order,
                self.stack_goal_xyz,
                self.active_task,
                self.target_button,
                self.target_slide_pos,
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
                float(env.cpu_env._cube_size),
                env._drawer_qpos_addr,
                env._window_qpos_addr,
                env._drawer_site_id,
                env._window_site_id,
                self.button0_site_id,
                self.button1_site_id,
                env._drawer_base_mocap_id,
                self.drawer_open_threshold,
                self.drawer_slide_min,
                self.drawer_slide_max,
                self.window_slide_min,
                self.window_slide_max,
                self.p_mistake,
                self.cube_settle_steps,
                self.nominal_drawer_pos,
                self.wp.vec3(
                    float(env.cpu_env._target_sampling_bounds[0, 0]),
                    float(env.cpu_env._target_sampling_bounds[0, 1]),
                    float(env.cpu_env._cube_size),
                ),
                self.wp.vec3(
                    float(env.cpu_env._target_sampling_bounds[1, 0]),
                    float(env.cpu_env._target_sampling_bounds[1, 1]),
                    float(env.cpu_env._cube_size),
                ),
                self.arm_lo,
                self.arm_hi,
                self.workspace_lo,
                self.workspace_hi,
                self.ee_low,
                self.ee_high,
                env._pinch_site_id,
                env._gripper_opening_joint_id,
                float(env.cpu_env._control_timestep),
                1000,
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
