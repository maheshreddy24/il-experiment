from __future__ import annotations

import numpy as np

from ocbench.mjwarp.controllers import hanoi_kernels as kernels
from ocbench.mjwarp.controllers.controller_kernels import update_done
from ocbench.mjwarp.primitives.hanoi import HanoiMjWarpPrimitive
from ocbench.mjwarp.primitives.primitive import MjWarpPrimitive


class HanoiMjWarpController:
    def __init__(
        self,
        env,
        seed,
        max_steps: int,
        p_mistake: float = 0.002,
        max_subgoal_steps: int = 1500,
    ):
        if env._data is None:
            raise ValueError('Call `env.reset` before creating HanoiMjWarpController.')

        self.env = env
        self.max_steps = int(max_steps)
        self.p_mistake = float(p_mistake)
        self.max_subgoal_steps = int(max_subgoal_steps)
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
        self.tilt_randomization = int(not env.cpu_env._lite)
        self.segment_dt_scale = 0.5 if env.cpu_env._lite else 1.0

    def _allocate_state(self):
        wp = self.wp
        device = self.device
        nworld = self.nworld

        self.done = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.episode_length = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.episode_success = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.controller_done = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.agent_done = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.active_steps = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.is_mistake = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.rng_counter = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.speed_dt = wp.zeros(nworld, dtype=wp.float32, device=device)

        self.target_peg = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.target_disk = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.target_source = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.target_pos = wp.zeros(nworld, dtype=wp.vec3f, device=device)
        self.target_yaw = wp.zeros(nworld, dtype=wp.float32, device=device)

        self.primitive = MjWarpPrimitive(self.env)
        self.primitive.bind_to(self)
        self.hanoi_primitive = HanoiMjWarpPrimitive(self.env)
        self.hanoi_primitive.bind_to(self)

    def _goal_pegs(self):
        goal_pegs = np.asarray(self.env.cpu_env.cur_task_info['goal_pegs'], dtype=np.int32)
        return int(goal_pegs[0]), int(goal_pegs[1])

    def reset(self):
        env = self.env
        wp = self.wp
        goal_peg0, goal_peg1 = self._goal_pegs()
        wp.launch(
            kernels.reset_controller,
            dim=self.nworld,
            inputs=[
                env.data.qpos,
                env.data.site_xpos,
                env.data.site_xmat,
                env._gpu_disk_qpos_addrs,
                env._gpu_peg_pos,
                env._gpu_peg_yaw,
                self.rng_counter,
                self.is_mistake,
                self.done,
                self.episode_length,
                self.episode_success,
                self.controller_done,
                self.agent_done,
                self.active_steps,
                self.speed_dt,
                self.target_peg,
                self.active_disk,
                self.target_disk,
                self.target_source,
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
                self.vertical_start,
                self.vertical_end,
                self.vertical_xy,
                self.vertical_yaw,
                self.vertical_c1,
                self.vertical_c2,
                self.plan_time,
                self.last_time,
                self.seeds,
                env.cpu_env._num_disks,
                env.cpu_env._num_pegs,
                goal_peg0,
                goal_peg1,
                float(env.cpu_env._disk_half_height),
                float(env.cpu_env._disk_gap),
                float(env.cpu_env._peg_half_height),
                float(env.cpu_env._disk_hole_half_size),
                float(env.cpu_env._disk_min_long_half_size),
                float(env.cpu_env._disk_long_size_step),
                self.p_mistake,
                env._pinch_site_id,
                self.arm_lo,
                self.arm_hi,
                self.segment_dt_scale,
                self.tilt_randomization,
            ],
            device=self.device,
        )

    def make_targets(self):
        env = self.env
        wp = self.wp
        goal_peg0, goal_peg1 = self._goal_pegs()
        wp.launch(
            kernels.make_targets,
            dim=self.nworld,
            inputs=[
                env.data.qpos,
                env.data.site_xpos,
                env.data.site_xmat,
                env._gpu_disk_qpos_addrs,
                env._gpu_peg_pos,
                env._gpu_peg_yaw,
                self.rng_counter,
                self.is_mistake,
                self.done,
                self.controller_done,
                self.agent_done,
                self.active_steps,
                self.speed_dt,
                self.target_peg,
                self.active_disk,
                self.target_disk,
                self.target_source,
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
                self.vertical_start,
                self.vertical_end,
                self.vertical_xy,
                self.vertical_yaw,
                self.vertical_c1,
                self.vertical_c2,
                self.plan_time,
                self.last_time,
                self.target_attach_pos,
                self.target_attach_xmat,
                self.target_gripper,
                self.down_xmat,
                self.down_xmat_inv,
                self.t_pa_rot,
                self.t_pa_translation,
                self.seeds,
                env.cpu_env._num_disks,
                env.cpu_env._num_pegs,
                goal_peg0,
                goal_peg1,
                float(env.cpu_env._disk_half_height),
                float(env.cpu_env._disk_gap),
                float(env.cpu_env._peg_half_height),
                float(env.cpu_env._disk_hole_half_size),
                float(env.cpu_env._disk_min_long_half_size),
                float(env.cpu_env._disk_long_size_step),
                self.p_mistake,
                self.arm_lo,
                self.arm_hi,
                self.workspace_lo,
                self.workspace_hi,
                self.ee_low,
                self.ee_high,
                env._pinch_site_id,
                env._gripper_opening_joint_id,
                float(env.cpu_env._control_timestep),
                self.max_subgoal_steps,
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
                self.env._gpu_alive,
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
