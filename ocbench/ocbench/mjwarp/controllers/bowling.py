from __future__ import annotations

import numpy as np

from ocbench.mjwarp.controllers import bowling_kernels as kernels
from ocbench.mjwarp.primitives.bowling import BowlingMjWarpPrimitive
from ocbench.mjwarp.primitives.primitive import MjWarpPrimitive


class BowlingMjWarpController:
    def __init__(
        self,
        env,
        seed,
        max_steps: int,
        max_subgoal_steps: int = 500,
    ):
        if env._data is None:
            raise ValueError('Call `env.reset` before creating BowlingMjWarpController.')

        self.env = env
        self.max_steps = int(max_steps)
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

    def _allocate_state(self):
        wp = self.wp
        device = self.device
        nworld = self.nworld

        self.done = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.episode_length = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.episode_success = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.episode_score = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.controller_done = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.active_steps = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.rng_counter = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.speed_dt = wp.zeros(nworld, dtype=wp.float32, device=device)

        self.primitive = MjWarpPrimitive(self.env)
        self.primitive.bind_to(self)
        self.bowling_primitive = BowlingMjWarpPrimitive(self.env)
        self.bowling_primitive.bind_to(self)

    def reset(self):
        env = self.env
        self.wp.launch(
            kernels.reset_controller,
            dim=self.nworld,
            inputs=[
                env.data.qpos,
                env.data.site_xpos,
                env._ball_qpos_addr,
                env._gpu_pin_qpos_addrs,
                self.rng_counter,
                self.done,
                self.episode_length,
                self.episode_score,
                self.controller_done,
                self.active_steps,
                self.speed_dt,
                self.phase,
                self.phase_steps,
                self.ball_start_pos,
                self.push_speed,
                self.push_steps,
                self.push_ramp_steps,
                self.prepush_dist,
                self.push_z,
                self.gripper,
                self.side_offset,
                self.push_dir,
                self.side_dir,
                self.yaw,
                self.tilt,
                self.tilt_drift,
                self.seeds,
                env._pinch_site_id,
                env.cpu_env._num_pins,
            ],
            device=self.device,
        )

    def make_targets(self):
        env = self.env
        self.wp.launch(
            kernels.make_targets,
            dim=self.nworld,
            inputs=[
                env.data.qpos,
                env.data.site_xpos,
                env.data.site_xmat,
                self.done,
                self.controller_done,
                self.active_steps,
                self.phase,
                self.phase_steps,
                self.ball_start_pos,
                self.push_speed,
                self.push_steps,
                self.push_ramp_steps,
                self.prepush_dist,
                self.push_z,
                self.gripper,
                self.side_offset,
                self.push_dir,
                self.side_dir,
                self.yaw,
                self.tilt,
                self.tilt_drift,
                self.target_attach_pos,
                self.target_attach_xmat,
                self.target_gripper,
                self.down_xmat,
                self.down_xmat_inv,
                self.t_pa_rot,
                self.t_pa_translation,
                self.workspace_lo,
                self.workspace_hi,
                self.ee_high,
                env._pinch_site_id,
                env._gripper_opening_joint_id,
                self.max_subgoal_steps,
            ],
            device=self.device,
        )
        return self.target_attach_pos, self.target_attach_xmat, self.target_gripper

    def update_done(self):
        self.wp.launch(
            kernels.update_done,
            dim=self.nworld,
            inputs=[
                self.env._gpu_healthy,
                self.env._gpu_num_knocked_pins,
                self.done,
                self.episode_length,
                self.episode_score,
                self.max_steps,
            ],
            device=self.device,
        )
        self.env.park_done_worlds_gpu(self.done)

    def stats(self):
        self.wp.synchronize()
        lengths = self.episode_length.numpy().copy()
        scores = self.episode_score.numpy().copy()
        done = self.done.numpy().copy().astype(bool)
        return dict(
            completed=int(done.sum()),
            transitions=int(lengths.sum()),
            avg_knocked_pins=float(scores.mean()),
            avg_episode_length=float(lengths.mean()),
            min_episode_length=int(lengths.min()),
            max_episode_length=int(lengths.max()),
        )
