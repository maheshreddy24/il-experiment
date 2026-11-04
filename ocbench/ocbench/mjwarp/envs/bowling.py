from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ocbench.envs.bowling_env import BowlingEnv
from ocbench.mjwarp.envs.manipulation import ManipulationMjWarpEnv


class BowlingMjWarpEnv(ManipulationMjWarpEnv):
    """Batched BowlingEnv backed by MuJoCo Warp physics."""

    KERNEL_MODULE = 'ocbench.mjwarp.envs.bowling_kernels'

    def __init__(
        self,
        nworld: int = 1024,
        nconmax: int = 64,
        njmax: int = 512,
        nccdmax: Optional[int] = None,
        use_cuda_graph: bool = True,
        task_id=1,
        terminate_at_success=True,
        **kwargs,
    ):
        cpu_env = BowlingEnv(
            task_id=task_id,
            terminate_at_success=terminate_at_success,
            **kwargs,
        )
        super().__init__(
            cpu_env=cpu_env,
            nworld=nworld,
            nconmax=nconmax,
            njmax=njmax,
            nccdmax=nccdmax,
            use_cuda_graph=use_cuda_graph,
            terminate_at_success=terminate_at_success,
        )

        self._pin_knocked = None
        self._prev_num_knocked_pins = None
        self._num_knocked_pins = None
        self._elapsed_steps = None
        self._gpu_pin_knocked = None
        self._gpu_prev_num_knocked_pins = None
        self._gpu_num_knocked_pins = None
        self._gpu_pin_qpos_addrs = None

    def _cache_task_ids(self, model):
        self._ball_qpos_addr = int(model.jnt_qposadr[model.joint('ball_joint').id])
        self._ball_qvel_addr = int(model.jnt_dofadr[model.joint('ball_joint').id])
        self._pin_qpos_addrs = np.asarray(
            [model.jnt_qposadr[model.joint(f'pin_joint_{i}').id] for i in range(self._cpu_env._num_pins)],
            dtype=np.int32,
        )

    def _ensure_task_gpu_step_buffers(self, device):
        wp = self._wp
        self._gpu_pin_knocked = wp.zeros((self._nworld, self._cpu_env._num_pins), dtype=wp.int32, device=device)
        self._gpu_prev_num_knocked_pins = wp.zeros(self._nworld, dtype=wp.int32, device=device)
        self._gpu_num_knocked_pins = wp.zeros(self._nworld, dtype=wp.int32, device=device)
        self._gpu_pin_qpos_addrs = wp.array(self._pin_qpos_addrs.astype(np.int32), dtype=wp.int32, device=device)

    def _reset_worlds_gpu(self, world_ids, seeds, full_reset=False):
        self._ensure_ik_buffers()
        self._ensure_gpu_step_buffers()
        wp = self._wp
        device = self._data.qpos.device
        world_ids = np.asarray(world_ids, dtype=np.int32)
        seeds = np.asarray(seeds, dtype=np.uint32)
        if full_reset:
            self._mjw.reset_data(self._model, self._data)
        else:
            self._mjw.kinematics(self._model, self._data)
            self._mjw.com_pos(self._model, self._data)
            wp.launch(
                self._ik_kernels.hold_current_attach_targets,
                dim=self._nworld,
                inputs=[
                    self._data.qpos,
                    self._data.site_xpos,
                    self._data.site_xmat,
                    self._ik_target_pos,
                    self._ik_target_xmat,
                    self._ik_target_gripper,
                    self._attach_site_id,
                    self._gripper_opening_joint_id,
                ],
                device=device,
            )

        pin_init_pos = np.column_stack(
            [
                self._cpu_env._pin_init_xys,
                np.full(self._cpu_env._num_pins, self._cpu_env._pin_half_height),
            ]
        ).astype(np.float32)
        wp.launch(
            self._kernels.reset_bowling_worlds,
            dim=len(world_ids),
            inputs=[
                self._data.qpos,
                self._data.qvel,
                self._data.ctrl,
                self._data.time,
                wp.array(world_ids, dtype=wp.int32, device=device),
                wp.array(seeds.astype(np.int32), dtype=wp.int32, device=device),
                self._ik_target_pos,
                self._ik_target_xmat,
                self._ik_target_gripper,
                self._ik_arm_qpos_ids,
                self._ball_qpos_addr,
                self._gpu_pin_qpos_addrs,
                self._gpu_pin_knocked,
                self._gpu_prev_num_knocked_pins,
                self._gpu_num_knocked_pins,
                wp.array(self._cpu_env._home_qpos.astype(np.float32), dtype=wp.float32, device=device),
                wp.array(pin_init_pos, dtype=wp.vec3f, device=device),
                wp.array(self._cpu_env._effector_down_rotation.as_matrix()[None].astype(np.float32), dtype=wp.mat33f, device=device),
                wp.array(self._T_pa_rotation[None].astype(np.float32), dtype=wp.mat33f, device=device),
                wp.array(self._T_pa_translation[None].astype(np.float32), dtype=wp.vec3f, device=device),
                self._cpu_env.model.nq,
                self._cpu_env.model.nv,
                self._cpu_env.model.nu,
                self._cpu_env._num_pins,
                self._wp.vec3(*self._cpu_env._ball_init_pos.astype(np.float32)),
                self._wp.vec3(float(self._cpu_env._ball_init_xy_randomization[0, 0]), float(self._cpu_env._ball_init_xy_randomization[0, 1]), 0.0),
                self._wp.vec3(float(self._cpu_env._ball_init_xy_randomization[1, 0]), float(self._cpu_env._ball_init_xy_randomization[1, 1]), 0.0),
                self._wp.vec3(float(self._cpu_env._pin_rack_xy_randomization[0, 0]), float(self._cpu_env._pin_rack_xy_randomization[0, 1]), 0.0),
                self._wp.vec3(float(self._cpu_env._pin_rack_xy_randomization[1, 0]), float(self._cpu_env._pin_rack_xy_randomization[1, 1]), 0.0),
                self._wp.vec3(float(self._cpu_env._arm_sampling_bounds[0, 0]), float(self._cpu_env._arm_sampling_bounds[0, 1]), float(self._cpu_env._arm_sampling_bounds[0, 2])),
                self._wp.vec3(float(self._cpu_env._arm_sampling_bounds[1, 0]), float(self._cpu_env._arm_sampling_bounds[1, 1]), float(self._cpu_env._arm_sampling_bounds[1, 2])),
            ],
            device=device,
        )
        self._solve_reset_arm_ik_gpu()

    def _finish_reset(self, world_ids=None):
        self.compute_success_gpu()
        arrays = self._get_arrays()
        if world_ids is None:
            self._prev_qpos = arrays['qpos'].copy()
            self._prev_qvel = arrays['qvel'].copy()
            self._elapsed_steps = np.zeros(self._nworld, dtype=np.int32)
        else:
            self._prev_qpos[world_ids] = arrays['qpos'][world_ids]
            self._prev_qvel[world_ids] = arrays['qvel'][world_ids]
            self._elapsed_steps[world_ids] = 0

        self._post_step(arrays)
        info = self.compute_ob_info(arrays)
        ob = self.compute_observation(info, arrays=arrays)
        info['success'] = self._success.copy()
        info['failure'] = self._failure.copy()
        info['healthy'] = self._healthy.copy()
        return ob, info

    def _save_prev_num_knocked_pins_gpu(self):
        self._wp.launch(
            self._kernels.save_bowling_prev_num_knocked,
            dim=self._nworld,
            inputs=[
                self._gpu_prev_num_knocked_pins,
                self._gpu_num_knocked_pins,
            ],
            device=self._data.qpos.device,
        )

    def _before_step_joint_actions_gpu(self):
        self._save_prev_num_knocked_pins_gpu()

    def compute_success_gpu(self):
        if self._data is None:
            raise ValueError('Call `reset` before computing success.')
        self._ensure_gpu_step_buffers()
        lower_z = float(self._cpu_env._workspace_bounds[0, 2])
        upper_z = float(self._cpu_env._workspace_bounds[1, 2])
        self._wp.launch(
            self._kernels.bowling_score,
            dim=self._nworld,
            inputs=[
                self._data.qpos,
                self._ball_qpos_addr,
                self._gpu_pin_qpos_addrs,
                self._gpu_pin_knocked,
                self._gpu_num_knocked_pins,
                self._gpu_success,
                self._gpu_healthy,
                self._cpu_env._num_pins,
                float(self._cpu_env._workspace_half_size),
                lower_z,
                upper_z,
                0.2,
            ],
            device=self._data.qpos.device,
        )
        return self._gpu_success

    def park_done_worlds_gpu(self, done):
        if self._data is None:
            raise ValueError('Call `reset` before parking worlds.')
        self._ensure_ik_buffers()
        self._ensure_gpu_step_buffers()
        self._wp.launch(
            self._kernels.park_done_bowling_worlds,
            dim=self._nworld,
            inputs=[
                self._data.qpos,
                self._data.qvel,
                self._data.ctrl,
                done,
                self._ball_qpos_addr,
                self._gpu_pin_qpos_addrs,
                self._ik_arm_qpos_ids,
                self._gpu_arm_actuator_ids,
                self._gpu_gripper_actuator_ids,
                self._gripper_opening_joint_id,
                self._cpu_env._num_pins,
                self._cpu_env.model.nv,
                len(self._gripper_actuator_ids),
            ],
            device=self._data.qpos.device,
        )

    def step(self, action):
        if self._data is None:
            self.reset()

        arrays = self._get_arrays()
        self._prev_qpos = arrays['qpos'].copy()
        self._prev_qvel = arrays['qvel'].copy()
        self._save_prev_num_knocked_pins_gpu()
        self._set_control(action, arrays, ctrl=arrays['ctrl'])

        self._ensure_step_graph()
        self._run_mjwarp_control_step()
        self.compute_success_gpu()
        arrays = self._get_arrays()
        self._elapsed_steps += 1
        self._post_step(arrays)

        ob_info = self.compute_ob_info(arrays)
        ob = self.compute_state_observation(ob_info)
        reward = 0.1 * np.maximum(self._num_knocked_pins - self._prev_num_knocked_pins, 0).astype(np.float32)
        terminated = (~self._healthy) | (self._elapsed_steps >= self._cpu_env._max_episode_steps)
        truncated = np.zeros(self._nworld, dtype=bool)
        return self._finalize_step(action, arrays, ob_info, ob, reward, terminated, truncated)

    def _post_step(self, arrays=None):
        if arrays is None:
            arrays = self._get_arrays()
        self._pin_knocked = self._gpu_pin_knocked.numpy().copy().astype(bool)
        self._prev_num_knocked_pins = self._gpu_prev_num_knocked_pins.numpy().copy()
        self._num_knocked_pins = self._gpu_num_knocked_pins.numpy().copy()
        self._healthy = self._gpu_healthy.numpy().copy().astype(bool)
        self._success = np.zeros(self._nworld, dtype=bool)
        self._failure = np.zeros(self._nworld, dtype=bool)

    def compute_ob_info(self, arrays=None):
        if arrays is None:
            arrays = self._get_arrays()
        qpos = arrays['qpos']
        qvel = arrays['qvel']
        site_yaw = self._site_yaw(arrays['site_xmat'])

        ob_info: dict[str, Any] = {}
        ob_info['proprio/joint_pos'] = qpos[:, self._arm_joint_ids].copy()
        ob_info['proprio/joint_vel'] = qvel[:, self._arm_joint_ids].copy()
        ob_info['proprio/effector_pos'] = arrays['site_xpos'][:, self._pinch_site_id].copy()
        ob_info['proprio/effector_xmat'] = arrays['site_xmat'][:, self._pinch_site_id].reshape(self._nworld, 3, 3).copy()
        ob_info['proprio/effector_yaw'] = site_yaw
        ob_info['proprio/gripper_opening'] = np.clip(qpos[:, [self._gripper_opening_joint_id]] / 0.8, 0, 1)
        ob_info['proprio/gripper_vel'] = qvel[:, [self._gripper_opening_joint_id]].copy()
        ob_info['proprio/gripper_contact'] = np.clip(
            np.linalg.norm(arrays['cfrc_ext'][:, self._right_pad_body_id], axis=1, keepdims=True) / 50,
            0,
            1,
        )

        ball_qpos = qpos[:, self._ball_qpos_addr : self._ball_qpos_addr + 7]
        ball_qvel = qvel[:, self._ball_qvel_addr : self._ball_qvel_addr + 6]
        ob_info['privileged/ball_pos'] = ball_qpos[:, :3].copy()
        ob_info['privileged/ball_quat'] = ball_qpos[:, 3:].copy()
        ob_info['privileged/ball_vel'] = ball_qvel[:, :3].copy()
        ob_info['privileged/ball_angvel'] = ball_qvel[:, 3:].copy()
        for i, addr in enumerate(self._pin_qpos_addrs):
            pin_qpos = qpos[:, addr : addr + 7]
            ob_info[f'privileged/pin_{i}_pos'] = pin_qpos[:, :3].copy()
            ob_info[f'privileged/pin_{i}_quat'] = pin_qpos[:, 3:].copy()

        ob_info['prev_qpos'] = self._prev_qpos.copy()
        ob_info['prev_qvel'] = self._prev_qvel.copy()
        ob_info['qpos'] = qpos.copy()
        ob_info['qvel'] = qvel.copy()
        ob_info['control'] = arrays['ctrl'].copy()
        ob_info['time'] = arrays['time'][:, None].copy()
        return ob_info

    def compute_state_observation(self, ob_info=None):
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        gripper_scaler = 3.0

        if ob_info is None:
            ob_info = self.compute_ob_info()
        ob = [
            ob_info['proprio/joint_pos'],
            ob_info['proprio/joint_vel'],
            (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
            np.cos(ob_info['proprio/effector_yaw']),
            np.sin(ob_info['proprio/effector_yaw']),
            ob_info['proprio/gripper_opening'] * gripper_scaler,
            ob_info['proprio/gripper_contact'],
            (ob_info['privileged/ball_pos'] - xyz_center) * xyz_scaler,
            ob_info['privileged/ball_quat'],
            ob_info['privileged/ball_vel'] * xyz_scaler,
            ob_info['privileged/ball_angvel'],
        ]
        for i in range(self._cpu_env._num_pins):
            ob.extend(
                [
                    (ob_info[f'privileged/pin_{i}_pos'] - xyz_center) * xyz_scaler,
                    ob_info[f'privileged/pin_{i}_quat'],
                ]
            )
        return np.concatenate(ob, axis=1)

    def copy_world_to_cpu_env(self, world_id: int, *, arrays=None):
        if self._data is None:
            raise ValueError('Call `reset` before copying a world.')
        import mujoco

        if arrays is None:
            arrays = self._get_arrays()
        world_id = int(world_id)
        self._cpu_env.data.qpos[:] = arrays['qpos'][world_id]
        self._cpu_env.data.qvel[:] = arrays['qvel'][world_id]
        self._cpu_env.data.ctrl[:] = arrays['ctrl'][world_id]
        mujoco.mj_forward(self._cpu_env.model, self._cpu_env.data)
