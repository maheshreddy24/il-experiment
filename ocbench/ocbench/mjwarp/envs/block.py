from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ocbench.envs.block_env import BlockEnv
from ocbench.mjwarp.envs.manipulation import ManipulationMjWarpEnv


class BlockMjWarpEnv(ManipulationMjWarpEnv):
    """Batched BlockEnv backed by MuJoCo Warp physics."""

    KERNEL_MODULE = 'ocbench.mjwarp.envs.block_kernels'

    def __init__(
        self,
        nworld: int = 1024,
        nconmax: int = 64,
        njmax: int = 512,
        nccdmax: Optional[int] = None,
        use_cuda_graph: bool = True,
        env_type='single',
        task_id=1,
        terminate_at_success=True,
        **kwargs,
    ):
        cpu_env = BlockEnv(
            env_type=env_type,
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

        self._gpu_object_qpos_addrs = None
        self._gpu_cube_target_mocap_ids = None

    def _cache_task_ids(self, model):
        self._cube_target_mocap_ids = np.asarray(self._cpu_env._cube_target_mocap_ids, dtype=np.int32)
        self._object_qpos_addrs = np.asarray(
            [model.jnt_qposadr[model.joint(f'object_joint_{i}').id] for i in range(self._cpu_env._num_cubes)],
            dtype=np.int32,
        )

    def _ensure_task_gpu_step_buffers(self, device):
        wp = self._wp
        self._gpu_object_qpos_addrs = wp.array(self._object_qpos_addrs.astype(np.int32), dtype=wp.int32, device=device)
        self._gpu_cube_target_mocap_ids = wp.array(self._cube_target_mocap_ids.astype(np.int32), dtype=wp.int32, device=device)

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

        task_info = self._cpu_env.cur_task_info
        has_fixed_targets = int(not self._cpu_env._is_stack_anywhere_task())
        if has_fixed_targets:
            goal_xyzs = np.asarray(task_info['goal_xyzs'], dtype=np.float32)
        else:
            goal_xyzs = np.zeros((self._cpu_env._num_cubes, 3), dtype=np.float32)

        wp.launch(
            self._kernels.reset_block_worlds,
            dim=len(world_ids),
            inputs=[
                self._data.qpos,
                self._data.qvel,
                self._data.ctrl,
                self._data.time,
                self._data.mocap_pos,
                self._data.mocap_quat,
                wp.array(world_ids, dtype=wp.int32, device=device),
                wp.array(seeds.astype(np.int32), dtype=wp.int32, device=device),
                self._ik_target_pos,
                self._ik_target_xmat,
                self._ik_target_gripper,
                self._ik_arm_qpos_ids,
                self._gpu_object_qpos_addrs,
                self._gpu_cube_target_mocap_ids,
                wp.array(self._cpu_env._home_qpos.astype(np.float32), dtype=wp.float32, device=device),
                wp.array(goal_xyzs, dtype=wp.vec3f, device=device),
                wp.array(self._cpu_env._effector_down_rotation.as_matrix()[None].astype(np.float32), dtype=wp.mat33f, device=device),
                wp.array(self._T_pa_rotation[None].astype(np.float32), dtype=wp.mat33f, device=device),
                wp.array(self._T_pa_translation[None].astype(np.float32), dtype=wp.vec3f, device=device),
                self._cpu_env.model.nq,
                self._cpu_env.model.nv,
                self._cpu_env.model.nu,
                self._cpu_env._num_cubes,
                has_fixed_targets,
                float(self._cpu_env._cube_size),
                float(self._cpu_env._min_object_init_dist),
                wp.vec2(
                    float(self._cpu_env._object_sampling_bounds[0, 0]),
                    float(self._cpu_env._object_sampling_bounds[0, 1]),
                ),
                wp.vec2(
                    float(self._cpu_env._object_sampling_bounds[1, 0]),
                    float(self._cpu_env._object_sampling_bounds[1, 1]),
                ),
                wp.vec3(
                    float(self._cpu_env._arm_sampling_bounds[0, 0]),
                    float(self._cpu_env._arm_sampling_bounds[0, 1]),
                    float(self._cpu_env._arm_sampling_bounds[0, 2]),
                ),
                wp.vec3(
                    float(self._cpu_env._arm_sampling_bounds[1, 0]),
                    float(self._cpu_env._arm_sampling_bounds[1, 1]),
                    float(self._cpu_env._arm_sampling_bounds[1, 2]),
                ),
            ],
            device=device,
        )
        self._solve_reset_arm_ik_gpu()

    def compute_success_gpu(self):
        if self._data is None:
            raise ValueError('Call `reset` before computing success.')
        self._ensure_gpu_step_buffers()
        lower = (self._cpu_env._workspace_bounds[0] - 0.2).astype(np.float32)
        upper = (self._cpu_env._workspace_bounds[1] + 0.2).astype(np.float32)
        self._wp.launch(
            self._kernels.block_success,
            dim=self._nworld,
            inputs=[
                self._data.qpos,
                self._data.mocap_pos,
                self._gpu_object_qpos_addrs,
                self._gpu_cube_target_mocap_ids,
                self._gpu_success,
                self._gpu_healthy,
                self._cpu_env._num_cubes,
                int(self._cpu_env._is_stack_anywhere_task()),
                float(self._cpu_env._cube_size),
                self._wp.vec3(float(lower[0]), float(lower[1]), float(lower[2])),
                self._wp.vec3(float(upper[0]), float(upper[1]), float(upper[2])),
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
            self._kernels.park_done_worlds,
            dim=self._nworld,
            inputs=[
                self._data.qpos,
                self._data.qvel,
                self._data.ctrl,
                self._data.mocap_pos,
                self._data.mocap_quat,
                done,
                self._gpu_object_qpos_addrs,
                self._gpu_cube_target_mocap_ids,
                self._ik_arm_qpos_ids,
                self._gpu_arm_actuator_ids,
                self._gpu_gripper_actuator_ids,
                self._gripper_opening_joint_id,
                self._cpu_env._num_cubes,
                self._cpu_env.model.nv,
                len(self._gripper_actuator_ids),
            ],
            device=self._data.qpos.device,
        )

    def _compute_stack_anywhere_successes(self, block_xyzs):
        order = np.argsort(block_xyzs[:, :, 2], axis=1)
        sorted_xyzs = np.take_along_axis(block_xyzs, order[:, :, None], axis=1)
        target_zs = self._cpu_env._cube_size * (2 * np.arange(self._cpu_env._num_cubes) + 1)

        xy_success = np.all(np.linalg.norm(block_xyzs[:, :, :2] - sorted_xyzs[:, :1, :2], axis=2) <= 0.04, axis=1)
        z_success = np.all(np.abs(sorted_xyzs[:, :, 2] - target_zs[None, :]) <= 0.03, axis=1)
        return xy_success & z_success

    def _post_step(self, arrays=None):
        if arrays is None:
            arrays = self._get_arrays()
        qpos = arrays['qpos']
        block_xyzs = np.stack([qpos[:, addr : addr + 3] for addr in self._object_qpos_addrs], axis=1)

        lower = self._cpu_env._workspace_bounds[0] - 0.2
        upper = self._cpu_env._workspace_bounds[1] + 0.2
        self._healthy = np.all((block_xyzs > lower) & (block_xyzs < upper), axis=(1, 2))

        if self._cpu_env._is_stack_anywhere_task():
            cube_successes = self._compute_stack_anywhere_successes(block_xyzs)
        else:
            target_xyzs = arrays['mocap_pos'][:, self._cube_target_mocap_ids]
            cube_successes = np.all(np.linalg.norm(block_xyzs - target_xyzs, axis=2) <= 0.04, axis=1)
        self._success = self._healthy & cube_successes

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

        for i, addr in enumerate(self._object_qpos_addrs):
            quat = qpos[:, addr + 3 : addr + 7]
            ob_info[f'privileged/block_{i}_pos'] = qpos[:, addr : addr + 3].copy()
            ob_info[f'privileged/block_{i}_quat'] = quat.copy()
            ob_info[f'privileged/block_{i}_yaw'] = self._quat_yaw(quat)

        ob_info['prev_qpos'] = self._prev_qpos.copy()
        ob_info['prev_qvel'] = self._prev_qvel.copy()
        ob_info['qpos'] = qpos.copy()
        ob_info['qvel'] = qvel.copy()
        ob_info['control'] = arrays['ctrl'].copy()
        ob_info['mocap_pos'] = arrays['mocap_pos'].copy()
        ob_info['mocap_quat'] = arrays['mocap_quat'].copy()
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
        ]
        for i in range(self._cpu_env._num_cubes):
            ob.extend(
                [
                    (ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler,
                    ob_info[f'privileged/block_{i}_quat'],
                    np.cos(ob_info[f'privileged/block_{i}_yaw']),
                    np.sin(ob_info[f'privileged/block_{i}_yaw']),
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
        self._cpu_env.data.mocap_pos[:] = arrays['mocap_pos'][world_id]
        self._cpu_env.data.mocap_quat[:] = arrays['mocap_quat'][world_id]
        mujoco.mj_forward(self._cpu_env.model, self._cpu_env.data)
