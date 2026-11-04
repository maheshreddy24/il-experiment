from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ocbench.envs.hanoi_env import HanoiEnv
from ocbench.mjwarp.envs.manipulation import ManipulationMjWarpEnv


class HanoiMjWarpEnv(ManipulationMjWarpEnv):
    """Batched HanoiEnv backed by MuJoCo Warp physics."""

    KERNEL_MODULE = 'ocbench.mjwarp.envs.hanoi_kernels'

    def __init__(
        self,
        nworld: int = 1024,
        nconmax: int = 128,
        njmax: int = 1024,
        nccdmax: Optional[int] = None,
        use_cuda_graph: bool = True,
        env_type='double',
        task_id=1,
        terminate_at_success=True,
        **kwargs,
    ):
        cpu_env = HanoiEnv(
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

        self._prev_peg_pos = None
        self._prev_peg_yaw = None
        self._gpu_failure = None
        self._gpu_alive = None
        self._gpu_bad_floor_contact = None
        self._gpu_disk_qpos_addrs = None
        self._gpu_disk_geom_ids = None
        self._gpu_disk_geom_counts = None
        self._gpu_peg_mocap_ids = None
        self._gpu_peg_pos = None
        self._gpu_peg_yaw = None

    def _cache_task_ids(self, model):
        self._disk_qpos_addrs = np.asarray(
            [model.jnt_qposadr[model.joint(f'disk_joint_{i}').id] for i in range(self._cpu_env._num_disks)],
            dtype=np.int32,
        )
        max_disk_geoms = 8
        self._disk_geom_ids = np.full((self._cpu_env._num_disks, max_disk_geoms), -1, dtype=np.int32)
        self._disk_geom_counts = np.asarray([len(geom_ids) for geom_ids in self._cpu_env._disk_geom_ids_list], dtype=np.int32)
        for disk, geom_ids in enumerate(self._cpu_env._disk_geom_ids_list):
            self._disk_geom_ids[disk, : len(geom_ids)] = np.asarray(geom_ids, dtype=np.int32)
        self._floor_geom_id = self._cpu_env._floor_geom_id
        self._peg_mocap_ids = np.asarray(self._cpu_env._peg_body_mocap_ids, dtype=np.int32)

    def _ensure_task_gpu_step_buffers(self, device):
        wp = self._wp
        self._gpu_failure = wp.zeros(self._nworld, dtype=wp.int32, device=device)
        self._gpu_alive = wp.zeros(self._nworld, dtype=wp.int32, device=device)
        self._gpu_bad_floor_contact = wp.zeros(self._nworld, dtype=wp.int32, device=device)
        self._gpu_disk_qpos_addrs = wp.array(self._disk_qpos_addrs.astype(np.int32), dtype=wp.int32, device=device)
        self._gpu_disk_geom_ids = wp.array(self._disk_geom_ids.astype(np.int32), dtype=wp.int32, device=device)
        self._gpu_disk_geom_counts = wp.array(self._disk_geom_counts.astype(np.int32), dtype=wp.int32, device=device)
        self._gpu_peg_mocap_ids = wp.array(self._peg_mocap_ids.astype(np.int32), dtype=wp.int32, device=device)
        self._gpu_peg_pos = wp.zeros((self._nworld, self._cpu_env._num_pegs), dtype=wp.vec3f, device=device)
        self._gpu_peg_yaw = wp.zeros((self._nworld, self._cpu_env._num_pegs), dtype=wp.float32, device=device)

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

        base_peg_pos = np.column_stack(
            [
                self._cpu_env._base_peg_positions,
                np.full(self._cpu_env._num_pegs, self._cpu_env._peg_half_height),
            ]
        ).astype(np.float32)
        wp.launch(
            self._kernels.reset_hanoi_worlds,
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
                self._gpu_disk_qpos_addrs,
                self._gpu_peg_mocap_ids,
                self._gpu_peg_pos,
                self._gpu_peg_yaw,
                wp.array(self._cpu_env._home_qpos.astype(np.float32), dtype=wp.float32, device=device),
                wp.array(base_peg_pos, dtype=wp.vec3f, device=device),
                wp.array(self._cpu_env._effector_down_rotation.as_matrix()[None].astype(np.float32), dtype=wp.mat33f, device=device),
                wp.array(self._T_pa_rotation[None].astype(np.float32), dtype=wp.mat33f, device=device),
                wp.array(self._T_pa_translation[None].astype(np.float32), dtype=wp.vec3f, device=device),
                self._cpu_env.model.nq,
                self._cpu_env.model.nv,
                self._cpu_env.model.nu,
                self._cpu_env._num_disks,
                self._cpu_env._num_pegs,
                int(self._cpu_env.cur_task_info['init_peg']),
                int(self._cpu_env.cur_task_info['task_name'] == 'random'),
                float(self._cpu_env._disk_half_height),
                float(self._cpu_env._disk_gap),
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

    def _finish_reset(self, world_ids=None):
        self.compute_success_gpu()
        arrays = self._get_arrays()
        peg_pos = self._gpu_peg_pos.numpy().copy()
        peg_yaw = self._gpu_peg_yaw.numpy().copy()
        if world_ids is None:
            self._prev_qpos = arrays['qpos'].copy()
            self._prev_qvel = arrays['qvel'].copy()
            self._prev_peg_pos = peg_pos.copy()
            self._prev_peg_yaw = peg_yaw.copy()
        else:
            self._prev_qpos[world_ids] = arrays['qpos'][world_ids]
            self._prev_qvel[world_ids] = arrays['qvel'][world_ids]
            self._prev_peg_pos[world_ids] = peg_pos[world_ids]
            self._prev_peg_yaw[world_ids] = peg_yaw[world_ids]

        self._post_step(arrays, peg_pos=peg_pos)
        info = self.compute_ob_info(arrays, peg_pos=peg_pos, peg_yaw=peg_yaw)
        ob = self.compute_observation(info, arrays=arrays)
        info['success'] = self._success.copy()
        info['failure'] = self._failure.copy()
        info['healthy'] = self._healthy.copy()
        return ob, info

    def compute_success_gpu(self):
        if self._data is None:
            raise ValueError('Call `reset` before computing success.')
        self._ensure_gpu_step_buffers()
        lower = (self._cpu_env._workspace_bounds[0] - 0.2).astype(np.float32)
        upper = (self._cpu_env._workspace_bounds[1] + 0.2).astype(np.float32)
        goal_pegs = np.asarray(self._cpu_env.cur_task_info['goal_pegs'], dtype=np.int32)
        device = self._data.qpos.device
        self._wp.launch(
            self._kernels.clear_int_flags,
            dim=self._nworld,
            inputs=[self._gpu_bad_floor_contact],
            device=device,
        )
        self._wp.launch(
            self._kernels.hanoi_bad_floor_contacts,
            dim=int(self._data.contact.dim.shape[0]),
            inputs=[
                self._data.qpos,
                self._data.contact.geom,
                self._data.contact.dim,
                self._data.contact.worldid,
                self._data.nacon,
                self._gpu_disk_qpos_addrs,
                self._gpu_disk_geom_ids,
                self._gpu_disk_geom_counts,
                self._gpu_peg_pos,
                self._gpu_bad_floor_contact,
                self._cpu_env._num_disks,
                self._cpu_env._num_pegs,
                int(self._floor_geom_id),
                int(self._data.contact.dim.shape[0]),
            ],
            device=device,
        )
        self._wp.launch(
            self._kernels.hanoi_success,
            dim=self._nworld,
            inputs=[
                self._data.qpos,
                self._gpu_bad_floor_contact,
                self._gpu_disk_qpos_addrs,
                self._gpu_peg_pos,
                self._gpu_success,
                self._gpu_healthy,
                self._gpu_failure,
                self._gpu_alive,
                self._cpu_env._num_disks,
                self._cpu_env._num_pegs,
                int(goal_pegs[0]),
                int(goal_pegs[1]),
                float(self._cpu_env._disk_half_height),
                float(self._cpu_env._disk_gap),
                self._wp.vec3(float(lower[0]), float(lower[1]), float(lower[2])),
                self._wp.vec3(float(upper[0]), float(upper[1]), float(upper[2])),
            ],
            device=device,
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
                done,
                self._gpu_disk_qpos_addrs,
                self._ik_arm_qpos_ids,
                self._gpu_arm_actuator_ids,
                self._gpu_gripper_actuator_ids,
                self._gripper_opening_joint_id,
                self._cpu_env._num_disks,
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
        self._prev_peg_pos = self._gpu_peg_pos.numpy().copy()
        self._prev_peg_yaw = self._gpu_peg_yaw.numpy().copy()
        self._set_control(action, arrays, ctrl=arrays['ctrl'])

        self._ensure_step_graph()
        self._run_mjwarp_control_step()
        self.compute_success_gpu()
        arrays = self._get_arrays()
        peg_pos = self._gpu_peg_pos.numpy()
        peg_yaw = self._gpu_peg_yaw.numpy()
        self._post_step(arrays, peg_pos=peg_pos)

        ob_info = self.compute_ob_info(arrays, peg_pos=peg_pos, peg_yaw=peg_yaw)
        ob = self.compute_state_observation(ob_info)
        reward = (self._success & ~self._failure & self._healthy).astype(np.float32)
        terminated = (self._failure | ~self._healthy) | (self._terminate_at_success & self._success)
        truncated = np.zeros(self._nworld, dtype=bool)
        return self._finalize_step(action, arrays, ob_info, ob, reward, terminated, truncated)

    def _post_step(self, arrays=None, peg_pos=None):
        self._success = self._gpu_success.numpy().copy().astype(bool)
        self._healthy = self._gpu_healthy.numpy().copy().astype(bool)
        self._failure = self._gpu_failure.numpy().copy().astype(bool)

    def compute_ob_info(self, arrays=None, peg_pos=None, peg_yaw=None):
        if arrays is None:
            arrays = self._get_arrays()
        if peg_pos is None:
            peg_pos = self._gpu_peg_pos.numpy()
        if peg_yaw is None:
            peg_yaw = self._gpu_peg_yaw.numpy()
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

        for i, addr in enumerate(self._disk_qpos_addrs):
            quat = qpos[:, addr + 3 : addr + 7]
            ob_info[f'privileged/disk_{i}_pos'] = qpos[:, addr : addr + 3].copy()
            ob_info[f'privileged/disk_{i}_quat'] = quat.copy()
            ob_info[f'privileged/disk_{i}_yaw'] = self._quat_yaw(quat)
        for i in range(self._cpu_env._num_pegs):
            ob_info[f'privileged/peg_{i}_pos'] = peg_pos[:, i].copy()
            ob_info[f'privileged/peg_{i}_yaw'] = peg_yaw[:, [i]].copy()

        ob_info['prev_qpos'] = self._prev_qpos.copy()
        ob_info['prev_qvel'] = self._prev_qvel.copy()
        ob_info['qpos'] = qpos.copy()
        ob_info['qvel'] = qvel.copy()
        ob_info['control'] = arrays['ctrl'].copy()
        ob_info['mocap_pos'] = arrays['mocap_pos'].copy()
        ob_info['mocap_quat'] = arrays['mocap_quat'].copy()
        ob_info['time'] = arrays['time'][:, None].copy()
        ob_info['prev_dynamics_info'] = np.concatenate([self._prev_peg_pos, self._prev_peg_yaw[:, :, None]], axis=2).reshape(self._nworld, -1)
        ob_info['dynamics_info'] = np.concatenate([peg_pos, peg_yaw[:, :, None]], axis=2).reshape(self._nworld, -1)
        return ob_info

    def _get_task_arrays(self):
        return dict(peg_pos=self._gpu_peg_pos.numpy(), peg_yaw=self._gpu_peg_yaw.numpy())

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
        for i in range(self._cpu_env._num_disks):
            ob.extend(
                [
                    (ob_info[f'privileged/disk_{i}_pos'] - xyz_center) * xyz_scaler,
                    np.cos(ob_info[f'privileged/disk_{i}_yaw']),
                    np.sin(ob_info[f'privileged/disk_{i}_yaw']),
                ]
            )
        for i in range(self._cpu_env._num_pegs):
            ob.extend(
                [
                    (ob_info[f'privileged/peg_{i}_pos'] - xyz_center) * xyz_scaler,
                    np.cos(ob_info[f'privileged/peg_{i}_yaw']),
                    np.sin(ob_info[f'privileged/peg_{i}_yaw']),
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
        self._cpu_env._set_peg_layout(arrays['peg_pos'][world_id], arrays['peg_yaw'][world_id])
        mujoco.mj_forward(self._cpu_env.model, self._cpu_env.data)
