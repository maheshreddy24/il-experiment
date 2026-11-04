from __future__ import annotations

from importlib import import_module
from typing import Optional

import mujoco
import numpy as np
from gymnasium.spaces import Box

from ocbench.mjwarp import IK_ITERS


class ManipulationMjWarpEnv:
    """Base class for batched MuJoCo Warp manipulation environments."""

    KERNEL_MODULE = None

    def __init__(
        self,
        *,
        cpu_env,
        nworld: int,
        nconmax: int,
        njmax: int,
        nccdmax: Optional[int],
        use_cuda_graph: bool,
        terminate_at_success: bool,
    ):
        self._nworld = int(nworld)
        self._nconmax = int(nconmax)
        self._njmax = int(njmax)
        self._nccdmax = nccdmax
        self._use_cuda_graph = use_cuda_graph
        self._terminate_at_success = terminate_at_success

        self._cpu_env = cpu_env

        self._mjw = None
        self._wp = None
        self._model = None
        self._data = None
        self._ik_data = None
        self._prev_qpos = None
        self._prev_qvel = None
        self._success = None
        self._failure = None
        self._healthy = None
        self._step_graph = None
        self._can_capture_step_graph = False
        self._ik_kernels = None
        self._kernels = None
        self._ik_target_pos = None
        self._ik_target_xmat = None
        self._ik_target_gripper = None
        self._ik_jac_point = None
        self._ik_body = None
        self._ik_jacp = None
        self._ik_jacr = None
        self._ik_action = None
        self._ik_arm_qpos_ids = None
        self._ik_arm_dof_ids = None
        self._ik_joint_action_delta = None
        self._gpu_done = None
        self._gpu_success = None
        self._gpu_healthy = None
        self._gpu_arm_actuator_ids = None
        self._gpu_gripper_actuator_ids = None
        self._gpu_arm_ctrl_low = None
        self._gpu_arm_ctrl_high = None

    @property
    def nworld(self) -> int:
        return self._nworld

    @property
    def single_action_space(self):
        return self._cpu_env.action_space

    @property
    def single_observation_space(self):
        return self._cpu_env.observation_space

    @property
    def action_space(self):
        single = self.single_action_space
        return Box(
            low=np.broadcast_to(single.low, (self._nworld, *single.shape)).copy(),
            high=np.broadcast_to(single.high, (self._nworld, *single.shape)).copy(),
            dtype=single.dtype,
        )

    @property
    def model(self):
        if self._model is None:
            raise ValueError('MJWarp model not yet initialized. Call `reset` to initialize.')
        return self._model

    @property
    def data(self):
        if self._data is None:
            raise ValueError('MJWarp data not yet initialized. Call `reset` to initialize.')
        return self._data

    @property
    def cpu_env(self):
        return self._cpu_env

    @property
    def unwrapped(self):
        return self

    def _ensure_mjwarp(self):
        if self._mjw is not None:
            return
        try:
            import mujoco_warp as mjw
            import warp as wp
        except ImportError as e:
            raise ImportError('Install MuJoCo Warp with `pip install mujoco-warp` or `pip install -e .[mjwarp]`.') from e

        self._mjw = mjw
        self._wp = wp
        self._ik_kernels = import_module('ocbench.mjwarp.envs.manipulation_kernels')
        self._kernels = import_module(self.KERNEL_MODULE)

    def _cache_ids(self):
        model = self._cpu_env.model
        self._arm_joint_ids = self._cpu_env._arm_joint_ids
        self._arm_actuator_ids = self._cpu_env._arm_actuator_ids
        self._gripper_actuator_ids = self._cpu_env._gripper_actuator_ids
        self._gripper_opening_joint_id = self._cpu_env._gripper_opening_joint_id
        self._pinch_site_id = self._cpu_env._pinch_site_id
        self._attach_site_id = self._cpu_env._attach_site_id
        self._attach_body_id = model.site_bodyid[self._attach_site_id]
        self._right_pad_body_id = model.body('ur5e/robotiq/right_pad').id
        self._arm_qpos_ids = model.jnt_qposadr[self._arm_joint_ids]
        self._arm_dof_ids = model.jnt_dofadr[self._arm_joint_ids]
        self._arm_ctrlrange = model.actuator_ctrlrange[self._arm_actuator_ids].copy()
        self._T_pa_rotation = self._cpu_env._T_pa.rotation().as_matrix()
        self._T_pa_translation = self._cpu_env._T_pa.translation()
        self._cache_task_ids(model)

    def _cache_task_ids(self, model):
        pass

    def _make_data(self):
        kwargs = dict(nworld=self._nworld, nconmax=self._nconmax, njmax=self._njmax)
        if self._nccdmax is not None:
            kwargs['nccdmax'] = int(self._nccdmax)
        opt = self._cpu_env.model.opt
        opt.iterations = 6
        opt.tolerance = 1e-5
        opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
        self._model = self._mjw.put_model(self._cpu_env.model)
        self._data = self._mjw.make_data(self._cpu_env.model, **kwargs)
        self._ik_data = self._mjw.make_data(self._cpu_env.model, **kwargs)
        self._step_graph = None
        self._can_capture_step_graph = False
        self._reset_ik_buffers()

    def _reset_ik_buffers(self):
        self._ik_target_pos = None
        self._ik_target_xmat = None
        self._ik_target_gripper = None
        self._ik_jac_point = None
        self._ik_body = None
        self._ik_jacp = None
        self._ik_jacr = None
        self._ik_action = None
        self._ik_arm_qpos_ids = None
        self._ik_arm_dof_ids = None
        self._ik_joint_action_delta = None

    def _ensure_gpu_step_buffers(self):
        if self._gpu_done is not None:
            return

        wp = self._wp
        device = self._data.qpos.device
        self._gpu_done = wp.zeros(self._nworld, dtype=wp.int32, device=device)
        self._gpu_success = wp.zeros(self._nworld, dtype=wp.int32, device=device)
        self._gpu_healthy = wp.zeros(self._nworld, dtype=wp.int32, device=device)
        self._gpu_arm_actuator_ids = wp.array(self._arm_actuator_ids.astype(np.int32), dtype=wp.int32, device=device)
        self._gpu_gripper_actuator_ids = wp.array(self._gripper_actuator_ids.astype(np.int32), dtype=wp.int32, device=device)
        self._gpu_arm_ctrl_low = wp.array(self._arm_ctrlrange[:, 0].astype(np.float32), dtype=wp.float32, device=device)
        self._gpu_arm_ctrl_high = wp.array(self._arm_ctrlrange[:, 1].astype(np.float32), dtype=wp.float32, device=device)
        self._ensure_task_gpu_step_buffers(device)

    def _ensure_task_gpu_step_buffers(self, device):
        pass

    def _ensure_ik_buffers(self):
        if self._ik_action is not None:
            return

        wp = self._wp
        device = self._data.qpos.device
        self._ik_target_pos = wp.empty(self._nworld, dtype=wp.vec3f, device=device)
        self._ik_target_xmat = wp.empty(self._nworld, dtype=wp.mat33f, device=device)
        self._ik_target_gripper = wp.empty(self._nworld, dtype=wp.float32, device=device)
        self._ik_jac_point = wp.empty(self._nworld, dtype=wp.vec3f, device=device)
        self._ik_body = wp.empty(self._nworld, dtype=wp.int32, device=device)
        self._ik_jacp = wp.empty((self._nworld, 3, self._cpu_env.model.nv), dtype=wp.float32, device=device)
        self._ik_jacr = wp.empty((self._nworld, 3, self._cpu_env.model.nv), dtype=wp.float32, device=device)
        self._ik_action = wp.empty((self._nworld, 7), dtype=wp.float32, device=device)
        self._ik_arm_qpos_ids = wp.array(self._arm_qpos_ids.astype(np.int32), dtype=wp.int32, device=device)
        self._ik_arm_dof_ids = wp.array(self._arm_dof_ids.astype(np.int32), dtype=wp.int32, device=device)
        self._ik_joint_action_delta = wp.array(self._cpu_env._joint_action_delta.astype(np.float32), dtype=wp.float32, device=device)

    def _reset_seeds(self, seed, seeds, count):
        if seeds is not None:
            return np.asarray(seeds, dtype=np.uint32)
        if seed is None:
            return np.random.SeedSequence().generate_state(count, dtype=np.uint32)
        return np.random.SeedSequence(int(seed)).generate_state(count, dtype=np.uint32)

    def _ensure_reset_model(self, seed=None, options=None):
        if self._data is None or options is not None:
            self._cpu_env.reset(seed=seed, options=options)
            self._after_cpu_reset_model()
            if self._data is None:
                self._cache_ids()
                self._make_data()

    def _after_cpu_reset_model(self):
        pass

    def _solve_reset_arm_ik_gpu(
        self,
        ik_iters: int = IK_ITERS,
        damping_coeff: float = 1e-12,
        max_angle_change: float = np.radians(45),
    ):
        self._ensure_ik_buffers()
        device = self._data.qpos.device
        for _ in range(ik_iters):
            self._mjw.kinematics(self._model, self._data)
            self._mjw.com_pos(self._model, self._data)
            self._wp.launch(
                self._ik_kernels.set_jac_points,
                dim=self._nworld,
                inputs=[
                    self._data.site_xpos,
                    self._ik_jac_point,
                    self._ik_body,
                    self._attach_site_id,
                    self._attach_body_id,
                ],
                device=device,
            )
            self._mjw.jac(self._model, self._data, self._ik_jacp, self._ik_jacr, self._ik_jac_point, self._ik_body)
            self._wp.launch(
                self._ik_kernels.ik_update,
                dim=self._nworld,
                inputs=[
                    self._data.qpos,
                    self._data.site_xpos,
                    self._data.site_xmat,
                    self._ik_jacp,
                    self._ik_jacr,
                    self._ik_target_pos,
                    self._ik_target_xmat,
                    self._ik_arm_qpos_ids,
                    self._ik_arm_dof_ids,
                    self._attach_site_id,
                    float(damping_coeff),
                    float(max_angle_change),
                ],
                device=device,
            )

        self._mjw.forward(self._model, self._data)
        self._mjw.rne_postconstraint(self._model, self._data)

    def _finish_reset(self, world_ids=None):
        arrays = self._get_arrays()
        if world_ids is None:
            self._prev_qpos = arrays['qpos'].copy()
            self._prev_qvel = arrays['qvel'].copy()
            self._failure = np.zeros(self._nworld, dtype=bool)
        else:
            self._prev_qpos[world_ids] = arrays['qpos'][world_ids]
            self._prev_qvel[world_ids] = arrays['qvel'][world_ids]
            self._failure[world_ids] = False

        self._post_step(arrays)
        info = self.compute_ob_info(arrays)
        ob = self.compute_observation(info, arrays=arrays)
        info['success'] = self._success.copy()
        info['failure'] = self._failure.copy()
        info['healthy'] = self._healthy.copy()
        return ob, info

    def _launch_mjwarp_control_step_sequence(self):
        for _ in range(self._cpu_env._n_steps):
            self._mjw.step(self._model, self._data)
        self._mjw.rne_postconstraint(self._model, self._data)

    def _run_mjwarp_control_step(self):
        if self._use_cuda_graph and self._step_graph is not None:
            self._wp.capture_launch(self._step_graph)
            return

        self._launch_mjwarp_control_step_sequence()
        self._can_capture_step_graph = True

    def _ensure_step_graph(self):
        if not self._use_cuda_graph or self._step_graph is not None or not self._can_capture_step_graph:
            return

        with self._wp.ScopedCapture() as capture:
            self._launch_mjwarp_control_step_sequence()
        self._step_graph = capture.graph

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, seeds=None):
        self._ensure_mjwarp()
        if seed is not None and seeds is not None:
            raise ValueError('Pass either `seed` or `seeds`, not both.')
        if seeds is not None:
            seeds = np.asarray(seeds, dtype=np.uint32)
            if seeds.shape != (self._nworld,):
                raise ValueError(f'Expected {self._nworld} reset seeds, got shape {seeds.shape}.')
        reset_seeds = self._reset_seeds(seed, seeds, self._nworld)
        self._ensure_reset_model(seed=int(reset_seeds[0]), options=options)
        self._reset_worlds_gpu(np.arange(self._nworld, dtype=np.int32), reset_seeds, full_reset=True)
        return self._finish_reset()

    def reset_worlds(self, world_ids, options: Optional[dict] = None, seeds=None):
        if self._data is None:
            return self.reset(options=options, seeds=seeds)

        world_ids = np.asarray(world_ids, dtype=np.int32)
        if seeds is not None:
            seeds = np.asarray(seeds, dtype=np.uint32)
            if seeds.shape != (len(world_ids),):
                raise ValueError(f'Expected {len(world_ids)} reset seeds, got shape {seeds.shape}.')
        if len(world_ids) == 0:
            ob = self.compute_observation()
            info = self.get_info()
            return ob, info
        reset_seeds = self._reset_seeds(None, seeds, len(world_ids))
        self._ensure_reset_model(seed=int(reset_seeds[0]), options=options)
        self._reset_worlds_gpu(world_ids, reset_seeds)
        return self._finish_reset(world_ids)

    def _get_arrays(self):
        self._wp.synchronize()
        arrays = dict(
            qpos=self._data.qpos.numpy(),
            qvel=self._data.qvel.numpy(),
            ctrl=self._data.ctrl.numpy(),
            time=self._data.time.numpy(),
            site_xpos=self._data.site_xpos.numpy(),
            site_xmat=self._data.site_xmat.numpy(),
            cfrc_ext=self._data.cfrc_ext.numpy(),
            mocap_pos=self._data.mocap_pos.numpy(),
            mocap_quat=self._data.mocap_quat.numpy(),
        )
        arrays.update(self._get_task_arrays())
        return arrays

    def _get_task_arrays(self):
        return {}

    def _set_control(self, action, arrays, ctrl=None):
        qpos = arrays['qpos']
        action = np.asarray(action, dtype=np.float32)
        if action.shape == self.single_action_space.shape:
            action = np.broadcast_to(action, (self._nworld, *self.single_action_space.shape)).copy()
        if action.shape != (self._nworld, *self.single_action_space.shape):
            raise ValueError(f'Expected action shape {(self._nworld, *self.single_action_space.shape)} but got {action.shape}.')

        action = np.clip(action, -1.0, 1.0) * self._cpu_env._joint_action_delta
        qpos_target = qpos[:, self._arm_joint_ids] + action[:, :6]
        qpos_target = np.clip(qpos_target, self._arm_ctrlrange[:, 0], self._arm_ctrlrange[:, 1])

        gripper_opening = qpos[:, self._gripper_opening_joint_id] / 0.8
        target_gripper_opening = np.clip(gripper_opening + action[:, 6], 0.0, 1.0)

        if ctrl is None:
            ctrl = self._data.ctrl.numpy()
        ctrl[:, self._arm_actuator_ids] = qpos_target
        ctrl[:, self._gripper_actuator_ids] = 255.0 * target_gripper_opening[:, None]
        self._wp.copy(self._data.ctrl, self._wp.array(ctrl.astype(np.float32), dtype=self._wp.float32))

    def _finalize_step(self, action, arrays, ob_info, ob, reward, terminated, truncated):
        action = np.asarray(action)
        if action.shape == self.single_action_space.shape:
            action = np.broadcast_to(action, (self._nworld, *self.single_action_space.shape))
        finite_action = np.all(np.isfinite(action).reshape(self._nworld, -1), axis=1)

        finite_transition = np.ones(self._nworld, dtype=bool)
        for value in (*arrays.values(), ob, reward):
            value = np.asarray(value)
            finite_transition &= np.all(np.isfinite(value).reshape(self._nworld, -1), axis=1)
        finite_transition &= self._data.nefc.numpy() <= self._njmax
        numerical_failure = finite_action & ~finite_transition

        self._success[numerical_failure] = False
        self._failure[numerical_failure] = True
        self._healthy[numerical_failure] = False
        reward[numerical_failure] = 0.0
        terminated[numerical_failure] = True

        info = ob_info
        info['success'] = self._success.copy()
        info['failure'] = self._failure.copy()
        info['healthy'] = self._healthy.copy()
        info['numerical_failure'] = numerical_failure
        if self._cpu_env._ob_type == 'pixels':
            ob = self.get_pixel_observation(arrays=arrays)
        return ob, reward, terminated, truncated, info

    def _target_attach_pose(self, effector_pos, effector_xmat):
        target_pos = effector_pos + np.einsum('nij,j->ni', effector_xmat, self._T_pa_translation)
        target_xmat = effector_xmat @ self._T_pa_rotation
        return target_pos, target_xmat

    def ee_targets_to_joint_actions(
        self,
        effector_pos,
        effector_xmat,
        gripper_opening,
        ik_iters: int = IK_ITERS,
        damping_coeff: float = 1e-12,
        max_angle_change: float = np.radians(45),
    ):
        """Convert batched end-effector targets to normalized joint actions."""
        if self._data is None:
            raise ValueError('Call `reset` before computing actions.')
        wp = self._wp
        self._ensure_ik_buffers()

        effector_pos = np.asarray(effector_pos, dtype=np.float32)
        effector_xmat = np.asarray(effector_xmat, dtype=np.float32)
        gripper_opening = np.asarray(gripper_opening, dtype=np.float32).reshape(self._nworld)
        target_pos, target_xmat = self._target_attach_pose(effector_pos, effector_xmat)

        device = self._data.qpos.device
        wp.copy(self._ik_target_pos, wp.array(target_pos.astype(np.float32), dtype=wp.vec3f, device=device))
        wp.copy(self._ik_target_xmat, wp.array(target_xmat.astype(np.float32), dtype=wp.mat33f, device=device))
        wp.copy(self._ik_target_gripper, wp.array(gripper_opening.astype(np.float32), dtype=wp.float32, device=device))

        wp.copy(self._ik_data.qpos, self._data.qpos)
        wp.copy(self._ik_data.qvel, self._data.qvel)
        wp.copy(self._ik_data.ctrl, self._data.ctrl)
        if self._data.mocap_pos.shape[1] > 0:
            wp.copy(self._ik_data.mocap_pos, self._data.mocap_pos)
            wp.copy(self._ik_data.mocap_quat, self._data.mocap_quat)

        for _ in range(ik_iters):
            self._mjw.kinematics(self._model, self._ik_data)
            self._mjw.com_pos(self._model, self._ik_data)
            wp.launch(
                self._ik_kernels.set_jac_points,
                dim=self._nworld,
                inputs=[
                    self._ik_data.site_xpos,
                    self._ik_jac_point,
                    self._ik_body,
                    self._attach_site_id,
                    self._attach_body_id,
                ],
                device=device,
            )
            self._mjw.jac(self._model, self._ik_data, self._ik_jacp, self._ik_jacr, self._ik_jac_point, self._ik_body)
            wp.launch(
                self._ik_kernels.ik_update,
                dim=self._nworld,
                inputs=[
                    self._ik_data.qpos,
                    self._ik_data.site_xpos,
                    self._ik_data.site_xmat,
                    self._ik_jacp,
                    self._ik_jacr,
                    self._ik_target_pos,
                    self._ik_target_xmat,
                    self._ik_arm_qpos_ids,
                    self._ik_arm_dof_ids,
                    self._attach_site_id,
                    float(damping_coeff),
                    float(max_angle_change),
                ],
                device=device,
            )

        wp.launch(
            self._ik_kernels.ik_action,
            dim=self._nworld,
            inputs=[
                self._ik_data.qpos,
                self._data.qpos,
                self._ik_target_gripper,
                self._ik_action,
                self._ik_arm_qpos_ids,
                self._gripper_opening_joint_id,
                self._ik_joint_action_delta,
            ],
            device=device,
        )
        return self._ik_action.numpy().copy()

    def ee_target_arrays_to_joint_actions_gpu(
        self,
        target_pos,
        target_xmat,
        target_gripper,
        ik_iters: int = IK_ITERS,
        damping_coeff: float = 1e-12,
        max_angle_change: float = np.radians(45),
    ):
        if self._data is None:
            raise ValueError('Call `reset` before computing actions.')
        wp = self._wp
        self._ensure_ik_buffers()

        device = self._data.qpos.device
        wp.copy(self._ik_data.qpos, self._data.qpos)
        wp.copy(self._ik_data.qvel, self._data.qvel)
        wp.copy(self._ik_data.ctrl, self._data.ctrl)
        if self._data.mocap_pos.shape[1] > 0:
            wp.copy(self._ik_data.mocap_pos, self._data.mocap_pos)
            wp.copy(self._ik_data.mocap_quat, self._data.mocap_quat)

        for _ in range(ik_iters):
            self._mjw.kinematics(self._model, self._ik_data)
            self._mjw.com_pos(self._model, self._ik_data)
            wp.launch(
                self._ik_kernels.set_jac_points,
                dim=self._nworld,
                inputs=[
                    self._ik_data.site_xpos,
                    self._ik_jac_point,
                    self._ik_body,
                    self._attach_site_id,
                    self._attach_body_id,
                ],
                device=device,
            )
            self._mjw.jac(self._model, self._ik_data, self._ik_jacp, self._ik_jacr, self._ik_jac_point, self._ik_body)
            wp.launch(
                self._ik_kernels.ik_update,
                dim=self._nworld,
                inputs=[
                    self._ik_data.qpos,
                    self._ik_data.site_xpos,
                    self._ik_data.site_xmat,
                    self._ik_jacp,
                    self._ik_jacr,
                    target_pos,
                    target_xmat,
                    self._ik_arm_qpos_ids,
                    self._ik_arm_dof_ids,
                    self._attach_site_id,
                    float(damping_coeff),
                    float(max_angle_change),
                ],
                device=device,
            )

        wp.launch(
            self._ik_kernels.ik_action,
            dim=self._nworld,
            inputs=[
                self._ik_data.qpos,
                self._data.qpos,
                target_gripper,
                self._ik_action,
                self._ik_arm_qpos_ids,
                self._gripper_opening_joint_id,
                self._ik_joint_action_delta,
            ],
            device=device,
        )
        return self._ik_action

    def _before_step_joint_actions_gpu(self):
        pass

    def step_joint_actions_gpu(self, action, done=None):
        if self._data is None:
            self.reset()
        self._ensure_ik_buffers()
        self._ensure_gpu_step_buffers()

        self._before_step_joint_actions_gpu()
        self.set_joint_action_control_gpu(action, done)
        self.advance_physics_gpu()
        self.compute_success_gpu()
        return self._gpu_success

    def set_joint_action_control_gpu(self, action, done=None):
        if self._data is None:
            self.reset()
        self._ensure_ik_buffers()
        self._ensure_gpu_step_buffers()

        wp = self._wp
        device = self._data.qpos.device
        if done is None:
            wp.copy(self._gpu_done, wp.zeros(self._nworld, dtype=wp.int32, device=device))
        else:
            wp.copy(self._gpu_done, done)
        wp.launch(
            self._ik_kernels.set_control_from_action,
            dim=self._nworld,
            inputs=[
                self._data.qpos,
                self._data.ctrl,
                action,
                self._gpu_done,
                self._ik_arm_qpos_ids,
                self._gpu_arm_actuator_ids,
                self._gpu_gripper_actuator_ids,
                self._gpu_arm_ctrl_low,
                self._gpu_arm_ctrl_high,
                self._ik_joint_action_delta,
                self._gripper_opening_joint_id,
                len(self._gripper_actuator_ids),
            ],
            device=device,
        )

    def advance_physics_gpu(self):
        self._ensure_step_graph()
        self._run_mjwarp_control_step()

    def step(self, action):
        if self._data is None:
            self.reset()

        arrays = self._get_arrays()
        self._prev_qpos = arrays['qpos'].copy()
        self._prev_qvel = arrays['qvel'].copy()
        self._set_control(action, arrays, ctrl=arrays['ctrl'])

        self._ensure_step_graph()
        self._run_mjwarp_control_step()
        arrays = self._get_arrays()
        self._post_step(arrays)

        ob_info = self.compute_ob_info(arrays)
        ob = self.compute_state_observation(ob_info)
        reward = (self._success & ~self._failure & self._healthy).astype(np.float32)
        terminated = (self._failure | ~self._healthy) | (self._terminate_at_success & self._success)
        truncated = np.zeros(self._nworld, dtype=bool)
        return self._finalize_step(action, arrays, ob_info, ob, reward, terminated, truncated)

    def _site_yaw(self, site_xmat):
        site_mat = site_xmat[:, self._pinch_site_id]
        if site_mat.ndim == 2:
            site_mat = site_mat.reshape(self._nworld, 3, 3)
        return np.arctan2(site_mat[:, 1, 0], site_mat[:, 0, 0])[:, None]

    def _quat_yaw(self, quat):
        q0 = quat[:, 0]
        q1 = quat[:, 1]
        q2 = quat[:, 2]
        q3 = quat[:, 3]
        return np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))[:, None]

    def get_info(self):
        info = self.compute_ob_info()
        info['success'] = self._success.copy()
        info['failure'] = self._failure.copy()
        info['healthy'] = self._healthy.copy()
        return info

    def compute_observation(self, ob_info=None, *, arrays=None):
        if self._cpu_env._ob_type == 'pixels':
            return self.get_pixel_observation(arrays=arrays)
        return self.compute_state_observation(ob_info)

    def get_pixel_observation(self, world_ids=None, *, arrays=None):
        if world_ids is None:
            world_ids = range(self._nworld)
        if arrays is None:
            arrays = self._get_arrays()
        images = []
        for world_id in world_ids:
            self.copy_world_to_cpu_env(world_id, arrays=arrays)
            images.append(self._cpu_env.get_pixel_observation())
        return np.stack(images)

    def render_world(self, world_id: int = 0, *args, **kwargs):
        self.copy_world_to_cpu_env(world_id)
        return self._cpu_env.render(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.render_world(0, *args, **kwargs)

    def close(self):
        self._cpu_env.close()
