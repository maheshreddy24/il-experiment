from __future__ import annotations

import numpy as np

from ocbench.mjwarp import MAX_PLAN_KEYS


class MjWarpPrimitive:
    state_names = (
        'key_count',
        'key_time',
        'key_xyz',
        'key_quat',
        'key_grasp',
        'key_stop',
        'key_tangent',
        'plan_time',
        'last_time',
        'target_attach_pos',
        'target_attach_xmat',
        'target_gripper',
    )
    constant_names = (
        'down_xmat',
        'down_xmat_inv',
        't_pa_rot',
        't_pa_translation',
        'ee_low',
        'ee_high',
        'arm_lo',
        'arm_hi',
        'workspace_lo',
        'workspace_hi',
    )

    def __init__(self, env):
        self.env = env
        self.wp = env._wp
        self.device = env.data.qpos.device
        self.nworld = env.nworld
        self._allocate_state()
        self._allocate_constants()

    def _allocate_state(self):
        wp = self.wp
        nworld = self.nworld
        device = self.device

        self.key_count = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.key_time = wp.zeros((nworld, MAX_PLAN_KEYS), dtype=wp.float32, device=device)
        self.key_xyz = wp.zeros((nworld, MAX_PLAN_KEYS), dtype=wp.vec3f, device=device)
        self.key_quat = wp.zeros((nworld, MAX_PLAN_KEYS), dtype=wp.quatf, device=device)
        self.key_grasp = wp.zeros((nworld, MAX_PLAN_KEYS), dtype=wp.float32, device=device)
        self.key_stop = wp.zeros((nworld, MAX_PLAN_KEYS), dtype=wp.int32, device=device)
        self.key_tangent = wp.zeros((nworld, MAX_PLAN_KEYS), dtype=wp.vec3f, device=device)
        # Keep control-grid selection stable across long plans and gate pauses.
        self.plan_time = wp.zeros(nworld, dtype=wp.float64, device=device)
        self.last_time = wp.zeros(nworld, dtype=wp.float32, device=device)

        self.target_attach_pos = wp.zeros(nworld, dtype=wp.vec3f, device=device)
        self.target_attach_xmat = wp.zeros(nworld, dtype=wp.mat33f, device=device)
        self.target_gripper = wp.zeros(nworld, dtype=wp.float32, device=device)

    def _allocate_constants(self):
        wp = self.wp
        env = self.env
        cpu_env = env.cpu_env
        device = self.device

        self.down_xmat = wp.array(
            cpu_env._effector_down_rotation.as_matrix()[None].astype(np.float32),
            dtype=wp.mat33f,
            device=device,
        )
        self.down_xmat_inv = wp.array(
            cpu_env._effector_down_rotation.as_matrix().T[None].astype(np.float32),
            dtype=wp.mat33f,
            device=device,
        )
        self.t_pa_rot = wp.array(env._T_pa_rotation[None].astype(np.float32), dtype=wp.mat33f, device=device)
        self.t_pa_translation = wp.array(env._T_pa_translation[None].astype(np.float32), dtype=wp.vec3f, device=device)
        ee_delta = cpu_env._ee_action_delta.astype(np.float32)
        self.ee_low = wp.array(-ee_delta, dtype=wp.float32, device=device)
        self.ee_high = wp.array(ee_delta, dtype=wp.float32, device=device)
        self.arm_lo = self._vec3(cpu_env._arm_sampling_bounds[0])
        self.arm_hi = self._vec3(cpu_env._arm_sampling_bounds[1])
        self.workspace_lo = self._vec3(cpu_env._workspace_bounds[0])
        self.workspace_hi = self._vec3(cpu_env._workspace_bounds[1])

    def _vec3(self, value):
        return self.wp.vec3(float(value[0]), float(value[1]), float(value[2]))

    def bind_to(self, owner):
        for name in self.state_names + self.constant_names:
            setattr(owner, name, getattr(self, name))
