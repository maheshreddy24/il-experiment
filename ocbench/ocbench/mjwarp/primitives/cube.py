from __future__ import annotations

from ocbench.mjwarp.primitives import cube_kernels as kernels


class CubeMjWarpPrimitive:
    state_names = (
        'active_block',
        'gate_time',
        'gate_xyz',
        'gate_passed',
        'phase_start',
        'phase_end',
        'phase_c1',
        'phase_c2',
        'pick_check_time',
        'pick_source_z',
        'pick_checked',
        'num_pick_retries',
    )
    constant_names = ()

    def __init__(self, env):
        self.env = env
        self.wp = env._wp
        self.device = env.data.qpos.device
        self.nworld = env.nworld
        self._allocate_state()

    def _allocate_state(self):
        wp = self.wp
        nworld = self.nworld
        device = self.device

        self.active_block = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.gate_time = wp.zeros((nworld, 2), dtype=wp.float32, device=device)
        self.gate_xyz = wp.zeros((nworld, 2), dtype=wp.vec3f, device=device)
        self.gate_passed = wp.zeros((nworld, 2), dtype=wp.int32, device=device)
        self.phase_start = wp.zeros((nworld, kernels.MAX_PHASES), dtype=wp.float32, device=device)
        self.phase_end = wp.zeros((nworld, kernels.MAX_PHASES), dtype=wp.float32, device=device)
        self.phase_c1 = wp.zeros((nworld, kernels.MAX_PHASES), dtype=wp.vec3f, device=device)
        self.phase_c2 = wp.zeros((nworld, kernels.MAX_PHASES), dtype=wp.vec3f, device=device)
        self.pick_check_time = wp.zeros(nworld, dtype=wp.float32, device=device)
        self.pick_source_z = wp.zeros(nworld, dtype=wp.float32, device=device)
        self.pick_checked = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.num_pick_retries = wp.zeros(nworld, dtype=wp.int32, device=device)

    def bind_to(self, owner):
        for name in self.state_names + self.constant_names:
            setattr(owner, name, getattr(self, name))
