from __future__ import annotations


class HanoiMjWarpPrimitive:
    state_names = (
        'active_disk',
        'gate_time',
        'gate_xyz',
        'gate_passed',
        'vertical_start',
        'vertical_end',
        'vertical_xy',
        'vertical_yaw',
        'vertical_c1',
        'vertical_c2',
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

        self.active_disk = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.gate_time = wp.zeros((nworld, 6), dtype=wp.float32, device=device)
        self.gate_xyz = wp.zeros((nworld, 6), dtype=wp.vec3f, device=device)
        self.gate_passed = wp.zeros((nworld, 6), dtype=wp.int32, device=device)
        self.vertical_start = wp.zeros((nworld, 2), dtype=wp.float32, device=device)
        self.vertical_end = wp.zeros((nworld, 2), dtype=wp.float32, device=device)
        self.vertical_xy = wp.zeros((nworld, 2), dtype=wp.vec3f, device=device)
        self.vertical_yaw = wp.zeros((nworld, 2), dtype=wp.float32, device=device)
        self.vertical_c1 = wp.zeros((nworld, 2), dtype=wp.vec3f, device=device)
        self.vertical_c2 = wp.zeros((nworld, 2), dtype=wp.vec3f, device=device)

    def bind_to(self, owner):
        for name in self.state_names + self.constant_names:
            setattr(owner, name, getattr(self, name))
