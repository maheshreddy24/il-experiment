from __future__ import annotations


class BowlingMjWarpPrimitive:
    state_names = (
        'phase',
        'phase_steps',
        'ball_start_pos',
        'push_speed',
        'push_steps',
        'push_ramp_steps',
        'prepush_dist',
        'push_z',
        'gripper',
        'side_offset',
        'push_dir',
        'side_dir',
        'yaw',
        'tilt',
        'tilt_drift',
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

        self.phase = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.phase_steps = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.ball_start_pos = wp.zeros(nworld, dtype=wp.vec3f, device=device)
        self.push_speed = wp.zeros(nworld, dtype=wp.float32, device=device)
        self.push_steps = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.push_ramp_steps = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.prepush_dist = wp.zeros(nworld, dtype=wp.float32, device=device)
        self.push_z = wp.zeros(nworld, dtype=wp.float32, device=device)
        self.gripper = wp.zeros(nworld, dtype=wp.float32, device=device)
        self.side_offset = wp.zeros(nworld, dtype=wp.float32, device=device)
        self.push_dir = wp.zeros(nworld, dtype=wp.vec3f, device=device)
        self.side_dir = wp.zeros(nworld, dtype=wp.vec3f, device=device)
        self.yaw = wp.zeros(nworld, dtype=wp.float32, device=device)
        self.tilt = wp.zeros(nworld, dtype=wp.vec3f, device=device)
        self.tilt_drift = wp.zeros(nworld, dtype=wp.vec3f, device=device)

    def bind_to(self, owner):
        for name in self.state_names + self.constant_names:
            setattr(owner, name, getattr(self, name))
