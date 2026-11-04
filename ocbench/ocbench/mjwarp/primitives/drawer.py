from __future__ import annotations


class DrawerMjWarpPrimitive:
    state_names = ()
    constant_names = ('drawer_open_threshold', 'drawer_slide_min', 'drawer_slide_max', 'nominal_drawer_pos')

    def __init__(self, env):
        cpu_env = env.cpu_env
        wp = env._wp
        self.drawer_open_threshold = float(cpu_env._drawer_open_threshold)
        self.drawer_slide_min = float(cpu_env._drawer_slide_min)
        self.drawer_slide_max = float(cpu_env._drawer_slide_max)
        self.nominal_drawer_pos = wp.vec3(*cpu_env._nominal_drawer_base_pos.astype('float32'))

    def bind_to(self, owner):
        for name in self.constant_names:
            setattr(owner, name, getattr(self, name))
