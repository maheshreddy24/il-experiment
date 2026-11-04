from __future__ import annotations


class WindowMjWarpPrimitive:
    state_names = ()
    constant_names = ('window_slide_min', 'window_slide_max')

    def __init__(self, env):
        cpu_env = env.cpu_env
        self.window_slide_min = float(cpu_env._window_slide_min)
        self.window_slide_max = float(cpu_env._window_slide_max)

    def bind_to(self, owner):
        for name in self.constant_names:
            setattr(owner, name, getattr(self, name))
