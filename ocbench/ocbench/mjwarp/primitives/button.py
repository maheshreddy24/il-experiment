from __future__ import annotations

import numpy as np


class ButtonMjWarpPrimitive:
    state_names = ()
    constant_names = ('button_site_ids', 'button0_site_id', 'button1_site_id')

    def __init__(self, env):
        site_ids = np.asarray(env.cpu_env._button_site_ids, dtype=np.int32)
        self.button_site_ids = env._wp.array(site_ids, dtype=env._wp.int32, device=env.data.qpos.device)
        self.button0_site_id = int(site_ids[0])
        self.button1_site_id = int(site_ids[1] if len(site_ids) > 1 else site_ids[0])

    def bind_to(self, owner):
        for name in self.constant_names:
            setattr(owner, name, getattr(self, name))
