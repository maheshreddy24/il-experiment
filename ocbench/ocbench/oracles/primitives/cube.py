import numpy as np

from ocbench import lie
from ocbench.oracles.primitives.primitive import Primitive


class CubePrimitive(Primitive):
    def __init__(
        self,
        env,
        segment_dt=1.0,
        target_peak_xyz_speed=0.3,
    ):
        super().__init__(
            env=env,
            segment_dt=segment_dt,
            target_peak_xyz_speed=target_peak_xyz_speed,
        )
        self._smooth_plan_stop_names = {'initial', 'pick_hold', 'place_start', 'place_end', 'final'}
        self._max_pick_retries = 2
        self._descent_height_range = (0.06, 0.22)
        self._descent_xy_range = (0.016, 0.06)
        self._num_pick_retries = 0
        self._pick_checked = False
        self._pick_check_time = None
        self._pick_source_z = None
        self._block = None
        self._target = None

    def compute_keyframes(self, plan_input):
        tilt_randomization = self._tilt_randomization
        noise_scale = 0.75**self._num_pick_retries
        use_pick_hold = np.random.uniform() < 0.4 * noise_scale

        # Poses.
        poses = {}

        # Pick.
        block_initial_base = self.shortest_yaw(
            eff_yaw=self.get_yaw(plan_input['effector_initial']),
            obj_yaw=self.get_yaw(plan_input['block_initial']),
            translation=plan_input['block_initial'].translation(),
        )
        yaw_mode = np.random.choice([-1, 0, 1], p=[0.15, 0.7, 0.15]) * np.pi / 2
        yaw_offset = self.yaw_error(self.get_yaw(block_initial_base), self.get_yaw(plan_input['block_initial']))
        yaw_offset += yaw_mode + np.random.uniform(-0.04, 0.04) * noise_scale
        if tilt_randomization:
            pick_tilt_angle = np.random.uniform(-np.pi, np.pi)
            pick_tilt_mag = np.random.uniform(0.0, 25.0 * np.pi / 180.0) * noise_scale
            pick_tilt = pick_tilt_mag * np.array([np.cos(pick_tilt_angle), np.sin(pick_tilt_angle)])
            place_tilt = 0.2 * pick_tilt + np.random.uniform(-0.06, 0.06, size=2) * noise_scale
        else:
            pick_tilt = np.zeros(2)
            place_tilt = np.zeros(2)
        block_initial = self.to_pose(
            pos=plan_input['block_initial'].translation(),
            yaw=self.get_yaw(plan_input['block_initial']) + yaw_offset,
            roll=pick_tilt[0],
            pitch=pick_tilt[1],
        )
        poses['initial'] = plan_input['effector_initial']
        poses['pick'] = self.above(block_initial, 0.1 + np.random.uniform(0, 0.1))
        poses['pick_start'] = block_initial
        poses['pick_end'] = block_initial
        if use_pick_hold:
            poses['pick_hold'] = block_initial
        poses['postpick'] = poses['pick']

        # Place.
        block_goal = self.to_pose(
            pos=plan_input['block_goal'].translation(),
            yaw=self.get_yaw(plan_input['block_goal']) + yaw_offset,
            roll=place_tilt[0],
            pitch=place_tilt[1],
        )
        poses['place'] = self.above(block_goal, 0.1 + np.random.uniform(0, 0.1))
        poses['place_start'] = block_goal
        poses['place_end'] = block_goal
        poses['postplace'] = poses['place']
        poses['final'] = plan_input['effector_goal']

        # Approach.
        pick_start = poses['pick_start'].translation()
        place_start = poses['place_start'].translation()
        side_sign = np.random.choice([-1, 1])
        place_yaw = self.get_yaw(poses['place_start'])
        place_forward = np.array([np.cos(place_yaw), np.sin(place_yaw)])
        place_dir = np.array([-np.sin(place_yaw), np.cos(place_yaw)]) * side_sign
        descent_xy_range = np.asarray(self._descent_xy_range) * noise_scale
        place_offset = (
            np.random.uniform(*descent_xy_range) * place_dir[:2]
            + np.random.uniform(-0.02, 0.02) * noise_scale * place_forward
        )
        pick_axis = poses['pick_start'].rotation() @ np.array([0.0, 0.0, 1.0])
        contact_angle = np.random.uniform(-np.pi, np.pi)
        contact_dir = poses['pick_start'].rotation() @ np.array(
            [np.cos(contact_angle), np.sin(contact_angle), 0.0]
        )
        contact_shift = np.random.uniform(0.0, 0.012) * noise_scale
        start_lift = np.random.uniform(0.0, 0.012) * noise_scale
        end_depth = np.random.uniform(-0.004, 0.006) * noise_scale
        pick_start_pos = pick_start + start_lift * pick_axis - 0.5 * contact_shift * contact_dir
        pick_end_pos = pick_start + end_depth * pick_axis + 0.5 * contact_shift * contact_dir
        pick_end_tilt = pick_tilt
        if tilt_randomization:
            pick_end_tilt = pick_end_tilt + np.random.uniform(-0.035, 0.035, size=2) * noise_scale
        pick_end_yaw = self.get_yaw(poses['pick_start']) + np.random.uniform(-0.06, 0.06) * noise_scale
        poses['pick_start'] = lie.SE3.from_rotation_and_translation(
            rotation=poses['pick_start'].rotation(),
            translation=pick_start_pos,
        )
        poses['pick_end'] = self.to_pose(
            pos=pick_end_pos,
            yaw=pick_end_yaw,
            roll=pick_end_tilt[0],
            pitch=pick_end_tilt[1],
        )
        if use_pick_hold:
            poses['pick_hold'] = poses['pick_end']
        pick_pos = pick_start_pos + np.random.uniform(*self._descent_height_range) * pick_axis
        pick_z = pick_pos[2]
        place_z = place_start[2] + np.random.uniform(*self._descent_height_range)

        poses['pick'] = lie.SE3.from_rotation_and_translation(
            rotation=poses['pick'].rotation(),
            translation=pick_pos,
        )
        poses['postpick'] = poses['pick']
        poses['place'] = lie.SE3.from_rotation_and_translation(
            rotation=poses['place'].rotation(),
            translation=np.array([*(place_start[:2] + place_offset), place_z]),
        )
        poses['postplace'] = poses['place']

        # Carry midpoints.
        midpoint_names = []
        num_midpoints = np.random.randint(1, 2)
        postpick_pos = poses['postpick'].translation()
        place_pos = poses['place'].translation()
        diff = place_pos - postpick_pos
        xy_norm = np.linalg.norm(diff[:2])
        if xy_norm > 1e-6:
            side = np.array([-diff[1], diff[0]]) / xy_norm
        else:
            yaw = np.random.uniform(-np.pi, np.pi)
            forward = np.array([np.cos(yaw), np.sin(yaw)])
            side = np.array([-forward[1], forward[0]])
        alphas = np.linspace(0.0, 1.0, num_midpoints + 2)[1:-1]
        if np.random.uniform() < 0.35:
            alphas[-1] = np.random.uniform(1.15, 1.35) * noise_scale + 1.0 * (1.0 - noise_scale)
        else:
            alphas = np.sort(
                np.clip(alphas + np.random.uniform(-0.08, 0.08, size=num_midpoints) * noise_scale, 0.15, 0.85)
            )
        carry_side = np.random.choice([-1, 1])
        side_offset = np.random.uniform(0.04, 0.14) * noise_scale
        min_bounds, max_bounds = self._env.unwrapped._arm_sampling_bounds
        base_yaw = self.get_yaw(poses['postpick'])
        yaw_delta = self.yaw_error(self.get_yaw(poses['place']), base_yaw)
        for i, alpha in enumerate(alphas):
            name = f'clearance_{i}'
            midpoint_names.append(name)
            xy = (
                postpick_pos[:2]
                + alpha * diff[:2]
                + carry_side * side * side_offset
                + np.random.uniform(-0.02, 0.02, size=2) * noise_scale
            )
            z = max(pick_z, place_z) + np.random.uniform(0.03, 0.12)
            pos = np.clip(np.array([xy[0], xy[1], z]), min_bounds, max_bounds)
            yaw = base_yaw + alpha * yaw_delta + np.random.uniform(-0.4, 0.4) * noise_scale
            tilt = (1 - alpha) * pick_tilt + alpha * place_tilt
            if tilt_randomization:
                tilt = tilt + np.random.uniform(-0.05, 0.05, size=2) * noise_scale
            poses[name] = self.to_pose(pos=pos, yaw=yaw, roll=tilt[0], pitch=tilt[1])

        # Times.
        times = {}
        times['initial'] = 0.0
        times['pick'] = times['initial'] + self._dt
        times['pick_start'] = times['pick'] + self._dt * 1.8
        times['pick_end'] = times['pick_start'] + self._dt * np.random.uniform(0.75, 1.25)
        if use_pick_hold:
            times['pick_hold'] = times['pick_end'] + self._dt * np.random.uniform(0.1, 0.22)
        times['postpick'] = times[next(reversed(times))] + self._dt * np.random.uniform(0.75, 1.05)
        for name in midpoint_names:
            times[name] = times[next(reversed(times))] + self._dt
        times['place'] = times[next(reversed(times))] + self._dt
        times['place_start'] = times['place'] + self._dt * 1.5
        times['place_end'] = times['place_start'] + self._dt * np.random.uniform(0.75, 1.1)
        times['postplace'] = times['place_end'] + self._dt * np.random.uniform(0.8, 1.15)
        times['final'] = times['postplace'] + self._dt
        for name in times.keys():
            if name != 'initial':
                times[name] += np.random.uniform(-1, 1) * self._dt * 0.03

        # Grasps.
        g = 0.0
        grasps = {}
        if use_pick_hold:
            close_name = np.random.choice(['pick_end', 'pick_hold'], p=[0.65, 0.35])
        else:
            close_name = np.random.choice(['pick_start', 'pick_end'], p=[0.1, 0.9])
        open_name = 'place_end'
        for name in times.keys():
            if name == close_name:
                g = 1.0
            elif name == open_name:
                g = 0.0
            grasps[name] = g

        times = self._smooth_keyframe_times(times, poses)

        # Gates.
        self._gates = [
            self.gate(times['pick'], poses['pick']),
            self.gate(times['place'], poses['place']),
        ]
        pick_retreat_start = 'pick_hold' if use_pick_hold else 'pick_end'
        self._vertical_phases = [
            dict(
                start=times[start],
                end=times[end],
                xy_curve=np.random.uniform(
                    -0.025 * noise_scale,
                    0.025 * noise_scale,
                    size=(2, 2),
                ),
            )
            for start, end in [
                ('pick', 'pick_start'),
                (pick_retreat_start, 'postpick'),
                ('place', 'place_start'),
                ('place_end', 'postplace'),
            ]
        ]

        return times, poses, grasps

    def _reset_to_target(self, info, target):
        block = target['block']
        self._block = block
        self._target = dict(
            block=block,
            block_pos=np.asarray(target['block_pos']).copy(),
            block_yaw=target['block_yaw'],
        )
        self._pick_source_z = info[f'privileged/block_{block}_pos'][2]
        self._pick_checked = False
        effector_initial = self.to_pose(
            pos=info['proprio/effector_pos'],
            yaw=info['proprio/effector_yaw'][0],
        )
        if self._num_pick_retries > 0:
            effector_orientation = lie.SO3.from_matrix(
                self._env.unwrapped._data.site_xmat[self._env.unwrapped._pinch_site_id].copy().reshape(3, 3)
            )
            effector_initial = lie.SE3.from_rotation_and_translation(
                rotation=effector_orientation @ self._env.unwrapped._effector_down_rotation.inverse(),
                translation=info['proprio/effector_pos'],
            )
        plan_input = {
            'effector_initial': effector_initial,
            'effector_goal': self.sample_effector_goal(),
            'block_initial': self.to_pose(
                pos=info[f'privileged/block_{block}_pos'],
                yaw=info[f'privileged/block_{block}_yaw'][0],
            ),
            'block_goal': self.to_pose(
                pos=target['block_pos'],
                yaw=target['block_yaw'],
            ),
        }

        times, poses, grasps = self.compute_keyframes(plan_input)
        self._pick_check_time = times['postpick']
        self.reset_plan(info, times, poses, grasps)

    def reset(self, ob, info, target):
        self._num_pick_retries = 0
        self._reset_to_target(info, target)

    def _active_vertical_phase(self):
        for phase in self._vertical_phases:
            if phase['start'] <= self._plan_time <= phase['end']:
                return phase
        return None

    def select_action(self, ob, info):
        ab_action = self.current_plan_action(info)
        if (
            not self._pick_checked
            and self._plan_time >= self._pick_check_time
            and self._num_pick_retries < self._max_pick_retries
        ):
            self._pick_checked = True
            block_z = info[f'privileged/block_{self._block}_pos'][2]
            if block_z < self._pick_source_z + 0.025:
                self._num_pick_retries += 1
                self._reset_to_target(info, self._target)
                ab_action = self.current_plan_action(info)
        vertical_phase = self._active_vertical_phase()

        if vertical_phase is not None:
            u = (self._plan_time - vertical_phase['start']) / (vertical_phase['end'] - vertical_phase['start'])
            u = np.clip(u, 0.0, 1.0)
            c1, c2 = vertical_phase['xy_curve']
            ab_action[:2] += 16 * u**2 * (1 - u) ** 2 * ((1 - u) * c1 + u * c2)

        return self.plan_action_to_action(ab_action, info)
