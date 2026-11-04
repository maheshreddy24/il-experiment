import numpy as np

from ocbench.oracles.primitives.primitive import Primitive


class HanoiPrimitive(Primitive):
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
        self._smooth_plan_stop_names = {
            'initial',
            'pick_start',
            'pick_end',
            'extract',
            'insert',
            'place_start',
            'place_end',
            'final',
        }
        self._gate_pos_tol = 0.012
        self._max_xyz_delta = 0.018

    def compute_keyframes(self, plan_input):
        # Poses.
        poses = {}
        env = self._env.unwrapped
        disk_initial = plan_input['disk_initial']
        disk_goal = plan_input['disk_goal']
        grasp_yaw = self.get_yaw(disk_initial)
        target_yaw = self.get_yaw(disk_goal)
        clearance_z = max(
            2 * env._peg_half_height + 0.12,
            disk_initial.translation()[2] + 0.08,
            disk_goal.translation()[2] + 0.08,
        )
        extract_z = min(
            clearance_z,
            max(
                2 * env._peg_half_height + env._disk_half_height * 1.8 + 0.025,
                disk_initial.translation()[2] + 0.06,
                disk_goal.translation()[2] + 0.06,
            ),
        )

        poses['initial'] = plan_input['effector_initial']
        poses['lift'] = self.to_pose(
            pos=np.array([*poses['initial'].translation()[:2], clearance_z]),
            yaw=self.get_yaw(poses['initial']),
        )
        poses['pick'] = self.to_pose(
            pos=np.array([*disk_initial.translation()[:2], clearance_z]),
            yaw=grasp_yaw,
        )
        poses['pick_start'] = disk_initial
        poses['pick_end'] = disk_initial
        poses['extract'] = self.to_pose(
            pos=np.array([*disk_initial.translation()[:2], extract_z]),
            yaw=grasp_yaw,
        )
        poses['postpick'] = poses['pick']
        midpoint_names = []
        num_midpoints = np.random.randint(1, 3)
        carry_start = poses['postpick'].translation()
        carry_end = np.array([*disk_goal.translation()[:2], clearance_z])
        carry_delta = carry_end - carry_start
        xy_norm = np.linalg.norm(carry_delta[:2])
        if xy_norm > 1e-6:
            side_dir = np.array([-carry_delta[1], carry_delta[0]]) / xy_norm
        else:
            side_dir = np.array([-np.sin(grasp_yaw), np.cos(grasp_yaw)])
        yaw_delta = self.yaw_error(target_yaw, grasp_yaw)
        alphas = np.linspace(0.0, 1.0, num_midpoints + 2)[1:-1]
        alphas = np.sort(np.clip(alphas + np.random.uniform(-0.08, 0.08, size=num_midpoints), 0.2, 0.8))
        for i, alpha in enumerate(alphas):
            name = f'clearance_{i}'
            midpoint_names.append(name)
            pos = carry_start + alpha * carry_delta
            pos[:2] += side_dir * np.random.uniform(-0.05, 0.05)
            pos[2] = clearance_z + np.random.uniform(0.0, 0.04)
            yaw = grasp_yaw + alpha * yaw_delta + np.random.uniform(-0.1, 0.1)
            if self._tilt_randomization:
                tilt = np.random.uniform(-0.03, 0.03, size=2)
            else:
                tilt = np.zeros(2)
            poses[name] = self.to_pose(pos=pos, yaw=yaw, roll=tilt[0], pitch=tilt[1])
        poses['place'] = self.to_pose(
            pos=np.array([*disk_goal.translation()[:2], clearance_z]),
            yaw=target_yaw,
        )
        poses['insert'] = self.to_pose(
            pos=np.array([*disk_goal.translation()[:2], extract_z]),
            yaw=target_yaw,
        )
        release_z = disk_goal.translation()[2]
        peg_top_z = 2 * env._peg_half_height
        if release_z < peg_top_z:
            release_z = np.random.uniform(release_z, peg_top_z)
        poses['place_start'] = self.to_pose(
            pos=np.array([*disk_goal.translation()[:2], release_z]),
            yaw=target_yaw,
        )
        poses['place_end'] = poses['place_start']
        poses['postplace'] = poses['insert']
        poses['final'] = plan_input['effector_goal']

        # Times.
        times = {}
        times['initial'] = 0.0
        times['lift'] = times['initial'] + self._dt * 0.8
        times['pick'] = times['lift'] + self._dt
        times['pick_start'] = times['pick'] + self._dt * 1.4
        times['pick_end'] = times['pick_start'] + self._dt * 0.9
        times['extract'] = times['pick_end'] + self._dt
        times['postpick'] = times['extract'] + self._dt * 0.6
        carry_names = [*midpoint_names, 'place']
        for i, name in enumerate(carry_names):
            times[name] = times['postpick'] + self._dt * 2.0 * (i + 1) / len(carry_names)
        times['insert'] = times['place'] + self._dt * 0.8
        times['place_start'] = times['insert'] + self._dt * 1.2
        times['place_end'] = times['place_start'] + self._dt * 0.9
        times['postplace'] = times['place_end'] + self._dt * 0.8
        times['final'] = times['postplace'] + self._dt
        for time in times.keys():
            if time != 'initial':
                times[time] += np.random.uniform(-1, 1) * self._dt * 0.05

        # Grasps.
        g = 0.0
        grasp = plan_input['grasp']
        grasps = {}
        for name in times.keys():
            if name == 'pick_end':
                g = grasp
            elif name == 'place_end':
                g = 0.0
            grasps[name] = g

        times = self._smooth_keyframe_times(times, poses)

        # Gates.
        self._gates = [
            self.gate(times['pick'], poses['pick'], pos_tol=self._gate_pos_tol),
            self.gate(times['pick_start'], poses['pick_start'], pos_tol=self._gate_pos_tol),
            self.gate(times['extract'], poses['extract'], pos_tol=self._gate_pos_tol),
            self.gate(times['place'], poses['place'], pos_tol=self._gate_pos_tol),
            self.gate(times['insert'], poses['insert'], pos_tol=self._gate_pos_tol),
            self.gate(times['place_start'], poses['place_start'], pos_tol=self._gate_pos_tol),
        ]
        self._vertical_phases = [
            dict(
                start=times['pick_end'],
                end=times['extract'],
                xy=poses['extract'].translation()[:2].copy(),
                yaw=grasp_yaw,
                xy_curve=np.random.uniform(
                    -0.01,
                    0.01,
                    size=(2, 2),
                ),
            ),
            dict(
                start=times['insert'],
                end=times['place_start'],
                xy=poses['insert'].translation()[:2].copy(),
                yaw=target_yaw,
                xy_curve=np.random.uniform(
                    -0.01,
                    0.01,
                    size=(2, 2),
                ),
            ),
        ]

        return times, poses, grasps

    def reset(self, ob, info, target):
        env = self._env.unwrapped
        disk = target['disk']
        grasp_z_offset = env._disk_half_height * 0.8
        grasp_sign = int(np.random.choice([-1, 1]))
        grasp_fraction = np.random.uniform(0.7, 0.85)
        disk_initial_pos = info[f'privileged/disk_{disk}_pos'].copy()
        disk_initial_yaw = info[f'privileged/disk_{disk}_yaw'][0]
        disk_initial_pos[:2] += env._disk_grasp_offset(disk, disk_initial_yaw, grasp_sign, grasp_fraction)
        disk_initial_pos[2] += grasp_z_offset
        disk_initial = self.shortest_yaw(
            eff_yaw=info['proprio/effector_yaw'][0],
            obj_yaw=disk_initial_yaw + np.pi / 2,
            translation=disk_initial_pos,
            n=2,
        )
        disk_initial = self.to_pose(
            pos=disk_initial.translation(),
            yaw=self.get_yaw(disk_initial) + np.random.uniform(-0.02, 0.02),
        )
        disk_goal_pos = target['disk_goal_pos'].copy()
        disk_goal_yaw = target['disk_goal_yaw']
        disk_goal_pos[:2] += env._disk_grasp_offset(disk, disk_goal_yaw, grasp_sign, grasp_fraction)
        disk_goal_pos[2] += grasp_z_offset
        disk_goal = self.shortest_yaw(
            eff_yaw=self.get_yaw(disk_initial),
            obj_yaw=disk_goal_yaw + np.pi / 2,
            translation=disk_goal_pos,
            n=2,
        )

        plan_input = {
            'effector_initial': self.to_pose(
                pos=info['proprio/effector_pos'],
                yaw=info['proprio/effector_yaw'][0],
            ),
            'effector_goal': self.sample_effector_goal(),
            'disk_initial': disk_initial,
            'disk_goal': disk_goal,
            'grasp': target['disk_grasp'],
        }

        times, poses, grasps = self.compute_keyframes(plan_input)
        self.reset_plan(info, times, poses, grasps)

    def _active_vertical_phase(self):
        for phase in self._vertical_phases:
            if phase['start'] <= self._plan_time <= phase['end']:
                return phase
        return None

    def select_action(self, ob, info):
        ab_action = self.current_plan_action(info)
        vertical_phase = self._active_vertical_phase()

        if vertical_phase is not None:
            u = (self._plan_time - vertical_phase['start']) / (vertical_phase['end'] - vertical_phase['start'])
            u = np.clip(u, 0.0, 1.0)
            c1, c2 = vertical_phase['xy_curve']
            xy_offset = 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u**2 * c2
            ab_action[:2] = vertical_phase['xy'] + xy_offset
            ab_action[3] = vertical_phase['yaw']
        action = np.zeros(5)
        action[:3] = ab_action[:3] - info['proprio/effector_pos']
        if vertical_phase is not None:
            action[2] = np.clip(action[2], -0.65 * self._max_xyz_delta, 0.65 * self._max_xyz_delta)
            xy_norm = np.linalg.norm(action[:2])
            xy_budget = np.sqrt(max(self._max_xyz_delta**2 - action[2] ** 2, 0.0))
            if xy_norm > xy_budget:
                action[:2] = action[:2] / xy_norm * xy_budget
        else:
            norm = np.linalg.norm(action[:3])
            if norm > self._max_xyz_delta:
                action[:3] = action[:3] / norm * self._max_xyz_delta
        action[3] = self.yaw_error(ab_action[3], info['proprio/effector_yaw'][0])
        action[4] = ab_action[4] - info['proprio/gripper_opening'][0]

        return self.ee_delta_to_action(action, info, self._current_plan_rotation)
