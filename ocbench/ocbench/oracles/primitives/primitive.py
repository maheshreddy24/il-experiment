import numpy as np

from ocbench import lie


class Primitive:
    def __init__(
        self,
        env,
        segment_dt=1.0,
        target_peak_xyz_speed=None,
    ):
        self._env = env
        self._env_dt = self._env.unwrapped._control_timestep
        self._dt = segment_dt
        self._target_peak_xyz_speed = target_peak_xyz_speed
        self._smooth_plan_stop_names = None
        self._tilt_randomization = not self._env.unwrapped._lite
        self._ee_delta_high = self._env.unwrapped._ee_action_delta.copy()
        self._ee_delta_low = -self._ee_delta_high

        self._done = False
        self._t_max = None
        self._plan = None
        self._plan_rotations = None
        self._plan_keyframe_names = None
        self._current_plan_rotation = None
        self._plan_time = 0.0
        self._last_time = None
        self._gates = []

    @property
    def done(self):
        return self._done

    def above(self, pose, z):
        return (
            lie.SE3.from_rotation_and_translation(
                rotation=lie.SO3.identity(),
                translation=np.array([0.0, 0.0, z]),
            )
            @ pose
        )

    def to_pose(self, pos, yaw, roll=0.0, pitch=0.0):
        return lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_rpy_radians(roll=roll, pitch=pitch, yaw=yaw),
            translation=pos,
        )

    def get_yaw(self, pose):
        yaw = pose.rotation().compute_yaw_radians()
        if yaw < 0.0:
            return yaw + 2 * np.pi
        return yaw

    def shortest_yaw(self, eff_yaw, obj_yaw, translation, n=4):
        symmetries = np.array([i * 2 * np.pi / n + obj_yaw for i in range(-n, n + 1)])
        d = np.argmin(np.abs(eff_yaw - symmetries))
        return lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_z_radians(symmetries[d]),
            translation=translation,
        )

    def sample_effector_goal(self):
        return self.to_pose(
            pos=np.random.uniform(*self._env.unwrapped._arm_sampling_bounds),
            yaw=np.random.uniform(-np.pi, np.pi),
        )

    def gate(self, time, pose, pos_tol=0.04):
        return dict(time=time, xyz=pose.translation().copy(), pos_tol=pos_tol, passed=False)

    def _smooth_keyframe_times(self, times, poses):
        if self._target_peak_xyz_speed is None:
            return times

        names = list(times.keys())
        new_times = {names[0]: 0.0}
        peak_xyz_speed = self._target_peak_xyz_speed / self._dt
        for name, next_name in zip(names[:-1], names[1:]):
            dt = times[next_name] - times[name]
            dist = np.linalg.norm(poses[next_name].translation() - poses[name].translation())
            min_dt = 1.875 * dist / peak_xyz_speed
            new_times[next_name] = new_times[name] + max(dt, min_dt)
        return new_times

    def reset_plan(self, info, times, poses, grasps):
        self._plan_keyframe_names = list(times.keys())
        poses = [poses[name] for name in times.keys()]
        grasps = [grasps[name] for name in times.keys()]
        times = list(times.values())

        self._t_max = times[-1]
        self._done = False
        self._plan = self.compute_plan(times, poses, grasps)

    def compute_plan(self, times, poses, grasps):
        if self._smooth_plan_stop_names is not None:
            return self._compute_smooth_plan(times, poses, grasps)

        xyzs = [p.translation() for p in poses]
        quats = [p.rotation() for p in poses]

        # Curves.
        curve_offsets = []
        for i in range(len(times) - 1):
            p0, p1 = xyzs[i], xyzs[i + 1]
            diff = p1 - p0
            dist = np.linalg.norm(diff)
            xy_norm = np.linalg.norm(diff[:2])
            if xy_norm > 0.05:
                side = np.array([-diff[1], diff[0]]) / xy_norm
                sign = np.random.choice([-1, 1])
                curve = min(0.08, 0.45 * dist) * sign
                offset1 = np.array(
                    [
                        *(side * curve * np.random.uniform(0.7, 1.2)),
                        np.random.uniform(0.0, 0.03),
                    ]
                )
                offset2 = np.array(
                    [
                        *(side * curve * np.random.uniform(0.7, 1.2)),
                        np.random.uniform(0.0, 0.03),
                    ]
                )
            else:
                offset1 = np.zeros(3)
                offset2 = np.zeros(3)
            curve_offsets.append((offset1, offset2))

        # Plan.
        plan = []
        plan_rotations = []
        for step in range(int(np.ceil(self._t_max / self._env_dt))):
            t = step * self._env_dt
            s = np.searchsorted(times, t, side='right') - 1
            s = np.clip(s, 0, len(times) - 2)
            u = (t - times[s]) / (times[s + 1] - times[s])
            u = np.clip(u, 0.0, 1.0)
            u = u**3 * (10 - 15 * u + 6 * u**2)
            u = np.clip(u, 0.0, 1.0)
            p0, p3 = xyzs[s], xyzs[s + 1]
            p1 = p0 + (p3 - p0) / 3 + curve_offsets[s][0]
            p2 = p0 + 2 * (p3 - p0) / 3 + curve_offsets[s][1]
            xyz = (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u**2 * p2 + u**3 * p3

            action = np.zeros(5)
            action[:3] = xyz
            rotation = lie.interpolate(quats[s], quats[s + 1], u)
            action[3] = rotation.compute_yaw_radians()
            action[4] = (1 - u) * grasps[s] + u * grasps[s + 1]
            plan.append(action)
            plan_rotations.append(rotation)

        plan = np.array(plan)
        self._plan_rotations = plan_rotations
        self._plan_time = 0.0
        self._last_time = None
        return plan

    def _compute_smooth_plan(self, times, poses, grasps):
        xyzs = [p.translation() for p in poses]
        quats = [p.rotation() for p in poses]
        names = self._plan_keyframe_names
        if names is None or len(names) != len(times):
            names = [None] * len(times)

        tangents = []
        for i, name in enumerate(names):
            if i == 0 or i == len(times) - 1 or name in self._smooth_plan_stop_names:
                tangents.append(np.zeros(3))
                continue

            tangent = (xyzs[i + 1] - xyzs[i - 1]) / (times[i + 1] - times[i - 1])
            prev_speed = np.linalg.norm((xyzs[i] - xyzs[i - 1]) / (times[i] - times[i - 1]))
            next_speed = np.linalg.norm((xyzs[i + 1] - xyzs[i]) / (times[i + 1] - times[i]))
            max_speed = 1.25 * max(prev_speed, next_speed)
            speed = np.linalg.norm(tangent)
            if speed > max_speed:
                tangent *= max_speed / speed
            tangents.append(tangent)

        plan = []
        plan_rotations = []
        for step in range(int(np.ceil(self._t_max / self._env_dt))):
            t = step * self._env_dt
            s = np.searchsorted(times, t, side='right') - 1
            s = np.clip(s, 0, len(times) - 2)
            dt = times[s + 1] - times[s]
            u = (t - times[s]) / dt
            u = np.clip(u, 0.0, 1.0)

            p0, p1 = xyzs[s], xyzs[s + 1]
            m0, m1 = tangents[s], tangents[s + 1]
            u2 = u * u
            u3 = u2 * u
            xyz = (
                (2 * u3 - 3 * u2 + 1) * p0
                + (u3 - 2 * u2 + u) * dt * m0
                + (-2 * u3 + 3 * u2) * p1
                + (u3 - u2) * dt * m1
            )

            rot_u = u**3 * (10 - 15 * u + 6 * u**2)
            rot_u = np.clip(rot_u, 0.0, 1.0)
            rotation = lie.interpolate(quats[s], quats[s + 1], rot_u)

            action = np.zeros(5)
            action[:3] = xyz
            action[3] = rotation.compute_yaw_radians()
            action[4] = (1 - rot_u) * grasps[s] + rot_u * grasps[s + 1]
            plan.append(action)
            plan_rotations.append(rotation)

        plan = np.array(plan)
        self._plan_rotations = plan_rotations
        self._plan_time = 0.0
        self._last_time = None
        return plan

    def current_plan_action(self, info):
        time = info['time'][0]
        if self._last_time is not None:
            self._plan_time += max(time - self._last_time, 0.0)
        self._last_time = time

        blocked_gate = None
        for gate in self._gates:
            if not gate['passed'] and self._plan_time >= gate['time']:
                if np.linalg.norm(info['proprio/effector_pos'] - gate['xyz']) > gate['pos_tol']:
                    blocked_gate = gate
                    self._plan_time = gate['time']
                else:
                    gate['passed'] = True
                break

        cur_plan_idx = int((self._plan_time + 1e-7) // self._env_dt)
        if cur_plan_idx >= len(self._plan) - 1:
            cur_plan_idx = len(self._plan) - 1
            self._done = True

        ab_action = self._plan[cur_plan_idx].copy()
        self._current_plan_rotation = self._plan_rotations[cur_plan_idx]
        if blocked_gate is not None:
            ab_action[:3] = blocked_gate['xyz']
        return ab_action

    def yaw_error(self, target_yaw, yaw):
        return (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi

    def _limit_effector_rotation(self, effector_rotation):
        env = self._env.unwrapped
        current_effector_orientation = lie.SO3.from_matrix(
            env._data.site_xmat[env._pinch_site_id].copy().reshape(3, 3)
        )
        current_effector_rotation = current_effector_orientation @ env._effector_down_rotation.inverse()
        rotation_delta = (current_effector_rotation.inverse() @ effector_rotation).log()
        rotation_norm = np.linalg.norm(rotation_delta)
        max_rotation_delta = self._ee_delta_high[3]
        if rotation_norm > max_rotation_delta:
            effector_rotation = current_effector_rotation @ lie.SO3.exp(
                rotation_delta / rotation_norm * max_rotation_delta
            )
        return effector_rotation

    def ee_delta_to_action(self, action, info, effector_rotation=None):
        action = np.clip(action, self._ee_delta_low, self._ee_delta_high)
        target_effector_translation = info['proprio/effector_pos'] + action[:3]
        target_gripper_opening = info['proprio/gripper_opening'][0] + action[4]

        if effector_rotation is not None:
            return self._env.unwrapped.joint_position_action(
                target_effector_translation,
                None,
                target_gripper_opening,
                effector_rotation=self._limit_effector_rotation(effector_rotation),
            )

        target_effector_yaw = info['proprio/effector_yaw'][0] + action[3]
        return self._env.unwrapped.joint_position_action(
            target_effector_translation,
            target_effector_yaw,
            target_gripper_opening,
        )

    def plan_action_to_action(self, ab_action, info):
        action = np.zeros(5)
        action[:3] = ab_action[:3] - info['proprio/effector_pos']
        action[3] = self.yaw_error(ab_action[3], info['proprio/effector_yaw'][0])
        action[4] = ab_action[4] - info['proprio/gripper_opening'][0]
        return self.ee_delta_to_action(action, info, self._current_plan_rotation)

    def select_action(self, ob, info):
        ab_action = self.current_plan_action(info)
        return self.plan_action_to_action(ab_action, info)
