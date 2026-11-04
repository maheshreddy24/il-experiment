import numpy as np

from ocbench.oracles.primitives.primitive import Primitive


class DrawerPrimitive(Primitive):
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
        self._smooth_plan_stop_names = {'initial', 'grasp_start', 'grasp_end', 'move', 'release', 'final'}

    def compute_keyframes(self, plan_input):
        # Poses.
        poses = {}
        drawer_initial = self.shortest_yaw(
            eff_yaw=self.get_yaw(plan_input['effector_initial']),
            obj_yaw=self.get_yaw(plan_input['drawer_initial']),
            translation=plan_input['drawer_initial'].translation(),
            n=2,
        )
        drawer_goal = self.shortest_yaw(
            eff_yaw=self.get_yaw(plan_input['effector_initial']),
            obj_yaw=self.get_yaw(plan_input['drawer_initial']),
            translation=plan_input['drawer_goal'].translation(),
            n=2,
        )
        yaw_offset = np.random.choice([0.0, np.pi], p=[0.85, 0.15])
        yaw_offset += np.random.uniform(-0.08, 0.08)
        if self._tilt_randomization:
            grasp_tilt = np.random.uniform(-0.06, 0.06, size=2)
            move_tilt = 0.7 * grasp_tilt + np.random.uniform(-0.04, 0.04, size=2)
        else:
            grasp_tilt = np.zeros(2)
            move_tilt = np.zeros(2)
        grasp_yaw = self.get_yaw(drawer_initial) + yaw_offset
        move_yaw = self.get_yaw(drawer_goal) + yaw_offset + np.random.uniform(-0.04, 0.04)
        drawer_initial = self.to_pose(
            pos=drawer_initial.translation(),
            yaw=grasp_yaw,
            roll=grasp_tilt[0],
            pitch=grasp_tilt[1],
        )
        drawer_goal = self.to_pose(
            pos=drawer_goal.translation(),
            yaw=move_yaw,
            roll=move_tilt[0],
            pitch=move_tilt[1],
        )
        move_delta = drawer_goal.translation() - drawer_initial.translation()
        move_dist = np.linalg.norm(move_delta[:2])
        if move_dist > 1e-6:
            slide_dir = move_delta[:2] / move_dist
        else:
            slide_dir = np.array([np.cos(grasp_yaw), np.sin(grasp_yaw)])
        side_dir = np.array([-slide_dir[1], slide_dir[0]])
        side_sign = np.random.choice([-1, 1])
        approach_offset = (
            side_sign * np.random.uniform(0.015, 0.04) * side_dir
            + np.random.uniform(-0.015, 0.015) * slide_dir
        )
        clearance_offset = (
            side_sign * np.random.uniform(0.015, 0.04) * side_dir
            + np.random.uniform(-0.015, 0.015) * slide_dir
        )
        poses['initial'] = plan_input['effector_initial']
        poses['approach'] = self.to_pose(
            pos=drawer_initial.translation() + np.array([*approach_offset, np.random.uniform(0.09, 0.14)]),
            yaw=grasp_yaw,
            roll=grasp_tilt[0],
            pitch=grasp_tilt[1],
        )
        poses['grasp_start'] = drawer_initial
        poses['grasp_end'] = drawer_initial
        midpoint_names = []
        num_midpoints = np.random.randint(1, 3)
        alphas = np.linspace(0.0, 1.0, num_midpoints + 2)[1:-1]
        alphas = np.sort(np.clip(alphas + np.random.uniform(-0.08, 0.08, size=num_midpoints), 0.2, 0.8))
        for i, alpha in enumerate(alphas):
            name = f'move_midpoint_{i}'
            midpoint_names.append(name)
            pos = drawer_initial.translation() + alpha * move_delta
            pos[:2] += side_dir * np.random.uniform(-0.005, 0.005)
            pos[2] += np.random.uniform(-0.003, 0.006)
            yaw = grasp_yaw + alpha * self.yaw_error(move_yaw, grasp_yaw) + np.random.uniform(-0.04, 0.04)
            if self._tilt_randomization:
                tilt = (1 - alpha) * grasp_tilt + alpha * move_tilt + np.random.uniform(-0.02, 0.02, size=2)
            else:
                tilt = np.zeros(2)
            poses[name] = self.to_pose(pos=pos, yaw=yaw, roll=tilt[0], pitch=tilt[1])
        poses['move'] = drawer_goal
        poses['release'] = drawer_goal
        poses['clearance'] = self.to_pose(
            pos=drawer_goal.translation() + np.array([*clearance_offset, np.random.uniform(0.09, 0.14)]),
            yaw=move_yaw,
            roll=move_tilt[0],
            pitch=move_tilt[1],
        )
        poses['final'] = plan_input['effector_goal']

        # Times.
        times = {}
        times['initial'] = 0.0
        times['approach'] = times['initial'] + self._dt
        times['grasp_start'] = times['approach'] + self._dt * 0.5
        times['grasp_end'] = times['grasp_start'] + self._dt * 0.5
        for name in midpoint_names:
            times[name] = times[next(reversed(times))] + self._dt * 0.5
        times['move'] = times[next(reversed(times))] + self._dt * 0.5
        times['release'] = times['move'] + self._dt * np.random.uniform(0.3, 0.7)
        times['clearance'] = times['release'] + self._dt * 0.5
        times['final'] = times['clearance'] + self._dt
        for time in times.keys():
            if time != 'initial':
                times[time] += np.random.uniform(-1, 1) * self._dt * 0.1

        # Grasps.
        grasps = {}
        g = 0.0
        for name in times.keys():
            if name in {'grasp_end', 'release'}:
                g = 1.0 - g
            grasps[name] = g

        times = self._smooth_keyframe_times(times, poses)

        # Gates.
        self._gates = [
            self.gate(times['approach'], poses['approach']),
            self.gate(times['grasp_start'], poses['grasp_start']),
        ]

        return times, poses, grasps

    def reset(self, ob, info, target):
        plan_input = {
            'effector_initial': self.to_pose(
                pos=info['proprio/effector_pos'],
                yaw=info['proprio/effector_yaw'][0],
            ),
            'effector_goal': self.sample_effector_goal(),
            'drawer_initial': self.to_pose(
                pos=info['privileged/drawer_handle_pos'],
                yaw=info['privileged/drawer_handle_yaw'][0],
            ),
            'drawer_goal': self.to_pose(
                pos=target['drawer_handle_pos'],
                yaw=info['privileged/drawer_handle_yaw'][0],
            ),
        }

        times, poses, grasps = self.compute_keyframes(plan_input)
        self.reset_plan(info, times, poses, grasps)
