import numpy as np

from ocbench.oracles.primitives.primitive import Primitive


class ButtonPrimitive(Primitive):
    def __init__(
        self,
        env,
        segment_dt=1.0,
        target_peak_xyz_speed=0.3,
        gripper_always_closed=False,
    ):
        super().__init__(
            env=env,
            segment_dt=segment_dt,
            target_peak_xyz_speed=target_peak_xyz_speed,
        )
        self._smooth_plan_stop_names = {'initial', 'press_start', 'press', 'press_end', 'final'}
        self._gripper_always_closed = gripper_always_closed

    def compute_keyframes(self, plan_input):
        # Poses.
        poses = {}
        button_pos = plan_input['button'].translation()
        button_yaw = self.get_yaw(plan_input['button'])
        yaw_offset = np.random.choice([-np.pi / 2, 0.0, np.pi / 2], p=[0.15, 0.7, 0.15])
        yaw_offset += np.random.uniform(-0.08, 0.08)
        press_yaw = button_yaw + yaw_offset
        if self._tilt_randomization:
            press_tilt = np.random.uniform(-0.015, 0.015, size=2)
            exit_tilt = 0.7 * press_tilt + np.random.uniform(-0.01, 0.01, size=2)
        else:
            press_tilt = np.zeros(2)
            exit_tilt = np.zeros(2)
        press_xy_offset = np.random.uniform(-0.0015, 0.0015, size=2)
        press_height = np.random.uniform(-0.026, -0.024)
        start_height = np.random.uniform(0.05, 0.08)
        press_start_pos = button_pos + np.array([*press_xy_offset, start_height])
        press_pos = button_pos + np.array([*press_xy_offset, press_height])
        press_end_pos = button_pos + np.array(
            [
                *(press_xy_offset + np.random.uniform(-0.001, 0.001, size=2)),
                start_height,
            ]
        )
        poses['initial'] = plan_input['effector_initial']
        midpoint_names = []
        num_midpoints = np.random.randint(0, 2)
        if num_midpoints:
            name = 'approach_midpoint_0'
            midpoint_names.append(name)
            alpha = np.random.uniform(0.35, 0.65)
            pos = (1 - alpha) * poses['initial'].translation() + alpha * press_start_pos
            pos[:2] += np.random.uniform(-0.015, 0.015, size=2)
            pos[2] = max(pos[2], button_pos[2] + np.random.uniform(0.08, 0.12))
            initial_yaw = self.get_yaw(poses['initial'])
            yaw = initial_yaw + alpha * self.yaw_error(press_yaw, initial_yaw)
            if self._tilt_randomization:
                tilt = alpha * press_tilt + np.random.uniform(-0.02, 0.02, size=2)
            else:
                tilt = np.zeros(2)
            poses[name] = self.to_pose(pos=pos, yaw=yaw, roll=tilt[0], pitch=tilt[1])
        poses['press_start'] = self.to_pose(
            pos=press_start_pos,
            yaw=press_yaw,
            roll=press_tilt[0],
            pitch=press_tilt[1],
        )
        poses['press'] = self.to_pose(
            pos=press_pos,
            yaw=press_yaw + np.random.uniform(-0.01, 0.01),
            roll=press_tilt[0],
            pitch=press_tilt[1],
        )
        poses['press_end'] = self.to_pose(
            pos=press_end_pos,
            yaw=press_yaw + np.random.uniform(-0.015, 0.015),
            roll=exit_tilt[0],
            pitch=exit_tilt[1],
        )
        poses['final'] = plan_input['effector_goal']

        # Times.
        times = {}
        distance = np.linalg.norm(poses['initial'].translation() - poses['press_start'].translation())
        times['initial'] = 0.0
        for name in midpoint_names:
            times[name] = times[next(reversed(times))] + self._dt * 0.5
        times['press_start'] = times[next(reversed(times))] + self._dt * (0.5 + distance * 4)
        times['press'] = times['press_start'] + self._dt * np.random.uniform(0.7, 0.9)
        times['press_end'] = times['press'] + self._dt * np.random.uniform(0.7, 0.9)
        times['final'] = times['press_end'] + self._dt * np.random.uniform(1.1, 1.4)
        for time in times.keys():
            if time != 'initial':
                times[time] += np.random.uniform(-1, 1) * self._dt * 0.1

        # Grasps.
        grasps = {}
        if self._gripper_always_closed:
            g = 1.0
        else:
            g = 0.0
        for name in times.keys():
            if not self._gripper_always_closed:
                if name in {'press_start', 'final'}:
                    g = 1.0 - g
            grasps[name] = g

        times = self._smooth_keyframe_times(times, poses)

        # Gates.
        self._gates = [
            self.gate(times['press_start'], poses['press_start']),
        ]

        return times, poses, grasps

    def reset(self, ob, info, target):
        plan_input = {
            'effector_initial': self.to_pose(
                pos=info['proprio/effector_pos'],
                yaw=info['proprio/effector_yaw'][0],
            ),
            'effector_goal': self.sample_effector_goal(),
            'button': self.to_pose(
                pos=target['button_top_pos'],
                yaw=target['button_top_yaw'],
            ),
        }

        times, poses, grasps = self.compute_keyframes(plan_input)
        self.reset_plan(info, times, poses, grasps)
