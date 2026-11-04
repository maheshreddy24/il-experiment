import numpy as np

from ocbench import lie
from ocbench.oracles.primitives.primitive import Primitive


class BowlingPrimitive(Primitive):
    def __init__(
        self,
        env,
        segment_dt=1.0,
    ):
        super().__init__(
            env=env,
            segment_dt=segment_dt,
        )
        self._phase = None
        self._phase_steps = 0
        self._ball_start_pos = None
        self._max_move_delta = 0.018
        self._max_push_delta = 0.045

    @property
    def push_speed(self):
        return self._push_speed

    def reset(self, ob, info):
        self._sample_episode_params(info)
        self._done = False
        self._phase = 'prepush'
        self._phase_steps = 0

    def _sample_episode_params(self, info=None):
        self._push_speed = np.random.uniform(0.03, 0.056)
        self._push_steps = int(np.random.randint(130, 221))
        self._push_ramp_steps = int(np.random.randint(30, 66))
        self._prepush_dist = np.random.uniform(0.10, 0.14)
        self._push_z = np.random.uniform(0.048, 0.064)
        self._gripper = np.random.uniform(0.06, 0.20)
        self._side_offset = np.random.uniform(-0.025, 0.025)
        angle_noise = np.random.uniform(-0.10, 0.10)
        yaw_noise = np.random.uniform(-0.08, 0.08)
        target_x_offset = np.random.uniform(-0.18, 0.18)

        ball_pos = info['privileged/ball_pos'] if info is not None else self._env.unwrapped._ball_init_pos
        pin_head_pos = (
            info['privileged/pin_0_pos'][:2]
            if info is not None and 'privileged/pin_0_pos' in info
            else self._env.unwrapped._pin_head_pos
        )
        self._pin_head_pos = pin_head_pos.copy()
        target_x = pin_head_pos[0] + target_x_offset
        angle = np.arctan2(target_x - ball_pos[0], pin_head_pos[1] - ball_pos[1]) + angle_noise
        self._ball_start_pos = ball_pos.copy()
        self._push_angle = np.clip(angle, -0.45, 0.45)
        self._push_dir = np.array([np.sin(self._push_angle), np.cos(self._push_angle)])
        self._side_dir = np.array([self._push_dir[1], -self._push_dir[0]])
        self._yaw = np.pi / 2 + self._push_angle + yaw_noise
        self._tilt = np.random.uniform(-0.08, 0.08, size=2)
        self._tilt_drift = np.random.uniform(-0.04, 0.04, size=2)

    def _push_yaw(self):
        return self._yaw

    def _target_tilt(self):
        if self._phase == 'push':
            u = np.clip((self._phase_steps + 1) / max(self._push_steps, 1), 0.0, 1.0)
            u = u * u * (3 - 2 * u)
            return self._tilt + u * self._tilt_drift
        return self._tilt

    def _target_rotation(self):
        tilt = self._target_tilt()
        return lie.SO3.from_rpy_radians(
            roll=tilt[0],
            pitch=tilt[1],
            yaw=self._push_yaw(),
        )

    def target_info(self):
        return dict(
            ball_pos=self._ball_start_pos.copy(),
            pin_head_pos=self._pin_head_pos.copy(),
            push_angle=float(self._push_angle),
            push_dir=self._push_dir.copy(),
            push_speed=float(self._push_speed),
            push_steps=self._push_steps,
            push_ramp_steps=self._push_ramp_steps,
            push_z=float(self._push_z),
            gripper=float(self._gripper),
            tilt=self._tilt.copy(),
            tilt_drift=self._tilt_drift.copy(),
        )

    def _limit_xyz_delta(self, action, max_xyz_delta):
        norm = np.linalg.norm(action[:3])
        if norm > max_xyz_delta:
            action[:3] *= max_xyz_delta / norm
        return action

    def _move_action(self, info, target_pos, gripper, max_xyz_delta=None):
        if max_xyz_delta is None:
            max_xyz_delta = self._max_move_delta
        raw_action = np.zeros(5)
        target_pos = np.asarray(target_pos).copy()
        target_pos[1] = min(target_pos[1], -0.015)
        raw_action[:3] = target_pos - info['proprio/effector_pos']
        raw_action = self._limit_xyz_delta(raw_action, max_xyz_delta)
        raw_action[3] = np.clip(
            self.yaw_error(self._push_yaw(), info['proprio/effector_yaw'][0]),
            -0.3,
            0.3,
        )
        raw_action[4] = np.clip(gripper - info['proprio/gripper_opening'][0], -1.0, 1.0)
        return self.ee_delta_to_action(raw_action, info, self._target_rotation())

    def _push_action(self, info, speed, gripper):
        raw_action = np.zeros(5)
        raw_action[:2] = speed * self._push_dir
        raw_action[1] = min(raw_action[1], -0.015 - info['proprio/effector_pos'][1])
        raw_action = self._limit_xyz_delta(raw_action, self._max_push_delta)
        raw_action[3] = np.clip(
            self.yaw_error(self._push_yaw(), info['proprio/effector_yaw'][0]),
            -0.3,
            0.3,
        )
        raw_action[4] = np.clip(gripper - info['proprio/gripper_opening'][0], -1.0, 1.0)
        return self.ee_delta_to_action(raw_action, info, self._target_rotation())

    def _ready(self, info, target_pos):
        return (
            np.linalg.norm(info['proprio/effector_pos'] - target_pos) <= 0.012
            and abs(self.yaw_error(self._yaw, info['proprio/effector_yaw'][0])) <= 0.08
            and abs(self._gripper - info['proprio/gripper_opening'][0]) <= 0.08
        )

    def _push_poses(self, ball_pos):
        xy = ball_pos[:2] - self._prepush_dist * self._push_dir + self._side_offset * self._side_dir
        prepush_pos = np.array([xy[0], xy[1], 0.18])
        push_pos = np.array([xy[0], xy[1], self._push_z])
        return prepush_pos, push_pos

    def select_action(self, ob, info):
        if self._done:
            return np.zeros(self._env.action_space.shape)

        prepush_pos, push_pos = self._push_poses(self._ball_start_pos)

        if self._phase == 'prepush':
            if self._ready(info, prepush_pos):
                self._phase = 'lower'
                self._phase_steps = 0
            action = self._move_action(info, prepush_pos, self._gripper)
        elif self._phase == 'lower':
            if self._ready(info, push_pos):
                self._phase = 'preload'
                self._phase_steps = 0
            action = self._move_action(info, push_pos, self._gripper, max_xyz_delta=0.012)
        elif self._phase == 'preload':
            if self._phase_steps >= 25:
                self._phase = 'push'
                self._phase_steps = 0
            action = self._push_action(info, 0.014, self._gripper)
        elif self._phase == 'push':
            if self._phase_steps >= self._push_steps or info['proprio/effector_pos'][1] >= -0.015 - 1e-3:
                self._phase = 'settle'
                self._phase_steps = 0
            push_scale = min(1.0, (self._phase_steps + 1) / self._push_ramp_steps)
            action = self._push_action(info, self._push_speed * push_scale, self._gripper)
        elif self._phase == 'settle':
            if self._phase_steps >= 220:
                self._done = True
            action = self._move_action(info, info['proprio/effector_pos'], self._gripper, max_xyz_delta=0.01)
        else:
            raise ValueError(f'Unknown BowlingPrimitive phase: {self._phase}')

        self._phase_steps += 1
        return action
