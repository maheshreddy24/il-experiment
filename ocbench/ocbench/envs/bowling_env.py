import mujoco
import numpy as np
from dm_control import mjcf

from ocbench import lie
from ocbench.envs.manipulation_env import ManipulationEnv


class BowlingEnv(ManipulationEnv):
    def __init__(
        self,
        task_id=1,
        **kwargs,
    ):
        self._ball_radius = 0.0425
        self._pin_radius = 0.0228
        self._pin_half_height = 0.075
        self._workspace_half_size = 1.0
        self._ball_init_pos = np.array([0.425, -0.38, self._ball_radius])
        self._ball_init_xy_randomization = np.array([[-0.15, -0.1], [0.15, 0.1]])
        self._pin_head_pos = np.array([0.425, 0.28])
        self._pin_rack_xy_randomization = np.array([[-0.1, -0.1], [0.1, 0.1]])
        self._pin_spacing = 0.08
        self._num_pins = 10
        self._max_episode_steps = 500
        self._elapsed_steps = 0
        self._pin_knocked = np.zeros(self._num_pins, dtype=bool)
        self._prev_num_knocked_pins = 0
        self._pin_init_xys = self._make_pin_init_xys()

        super().__init__(
            task_id=task_id,
            **kwargs,
        )

        self._arm_sampling_bounds = np.asarray([[0.20, -0.55, 0.20], [0.65, -0.20, 0.35]])
        self._object_sampling_bounds = np.asarray([[0.30, -0.45], [0.55, 0.35]])
        self._target_sampling_bounds = np.asarray([[0.28, -0.45], [0.57, 0.38]])

    def _make_pin_init_xys(self):
        x0, y0 = self._pin_head_pos
        xys = []
        for row in range(4):
            y = y0 + row * self._pin_spacing
            for col in range(row + 1):
                x = x0 + (col - row / 2) * self._pin_spacing
                xys.append([x, y])
        return np.asarray(xys)

    def set_tasks(self):
        self.task_infos = [
            dict(
                task_name='knock_down',
            )
        ]

    def add_objects(self, arena_mjcf):
        for wall in arena_mjcf.find_all('geom'):
            if wall.name in {f'wall_{i}' for i in range(4)}:
                wall.rgba = (0.0, 0.0, 0.0, 0.0)
                wall.conaffinity = 0

        # Add bowling defaults.
        bowling_outer_mjcf = mjcf.from_path((self._desc_dir / 'bowling_outer.xml').as_posix())
        arena_mjcf.include_copy(bowling_outer_mjcf)

        # Add object boundary walls.
        lower = np.array([-self._workspace_half_size, -self._workspace_half_size])
        upper = np.array([self._workspace_half_size, self._workspace_half_size])
        wall_thickness = 0.01
        wall_height = 0.15
        wall_rgba = (0.0, 0.0, 0.0, 0.0)
        wall_kwargs = dict(type='box', rgba=wall_rgba, contype=0, conaffinity=2, group=3)
        x_center = (lower[0] + upper[0]) / 2
        y_center = (lower[1] + upper[1]) / 2
        x_half = (upper[0] - lower[0]) / 2 + wall_thickness
        y_half = (upper[1] - lower[1]) / 2 + wall_thickness
        z_center = wall_height / 2
        arena_mjcf.worldbody.add(
            'geom',
            name='bowling_wall_x_low',
            pos=(lower[0] - wall_thickness, y_center, z_center),
            size=(wall_thickness, y_half, wall_height / 2),
            **wall_kwargs,
        )
        arena_mjcf.worldbody.add(
            'geom',
            name='bowling_wall_x_high',
            pos=(upper[0] + wall_thickness, y_center, z_center),
            size=(wall_thickness, y_half, wall_height / 2),
            **wall_kwargs,
        )
        arena_mjcf.worldbody.add(
            'geom',
            name='bowling_wall_y_low',
            pos=(x_center, lower[1] - wall_thickness, z_center),
            size=(x_half, wall_thickness, wall_height / 2),
            **wall_kwargs,
        )
        arena_mjcf.worldbody.add(
            'geom',
            name='bowling_wall_y_high',
            pos=(x_center, upper[1] + wall_thickness, z_center),
            size=(x_half, wall_thickness, wall_height / 2),
            **wall_kwargs,
        )
        arena_mjcf.worldbody.add(
            'geom',
            name='bowling_gripper_y_wall',
            type='plane',
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 1.0, 0.0, 0.0),
            size=(1.0, 1.0, 0.01),
            rgba=(0.0, 0.0, 0.0, 0.0),
            contype=4,
            conaffinity=0,
            group=3,
        )

        # Add ball.
        ball_body = arena_mjcf.worldbody.add('body', name='ball', pos=self._ball_init_pos)
        ball_body.add('freejoint', name='ball_joint')
        ball_body.add('geom', name='ball', dclass='bowling_ball', size=(self._ball_radius,))
        ball_body.add('site', name='ball_center', dclass='bowling_ball')

        # Add pins.
        self._pin_geoms = []
        for i, xy in enumerate(self._pin_init_xys):
            pin_pos = np.array([xy[0], xy[1], self._pin_half_height])
            pin_body = arena_mjcf.worldbody.add('body', name=f'pin_{i}', pos=pin_pos)
            pin_body.add('freejoint', name=f'pin_joint_{i}')
            pin_geom = pin_body.add(
                'geom',
                name=f'pin_{i}',
                dclass='bowling_pin',
            )
            pin_body.add('geom', name=f'pin_stripe_{i}', dclass='bowling_pin_stripe')
            pin_body.add('site', name=f'pin_top_{i}', pos=(0.0, 0.0, self._pin_half_height), dclass='bowling_pin')
            self._pin_geoms.append(pin_geom)

        # Add cameras.
        cameras = {
            'front': {
                'pos': (2.151, 0.000, 0.823),
                'xyaxes': (0.000, 1.000, 0.000, -0.342, 0.000, 0.940),
            },
            'front_pixels': {
                'pos': (1.053, -0.014, 0.639),
                'xyaxes': (0.000, 1.000, 0.000, -0.628, 0.001, 0.778),
            },
            'side': {
                'pos': (1.226, -1.603, 0.823),
                'xyaxes': (0.866, 0.500, -0.000, -0.171, 0.296, 0.940),
            },
        }
        for camera_name, camera_kwargs in cameras.items():
            arena_mjcf.worldbody.add('camera', name=camera_name, **camera_kwargs)

    def post_compilation_objects(self):
        self._ball_geom_id = self._model.geom('ball').id
        self._ball_body_id = self._model.body('ball').id
        self._pin_geom_ids = [self._model.geom(f'pin_{i}').id for i in range(self._num_pins)]
        self._pin_body_ids = [self._model.body(f'pin_{i}').id for i in range(self._num_pins)]
        self._gripper_wall_geom_id = self._model.geom('bowling_gripper_y_wall').id
        for name in [
            'ur5e/robotiq/right_pad1',
            'ur5e/robotiq/right_pad2',
            'ur5e/robotiq/left_pad1',
            'ur5e/robotiq/left_pad2',
        ]:
            self._model.geom_conaffinity[self._model.geom(name).id] |= 4

    def initialize_episode(self):
        self._elapsed_steps = 0
        self._pin_knocked = np.zeros(self._num_pins, dtype=bool)
        self._prev_num_knocked_pins = 0

        self._model.geom(self._ball_geom_id).rgba = self._colors['blue']
        for gid in self._pin_geom_ids:
            self._model.geom(gid).rgba = self._colors['white']

        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        self.initialize_arm()
        ball_init_pos = self._ball_init_pos.copy()
        ball_init_pos[:2] += self.np_random.uniform(*self._ball_init_xy_randomization)
        self._data.joint('ball_joint').qpos[:3] = ball_init_pos
        self._data.joint('ball_joint').qpos[3:] = lie.SO3.identity().wxyz
        pin_init_xys = self._pin_init_xys + self.np_random.uniform(*self._pin_rack_xy_randomization)
        for i, xy in enumerate(pin_init_xys):
            self._data.joint(f'pin_joint_{i}').qpos[:3] = np.array([xy[0], xy[1], self._pin_half_height])
            self._data.joint(f'pin_joint_{i}').qpos[3:] = lie.SO3.identity().wxyz

        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False

    def step(self, action):
        if self._reset_next_step:
            return self.reset()

        self._elapsed_steps += 1
        return super().step(action)

    def _pin_positions(self):
        return np.array([self._data.joint(f'pin_joint_{i}').qpos[:3] for i in range(self._num_pins)])

    def _compute_pin_knocked(self):
        knocked = []
        for i in range(self._num_pins):
            quat = self._data.joint(f'pin_joint_{i}').qpos[3:]
            _, x, y, _ = quat
            knocked.append(1 - 2 * (x * x + y * y) < 0.75)
        return np.asarray(knocked, dtype=bool)

    def _positions_in_floor_workspace(self, positions, margin=0.2):
        lower = np.array([-self._workspace_half_size, -self._workspace_half_size, self._workspace_bounds[0, 2]]) - margin
        upper = np.array([self._workspace_half_size, self._workspace_half_size, self._workspace_bounds[1, 2]]) + margin
        return bool(np.all(np.asarray(positions) > lower) and np.all(np.asarray(positions) < upper))

    def pre_step(self):
        self._prev_num_knocked_pins = int(self._pin_knocked.sum())
        super().pre_step()

    def post_step(self):
        ball_pos = self._data.joint('ball_joint').qpos[:3]
        self._healthy = self._positions_in_floor_workspace(np.vstack([ball_pos, self._pin_positions()]))
        self._pin_knocked |= self._compute_pin_knocked()
        self._success = False
        self._failure = False

    def compute_reward(self):
        return 0.1 * max(int(self._pin_knocked.sum()) - self._prev_num_knocked_pins, 0)

    def terminate_episode(self):
        return (not self._healthy) or self._elapsed_steps >= self._max_episode_steps

    def add_object_info(self, ob_info):
        ball_qpos = self._data.joint('ball_joint').qpos.copy()
        ball_qvel = self._data.joint('ball_joint').qvel.copy()
        ob_info['privileged/ball_pos'] = ball_qpos[:3]
        ob_info['privileged/ball_quat'] = ball_qpos[3:]
        ob_info['privileged/ball_vel'] = ball_qvel[:3]
        ob_info['privileged/ball_angvel'] = ball_qvel[3:]
        for i in range(self._num_pins):
            pin_qpos = self._data.joint(f'pin_joint_{i}').qpos.copy()
            ob_info[f'privileged/pin_{i}_pos'] = pin_qpos[:3]
            ob_info[f'privileged/pin_{i}_quat'] = pin_qpos[3:]

    def compute_observation(self):
        if self._ob_type == 'pixels':
            return self.get_pixel_observation()
        else:
            xyz_center = np.array([0.425, 0.0, 0.0])
            xyz_scaler = 10.0
            gripper_scaler = 3.0

            ob_info = self.compute_ob_info()
            ob = [
                ob_info['proprio/joint_pos'],
                ob_info['proprio/joint_vel'],
                (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
                np.cos(ob_info['proprio/effector_yaw']),
                np.sin(ob_info['proprio/effector_yaw']),
                ob_info['proprio/gripper_opening'] * gripper_scaler,
                ob_info['proprio/gripper_contact'],
                (ob_info['privileged/ball_pos'] - xyz_center) * xyz_scaler,
                ob_info['privileged/ball_quat'],
                ob_info['privileged/ball_vel'] * xyz_scaler,
                ob_info['privileged/ball_angvel'],
            ]
            for i in range(self._num_pins):
                ob.extend(
                    [
                        (ob_info[f'privileged/pin_{i}_pos'] - xyz_center) * xyz_scaler,
                        ob_info[f'privileged/pin_{i}_quat'],
                    ]
                )

            return np.concatenate(ob)
