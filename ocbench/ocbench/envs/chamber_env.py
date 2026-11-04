import mujoco
import numpy as np
from dm_control import mjcf

from ocbench import lie
from ocbench.envs.manipulation_env import ManipulationEnv


class ChamberEnv(ManipulationEnv):
    def __init__(
        self,
        env_type='easy',
        task_id=1,
        **kwargs,
    ):
        self._env_type = env_type
        self._cube_size = 0.03
        self._min_object_init_dist = 0.09
        if self._env_type == 'easy':
            self._num_cubes = 1
        elif self._env_type == 'medium':
            self._num_cubes = 2
        elif self._env_type == 'hard':
            self._num_cubes = 3
        else:
            raise ValueError(f'Invalid env_type: {env_type}')

        super().__init__(
            task_id=task_id,
            **kwargs,
        )

        # Adjust sampling bounds to a smaller region.
        self._arm_sampling_bounds = np.asarray([[0.25, -0.2, 0.20], [0.6, 0.2, 0.35]])
        self._object_sampling_bounds = np.asarray([[0.3, -0.07], [0.45, 0.18]])
        self._target_sampling_bounds = self._object_sampling_bounds

        # Define constants.
        self._cube_colors = np.array([self._colors['blue'], self._colors['red'], self._colors['orange']])
        self._cube_success_colors = np.array(
            [self._colors['lightblue'], self._colors['lightred'], self._colors['lightorange']]
        )
        self._num_buttons = 2
        self._num_button_states = 2
        self._cur_button_states = np.array([0] * self._num_buttons)

        self._target_button_states = np.array([0] * self._num_buttons)
        self._target_drawer_pos = 0.0
        self._target_window_pos = 0.0
        self._prev_dynamics_info = None
        self._mirrored = False

        self._drawer_slide_min = -0.16
        self._drawer_slide_max = 0.0
        self._drawer_open_threshold = -0.12
        self._window_slide_min = 0.0
        self._window_slide_max = 0.2
        self._drawer_base_randomization = np.asarray([[-0.05, -0.15, 0.0], [0.05, 0.0, 0.0]])
        self._window_base_randomization = np.asarray([[0.0, 0.0, 0.0], [0.05, 0.15, 0.0]])
        self._drawer_block_x_range = np.asarray([-0.06, 0.06])
        self._drawer_block_y_range = np.asarray([0.16, 0.18])

    def set_state(self, qpos, qvel, dynamics_info=None, button_states=None):
        apply_button_states = False
        if dynamics_info is not None:
            dynamics_info = np.asarray(dynamics_info)
            if dynamics_info.shape == (self._num_buttons,) and button_states is None:
                button_states = dynamics_info
                dynamics_info = None

        if dynamics_info is not None:
            self._set_dynamics_info(dynamics_info, apply_button_states=False)
            apply_button_states = True
        elif button_states is not None:
            self._cur_button_states = np.rint(button_states).astype(np.int64)
            apply_button_states = True
        super().set_state(qpos, qvel)
        if apply_button_states:
            self._apply_button_states(force=True)
        self._prev_button_states = self._cur_button_states.copy()
        self._prev_dynamics_info = self._get_dynamics_info()

    def set_tasks(self):
        if self._env_type == 'easy':
            self.task_infos = [
                dict(
                    task_name='open_all',
                    init=dict(
                        drawer_button_state=0,
                        drawer_pos_range=np.array([-0.06, 0]),
                    ),
                    goal=dict(
                        button_states=np.array([0, 0]),
                        drawer_pos=-0.16,
                        window_pos=0.2,
                    ),
                    cube_success_type='on_floor',
                ),
                dict(
                    task_name='put_in',
                    init=dict(
                        drawer_button_state=0,
                        drawer_pos_range=np.array([-0.06, 0]),
                    ),
                    goal=dict(
                        button_states=np.array([0, 0]),
                        drawer_pos=0.0,
                        window_pos=0.2,
                    ),
                    cube_success_type='in_drawer',
                ),
                dict(
                    task_name='take_out',
                    init=dict(
                        block_sampling='drawer',
                        drawer_button_state=0,
                        drawer_pos_range=np.array([-0.06, 0]),
                    ),
                    goal=dict(
                        button_states=np.array([0, 0]),
                        drawer_pos=0.0,
                        window_pos=0.0,
                    ),
                    cube_success_type='stack_anywhere',
                ),
            ]
        elif self._env_type == 'medium':
            self.task_infos = [
                dict(
                    task_name='put_all_in',
                    init=dict(
                        num_drawer_top_blocks_choices=np.array([1, 2]),
                        drawer_button_state=0,
                        drawer_pos_range=np.array([-0.06, 0]),
                    ),
                    goal=dict(
                        button_states=np.array([0, 0]),
                        drawer_pos=0.0,
                        window_pos=0.2,
                    ),
                    cube_success_type='in_drawer',
                ),
                dict(
                    task_name='stack_anywhere',
                    init=dict(
                        block_location_count_choices=np.array(
                            [
                                [1, 1, 0],
                                [2, 0, 0],
                            ]
                        ),
                        drawer_button_state=0,
                        drawer_pos_range=np.array([-0.06, 0]),
                    ),
                    goal=dict(
                        button_states=np.array([0, 0]),
                        drawer_pos=0.0,
                        window_pos=0.0,
                    ),
                    cube_success_type='stack_anywhere',
                ),
            ]
        elif self._env_type == 'hard':
            self.task_infos = [
                dict(
                    task_name='put_all_in',
                    init=dict(
                        num_drawer_top_blocks_choices=np.array([1, 2]),
                        drawer_button_state=0,
                        drawer_pos_range=np.array([-0.06, 0]),
                    ),
                    goal=dict(
                        button_states=np.array([0, 0]),
                        drawer_pos=0.0,
                        window_pos=0.2,
                    ),
                    cube_success_type='in_drawer',
                ),
                dict(
                    task_name='stack_anywhere',
                    init=dict(
                        block_location_count_choices=np.array(
                            [
                                [1, 0, 2],
                                [1, 1, 1],
                                [1, 2, 0],
                                [2, 0, 1],
                                [2, 1, 0],
                            ]
                        ),
                        drawer_button_state=0,
                        drawer_pos_range=np.array([-0.06, 0]),
                    ),
                    goal=dict(
                        button_states=np.array([0, 0]),
                        drawer_pos=0.0,
                        window_pos=0.0,
                    ),
                    cube_success_type='stack_anywhere',
                ),
            ]

    def add_objects(self, arena_mjcf):
        # Add cubes.
        cube_outer_mjcf = mjcf.from_path((self._desc_dir / 'cube_outer.xml').as_posix())
        cube_outer_mjcf.find('default', 'cube').geom.size = np.full(3, self._cube_size)
        arena_mjcf.include_copy(cube_outer_mjcf)
        distance = 0.05
        for i in range(self._num_cubes):
            cube_mjcf = mjcf.from_path((self._desc_dir / 'cube_inner.xml').as_posix())
            cube_mjcf.find('body', 'object_target_0').remove()
            cube_mjcf.find('body', 'object_0').pos[2] = self._cube_size
            pos = -distance * (self._num_cubes - 1) + 2 * distance * i
            cube_mjcf.find('body', 'object_0').pos[1] = pos
            for tag in ['body', 'joint', 'geom', 'site']:
                for item in cube_mjcf.find_all(tag):
                    if hasattr(item, 'name') and item.name is not None and item.name.endswith('_0'):
                        item.name = item.name[:-2] + f'_{i}'
            arena_mjcf.include_copy(cube_mjcf)

        # Add other objects to scene.
        button_mjcf = mjcf.from_path((self._desc_dir / 'buttons.xml').as_posix())
        arena_mjcf.include_copy(button_mjcf)
        drawer_mjcf = mjcf.from_path((self._desc_dir / 'drawer.xml').as_posix())
        arena_mjcf.include_copy(drawer_mjcf)
        window_mjcf = mjcf.from_path((self._desc_dir / 'window.xml').as_posix())
        arena_mjcf.include_copy(window_mjcf)
        drawer_lock_anchor = arena_mjcf.worldbody.add('body', name='drawer_lock_anchor', mocap=True, pos=(0, 0, -10))
        window_lock_anchor = arena_mjcf.worldbody.add('body', name='window_lock_anchor', mocap=True, pos=(0, 0, -10))
        arena_mjcf.equality.add(
            'weld',
            name='drawer_lock',
            body1=arena_mjcf.find('body', 'drawer_link'),
            body2=drawer_lock_anchor,
            active=False,
            relpose=(0, 0, 0, 1, 0, 0, 0),
            solimp=(0.95, 0.99, 0.001),
            solref=(0.005, 1),
        )
        arena_mjcf.equality.add(
            'weld',
            name='window_lock',
            body1=arena_mjcf.find('body', 'windowb_a'),
            body2=window_lock_anchor,
            active=False,
            relpose=(0, 0, 0, 1, 0, 0, 0),
            solimp=(0.95, 0.99, 0.001),
            solref=(0.005, 1),
        )

        # Save geoms.
        self._cube_geoms_list = []
        for i in range(self._num_cubes):
            self._cube_geoms_list.append(arena_mjcf.find('body', f'object_{i}').find_all('geom'))
        self._button_geoms_list = []
        for i in range(self._num_buttons):
            self._button_geoms_list.append([button_mjcf.find('geom', f'btngeom_{i}')])

        # Add cameras.
        cameras = {
            'front': {
                'pos': (1.447, 0.000, 1.069),
                'xyaxes': (0.000, 1.000, 0.000, -0.627, 0.000, 0.779),
            },
            'front_pixels': {
                'pos': (0.905, 0.000, 0.762),
                'xyaxes': (0.000, 1.000, 0.000, -0.771, 0.000, 0.637),
            },
            'side': {
                'pos': (1.294, 0.572, 1.069),
                'xyaxes': (-0.500, 0.866, 0.000, -0.543, -0.314, 0.779),
            },
        }
        for camera_name, camera_kwargs in cameras.items():
            arena_mjcf.worldbody.add('camera', name=camera_name, **camera_kwargs)

    def post_compilation_objects(self):
        # Cube geom IDs.
        self._cube_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in cube_geoms] for cube_geoms in self._cube_geoms_list
        ]

        # Button geom IDs.
        self._button_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in button_geoms]
            for button_geoms in self._button_geoms_list
        ]
        self._button_site_ids = [self._model.site(f'btntop_{i}').id for i in range(self._num_buttons)]
        self._button_base_body_ids = [self._model.body(f'buttonbox_{i}').id for i in range(self._num_buttons)]
        self._nominal_button_base_poss = self._model.body_pos[self._button_base_body_ids].copy()
        self._nominal_button_base_quats = self._model.body_quat[self._button_base_body_ids].copy()

        # Drawer and window site IDs.
        self._drawer_site_id = self._model.site('drawer_handle_center').id
        self._drawer_target_site_id = self._model.site('drawer_handle_center_target').id

        self._window_site_id = self._model.site('window_handle_center').id
        self._window_target_site_id = self._model.site('window_handle_center_target').id

        self._drawer_link_body_id = self._model.body('drawer_link').id
        self._window_link_body_id = self._model.body('windowb_a').id
        self._drawer_lock_anchor_body_id = self._model.body('drawer_lock_anchor').id
        self._window_lock_anchor_body_id = self._model.body('window_lock_anchor').id
        self._drawer_lock_anchor_mocap_id = self._model.body_mocapid[self._drawer_lock_anchor_body_id]
        self._window_lock_anchor_mocap_id = self._model.body_mocapid[self._window_lock_anchor_body_id]
        self._drawer_lock_eq_id = self._model.equality('drawer_lock').id
        self._window_lock_eq_id = self._model.equality('window_lock').id

        # Drawer and window base body IDs.
        self._drawer_base_body_id = self._model.body('drawer_base').id
        self._window_base_body_id = self._model.body('window').id
        self._nominal_drawer_base_pos = self._model.body_pos[self._drawer_base_body_id].copy()
        self._nominal_window_base_pos = self._model.body_pos[self._window_base_body_id].copy()
        self._drawer_base_quat = self._model.body_quat[self._drawer_base_body_id].copy()
        self._window_base_quat = self._model.body_quat[self._window_base_body_id].copy()
        self._mirrored_drawer_base_quat = lie.SO3.identity().wxyz
        self._mirrored_window_base_quat = lie.SO3.from_z_radians(np.pi).wxyz

    def _layout_body_pos(self, body_id):
        mocap_id = self._model.body_mocapid[body_id]
        if mocap_id >= 0:
            return self._data.mocap_pos[mocap_id]
        return self._model.body_pos[body_id]

    def _set_layout_body_pose(self, body_id, pos, quat):
        mocap_id = self._model.body_mocapid[body_id]
        if mocap_id >= 0:
            self._data.mocap_pos[mocap_id] = pos
            self._data.mocap_quat[mocap_id] = quat
        else:
            self._model.body_pos[body_id] = pos
            self._model.body_quat[body_id] = quat

    def _sample_base_pos(self, nominal_pos, randomization):
        pos = nominal_pos + self.np_random.uniform(*randomization)
        pos[2] = nominal_pos[2]
        return pos

    def _mirror_pos(self, pos):
        pos = np.asarray(pos).copy()
        pos[1] = -pos[1]
        return pos

    def _sample_floor_xy(self, bounds):
        xy = self.np_random.uniform(*bounds)
        if self._mirrored:
            xy[1] *= -1
        return xy

    def _sample_drawer_block_pos(self):
        drawer_base_pos = self._layout_body_pos(self._drawer_base_body_id)
        drawer_handle_pos = self._data.site_xpos[self._drawer_site_id]
        x = drawer_base_pos[0] + self.np_random.uniform(*self._drawer_block_x_range)
        y = drawer_handle_pos[1] + self._drawer_slide_y_sign() * self.np_random.uniform(*self._drawer_block_y_range)
        return np.array([x, y, 0.076])

    def _is_stack_anywhere_task(self):
        return self.cur_task_info is not None and self.cur_task_info['cube_success_type'] == 'stack_anywhere'

    def _set_layout(self, mirrored, button_base_poss, drawer_base_pos, window_base_pos):
        self._mirrored = bool(mirrored)
        button_base_poss = np.asarray(button_base_poss).copy()
        drawer_base_pos = np.asarray(drawer_base_pos).copy()
        window_base_pos = np.asarray(window_base_pos).copy()
        button_base_poss[:, 2] = self._nominal_button_base_poss[:, 2]
        drawer_base_pos[2] = self._nominal_drawer_base_pos[2]
        window_base_pos[2] = self._nominal_window_base_pos[2]
        for body_id, pos, quat in zip(
            self._button_base_body_ids, button_base_poss, self._nominal_button_base_quats
        ):
            self._set_layout_body_pose(body_id, pos, quat)
        self._set_layout_body_pose(
            self._drawer_base_body_id,
            drawer_base_pos,
            self._mirrored_drawer_base_quat if self._mirrored else self._drawer_base_quat,
        )
        self._set_layout_body_pose(
            self._window_base_body_id,
            window_base_pos,
            self._mirrored_window_base_quat if self._mirrored else self._window_base_quat,
        )

    def _drawer_slide_y_sign(self):
        return 1.0 if self._mirrored else -1.0

    def _window_slide_x_sign(self):
        return -1.0 if self._mirrored else 1.0

    def _get_dynamics_info(self):
        button_base_poss = np.array([self._layout_body_pos(body_id).copy() for body_id in self._button_base_body_ids])
        return np.concatenate(
            [
                self._cur_button_states.astype(np.float64),
                np.array([float(self._mirrored)]),
                button_base_poss.ravel(),
                self._layout_body_pos(self._drawer_base_body_id).copy(),
                self._layout_body_pos(self._window_base_body_id).copy(),
            ]
        )

    def _set_dynamics_info(self, dynamics_info, apply_button_states=True):
        dynamics_info = np.asarray(dynamics_info)
        self._cur_button_states = np.rint(dynamics_info[: self._num_buttons]).astype(np.int64)
        mirrored = bool(np.rint(dynamics_info[self._num_buttons]))
        base_positions = dynamics_info[self._num_buttons + 1 :].reshape(self._num_buttons + 2, 3)
        self._set_layout(mirrored, base_positions[: self._num_buttons], base_positions[-2], base_positions[-1])
        if apply_button_states:
            self._apply_button_states(force=True)

    def _set_lock_constraint(self, eq_id, mocap_id, body_id, active, force=False):
        was_active = bool(self._data.eq_active[eq_id])
        if active and (force or not was_active):
            self._data.mocap_pos[mocap_id] = self._data.xpos[body_id]
            self._data.mocap_quat[mocap_id] = self._data.xquat[body_id]
        self._data.eq_active[eq_id] = active

    def _apply_button_states(self, force=False):
        mujoco.mj_forward(self._model, self._data)

        # Adjust button colors based on the current state.
        for i in range(self._num_buttons):
            for gid in self._button_geom_ids_list[i]:
                self._model.geom(gid).rgba = self._colors['red' if self._cur_button_states[i] == 0 else 'white']

        self._model.material('drawer_handle').rgba = self._colors['red' if self._cur_button_states[0] == 0 else 'white']
        self._model.material('window_handle').rgba = self._colors['red' if self._cur_button_states[1] == 0 else 'white']
        self._set_lock_constraint(
            self._drawer_lock_eq_id,
            self._drawer_lock_anchor_mocap_id,
            self._drawer_link_body_id,
            self._cur_button_states[0] == 0,
            force=force,
        )
        self._set_lock_constraint(
            self._window_lock_eq_id,
            self._window_lock_anchor_mocap_id,
            self._window_link_body_id,
            self._cur_button_states[1] == 0,
            force=force,
        )

        mujoco.mj_forward(self._model, self._data)

    def initialize_episode(self):
        # Set cube colors.
        for i in range(self._num_cubes):
            for gid in self._cube_geom_ids_list[i]:
                self._model.geom(gid).rgba = self._cube_colors[i]

        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        mirrored = self.np_random.uniform() < 0.5
        button_base_poss = self._nominal_button_base_poss.copy()
        drawer_base_pos = self._sample_base_pos(self._nominal_drawer_base_pos, self._drawer_base_randomization)
        window_base_pos = self._sample_base_pos(self._nominal_window_base_pos, self._window_base_randomization)
        if mirrored:
            button_base_poss = np.array([self._mirror_pos(pos) for pos in button_base_poss])
            drawer_base_pos = self._mirror_pos(drawer_base_pos)
            window_base_pos = self._mirror_pos(window_base_pos)
        self._set_layout(mirrored, button_base_poss, drawer_base_pos, window_base_pos)
        mujoco.mj_forward(self._model, self._data)

        # Get the current task info for the other objects.
        goal_button_states = self.cur_task_info['goal']['button_states'].copy()
        goal_drawer_pos = self.cur_task_info['goal']['drawer_pos']
        goal_window_pos = self.cur_task_info['goal']['window_pos']
        init_button_states = self.np_random.choice(self._num_button_states, size=self._num_buttons)
        init_drawer_pos = self.np_random.uniform(self._drawer_slide_min, self._drawer_slide_max)
        init_window_pos = self.np_random.uniform(self._window_slide_min, self._window_slide_max)
        init_info = self.cur_task_info['init']
        if 'drawer_button_state' in init_info:
            init_button_states[0] = init_info['drawer_button_state']
        if 'drawer_pos_range' in init_info:
            init_drawer_pos = self.np_random.uniform(*init_info['drawer_pos_range'])

        self._data.joint('drawer_slide').qpos[0] = init_drawer_pos
        self._data.joint('window_slide').qpos[0] = init_window_pos
        mujoco.mj_forward(self._model, self._data)

        drawer_blocks = set()
        if 'block_location_count_choices' in init_info:
            count_choices = init_info['block_location_count_choices']
            num_drawer_blocks, num_drawer_top_blocks, _ = count_choices[self.np_random.integers(len(count_choices))]
            blocks = self.np_random.permutation(self._num_cubes)
            drawer_blocks = set(blocks[:num_drawer_blocks].tolist())
            drawer_top_blocks = blocks[num_drawer_blocks : num_drawer_blocks + num_drawer_top_blocks].tolist()
        elif init_info.get('block_sampling') == 'drawer':
            drawer_blocks = set(range(self._num_cubes))
            drawer_top_blocks = []
            num_drawer_top_blocks = 0
        elif 'num_drawer_top_blocks_choices' in init_info:
            num_drawer_top_blocks = int(self.np_random.choice(init_info['num_drawer_top_blocks_choices']))
            drawer_top_blocks = self.np_random.choice(
                self._num_cubes, size=num_drawer_top_blocks, replace=False
            ).tolist()
        else:
            num_drawer_top_blocks = init_info.get('num_drawer_top_blocks', 0)
            drawer_top_blocks = self.np_random.choice(
                self._num_cubes, size=num_drawer_top_blocks, replace=False
            ).tolist()
        drawer_x_offsets = {}
        if len(drawer_blocks) == 2:
            x_offsets = np.array([-0.05, 0.05]) + self.np_random.uniform(-0.005, 0.005, size=2)
            self.np_random.shuffle(x_offsets)
            drawer_x_offsets = dict(zip(drawer_blocks, x_offsets))
        drawer_top_x_offsets = {}
        if num_drawer_top_blocks == 2:
            x_offsets = np.array([-0.05, 0.05]) + self.np_random.uniform(-0.005, 0.005, size=2)
            self.np_random.shuffle(x_offsets)
            drawer_top_x_offsets = dict(zip(drawer_top_blocks, x_offsets))
        drawer_top_blocks = set(drawer_top_blocks)
        init_block_xyzs = np.zeros((self._num_cubes, 3))
        init_block_quats = np.zeros((self._num_cubes, 4))
        object_xys = []
        for i in range(self._num_cubes):
            # Randomize the position and orientation of the cube.
            if i in drawer_blocks:
                if i in drawer_x_offsets:
                    drawer_base_pos = self._layout_body_pos(self._drawer_base_body_id)
                    drawer_handle_pos = self._data.site_xpos[self._drawer_site_id]
                    x = drawer_base_pos[0] + drawer_x_offsets[i]
                    y = drawer_handle_pos[1] + self._drawer_slide_y_sign() * self.np_random.uniform(
                        *self._drawer_block_y_range
                    )
                    obj_pos = np.array([x, y, 0.076])
                else:
                    obj_pos = self._sample_drawer_block_pos()
            elif i in drawer_top_blocks:
                drawer_base_pos = self._layout_body_pos(self._drawer_base_body_id)
                x = drawer_base_pos[0] + drawer_top_x_offsets.get(i, self.np_random.uniform(-0.06, 0.06))
                y = drawer_base_pos[1] - self._drawer_slide_y_sign() * self.np_random.uniform(0.07, 0.1)
                xy = np.array([x, y])
                obj_pos = np.array([xy[0], xy[1], drawer_base_pos[2] + 0.108])
            else:
                for _ in range(100):
                    xy = self._sample_floor_xy(self._object_sampling_bounds)
                    obj_pos = np.array((*xy, self._cube_size))
                    if all(np.linalg.norm(xy - other_xy) >= self._min_object_init_dist for other_xy in object_xys):
                        break
                object_xys.append(xy)
            init_block_xyzs[i] = obj_pos
            yaw = self.np_random.uniform(0, 2 * np.pi)
            init_block_quats[i] = lie.SO3.from_z_radians(yaw).wxyz

        self.initialize_arm()
        self._data.joint('drawer_slide').qpos[0] = init_drawer_pos
        self._data.joint('window_slide').qpos[0] = init_window_pos
        mujoco.mj_forward(self._model, self._data)
        for i in range(self._num_cubes):
            self._data.joint(f'object_joint_{i}').qpos[:3] = init_block_xyzs[i]
            self._data.joint(f'object_joint_{i}').qpos[3:] = init_block_quats[i]

        # Set the button states.
        self._cur_button_states = init_button_states.copy()
        self._target_button_states = goal_button_states.copy()
        self._apply_button_states()

        # Set the drawer and window positions.
        self._model.site('drawer_handle_center_target').pos[1] = goal_drawer_pos
        self._target_drawer_pos = goal_drawer_pos
        self._model.site('window_handle_center_target').pos[0] = goal_window_pos
        self._target_window_pos = goal_window_pos

        # Forward kinematics to update site positions.
        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False

    def _is_in_drawer(self, obj_pos):
        """Check if the object is in the drawer."""
        drawer_base_pos = self._layout_body_pos(self._drawer_base_body_id)
        drawer_pos_y = self._data.site_xpos[self._drawer_site_id][1]
        if self._mirrored:
            drawer_low_y, drawer_high_y = drawer_pos_y + 0.11, drawer_pos_y + 0.31
        else:
            drawer_low_y, drawer_high_y = drawer_pos_y - 0.31, drawer_pos_y - 0.11
        drawer_low = np.array([drawer_base_pos[0] - 0.15, drawer_low_y, drawer_base_pos[2] - 0.044])
        drawer_high = np.array([drawer_base_pos[0] + 0.15, drawer_high_y, drawer_base_pos[2] + 0.046])
        return bool(np.all(drawer_low <= obj_pos) and np.all(obj_pos <= drawer_high))

    def _compute_stack_anywhere_successes(self, block_xyzs):
        order = np.argsort(block_xyzs[:, 2])
        sorted_xyzs = block_xyzs[order]
        target_zs = self._cube_size * (2 * np.arange(self._num_cubes) + 1)
        xy_success = np.all(np.linalg.norm(block_xyzs[:, :2] - sorted_xyzs[0, :2], axis=1) <= 0.04)
        z_success = np.all(np.abs(sorted_xyzs[:, 2] - target_zs) <= 0.03)
        outside_drawer = all(not self._is_in_drawer(obj_pos) for obj_pos in block_xyzs)
        success = bool(xy_success and z_success and outside_drawer)
        return [success] * self._num_cubes

    def _compute_cube_successes(self, block_xyzs=None):
        if block_xyzs is None:
            block_xyzs = np.array([self._data.joint(f'object_joint_{i}').qpos[:3] for i in range(self._num_cubes)])
        success_type = self.cur_task_info['cube_success_type']
        if success_type == 'in_drawer':
            return [self._is_in_drawer(obj_pos) for obj_pos in block_xyzs]
        if success_type == 'on_floor':
            return [
                abs(obj_pos[2] - self._cube_size) <= 0.03 and not self._is_in_drawer(obj_pos)
                for obj_pos in block_xyzs
            ]
        if success_type == 'stack_anywhere':
            return self._compute_stack_anywhere_successes(block_xyzs)
        raise ValueError(f'Unsupported cube success type: {success_type}')

    def pre_step(self):
        self._prev_button_states = self._cur_button_states.copy()
        self._prev_dynamics_info = self._get_dynamics_info()
        super().pre_step()

    def _compute_successes(self):
        """Compute object successes."""
        cube_successes = self._compute_cube_successes()
        button_successes = [
            (self._cur_button_states[i] == self._target_button_states[i]) for i in range(self._num_buttons)
        ]
        drawer_success = np.abs(self._data.joint('drawer_slide').qpos[0] - self._target_drawer_pos) <= 0.04
        window_success = np.abs(self._data.joint('window_slide').qpos[0] - self._target_window_pos) <= 0.04

        return cube_successes, button_successes, drawer_success, window_success

    def post_step(self):
        block_xyzs = np.array([self._data.joint(f'object_joint_{i}').qpos[:3] for i in range(self._num_cubes)])
        self._healthy = self._positions_in_workspace(block_xyzs)

        # Update button states.
        for i in range(self._num_buttons):
            prev_joint_pos = self._prev_ob_info[f'privileged/button_{i}_pos'][0]
            cur_joint_pos = self._data.joint(f'buttonbox_joint_{i}').qpos.copy()[0]
            if prev_joint_pos > -0.02 and cur_joint_pos <= -0.02:
                # Button pressed: change the state of the button.
                self._cur_button_states[i] = (self._cur_button_states[i] + 1) % self._num_button_states
        self._apply_button_states()

        # Evaluate successes.
        if self._healthy:
            cube_successes, button_successes, drawer_success, window_success = self._compute_successes()
        else:
            cube_successes = [False] * self._num_cubes
            button_successes = [False] * self._num_buttons
            drawer_success = False
            window_success = False
        self._success = (
            self._healthy and all(cube_successes) and all(button_successes) and drawer_success and window_success
        )

        # Adjust the colors of the cubes based on success.
        for i in range(self._num_cubes):
            if self._visualize_info and cube_successes[i]:
                for gid in self._cube_geom_ids_list[i]:
                    self._model.geom(gid).rgba[:3] = self._cube_success_colors[i, :3]
            else:
                for gid in self._cube_geom_ids_list[i]:
                    self._model.geom(gid).rgba[:3] = self._cube_colors[i, :3]

    def add_object_info(self, ob_info):
        # Cube positions and orientations.
        for i in range(self._num_cubes):
            ob_info[f'privileged/block_{i}_pos'] = self._data.joint(f'object_joint_{i}').qpos[:3].copy()
            ob_info[f'privileged/block_{i}_quat'] = self._data.joint(f'object_joint_{i}').qpos[3:].copy()
            ob_info[f'privileged/block_{i}_yaw'] = np.array(
                [lie.SO3(wxyz=self._data.joint(f'object_joint_{i}').qpos[3:]).compute_yaw_radians()]
            )

        # Button states.
        for i in range(self._num_buttons):
            ob_info[f'privileged/button_{i}_state'] = self._cur_button_states[i]
            ob_info[f'privileged/button_{i}_base_pos'] = self._layout_body_pos(self._button_base_body_ids[i]).copy()
            ob_info[f'privileged/button_{i}_pos'] = self._data.joint(f'buttonbox_joint_{i}').qpos.copy()
            ob_info[f'privileged/button_{i}_vel'] = self._data.joint(f'buttonbox_joint_{i}').qvel.copy()

        # Drawer states.
        ob_info['privileged/drawer_base_pos'] = self._layout_body_pos(self._drawer_base_body_id).copy()
        ob_info['privileged/drawer_pos'] = self._data.joint('drawer_slide').qpos.copy()
        ob_info['privileged/drawer_vel'] = self._data.joint('drawer_slide').qvel.copy()
        ob_info['privileged/drawer_handle_pos'] = self._data.site_xpos[self._drawer_site_id].copy()
        ob_info['privileged/drawer_handle_yaw'] = np.array(
            [lie.SO3.from_matrix(self._data.site_xmat[self._drawer_site_id].reshape(3, 3)).compute_yaw_radians()]
        )

        # Window states.
        ob_info['privileged/window_base_pos'] = self._layout_body_pos(self._window_base_body_id).copy()
        ob_info['privileged/window_pos'] = self._data.joint('window_slide').qpos.copy()
        ob_info['privileged/window_vel'] = self._data.joint('window_slide').qvel.copy()
        ob_info['privileged/window_handle_pos'] = self._data.site_xpos[self._window_site_id].copy()
        ob_info['privileged/window_handle_yaw'] = np.array(
            [lie.SO3.from_matrix(self._data.site_xmat[self._window_site_id].reshape(3, 3)).compute_yaw_radians()]
        )

        ob_info['prev_button_states'] = self._prev_button_states.copy()
        ob_info['button_states'] = self._cur_button_states.copy()
        ob_info['privileged/mirrored'] = np.array([float(self._mirrored)])
        prev_dynamics_info = self._prev_dynamics_info
        if prev_dynamics_info is None:
            prev_dynamics_info = self._get_dynamics_info()
        ob_info['prev_dynamics_info'] = prev_dynamics_info.copy()
        ob_info['dynamics_info'] = self._get_dynamics_info()

    def compute_observation(self):
        if self._ob_type == 'pixels':
            return self.get_pixel_observation()
        else:
            xyz_center = np.array([0.425, 0.0, 0.0])
            xyz_scaler = 10.0
            gripper_scaler = 3.0
            button_scaler = 120.0
            drawer_scaler = 18.0
            window_scaler = 15.0

            ob_info = self.compute_ob_info()
            ob = [
                ob_info['proprio/joint_pos'],
                ob_info['proprio/joint_vel'],
                (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
                np.cos(ob_info['proprio/effector_yaw']),
                np.sin(ob_info['proprio/effector_yaw']),
                ob_info['proprio/gripper_opening'] * gripper_scaler,
                ob_info['proprio/gripper_contact'],
            ]
            for i in range(self._num_cubes):
                ob.extend(
                    [
                        (ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler,
                        ob_info[f'privileged/block_{i}_quat'],
                        np.cos(ob_info[f'privileged/block_{i}_yaw']),
                        np.sin(ob_info[f'privileged/block_{i}_yaw']),
                    ]
                )
            for i in range(self._num_buttons):
                button_state = np.eye(self._num_button_states)[self._cur_button_states[i]]
                ob.extend(
                    [
                        button_state,
                        (ob_info[f'privileged/button_{i}_base_pos'] - xyz_center) * xyz_scaler,
                        ob_info[f'privileged/button_{i}_pos'] * button_scaler,
                        ob_info[f'privileged/button_{i}_vel'],
                    ]
                )
            ob.extend(
                [
                    ob_info['privileged/mirrored'],
                    ob_info['privileged/drawer_pos'] * drawer_scaler,
                    ob_info['privileged/drawer_vel'],
                    (ob_info['privileged/drawer_base_pos'] - xyz_center) * xyz_scaler,
                    ob_info['privileged/window_pos'] * window_scaler,
                    ob_info['privileged/window_vel'],
                    (ob_info['privileged/window_base_pos'] - xyz_center) * xyz_scaler,
                ]
            )

            return np.concatenate(ob)
