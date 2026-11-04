import mujoco
import numpy as np
from dm_control import mjcf

from ocbench import lie
from ocbench.envs.manipulation_env import ManipulationEnv


class HanoiEnv(ManipulationEnv):
    def __init__(
        self,
        env_type='triple',
        task_id=1,
        **kwargs,
    ):
        self._env_type = env_type
        self._num_pegs = 3
        if self._env_type == 'single':
            self._num_disks = 1
        elif self._env_type == 'double':
            self._num_disks = 2
        elif self._env_type == 'triple':
            self._num_disks = 3
        else:
            raise ValueError(f'Invalid env_type: {env_type}')

        self._peg_half_size = np.array([0.01, 0.01])
        self._peg_half_height = 0.084
        self._disk_half_height = 0.020
        self._disk_gap = 0.0
        self._disk_hole_half_size = 0.018
        self._disk_short_half_size = 0.028
        self._disk_min_long_half_size = 0.065
        self._disk_long_size_step = 0.015
        self._base_peg_positions = np.array(
            [
                [0.425, -0.3],
                [0.425, 0.0],
                [0.425, 0.3],
            ]
        )
        self._peg_positions = self._base_peg_positions.copy()
        self._peg_yaws = np.zeros(self._num_pegs)
        self._peg_randomization = np.asarray([[-0.05, -0.025], [0.05, 0.025]])

        super().__init__(
            task_id=task_id,
            **kwargs,
        )

        if self._env_type == 'single':
            self._disk_colors = np.array([self._colors['blue']])
        elif self._env_type == 'double':
            self._disk_colors = np.array([self._colors['red'], self._colors['blue']])
        else:
            self._disk_colors = np.array([self._colors['red'], self._colors['orange'], self._colors['blue']])
        self._arm_sampling_bounds = np.asarray([[0.18, -0.50, 0.20], [0.67, 0.50, 0.35]])
        self._prev_dynamics_info = None

    def set_tasks(self):
        self.task_infos = [
            dict(
                task_name='fixed',
                init_peg=1,
                goal_pegs=np.array([0, 2]),
            ),
            dict(
                task_name='random',
                init_peg=1,
                goal_pegs=np.array([0, 2]),
            ),
        ]

    def set_state(self, qpos, qvel, dynamics_info=None):
        if dynamics_info is not None:
            self._set_dynamics_info(dynamics_info)
        super().set_state(qpos, qvel)
        self._prev_dynamics_info = self._get_dynamics_info()

    def _disk_size_idx(self, disk):
        size_idx = disk
        return size_idx

    def _disk_half_extents(self, disk):
        return np.array(
            [
                self._disk_short_half_size,
                self._disk_min_long_half_size + self._disk_long_size_step * self._disk_size_idx(disk),
            ]
        )

    def _disk_grasp_width(self, disk):
        return 2 * self._disk_half_extents(disk)[0]

    def _disk_yaw_for_peg(self, peg):
        yaw = self._peg_yaws[peg]
        return (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4

    def _disk_grasp_offset(self, disk, yaw, sign, fraction=0.75):
        grasp_y = self._disk_hole_half_size + fraction * (self._disk_half_extents(disk)[1] - self._disk_hole_half_size)
        offset = np.array([0.0, sign * grasp_y])
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([c * offset[0] - s * offset[1], s * offset[0] + c * offset[1]])

    def _peg_body_positions(self):
        if hasattr(self, '_peg_body_mocap_ids'):
            return self._data.mocap_pos[self._peg_body_mocap_ids].copy()
        if hasattr(self, '_peg_body_ids'):
            return self._model.body_pos[self._peg_body_ids].copy()
        return np.column_stack([self._peg_positions, np.full(self._num_pegs, self._peg_half_height)])

    def _set_peg_layout(self, peg_body_positions, peg_yaws):
        peg_body_positions = np.asarray(peg_body_positions, dtype=np.float64)
        peg_yaws = np.asarray(peg_yaws, dtype=np.float64)
        self._peg_positions = peg_body_positions[:, :2].copy()
        self._peg_yaws = peg_yaws.copy()
        if hasattr(self, '_peg_body_mocap_ids'):
            for peg in range(self._num_pegs):
                self._data.mocap_pos[self._peg_body_mocap_ids[peg]] = peg_body_positions[peg]
                self._data.mocap_quat[self._peg_body_mocap_ids[peg]] = lie.SO3.from_z_radians(self._peg_yaws[peg]).wxyz
        elif hasattr(self, '_peg_body_ids'):
            for peg in range(self._num_pegs):
                self._model.body_pos[self._peg_body_ids[peg]] = peg_body_positions[peg]
                self._model.body_quat[self._peg_body_ids[peg]] = lie.SO3.from_z_radians(self._peg_yaws[peg]).wxyz

    def _sample_peg_layout(self):
        offsets = self.np_random.uniform(*self._peg_randomization, size=(self._num_pegs, 2))
        peg_body_positions = np.column_stack(
            [
                self._base_peg_positions + offsets,
                np.full(self._num_pegs, self._peg_half_height),
            ]
        )
        peg_yaws = self.np_random.uniform(0.0, np.pi / 2, size=self._num_pegs)
        return peg_body_positions, peg_yaws

    def _get_dynamics_info(self):
        return np.column_stack([self._peg_body_positions(), self._peg_yaws]).ravel().copy()

    def _set_dynamics_info(self, dynamics_info):
        dynamics_info = np.asarray(dynamics_info, dtype=np.float64).reshape(self._num_pegs, 4)
        self._set_peg_layout(dynamics_info[:, :3], dynamics_info[:, 3])

    def _configure_disk_mjcf(self, disk_mjcf, disk):
        outer_x, outer_y = self._disk_half_extents(disk)
        hole = self._disk_hole_half_size
        wall_x = (outer_x - hole) / 2
        wall_y = (outer_y - hole) / 2
        offset_x = (outer_x + hole) / 2
        offset_y = (outer_y + hole) / 2
        z = self._disk_half_height
        geom_specs = {
            'disk_left_0': ((wall_x, outer_y, z), (-offset_x, 0.0, 0.0)),
            'disk_right_0': ((wall_x, outer_y, z), (offset_x, 0.0, 0.0)),
            'disk_front_0': ((hole, wall_y, z), (0.0, -offset_y, 0.0)),
            'disk_back_0': ((hole, wall_y, z), (0.0, offset_y, 0.0)),
        }
        for geom_name, (size, pos) in geom_specs.items():
            geom = disk_mjcf.find('geom', geom_name)
            geom.size = size
            geom.pos = pos

    def _rename_suffix(self, root, suffix):
        for tag in ['body', 'joint', 'geom', 'site']:
            for item in root.find_all(tag):
                if hasattr(item, 'name') and item.name is not None and item.name.endswith('_0'):
                    item.name = item.name[:-2] + f'_{suffix}'

    def _disk_pos_on_peg(self, peg, level):
        z = self._disk_half_height + level * (2 * self._disk_half_height + self._disk_gap)
        return np.array([*self._peg_positions[peg], z])

    def _stack_disk_pos(self, peg, disk):
        level = self._num_disks - 1 - disk
        return self._disk_pos_on_peg(peg, level)

    def _set_disk_stack(self, peg):
        for disk in range(self._num_disks):
            qpos = self._data.joint(f'disk_joint_{disk}').qpos
            qvel = self._data.joint(f'disk_joint_{disk}').qvel
            qpos[:3] = self._stack_disk_pos(peg, disk)
            qpos[3:] = lie.SO3.from_z_radians(self._disk_yaw_for_peg(peg)).wxyz
            qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

    def add_objects(self, arena_mjcf):
        # Add Hanoi defaults.
        hanoi_outer_mjcf = mjcf.from_path((self._desc_dir / 'hanoi_outer.xml').as_posix())
        arena_mjcf.include_copy(hanoi_outer_mjcf)

        # Add pegs.
        for i, peg_xy in enumerate(self._peg_positions):
            peg_mjcf = mjcf.from_path((self._desc_dir / 'hanoi_peg.xml').as_posix())
            peg_body = peg_mjcf.find('body', 'peg_0')
            peg_body.mocap = True
            peg_body.pos = np.array([peg_xy[0], peg_xy[1], self._peg_half_height])
            peg_mjcf.find('geom', 'peg_0').size = (
                self._peg_half_size[0],
                self._peg_half_size[1],
                self._peg_half_height,
            )
            self._rename_suffix(peg_mjcf, i)
            arena_mjcf.include_copy(peg_mjcf)

        # Add disks.
        for i in range(self._num_disks):
            disk_mjcf = mjcf.from_path((self._desc_dir / 'hanoi_disk.xml').as_posix())
            self._configure_disk_mjcf(disk_mjcf, i)
            disk_mjcf.find('body', 'disk_0').pos = self._stack_disk_pos(0, i)
            self._rename_suffix(disk_mjcf, i)
            arena_mjcf.include_copy(disk_mjcf)

        # Save disk geoms.
        self._disk_geoms_list = []
        for i in range(self._num_disks):
            self._disk_geoms_list.append(arena_mjcf.find('body', f'disk_{i}').find_all('geom'))

        # Add cameras.
        cameras = {
            'front': {
                'pos': (1.534, 0.000, 0.598),
                'xyaxes': (0.000, 1.000, 0.000, -0.342, 0.000, 0.940),
            },
            'front_pixels': {
                'pos': (1.053, -0.014, 0.639),
                'xyaxes': (0.000, 1.000, 0.000, -0.628, 0.001, 0.778),
            },
            'side': {
                'pos': (0.917, -1.068, 0.598),
                'xyaxes': (0.866, 0.500, -0.000, -0.171, 0.296, 0.940),
            },
        }
        for camera_name, camera_kwargs in cameras.items():
            arena_mjcf.worldbody.add('camera', name=camera_name, **camera_kwargs)

    def post_compilation_objects(self):
        self._peg_body_ids = [self._model.body(f'peg_{i}').id for i in range(self._num_pegs)]
        self._peg_body_mocap_ids = np.asarray([self._model.body_mocapid[body_id] for body_id in self._peg_body_ids])

        # Disk geom IDs.
        self._disk_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in disk_geoms] for disk_geoms in self._disk_geoms_list
        ]
        self._disk_geom_id_sets = [set(geom_ids) for geom_ids in self._disk_geom_ids_list]
        self._floor_geom_id = self._model.geom('floor').id

    def initialize_episode(self):
        for i in range(self._num_disks):
            for gid in self._disk_geom_ids_list[i]:
                self._model.geom(gid).rgba = self._disk_colors[i]

        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        self.initialize_arm()
        if self.cur_task_info['task_name'] == 'random':
            peg_body_positions, peg_yaws = self._sample_peg_layout()
        else:
            peg_body_positions = np.column_stack(
                [self._base_peg_positions, np.full(self._num_pegs, self._peg_half_height)]
            )
            peg_yaws = np.zeros(self._num_pegs)
        self._set_peg_layout(peg_body_positions, peg_yaws)
        self._set_disk_stack(self.cur_task_info['init_peg'])

        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False

    def pre_step(self):
        self._prev_dynamics_info = self._get_dynamics_info()
        super().pre_step()

    def _is_tower_on_peg(self, peg):
        for disk in range(self._num_disks):
            disk_pos = self._data.joint(f'disk_joint_{disk}').qpos[:3]
            target_pos = self._stack_disk_pos(peg, disk)
            if np.linalg.norm(disk_pos[:2] - target_pos[:2]) > 0.04:
                return False
            if np.abs(disk_pos[2] - target_pos[2]) > 0.03:
                return False
        return True

    def _is_solved(self):
        return any(self._is_tower_on_peg(peg) for peg in self.cur_task_info['goal_pegs'])

    def _is_disk_touching_floor(self, disk):
        disk_geom_ids = self._disk_geom_id_sets[disk]
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            if (
                contact.geom1 == self._floor_geom_id
                and contact.geom2 in disk_geom_ids
                or contact.geom2 == self._floor_geom_id
                and contact.geom1 in disk_geom_ids
            ):
                return True
        return False

    def _is_disk_through_peg(self, disk):
        disk_pos = self._data.joint(f'disk_joint_{disk}').qpos[:3]
        return bool(np.min(np.linalg.norm(self._peg_positions - disk_pos[:2], axis=1)) <= 0.04)

    def _has_bad_floor_contact(self):
        for disk in range(self._num_disks):
            if self._is_disk_touching_floor(disk) and not self._is_disk_through_peg(disk):
                return True
        return False

    def _has_illegal_stack(self):
        stacks = [[] for _ in range(self._num_pegs)]
        disk_zs = np.zeros(self._num_disks)
        for disk in range(self._num_disks):
            disk_pos = self._data.joint(f'disk_joint_{disk}').qpos[:3]
            disk_zs[disk] = disk_pos[2]
            peg_dists = np.linalg.norm(self._peg_positions - disk_pos[:2], axis=1)
            peg = int(np.argmin(peg_dists))
            if peg_dists[peg] <= 0.04:
                stacks[peg].append(disk)

        for stack in stacks:
            stack.sort(key=lambda disk: disk_zs[disk])
            for i in range(len(stack) - 1):
                lower, upper = stack[i], stack[i + 1]
                z_gap = disk_zs[upper] - disk_zs[lower]
                if lower < upper and z_gap <= 2 * self._disk_half_height + 0.04:
                    return True
        return False

    def post_step(self):
        disk_xyzs = np.array([self._data.joint(f'disk_joint_{i}').qpos[:3] for i in range(self._num_disks)])
        healthy = self._positions_in_workspace(disk_xyzs)
        failure = healthy and (self._has_bad_floor_contact() or self._has_illegal_stack())
        success = healthy and not failure and self._is_solved()
        self._success = success
        self._failure = failure
        self._healthy = healthy

    def add_object_info(self, ob_info):
        # Disks.
        for i in range(self._num_disks):
            qpos = self._data.joint(f'disk_joint_{i}').qpos.copy()
            ob_info[f'privileged/disk_{i}_pos'] = qpos[:3]
            ob_info[f'privileged/disk_{i}_quat'] = qpos[3:]
            ob_info[f'privileged/disk_{i}_yaw'] = np.array([lie.SO3(wxyz=qpos[3:]).compute_yaw_radians()])

        # Pegs.
        for i in range(self._num_pegs):
            ob_info[f'privileged/peg_{i}_pos'] = self._peg_body_positions()[i].copy()
            ob_info[f'privileged/peg_{i}_yaw'] = np.array([self._peg_yaws[i]])

        ob_info['privileged/init_peg'] = np.array([self.cur_task_info['init_peg']])
        ob_info['privileged/valid_goal_pegs'] = self.cur_task_info['goal_pegs'].copy()
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
            for i in range(self._num_disks):
                disk_pos = ob_info[f'privileged/disk_{i}_pos']
                disk_quat = ob_info[f'privileged/disk_{i}_quat']
                disk_yaw = lie.SO3(wxyz=disk_quat).compute_yaw_radians()
                ob.extend(
                    [
                        (disk_pos - xyz_center) * xyz_scaler,
                        np.cos([disk_yaw]),
                        np.sin([disk_yaw]),
                    ]
                )
            for i in range(self._num_pegs):
                ob.extend(
                    [
                        (ob_info[f'privileged/peg_{i}_pos'] - xyz_center) * xyz_scaler,
                        np.cos(ob_info[f'privileged/peg_{i}_yaw']),
                        np.sin(ob_info[f'privileged/peg_{i}_yaw']),
                    ]
                )

            return np.concatenate(ob)
