from pathlib import Path

import mujoco
import numpy as np
from dm_control import mjcf
from gymnasium.spaces import Box

from ocbench import controllers, lie, mjcf_utils
from ocbench.envs.env import CustomMuJoCoEnv


class ManipulationEnv(CustomMuJoCoEnv):
    def __init__(
        self,
        ob_type='states',
        physics_timestep=0.002,
        control_timestep=0.02,
        terminate_at_success=True,
        visualize_info=True,
        pixel_cameras=None,
        action_delta_scale=1.0,
        lite=False,
        task_id=1,
        **kwargs,
    ):
        super().__init__(
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
            **kwargs,
        )

        # Define constants.
        self._desc_dir = Path(__file__).resolve().parent.parent / 'descriptions'
        self._home_qpos = np.asarray([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
        self._effector_down_rotation = lie.SO3(np.asarray([0.0, 1.0, 0.0, 0.0]))
        self._workspace_bounds = np.asarray([[0.10, -0.65, 0.02], [0.75, 0.65, 0.45]])
        self._arm_sampling_bounds = np.asarray([[0.25, -0.35, 0.20], [0.6, 0.35, 0.35]])
        self._object_sampling_bounds = np.asarray([[0.3, -0.3], [0.55, 0.3]])
        self._target_sampling_bounds = np.asarray([[0.3, -0.3], [0.55, 0.3]])
        self._colors = dict(
            red=np.array([0.96, 0.26, 0.33, 1.0]),
            orange=np.array([1.0, 0.69, 0.21, 1.0]),
            yellow=np.array([0.76, 0.96, 0.04, 1.0]),
            green=np.array([0.06, 0.74, 0.21, 1.0]),
            blue=np.array([0.35, 0.55, 0.91, 1.0]),
            purple=np.array([0.61, 0.28, 0.82, 1.0]),
            magenta=np.array([0.82, 0.28, 0.61, 1.0]),
            lightred=np.array([0.99, 0.85, 0.86, 1.0]),
            lightorange=np.array([1.0, 0.94, 0.84, 1.0]),
            lightyellow=np.array([0.95, 0.99, 0.8, 1.0]),
            lightgreen=np.array([0.77, 0.95, 0.81, 1.0]),
            lightblue=np.array([0.86, 0.9, 0.98, 1.0]),
            lightpurple=np.array([0.91, 0.84, 0.96, 1.0]),
            lightmagenta=np.array([0.96, 0.84, 0.91, 1.0]),
            white=np.array([0.9, 0.9, 0.9, 1.0]),
            lightgray=np.array([0.7, 0.7, 0.7, 1.0]),
            gray=np.array([0.5, 0.5, 0.5, 1.0]),
            darkgray=np.array([0.3, 0.3, 0.3, 1.0]),
            black=np.array([0.1, 0.1, 0.1, 1.0]),
        )

        self._ob_type = ob_type
        self._terminate_at_success = terminate_at_success
        self._visualize_info = visualize_info and ob_type == 'states'
        self._pixel_cameras = pixel_cameras
        self._lite = lite
        self._task_id = task_id

        assert ob_type in ['states', 'pixels']

        # Initialize inverse kinematics controller.
        ik_mjcf = mjcf.from_path((self._desc_dir / 'universal_robots_ur5e' / 'ur5e.xml'), escape_separators=True)
        xml_str = mjcf_utils.to_string(ik_mjcf)
        assets = mjcf_utils.get_assets(ik_mjcf)
        ik_model = mujoco.MjModel.from_xml_string(xml_str, assets)

        self._ik = controllers.DiffIKController(model=ik_model, sites=['attachment_site'])

        # Define action limits.
        self._joint_action_delta = np.array([0.18, 0.18, 0.18, 0.36, 0.36, 0.36, 0.12]) * action_delta_scale
        # Cartesian motion limits for oracle targets.
        self._ee_action_delta = np.array([0.05, 0.05, 0.05, 0.3, 0.12]) * action_delta_scale

        # Set task goals.
        self.task_infos = []
        self.cur_task_id = None
        self.cur_task_info = None
        self.set_tasks()
        self.num_tasks = len(self.task_infos)

        self._success = False
        self._failure = False
        self._healthy = True

    @property
    def observation_space(self):
        if self._model is None:
            self.reset()

        ex_ob = self.compute_observation()

        if self._ob_type == 'pixels':
            return Box(low=0, high=255, shape=ex_ob.shape, dtype=ex_ob.dtype)
        else:
            return Box(low=-np.inf, high=np.inf, shape=ex_ob.shape, dtype=ex_ob.dtype)

    @property
    def action_space(self):
        return Box(
            low=-np.ones(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32),
            shape=(7,),
            dtype=np.float32,
        )

    def normalize_action(self, action):
        action = np.asarray(action)
        return np.clip(action / self._joint_action_delta, -1, 1)

    def unnormalize_action(self, action):
        action = np.asarray(action)
        return np.clip(action, -1, 1) * self._joint_action_delta

    def set_tasks(self):
        pass

    def build_mjcf_model(self):
        # Set scene.
        arena_mjcf = mjcf.from_path((self._desc_dir / 'floor.xml').as_posix())

        # Add UR5e robot arm.
        ur5e_mjcf = mjcf.from_path((self._desc_dir / 'universal_robots_ur5e' / 'ur5e.xml'), escape_separators=True)
        ur5e_mjcf.model = 'ur5e'

        for light in ur5e_mjcf.find_all('light'):
            light.remove()
            del light

        # Attach the robotiq gripper to the UR5e flange.
        gripper_mjcf = mjcf.from_path((self._desc_dir / 'robotiq_2f85' / '2f85.xml'), escape_separators=True)
        gripper_mjcf.model = 'robotiq'
        mjcf_utils.attach(ur5e_mjcf, gripper_mjcf, 'attachment_site')

        # Wrist camera.
        ur5e_mjcf.find('body', 'wrist_3_link').add(
            'camera',
            name='wrist',
            pos=(0.0, 0.0, -0.1),
            xyaxes=(-1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
            fovy=75,
        )

        # Attach UR5e to the scene.
        mjcf_utils.attach(arena_mjcf, ur5e_mjcf)

        self.add_objects(arena_mjcf)

        # Cache joint and actuator elements.
        self._arm_jnts = mjcf_utils.safe_find_all(
            ur5e_mjcf,
            'joint',
            exclude_attachments=True,
        )
        self._arm_acts = mjcf_utils.safe_find_all(
            ur5e_mjcf,
            'actuator',
            exclude_attachments=True,
        )
        self._gripper_jnts = mjcf_utils.safe_find_all(gripper_mjcf, 'joint', exclude_attachments=True)
        self._gripper_acts = mjcf_utils.safe_find_all(gripper_mjcf, 'actuator', exclude_attachments=True)

        # Add bounding boxes to visualize the workspace and object sampling bounds.
        mjcf_utils.add_bounding_box_site(
            arena_mjcf.worldbody,
            lower=np.asarray((*self._target_sampling_bounds[0], 0.02)),
            upper=np.asarray((*self._target_sampling_bounds[1], 0.02)),
            rgba=(0.6, 0.3, 0.3, 0.2),
            group=4,
            name='object_bounds',
        )
        mjcf_utils.add_bounding_box_site(
            arena_mjcf.worldbody,
            lower=np.asarray(self._arm_sampling_bounds[0]),
            upper=np.asarray(self._arm_sampling_bounds[1]),
            rgba=(0.3, 0.6, 0.3, 0.2),
            group=4,
            name='arm_bounds',
        )

        return arena_mjcf

    def add_objects(self, arena_mjcf):
        pass

    def post_compilation(self):
        # Arm joint and actuator IDs.
        arm_joint_names = [j.full_identifier for j in self._arm_jnts]
        self._arm_joint_ids = np.asarray([self._model.joint(name).id for name in arm_joint_names])
        actuator_names = [a.full_identifier for a in self._arm_acts]
        self._arm_actuator_ids = np.asarray([self._model.actuator(name).id for name in actuator_names])
        gripper_actuator_names = [a.full_identifier for a in self._gripper_acts]
        self._gripper_actuator_ids = np.asarray([self._model.actuator(name).id for name in gripper_actuator_names])
        self._gripper_opening_joint_id = self._model.joint('ur5e/robotiq/right_driver_joint').id

        # Modify PD gains.
        self._model.actuator_gainprm[self._arm_actuator_ids, 0] = np.asarray([4500, 4500, 4500, 2000, 2000, 500])
        self._model.actuator_gainprm[self._arm_actuator_ids, 2] = np.asarray([-450, -450, -450, -200, -200, -50])
        self._model.actuator_biasprm[self._arm_actuator_ids, 1] = -np.asarray([4500, 4500, 4500, 2000, 2000, 500])

        # Site IDs.
        self._pinch_site_id = self._model.site('ur5e/robotiq/pinch').id
        self._attach_site_id = self._model.site('ur5e/attachment_site').id

        pinch_pose = lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_matrix(self._data.site_xmat[self._pinch_site_id].reshape(3, 3)),
            translation=self._data.site_xpos[self._pinch_site_id],
        )
        attach_pose = lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_matrix(self._data.site_xmat[self._attach_site_id].reshape(3, 3)),
            translation=self._data.site_xpos[self._attach_site_id],
        )
        self._T_pa = pinch_pose.inverse() @ attach_pose

        self.post_compilation_objects()

    def post_compilation_objects(self):
        pass

    def reset(self, options=None, *args, **kwargs):
        if options is None:
            options = {}

        if 'task_id' in options:
            task_id = options['task_id']
            assert 1 <= task_id <= self.num_tasks, f'Task ID must be in [1, {self.num_tasks}].'
            self.cur_task_id = task_id
            self.cur_task_info = self.task_infos[self.cur_task_id - 1]
        elif 'task_info' in options:
            self.cur_task_id = None
            self.cur_task_info = options['task_info']
        else:
            assert 1 <= self._task_id <= self.num_tasks, f'Task ID must be in [1, {self.num_tasks}].'
            self.cur_task_id = self._task_id
            self.cur_task_info = self.task_infos[self.cur_task_id - 1]

        self._success = False
        self._failure = False
        self._healthy = True
        return super().reset(*args, **kwargs)

    def step(self, action):
        if self._reset_next_step:
            return self.reset()

        ob, reward, terminated, truncated, info = super().step(action)
        info['success'] = self._success
        info['failure'] = self._failure
        info['healthy'] = self._healthy

        return ob, reward, terminated, truncated, info

    def initialize_arm(self):
        # Sample initial effector position and orientation.
        eff_pos = self.np_random.uniform(*self._arm_sampling_bounds)
        cur_ori = self._effector_down_rotation
        yaw = self.np_random.uniform(-np.pi, np.pi)
        rotz = lie.SO3.from_z_radians(yaw)
        eff_ori = rotz @ cur_ori

        # Solve for initial joint positions using IK.
        T_wp = lie.SE3.from_rotation_and_translation(eff_ori, eff_pos)
        T_wa = T_wp @ self._T_pa
        qpos_init = self._ik.solve(
            pos=T_wa.translation(),
            quat=T_wa.rotation().wxyz,
            curr_qpos=self._home_qpos,
        )

        self._data.qpos[self._arm_joint_ids] = qpos_init
        mujoco.mj_forward(self._model, self._data)

    def initialize_episode(self):
        pass

    def joint_position_action(self, effector_pos, effector_yaw, gripper_opening, effector_rotation=None):
        target_effector_translation = np.asarray(effector_pos).copy()
        np.clip(
            target_effector_translation,
            *self._workspace_bounds,
            out=target_effector_translation,
        )
        if effector_rotation is None:
            effector_rotation = lie.SO3.from_z_radians(effector_yaw)
        target_effector_orientation = effector_rotation @ self._effector_down_rotation
        target_gripper_opening = float(np.asarray(np.clip(gripper_opening, 0.0, 1.0)).squeeze())

        T_wp = lie.SE3.from_rotation_and_translation(
            rotation=target_effector_orientation,
            translation=target_effector_translation,
        )
        T_wa = T_wp @ self._T_pa
        qpos_target = self._ik.solve(
            pos=T_wa.translation(),
            quat=T_wa.rotation().wxyz,
            curr_qpos=self._data.qpos[self._arm_joint_ids],
        )

        arm_delta = qpos_target - self._data.qpos[self._arm_joint_ids]
        gripper_delta = target_gripper_opening - self._data.qpos[self._gripper_opening_joint_id] / 0.8
        return self.normalize_action(np.concatenate([arm_delta, [gripper_delta]]))

    def _set_joint_delta_control(self, action):
        arm_ctrlrange = self._model.actuator_ctrlrange[self._arm_actuator_ids]
        qpos_target = self._data.qpos[self._arm_joint_ids] + action[:6]
        qpos_target = np.clip(qpos_target, arm_ctrlrange[:, 0], arm_ctrlrange[:, 1])

        gripper_opening = self._data.qpos[self._gripper_opening_joint_id] / 0.8
        target_gripper_opening = np.clip(gripper_opening + action[6], 0.0, 1.0)

        self._data.ctrl[self._arm_actuator_ids] = qpos_target
        self._data.ctrl[self._gripper_actuator_ids] = 255.0 * target_gripper_opening

    def set_control(self, action):
        self._set_joint_delta_control(self.unnormalize_action(action))

    def pre_step(self):
        self._prev_qpos = self._data.qpos.copy()
        self._prev_qvel = self._data.qvel.copy()
        self._prev_ob_info = self.compute_ob_info()

    def compute_ob_info(self):
        ob_info = {}

        # Proprioceptive observations.
        ob_info['proprio/joint_pos'] = self._data.qpos[self._arm_joint_ids].copy()
        ob_info['proprio/joint_vel'] = self._data.qvel[self._arm_joint_ids].copy()
        ob_info['proprio/effector_pos'] = self._data.site_xpos[self._pinch_site_id].copy()
        ob_info['proprio/effector_yaw'] = np.array(
            [lie.SO3.from_matrix(self._data.site_xmat[self._pinch_site_id].copy().reshape(3, 3)).compute_yaw_radians()]
        )
        ob_info['proprio/gripper_opening'] = np.array(
            np.clip([self._data.qpos[self._gripper_opening_joint_id] / 0.8], 0, 1)
        )
        ob_info['proprio/gripper_vel'] = self._data.qvel[[self._gripper_opening_joint_id]].copy()
        ob_info['proprio/gripper_contact'] = np.array(
            [np.clip(np.linalg.norm(self._data.body('ur5e/robotiq/right_pad').cfrc_ext) / 50, 0, 1)]
        )

        self.add_object_info(ob_info)

        ob_info['prev_qpos'] = self._prev_qpos.copy()
        ob_info['prev_qvel'] = self._prev_qvel.copy()
        ob_info['qpos'] = self._data.qpos.copy()
        ob_info['qvel'] = self._data.qvel.copy()
        ob_info['control'] = self._data.ctrl.copy()
        ob_info['time'] = np.array([self._data.time])

        return ob_info

    def add_object_info(self, ob_info):
        pass

    def _positions_in_workspace(self, positions, margin=0.2):
        positions = np.asarray(positions)
        lower = self._workspace_bounds[0] - margin
        upper = self._workspace_bounds[1] + margin
        return bool(np.all(positions > lower) and np.all(positions < upper))

    def get_pixel_observation(self):
        if self._pixel_cameras is None:
            return self.render()
        return np.stack([self.render(camera=camera) for camera in self._pixel_cameras], axis=0)

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

            return np.concatenate(ob)

    def compute_reward(self):
        return 1.0 if self._success and not self._failure and self._healthy else 0.0

    def get_reset_info(self):
        info = self.compute_ob_info()
        info['success'] = self._success
        info['failure'] = self._failure
        info['healthy'] = self._healthy
        return info

    def get_step_info(self):
        ob_info = self.compute_ob_info()
        return ob_info

    def terminate_episode(self):
        return (self._terminate_at_success and self._success) or self._failure or (not self._healthy)

    def render(
        self,
        camera=None,
        *args,
        **kwargs,
    ):
        if camera is None:
            camera = 'front' if self._ob_type == 'states' else 'front_pixels'
            if self._ob_type == 'pixels' and self._pixel_cameras is not None:
                camera = self._pixel_cameras[0]

        return super().render(camera=camera, *args, **kwargs)
