from collections import defaultdict

import numpy as np

from ocbench.oracles.primitives.button import ButtonPrimitive
from ocbench.oracles.controllers.controller import Controller
from ocbench.oracles.primitives.cube import CubePrimitive
from ocbench.oracles.primitives.drawer import DrawerPrimitive
from ocbench.oracles.primitives.window import WindowPrimitive


class ChamberController(Controller):
    def __init__(
        self,
        env,
        p_mistake=0.2,
        max_subgoal_steps=1000,
        cube_settle_steps=25,
        speed_profile=(0.7, 1.0, 1.5),
    ):
        super().__init__(
            env=env,
            p_mistake=p_mistake,
            max_subgoal_steps=max_subgoal_steps,
            speed_profile=speed_profile,
        )
        self._cube_settle_steps = cube_settle_steps
        self._agents = None
        self._active_settle_steps = 0
        self._subgoal_counts = defaultdict(int)
        self._goal_order = None
        self._block_goal_xyzs = None

    def _goals(self):
        goal = self._env.unwrapped.cur_task_info['goal']
        return dict(
            block_xyzs=self._block_goal_xyzs,
            block_pos=self._block_goal_xyzs[0],
            button_states=goal['button_states'],
            drawer_pos=goal['drawer_pos'],
            window_pos=goal['window_pos'],
        )

    def _in_drawer_goal(self):
        env = self._env.unwrapped
        if env._env_type == 'easy':
            goal_block_xyzs = np.array([[0.33, -0.356, 0.076]])
        elif env._env_type == 'medium':
            goal_block_xyzs = np.array(
                [
                    [0.27, -0.356, 0.076],
                    [0.39, -0.356, 0.076],
                ]
            )
        else:
            goal_block_xyzs = np.array(
                [
                    [0.26, -0.356, 0.076],
                    [0.33, -0.356, 0.076],
                    [0.40, -0.356, 0.076],
                ]
            )
        nominal_drawer_base_pos = env._nominal_drawer_base_pos.copy()
        if env._mirrored:
            goal_block_xyzs[:, 1] *= -1
            nominal_drawer_base_pos = env._mirror_pos(nominal_drawer_base_pos)
        goal_block_xyzs += env._layout_body_pos(env._drawer_base_body_id) - nominal_drawer_base_pos
        return goal_block_xyzs

    def _sample_stack_anywhere_goal(self, info):
        env = self._env.unwrapped
        floor_blocks = [
            i
            for i in range(env._num_cubes)
            if abs(info[f'privileged/block_{i}_pos'][2] - env._cube_size) <= 0.03
            and not env._is_in_drawer(info[f'privileged/block_{i}_pos'])
        ]

        floor_block = None
        if len(floor_blocks) > 0:
            floor_block = int(np.random.choice(floor_blocks))
            xy = info[f'privileged/block_{floor_block}_pos'][:2].copy()
        else:
            xy = np.random.uniform(*env._target_sampling_bounds)
            if env._mirrored:
                xy[1] *= -1

        order = np.random.permutation(env._num_cubes)
        if floor_block is not None:
            order = np.concatenate([[floor_block], order[order != floor_block]])
        goal_block_xyzs = np.zeros((env._num_cubes, 3))
        for level, block_idx in enumerate(order):
            goal_block_xyzs[block_idx] = [xy[0], xy[1], env._cube_size * (2 * level + 1)]
        return goal_block_xyzs

    def _successes(self, info):
        goals = self._goals()
        cube_success = all(self._cube_successes(info))
        drawer_success = np.abs(info['privileged/drawer_pos'][0] - goals['drawer_pos']) <= 0.04
        window_success = np.abs(info['privileged/window_pos'][0] - goals['window_pos']) <= 0.04
        button_successes = [
            info[f'privileged/button_{i}_state'] == goals['button_states'][i]
            for i in range(self._env.unwrapped._num_buttons)
        ]
        return cube_success, button_successes, drawer_success, window_success

    def _cube_successes(self, info):
        block_xyzs = np.array([info[f'privileged/block_{i}_pos'] for i in range(self._env.unwrapped._num_cubes)])
        return self._env.unwrapped._compute_cube_successes(block_xyzs)

    def _target_successes(self, info):
        goals = self._goals()
        return [
            np.linalg.norm(info[f'privileged/block_{i}_pos'] - goals['block_xyzs'][i]) <= 0.04
            for i in range(self._env.unwrapped._num_cubes)
        ]

    def _open_drawer_cube_pos(self, info, block):
        goals = self._goals()
        target_pos = goals['block_xyzs'][block].copy()
        target_pos[1] += self._env.unwrapped._drawer_slide_y_sign() * (
            info['privileged/drawer_pos'][0] - goals['drawer_pos']
        )
        return target_pos

    def _drawer_subgoal(self, info):
        env = self._env.unwrapped
        goals = self._goals()
        cube_success, button_successes, drawer_success, _ = self._successes(info)
        drawer_locked = info['privileged/button_0_state'] == 0
        drawer_open = info['privileged/drawer_pos'][0] <= env._drawer_open_threshold
        task_name = env.cur_task_info['task_name']

        if task_name in {'put_in', 'put_all_in'}:
            if not cube_success:
                if not drawer_open:
                    if drawer_locked:
                        return dict(task='button', button=0, target_state=1, name='unlock_drawer')
                    return dict(task='drawer', target_pos=env._drawer_slide_min, name='open_drawer')
                cube_successes = self._cube_successes(info)
                block = next((int(i) for i in self._goal_order if not cube_successes[i]), 0)
                return dict(
                    task='cube',
                    block=block,
                    target_pos=self._open_drawer_cube_pos(info, block),
                    name='put_cube_in_drawer',
                )
        elif task_name in {'take_out', 'stack_anywhere'}:
            if not cube_success:
                target_successes = self._target_successes(info)
                target_zs = goals['block_xyzs'][:, 2]
                block = next((int(i) for i in np.argsort(target_zs) if not target_successes[i]), 0)
                block_in_drawer = env._is_in_drawer(info[f'privileged/block_{block}_pos'])
                if block_in_drawer and not drawer_open:
                    if drawer_locked:
                        return dict(task='button', button=0, target_state=1, name='unlock_drawer')
                    return dict(task='drawer', target_pos=env._drawer_slide_min, name='open_drawer')
                return dict(
                    task='cube',
                    block=block,
                    target_pos=goals['block_xyzs'][block].copy(),
                    name='take_cube_out',
                )
        elif task_name == 'open_all':
            pass
        else:
            raise ValueError(f'Unsupported Chamber task: {task_name}')

        if not drawer_success:
            if drawer_locked:
                return dict(task='button', button=0, target_state=1, name='unlock_drawer')
            return dict(task='drawer', target_pos=goals['drawer_pos'], name='close_drawer')
        if not button_successes[0]:
            return dict(task='button', button=0, target_state=goals['button_states'][0], name='lock_drawer')
        return None

    def _window_subgoal(self, info):
        goals = self._goals()
        _, button_successes, _, window_success = self._successes(info)
        window_locked = info['privileged/button_1_state'] == 0

        if not window_success:
            if window_locked:
                return dict(task='button', button=1, target_state=1, name='unlock_window')
            return dict(task='window', target_pos=goals['window_pos'], name='move_window')
        if not button_successes[1]:
            return dict(task='button', button=1, target_state=goals['button_states'][1], name='lock_window')
        return None

    def _mistake_subgoals(self, info):
        env = self._env.unwrapped
        subgoals = []
        for button in range(self._env.unwrapped._num_buttons):
            target_state = (info[f'privileged/button_{button}_state'] + 1) % self._env.unwrapped._num_button_states
            subgoals.append(dict(task='button', button=button, target_state=target_state, name='mistake_button'))

        if info['privileged/button_0_state'] == 1:
            drawer_pos = info['privileged/drawer_pos'][0]
            drawer_mid = (env._drawer_slide_min + env._drawer_slide_max) / 2
            target_pos = env._drawer_slide_min if drawer_pos >= drawer_mid else env._drawer_slide_max
            subgoals.append(dict(task='drawer', target_pos=target_pos, name='mistake_drawer'))
        if info['privileged/button_1_state'] == 1:
            window_pos = info['privileged/window_pos'][0]
            window_mid = (env._window_slide_min + env._window_slide_max) / 2
            target_pos = env._window_slide_max if window_pos <= window_mid else env._window_slide_min
            subgoals.append(dict(task='window', target_pos=target_pos, name='mistake_window'))
        cube_success_type = env.cur_task_info['cube_success_type']
        cube_successes = self._cube_successes(info)
        if cube_success_type == 'in_drawer' and not all(cube_successes):
            candidate_blocks = [i for i, success in enumerate(cube_successes) if success]
            if len(candidate_blocks) == 0:
                candidate_blocks = list(range(self._env.unwrapped._num_cubes))
            block = int(np.random.choice(candidate_blocks))
            xy = np.random.uniform(*env._target_sampling_bounds)
            if env._mirrored:
                xy[1] *= -1
            target_pos = np.array([*xy, self._env.unwrapped._cube_size])
            subgoals.append(dict(task='cube', block=block, target_pos=target_pos, name='mistake_cube'))
        elif (
            cube_success_type == 'stack_anywhere'
            and all(cube_successes)
            and info['privileged/drawer_pos'][0] <= env._drawer_open_threshold
        ):
            drawer_base_pos = env._layout_body_pos(env._drawer_base_body_id)
            drawer_handle_pos = env._data.site_xpos[env._drawer_site_id]
            x = drawer_base_pos[0] + np.random.uniform(*env._drawer_block_x_range)
            y = drawer_handle_pos[1] + env._drawer_slide_y_sign() * np.random.uniform(*env._drawer_block_y_range)
            target_pos = np.array([x, y, 0.076])
            subgoals.append(dict(task='cube', block=0, target_pos=target_pos, name='mistake_cube'))
        return subgoals

    def _choose_subgoal(self, info):
        goal_subgoals = [
            subgoal for subgoal in [self._drawer_subgoal(info), self._window_subgoal(info)] if subgoal is not None
        ]
        if len(goal_subgoals) == 0:
            return None

        mistake_subgoals = self._mistake_subgoals(info)
        if len(mistake_subgoals) > 0 and np.random.uniform() < self._p_mistake:
            return mistake_subgoals[int(np.random.randint(len(mistake_subgoals)))]
        return goal_subgoals[int(np.random.randint(len(goal_subgoals)))]

    def _planner_target(self, subgoal, info):
        task = subgoal['task']

        if task == 'button':
            button = subgoal['button']
            button_site_id = self._env.unwrapped._button_site_ids[button]
            button_site_xmat = self._env.unwrapped._data.site_xmat[button_site_id].reshape(3, 3)
            return dict(
                button=button,
                button_state=subgoal['target_state'],
                button_top_pos=self._env.unwrapped._data.site_xpos[button_site_id].copy(),
                button_top_yaw=np.arctan2(button_site_xmat[1, 0], button_site_xmat[0, 0]),
            )
        if task == 'drawer':
            target_pos = subgoal['target_pos']
            handle_pos = info['privileged/drawer_handle_pos'].copy()
            handle_pos[1] += self._env.unwrapped._drawer_slide_y_sign() * (
                target_pos - info['privileged/drawer_pos'][0]
            )
            return dict(drawer_pos=np.array([target_pos]), drawer_handle_pos=handle_pos)
        if task == 'window':
            target_pos = subgoal['target_pos']
            handle_pos = info['privileged/window_handle_pos'].copy()
            handle_pos[0] += self._env.unwrapped._window_slide_x_sign() * (
                target_pos - info['privileged/window_pos'][0]
            )
            return dict(window_pos=np.array([target_pos]), window_handle_pos=handle_pos)
        if task == 'cube':
            target_pos = np.asarray(subgoal['target_pos']).copy()
            if subgoal['name'] == 'put_cube_in_drawer':
                target_pos[2] += 0.09
            return dict(block=subgoal['block'], block_pos=target_pos, block_yaw=0.0)

        return {}

    def _target_info(self, subgoal, info):
        target = self._planner_target(subgoal, info)
        if subgoal['task'] == 'cube':
            target['block_pos'] = np.asarray(subgoal['target_pos']).copy()
        return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in target.items()}

    def _subtask_success(self, subgoal, info):
        task = subgoal['task']
        if task == 'button':
            return bool(info[f'privileged/button_{subgoal["button"]}_state'] == subgoal['target_state'])
        if task == 'drawer':
            return bool(np.abs(info['privileged/drawer_pos'][0] - subgoal['target_pos']) <= 0.04)
        if task == 'window':
            return bool(np.abs(info['privileged/window_pos'][0] - subgoal['target_pos']) <= 0.04)
        if task == 'cube':
            block = subgoal.get('block', 0)
            if subgoal['name'] == 'put_cube_in_drawer':
                return bool(self._cube_successes(info)[block])
            return bool(
                np.linalg.norm(info[f'privileged/block_{block}_pos'] - np.asarray(subgoal['target_pos'])) <= 0.04
            )
        return False

    def _start_segment(self, subgoal, info, transition_idx):
        super()._start_segment(
            subtask_name=subgoal['task'],
            transition_idx=transition_idx,
            target=self._target_info(subgoal, info),
            is_mistake=subgoal['name'].startswith('mistake'),
        )

    def _close_segment(self, end_idx, info):
        if self._active_segment is None:
            return
        super()._close_segment(end_idx, self._subtask_success(self._active_subgoal, info))

    def _reset_active(self, ob, info, transition_idx):
        subgoal = self._choose_subgoal(info)
        self._active_subgoal = subgoal
        if subgoal is None:
            self._done = True
            self._active_agent = None
            return

        self._done = False
        self._active_steps = 0
        self._active_settle_steps = 0
        self._subgoal_counts[subgoal['name']] += 1
        self._start_segment(subgoal, info, transition_idx)
        self._active_agent = self._agents[subgoal['task']]
        self._active_agent.reset(ob, info, self._planner_target(subgoal, info))

    def reset(self, ob, info, transition_idx=0):
        self._reset_episode_state(transition_idx)
        self._active_settle_steps = 0
        self._goal_order = np.random.permutation(self._env.unwrapped._num_cubes)
        self._block_goal_xyzs = None
        if self._env.unwrapped.cur_task_info['cube_success_type'] == 'stack_anywhere':
            self._block_goal_xyzs = self._sample_stack_anywhere_goal(info)
        elif self._env.unwrapped.cur_task_info['cube_success_type'] == 'in_drawer':
            self._block_goal_xyzs = self._in_drawer_goal()
        elif self._env.unwrapped.cur_task_info['cube_success_type'] == 'on_floor':
            self._block_goal_xyzs = np.array(
                [
                    info[f'privileged/block_{i}_pos']
                    for i in range(self._env.unwrapped._num_cubes)
                ]
            )
        self._agents = {
            'cube': CubePrimitive(
                env=self._env,
                segment_dt=self._segment_dt,
            ),
            'button': ButtonPrimitive(
                env=self._env,
                segment_dt=self._segment_dt,
            ),
            'drawer': DrawerPrimitive(
                env=self._env,
                segment_dt=self._segment_dt,
            ),
            'window': WindowPrimitive(
                env=self._env,
                segment_dt=self._segment_dt,
            ),
        }
        self._subgoal_counts.clear()
        self._reset_active(ob, info, transition_idx)

    def finish_episode(self, end_idx, info):
        self._close_segment(end_idx, info)

    def select_action(self, ob, info, transition_idx=None):
        transition_idx = self._next_transition_idx_or(transition_idx)

        if (
            self._active_agent is not None
            and self._active_agent.done
            and self._active_subgoal is not None
            and self._active_subgoal['name'] == 'put_cube_in_drawer'
            and self._active_settle_steps < self._cube_settle_steps
            and self._active_steps < self._max_subgoal_steps
            and not self._subtask_success(self._active_subgoal, info)
        ):
            self._active_settle_steps += 1
            self._active_steps += 1
            return np.zeros(self._env.action_space.shape)

        if self._active_agent is None or self._active_agent.done or self._active_steps >= self._max_subgoal_steps:
            self._close_segment(transition_idx - 1, info)
            self._reset_active(ob, info, transition_idx)
        if self._done:
            return np.zeros(self._env.action_space.shape)

        self._active_steps += 1
        return self._active_agent.select_action(ob, info)
