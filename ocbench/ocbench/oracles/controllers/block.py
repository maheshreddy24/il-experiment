import numpy as np

from ocbench import lie
from ocbench.oracles.controllers.controller import Controller
from ocbench.oracles.primitives.cube import CubePrimitive


class BlockController(Controller):
    def __init__(
        self,
        env,
        p_mistake=0.1,
        max_subgoal_steps=1250,
        speed_profile=(0.7, 1.0, 1.5),
    ):
        super().__init__(
            env=env,
            p_mistake=p_mistake,
            max_subgoal_steps=max_subgoal_steps,
            speed_profile=speed_profile,
        )
        self._agent = None
        self._goal_order = None
        self._stack_anywhere_goal_xyzs = None

    def _block_pos(self, info, block):
        return info[f'privileged/block_{block}_pos']

    def _target_pos(self, block):
        if self._stack_anywhere_goal_xyzs is not None:
            return self._stack_anywhere_goal_xyzs[block].copy()
        mocap_id = self._env.unwrapped._cube_target_mocap_ids[block]
        return self._env.unwrapped._data.mocap_pos[mocap_id].copy()

    def _target_yaw(self, block):
        if self._stack_anywhere_goal_xyzs is not None:
            return 0.0
        mocap_id = self._env.unwrapped._cube_target_mocap_ids[block]
        return lie.SO3(wxyz=self._env.unwrapped._data.mocap_quat[mocap_id]).compute_yaw_radians()

    def _sample_stack_anywhere_goal(self, info, p_existing_floor_xy=0.2):
        env = self._env.unwrapped
        floor_blocks = [i for i in range(env._num_cubes) if abs(self._block_pos(info, i)[2] - env._cube_size) <= 0.03]

        floor_block = None
        if len(floor_blocks) > 0 and np.random.uniform() < p_existing_floor_xy:
            floor_block = int(np.random.choice(floor_blocks))
            xy = self._block_pos(info, floor_block)[:2].copy()
        else:
            xy = np.random.uniform(*env._target_sampling_bounds)

        order = np.random.permutation(env._num_cubes)
        if floor_block is not None:
            order = np.concatenate([[floor_block], order[order != floor_block]])
        goal_xyzs = np.zeros((env._num_cubes, 3))
        for level, block_idx in enumerate(order):
            goal_xyzs[block_idx] = [xy[0], xy[1], env._cube_size * (2 * level + 1)]
        return goal_xyzs

    def _target_successes(self, info):
        return [
            np.linalg.norm(self._block_pos(info, i) - self._target_pos(i)) <= 0.04
            for i in range(self._env.unwrapped._num_cubes)
        ]

    def _top_blocks(self, info):
        block_xyzs = np.array([self._block_pos(info, i) for i in range(self._env.unwrapped._num_cubes)])
        top_blocks = []
        for i in range(self._env.unwrapped._num_cubes):
            for j in range(self._env.unwrapped._num_cubes):
                if i == j:
                    continue
                if (
                    block_xyzs[j][2] > block_xyzs[i][2]
                    and np.linalg.norm(block_xyzs[i][:2] - block_xyzs[j][:2]) < self._env.unwrapped._cube_size
                ):
                    break
            else:
                top_blocks.append(i)
        return top_blocks

    def _sample_clear_pos(self, info):
        env = self._env.unwrapped
        avoid_xys = [self._block_pos(info, i)[:2] for i in range(env._num_cubes)]
        avoid_xys.extend([self._target_pos(i)[:2] for i in range(env._num_cubes)])
        for _ in range(100):
            xy = np.random.uniform(*env._target_sampling_bounds)
            if all(np.linalg.norm(xy - other_xy) >= env._min_object_init_dist for other_xy in avoid_xys):
                break
        return np.array([*xy, env._cube_size])

    def _overlaps_pos(self, block_pos, target_pos):
        env = self._env.unwrapped
        xy_overlap = np.linalg.norm(block_pos[:2] - target_pos[:2]) <= env._cube_size
        z_overlap = abs(block_pos[2] - target_pos[2]) <= env._cube_size
        return xy_overlap and z_overlap

    def _clear_target_subgoal(self, info, target_block, successes):
        target_pos = self._target_pos(target_block)
        for block in self._top_blocks(info):
            if block == target_block or successes[block]:
                continue
            block_pos = self._block_pos(info, block)
            if self._overlaps_pos(block_pos, target_pos):
                return dict(
                    block=block,
                    target_pos=self._sample_clear_pos(info),
                    target_yaw=np.random.uniform(0, 2 * np.pi),
                    name='clear_obstacle',
                )
        return None

    def _make_goal_or_clear_subgoal(self, info, block, successes):
        clear_subgoal = self._clear_target_subgoal(info, block, successes)
        if clear_subgoal is not None:
            return clear_subgoal
        return self._make_goal_subgoal(block)

    def _goal_subgoal(self, info):
        if all(self._env.unwrapped._compute_successes()):
            return None

        task_name = self._env.unwrapped.cur_task_info['task_name']
        successes = self._target_successes(info)

        if task_name == 'move':
            return None if successes[0] else self._make_goal_or_clear_subgoal(info, 0, successes)
        if task_name == 'double_pnp':
            for block in self._goal_order:
                if not successes[block]:
                    return self._make_goal_or_clear_subgoal(info, block, successes)
            return None
        if task_name == 'stack_anywhere':
            target_zs = np.array([self._target_pos(i)[2] for i in range(self._env.unwrapped._num_cubes)])
            for block in np.argsort(target_zs):
                if not successes[block]:
                    return self._make_goal_subgoal(block)
            return None
        if task_name in {'stack', 'grid'}:
            target_zs = np.array([self._target_pos(i)[2] for i in range(self._env.unwrapped._num_cubes)])
            for block in np.argsort(target_zs):
                if not successes[block]:
                    return self._make_goal_or_clear_subgoal(info, block, successes)
            return None

        for block in self._goal_order:
            if not successes[block]:
                return self._make_goal_or_clear_subgoal(info, block, successes)
        return None

    def _make_goal_subgoal(self, block):
        return dict(
            block=block,
            target_pos=self._target_pos(block),
            target_yaw=self._target_yaw(block),
            name='goal',
        )

    def _mistake_subgoal(self, info):
        top_blocks = self._top_blocks(info)
        if len(top_blocks) == 0:
            return None

        block = int(np.random.choice(top_blocks))
        stack = len(top_blocks) >= 2 and np.random.uniform() < 0.5
        if stack:
            support_blocks = list(set(top_blocks) - {block})
            support_block = int(np.random.choice(support_blocks))
            support_pos = self._block_pos(info, support_block)
            target_pos = np.array(
                [
                    support_pos[0],
                    support_pos[1],
                    support_pos[2] + 2 * self._env.unwrapped._cube_size,
                ]
            )
        else:
            xy = np.random.uniform(*self._env.unwrapped._target_sampling_bounds)
            target_pos = np.array([*xy, self._env.unwrapped._cube_size])

        return dict(
            block=block,
            target_pos=target_pos,
            target_yaw=np.random.uniform(0, 2 * np.pi),
            name='mistake',
        )

    def _choose_subgoal(self, info):
        goal_subgoal = self._goal_subgoal(info)
        if goal_subgoal is None:
            return None

        mistake_subgoal = self._mistake_subgoal(info)
        if mistake_subgoal is not None and np.random.uniform() < self._p_mistake:
            return mistake_subgoal
        return goal_subgoal

    def _target_info(self, subgoal, info=None):
        return dict(
            block=subgoal['block'],
            block_pos=np.asarray(subgoal['target_pos']).copy(),
            block_yaw=subgoal['target_yaw'],
        )

    def _subtask_success(self, subgoal, info):
        block_pos = self._block_pos(info, subgoal['block'])
        return bool(np.linalg.norm(block_pos - np.asarray(subgoal['target_pos'])) <= 0.04)

    def _start_segment(self, subgoal, transition_idx):
        super()._start_segment(
            subtask_name='cube',
            transition_idx=transition_idx,
            target=self._target_info(subgoal),
            is_mistake=subgoal['name'] == 'mistake',
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
            return

        self._done = False
        self._active_steps = 0
        self._start_segment(subgoal, transition_idx)
        self._agent.reset(ob, info, self._target_info(subgoal))

    def reset(self, ob, info, transition_idx=0):
        self._reset_episode_state(transition_idx)
        self._stack_anywhere_goal_xyzs = None
        self._agent = CubePrimitive(
            env=self._env,
            segment_dt=self._segment_dt,
        )
        self._goal_order = np.random.permutation(self._env.unwrapped._num_cubes)
        if self._env.unwrapped.cur_task_info['task_name'] == 'stack_anywhere':
            self._stack_anywhere_goal_xyzs = self._sample_stack_anywhere_goal(info)
        self._reset_active(ob, info, transition_idx)

    def finish_episode(self, end_idx, info):
        self._close_segment(end_idx, info)

    def select_action(self, ob, info, transition_idx=None):
        transition_idx = self._next_transition_idx_or(transition_idx)

        if self._agent.done or self._active_steps >= self._max_subgoal_steps:
            self._close_segment(transition_idx - 1, info)
            self._reset_active(ob, info, transition_idx)
        if self._done:
            return np.zeros(self._env.action_space.shape)

        self._active_steps += 1
        return self._agent.select_action(ob, info)
