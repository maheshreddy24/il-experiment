import numpy as np

from ocbench.oracles.controllers.controller import Controller
from ocbench.oracles.primitives.hanoi import HanoiPrimitive


class HanoiController(Controller):
    def __init__(
        self,
        env,
        p_mistake=0.002,
        max_subgoal_steps=1500,
        speed_profile=(0.7, 1.0, 1.5),
    ):
        super().__init__(
            env=env,
            p_mistake=p_mistake,
            max_subgoal_steps=max_subgoal_steps,
            speed_profile=speed_profile,
        )
        self._agent = None
        self._target_peg = None

    def _disk_pos(self, info, disk):
        return info[f'privileged/disk_{disk}_pos']

    def _disk_yaw(self, info, disk):
        return info[f'privileged/disk_{disk}_yaw'][0]

    def _infer_stacks(self, info):
        env = self._env.unwrapped
        stacks = [[] for _ in range(env._num_pegs)]
        disk_zs = []
        for disk in range(env._num_disks):
            disk_pos = self._disk_pos(info, disk)
            peg = int(np.argmin(np.linalg.norm(env._peg_positions - disk_pos[:2], axis=1)))
            stacks[peg].append(disk)
            disk_zs.append(disk_pos[2])
        for peg in range(env._num_pegs):
            stacks[peg].sort(key=lambda disk: disk_zs[disk])
        return stacks

    def _tower_success(self, stacks):
        env = self._env.unwrapped
        goal_stack = list(range(env._num_disks - 1, -1, -1))
        return any(stacks[peg] == goal_stack for peg in env.cur_task_info['goal_pegs'])

    def _random_moves(self, stacks):
        moves = []
        for source in range(self._env.unwrapped._num_pegs):
            if len(stacks[source]) == 0:
                continue
            disk = stacks[source][-1]
            for target in range(self._env.unwrapped._num_pegs):
                if target != source:
                    moves.append(dict(disk=disk, source=source, target=target))
        return moves

    def _plan_to_target(self, stacks, target_peg):
        env = self._env.unwrapped
        positions = {}
        for peg, stack in enumerate(stacks):
            for disk in stack:
                positions[disk] = peg
        moves = []

        def move_stack(num_disks, target):
            if num_disks == 0:
                return
            disk = num_disks - 1
            if positions[disk] != target:
                source = positions[disk]
                aux = next(peg for peg in range(env._num_pegs) if peg not in {source, target})
                move_stack(num_disks - 1, aux)
                moves.append(dict(disk=disk, source=source, target=target))
                positions[disk] = target
            move_stack(num_disks - 1, target)

        move_stack(env._num_disks, target_peg)
        return moves

    def _make_subgoal(self, move, name, stacks, info):
        env = self._env.unwrapped
        target_level = len(stacks[move['target']])
        return dict(
            disk=move['disk'],
            source=move['source'],
            target=move['target'],
            target_pos=env._disk_pos_on_peg(move['target'], target_level),
            target_yaw=env._disk_yaw_for_peg(move['target']),
            grasp=1.0,
            name=name,
        )

    def _choose_subgoal(self, info):
        if self._env.unwrapped._failure:
            return None

        stacks = self._infer_stacks(info)
        if self._tower_success(stacks):
            return None

        moves = self._plan_to_target(stacks, self._target_peg)
        if len(moves) == 0:
            return None

        planned_move = moves[0]
        if np.random.uniform() < self._p_mistake:
            candidates = [
                move
                for move in self._random_moves(stacks)
                if not (
                    move['disk'] == planned_move['disk']
                    and move['source'] == planned_move['source']
                    and move['target'] == planned_move['target']
                )
            ]
            if len(candidates) > 0:
                move = candidates[int(np.random.randint(len(candidates)))]
                return self._make_subgoal(move, 'mistake', stacks, info)

        return self._make_subgoal(planned_move, 'goal', stacks, info)

    def _target_info(self, subgoal, info):
        return dict(
            disk=subgoal['disk'],
            source_peg=subgoal['source'],
            peg=subgoal['target'],
            disk_goal_pos=np.asarray(subgoal['target_pos']).copy(),
            disk_goal_yaw=subgoal['target_yaw'],
            disk_grasp=subgoal['grasp'],
        )

    def _subtask_success(self, subgoal, info):
        disk_pos = self._disk_pos(info, subgoal['disk'])
        target_pos = np.asarray(subgoal['target_pos'])
        xy_success = np.linalg.norm(disk_pos[:2] - target_pos[:2]) <= 0.04
        z_success = np.abs(disk_pos[2] - target_pos[2]) <= 0.03
        return bool(xy_success and z_success)

    def _start_segment(self, subgoal, transition_idx):
        super()._start_segment(
            subtask_name='hanoi',
            transition_idx=transition_idx,
            target=self._target_info(subgoal, None),
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
        self._agent.reset(ob, info, self._target_info(subgoal, info))

    def reset(self, ob, info, transition_idx=0):
        self._reset_episode_state(transition_idx)
        valid_goal_pegs = info['privileged/valid_goal_pegs'].astype(int)
        self._target_peg = int(np.random.choice(valid_goal_pegs))
        self._agent = HanoiPrimitive(
            env=self._env,
            segment_dt=self._segment_dt,
        )
        self._reset_active(ob, info, transition_idx)

    def finish_episode(self, end_idx, info):
        self._close_segment(end_idx, info)

    def select_action(self, ob, info, transition_idx=None):
        transition_idx = self._next_transition_idx_or(transition_idx)

        if self._env.unwrapped._failure:
            self._close_segment(transition_idx - 1, info)
            self._done = True
            return np.zeros(self._env.action_space.shape)

        if self._agent.done or self._active_steps >= self._max_subgoal_steps:
            self._close_segment(transition_idx - 1, info)
            self._reset_active(ob, info, transition_idx)
        if self._done:
            return np.zeros(self._env.action_space.shape)

        self._active_steps += 1
        return self._agent.select_action(ob, info)
