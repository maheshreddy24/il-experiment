import numpy as np

from ocbench.oracles.primitives.button import ButtonPrimitive
from ocbench.oracles.controllers.controller import Controller


class SwitchController(Controller):
    def __init__(
        self,
        env,
        p_mistake_range=(0.2, 1.0),
        max_subgoal_steps=450,
        speed_profile=(0.7, 1.0, 1.5),
    ):
        super().__init__(
            env=env,
            p_mistake=None,
            max_subgoal_steps=max_subgoal_steps,
            speed_profile=speed_profile,
        )
        self._p_mistake_range = p_mistake_range
        self._agent = None
        self._target_color = None
        self._pending_buttons = []
        self._last_button_states = None
        self._toggle_masks = self._make_toggle_masks()

    def _make_toggle_masks(self):
        env = self._env.unwrapped
        masks = np.zeros((env._num_buttons, env._num_buttons), dtype=np.uint8)
        for button in range(env._num_buttons):
            x, y = button // env._num_cols, button % env._num_cols
            for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < env._num_rows and 0 <= ny < env._num_cols:
                    masks[button, nx * env._num_cols + ny] = 1
        return masks

    def _active_buttons(self, button_states):
        return np.flatnonzero(np.asarray(button_states) != self._env.unwrapped._dropped_button_state)

    def _active_toggle_mask(self, button, button_states):
        mask = self._toggle_masks[button].copy()
        active_buttons = np.zeros(self._env.unwrapped._num_buttons, dtype=bool)
        active_buttons[self._active_buttons(button_states)] = True
        if not active_buttons[button]:
            mask[:] = 0
        else:
            mask[~active_buttons] = 0
        return mask

    def _solve(self, button_states, target_color):
        button_states = np.asarray(button_states, dtype=np.uint8)
        active_buttons = self._active_buttons(button_states)
        a = self._toggle_masks[np.ix_(active_buttons, active_buttons)].T.copy()
        b = (button_states[active_buttons] ^ np.uint8(target_color)).copy()
        n = a.shape[1]
        mat = np.concatenate([a, b[:, None]], axis=1)

        rank = 0
        pivots = []
        for col in range(n):
            pivot = None
            for row in range(rank, n):
                if mat[row, col]:
                    pivot = row
                    break
            if pivot is None:
                continue
            if pivot != rank:
                mat[[rank, pivot]] = mat[[pivot, rank]]
            for row in range(n):
                if row != rank and mat[row, col]:
                    mat[row] ^= mat[rank]
            pivots.append(col)
            rank += 1

        for row in range(rank, n):
            if not np.any(mat[row, :n]) and mat[row, n]:
                raise ValueError('Switch state is not solvable for the selected target color.')

        pivot_set = set(pivots)
        free_cols = [col for col in range(n) if col not in pivot_set]
        solution = np.zeros(n, dtype=np.uint8)
        for row, col in enumerate(pivots):
            solution[col] = mat[row, n]

        basis = []
        for free_col in free_cols:
            vec = np.zeros(n, dtype=np.uint8)
            vec[free_col] = 1
            for row, col in enumerate(pivots):
                vec[col] = mat[row, free_col]
            basis.append(vec)

        best = solution
        for mask in range(1, 1 << len(basis)):
            candidate = solution.copy()
            for i, vec in enumerate(basis):
                if (mask >> i) & 1:
                    candidate ^= vec
            if candidate.sum() < best.sum():
                best = candidate
        full_solution = np.zeros(self._env.unwrapped._num_buttons, dtype=np.uint8)
        full_solution[active_buttons] = best
        return full_solution

    def _reset_plan(self, info):
        self._target_color = int(np.random.randint(self._env.unwrapped._num_button_colors))
        solution = self._solve(info['button_states'], self._target_color)
        self._pending_buttons = np.flatnonzero(solution).astype(int).tolist()
        np.random.shuffle(self._pending_buttons)

    def _toggle_pending(self, button):
        if button in self._pending_buttons:
            self._pending_buttons.remove(button)
        else:
            idx = int(np.random.randint(len(self._pending_buttons) + 1))
            self._pending_buttons.insert(idx, button)

    def _pressed_button(self, prev_button_states, button_states):
        prev_button_states = np.asarray(prev_button_states, dtype=np.uint8)
        button_states = np.asarray(button_states, dtype=np.uint8)
        if np.any(
            (prev_button_states == self._env.unwrapped._dropped_button_state)
            != (button_states == self._env.unwrapped._dropped_button_state)
        ):
            return None

        active_buttons = self._active_buttons(button_states)
        diff = np.zeros(self._env.unwrapped._num_buttons, dtype=np.uint8)
        diff[active_buttons] = prev_button_states[active_buttons] ^ button_states[active_buttons]
        if not np.any(diff):
            return None
        matches = [
            button for button in active_buttons if np.all(self._active_toggle_mask(button, button_states) == diff)
        ]
        if len(matches) == 1:
            return int(matches[0])
        return None

    def _update_plan_from_info(self, info):
        if self._last_button_states is None:
            self._last_button_states = info['button_states'].copy()
            return

        pressed_button = self._pressed_button(self._last_button_states, info['button_states'])
        if pressed_button is not None:
            self._toggle_pending(pressed_button)
        elif np.any(self._last_button_states != info['button_states']):
            solution = self._solve(info['button_states'], self._target_color)
            self._pending_buttons = np.flatnonzero(solution).astype(int).tolist()
            np.random.shuffle(self._pending_buttons)
        self._last_button_states = info['button_states'].copy()

    def _choose_subgoal(self, info):
        if self._env.unwrapped._is_monochrome(info['button_states']):
            return None
        if len(self._pending_buttons) == 0:
            return None

        planned_button = self._pending_buttons[0]
        if np.random.uniform() < self._p_mistake:
            pending = set(self._pending_buttons)
            active_buttons = self._active_buttons(info['button_states']).astype(int).tolist()
            candidates = [button for button in active_buttons if button not in pending]
            if len(candidates) == 0:
                candidates = [button for button in active_buttons if button != planned_button]
            if len(candidates) > 0:
                button = int(np.random.choice(candidates))
                return dict(button=button, target_state=1 - info[f'privileged/button_{button}_state'], name='mistake')

        return dict(
            button=planned_button,
            target_state=1 - info[f'privileged/button_{planned_button}_state'],
            name='goal',
        )

    def _target_info(self, subgoal, info):
        button = subgoal['button']
        button_site_id = self._env.unwrapped._button_site_ids[button]
        button_site_xmat = self._env.unwrapped._data.site_xmat[button_site_id].reshape(3, 3)
        return dict(
            button=button,
            button_state=subgoal['target_state'],
            button_top_pos=self._env.unwrapped._data.site_xpos[button_site_id].copy(),
            button_top_yaw=np.arctan2(button_site_xmat[1, 0], button_site_xmat[0, 0]),
        )

    def _subtask_success(self, subgoal, info):
        return bool(info[f'privileged/button_{subgoal["button"]}_state'] == subgoal['target_state'])

    def _start_segment(self, subgoal, info, transition_idx):
        super()._start_segment(
            subtask_name='button',
            transition_idx=transition_idx,
            target=self._target_info(subgoal, info),
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
        self._start_segment(subgoal, info, transition_idx)
        self._agent.reset(ob, info, self._target_info(subgoal, info))

    def reset(self, ob, info, transition_idx=0):
        self._reset_episode_state(transition_idx)
        self._p_mistake = np.random.uniform(*self._p_mistake_range)
        self._agent = ButtonPrimitive(
            env=self._env,
            segment_dt=self._segment_dt,
            gripper_always_closed=True,
        )
        self._reset_plan(info)
        self._last_button_states = info['button_states'].copy()
        self._reset_active(ob, info, transition_idx)

    def finish_episode(self, end_idx, info):
        self._close_segment(end_idx, info)

    def select_action(self, ob, info, transition_idx=None):
        transition_idx = self._next_transition_idx_or(transition_idx)

        self._update_plan_from_info(info)
        if self._agent.done or self._active_steps >= self._max_subgoal_steps:
            self._close_segment(transition_idx - 1, info)
            self._reset_active(ob, info, transition_idx)
        if self._done:
            return np.zeros(self._env.action_space.shape)

        self._active_steps += 1
        return self._agent.select_action(ob, info)
