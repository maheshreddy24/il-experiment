import numpy as np

from ocbench.oracles.primitives.bowling import BowlingPrimitive
from ocbench.oracles.controllers.controller import Controller


class BowlingController(Controller):
    def __init__(
        self,
        env,
        max_subgoal_steps=500,
        speed_profile=(1.0,),
    ):
        super().__init__(
            env=env,
            p_mistake=None,
            max_subgoal_steps=max_subgoal_steps,
            speed_profile=speed_profile,
        )
        self._agent = None

    def _target_info(self):
        return self._agent.target_info()

    def _subtask_return(self):
        return 0.1 * int(self._env.unwrapped._pin_knocked.sum())

    def _start_segment(self, target, transition_idx):
        super()._start_segment(
            subtask_name='bowling',
            transition_idx=transition_idx,
            target=target,
            is_mistake=False,
        )

    def _close_segment(self, end_idx, info):
        if self._active_segment is None:
            return
        super()._close_segment(end_idx, False, self._subtask_return())

    def _reset_active(self, ob, info, transition_idx):
        self._done = False
        self._active_steps = 0
        self._agent.reset(ob, info)
        self._active_subgoal = self._target_info()
        self._start_segment(self._active_subgoal, transition_idx)

    def reset(self, ob, info, transition_idx=0):
        self._reset_episode_state(transition_idx)
        self._agent = BowlingPrimitive(
            env=self._env,
            segment_dt=self._segment_dt,
        )
        self._reset_active(ob, info, transition_idx)

    def finish_episode(self, end_idx, info):
        self._close_segment(end_idx, info)

    def select_action(self, ob, info, transition_idx=None):
        if self._done:
            return np.zeros(self._env.action_space.shape)

        self._next_transition_idx_or(transition_idx)
        self._active_steps += 1
        action = self._agent.select_action(ob, info)
        if self._active_steps >= self._max_subgoal_steps:
            self._done = True
        return action
