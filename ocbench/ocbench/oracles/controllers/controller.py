import numpy as np


class Controller:
    def __init__(
        self,
        env,
        p_mistake=None,
        max_subgoal_steps=1000,
        speed_profile=(0.7, 1.0, 1.5),
        lite_segment_dt_scale=0.5,
    ):
        self._env = env
        self._p_mistake = p_mistake
        self._max_subgoal_steps = max_subgoal_steps
        self._segment_dt_profile = np.asarray(sorted(speed_profile), dtype=np.float64)
        self._lite_segment_dt_scale = lite_segment_dt_scale
        self._segment_dt = None
        self._done = False
        self._active_subgoal = None
        self._active_agent = None
        self._active_steps = 0
        self._segments = []
        self._active_segment = None
        self._next_transition_idx = 0

    @property
    def done(self):
        return self._done

    @property
    def speed(self):
        return None if self._segment_dt is None else float(1.0 / self._segment_dt)

    @property
    def p_mistake(self):
        return self._p_mistake

    @property
    def segments(self):
        return self._segments

    def _sample_segment_dt(self):
        return np.random.uniform(self._segment_dt_profile[0], self._segment_dt_profile[-1])

    def _reset_episode_state(self, transition_idx):
        self._done = False
        self._active_subgoal = None
        self._active_agent = None
        self._active_steps = 0
        self._segments = []
        self._active_segment = None
        self._next_transition_idx = transition_idx
        self._segment_dt = self._sample_segment_dt()
        if self._env.unwrapped._lite:
            self._segment_dt *= self._lite_segment_dt_scale

    def _next_transition_idx_or(self, transition_idx):
        if transition_idx is None:
            transition_idx = self._next_transition_idx
            self._next_transition_idx += 1
        return transition_idx

    def _start_segment(self, subtask_name, transition_idx, target, is_mistake):
        self._active_segment = dict(
            segment_idx=len(self._segments),
            subtask_name=subtask_name,
            start=transition_idx,
            end=None,
            is_potential_high_level_mistake=is_mistake,
            subtask_success=None,
            subtask_return=None,
            target=target,
        )

    def _close_segment(self, end_idx, subtask_success, subtask_return=None):
        if self._active_segment is None:
            return

        self._active_segment['end'] = end_idx
        self._active_segment['subtask_success'] = subtask_success
        self._active_segment['subtask_return'] = subtask_return
        self._segments.append(self._active_segment)
        self._active_segment = None
