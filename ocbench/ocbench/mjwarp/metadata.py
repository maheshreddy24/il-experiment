from typing import Any

import numpy as np
import warp as wp

from ocbench.oracles.controllers.block import BlockController
from ocbench.oracles.controllers.chamber import ChamberController
from ocbench.oracles.controllers.hanoi import HanoiController
from ocbench.oracles.controllers.switch import SwitchController


@wp.kernel
def record_episode_outcomes(
    done: wp.array(dtype=int),
    healthy: wp.array(dtype=int),
    failure: wp.array(dtype=int),
    episode_healthy: wp.array(dtype=int),
    episode_failure: wp.array(dtype=int),
):
    world = wp.tid()
    if done[world] == 0:
        episode_healthy[world] = episode_healthy[world] & healthy[world]
        episode_failure[world] = episode_failure[world] | failure[world]


@wp.kernel
def record_final_state(values: wp.array2d(dtype=Any), done: wp.array(dtype=int), output: wp.array2d(dtype=Any)):
    world, col = wp.tid()
    if done[world] == 0:
        output[world, col] = values[world, col]


class MjWarpMetadata:
    """Record controller choices on GPU, then label segments with the CPU definitions."""

    def __init__(self, family, env, controller, max_steps, state_arrays):
        self.family = family
        self.env = env.cpu_env
        self.controller = controller
        self.state_arrays = state_arrays
        self.final_state = {key: wp.empty_like(value) for key, value in state_arrays.items()}
        self.fields = {}
        if family == 'bowling':
            self.bowling_targets = {
                key: getattr(controller, key).numpy()
                for key in (
                    'ball_start_pos',
                    'push_dir',
                    'push_speed',
                    'push_steps',
                    'push_ramp_steps',
                    'push_z',
                    'gripper',
                    'tilt',
                    'tilt_drift',
                )
            }
        else:
            names = ['active_steps', 'controller_done', 'is_mistake']
            names += {
                'block': ['target_block', 'target_pos', 'target_yaw'],
                'hanoi': ['target_disk', 'target_source', 'target_pos', 'target_yaw'],
                'switch': ['target_button', 'target_state'],
                'chamber': [
                    'active_task',
                    'target_button',
                    'target_slide_pos',
                    'target_block',
                    'target_pos',
                    'target_yaw',
                    'settle_after_cube',
                ],
            }[family]
            self.fields = {key: getattr(controller, key) for key in names}
            self.cpu_controller = {
                'block': BlockController,
                'chamber': ChamberController,
                'hanoi': HanoiController,
                'switch': SwitchController,
            }[family](self.env)
        self.buffers = {
            key: wp.empty((max_steps, env.nworld), dtype=value.dtype, device=value.device)
            for key, value in self.fields.items()
        }

    def record_targets(self, step):
        for key, value in self.fields.items():
            wp.copy(self.buffers[key], value, dest_offset=step * value.size, count=value.size)

    def record_state(self):
        for key, value in self.state_arrays.items():
            wp.launch(
                record_final_state,
                dim=value.shape,
                inputs=[value, self.controller.done, self.final_state[key]],
                device=value.device,
            )

    def _restore_state(self, state):
        self.env.set_state(**state)
        return self.env.compute_ob_info()

    def finish(self, dataset, lengths, healthy):
        traces = {key: value.numpy() for key, value in self.buffers.items()}
        final_state = {key: value.numpy() for key, value in self.final_state.items()}
        p_mistake = getattr(self.controller, 'p_mistake', 0.1 if self.family == 'block' else None)
        if self.family == 'switch':
            p_mistake = p_mistake.numpy()
        if self.family == 'bowling':
            bowling_scores = self.controller.episode_score.numpy()
        episodes = []
        offset = 0
        for world, length in enumerate(lengths):
            episode = dict(
                task_name=self.env.cur_task_info['task_name'],
                task_id=self.env.cur_task_id,
                p_mistake=float(p_mistake[world]) if self.family == 'switch' else p_mistake,
                segments=[],
            )
            # Invalid episodes will be replaced by the collector.
            state = {key: dataset[key][offset] for key in self.state_arrays}
            if 'dynamics_info' in dataset:
                state['dynamics_info'] = dataset['dynamics_info'][offset]
            finite = True
            for key, value in dataset.items():
                start = offset + world if key == 'observations' else offset
                count = length + 1 if key == 'observations' else length
                finite &= np.isfinite(value[start : start + count]).all()
            if not healthy[world] or not finite:
                episodes.append(episode)
                offset += length
                continue
            if self.family == 'bowling':
                values = {key: value[world] for key, value in self.bowling_targets.items()}
                pin_addr = self.env.model.jnt_qposadr[self.env.model.joint('pin_joint_0').id]
                target = dict(
                    ball_pos=values['ball_start_pos'].astype(np.float64),
                    pin_head_pos=state['qpos'][pin_addr : pin_addr + 2].astype(np.float64),
                    push_angle=float(np.arctan2(values['push_dir'][0], values['push_dir'][1])),
                    push_dir=values['push_dir'][:2].astype(np.float64),
                    push_speed=float(values['push_speed']),
                    push_steps=int(values['push_steps']),
                    push_ramp_steps=int(values['push_ramp_steps']),
                    push_z=float(values['push_z']),
                    gripper=float(values['gripper']),
                    tilt=values['tilt'][:2].astype(np.float64),
                    tilt_drift=values['tilt_drift'][:2].astype(np.float64),
                )
                episode['segments'].append(
                    dict(
                        segment_idx=0,
                        subtask_name='bowling',
                        start=0,
                        end=int(length - 1),
                        is_potential_high_level_mistake=False,
                        subtask_success=False,
                        subtask_return=0.1 * int(bowling_scores[world]),
                        target=target,
                    )
                )
            else:
                active = traces['controller_done'][:length, world] == 0
                # The first action of each new subgoal has active_steps == 1.
                starts = np.flatnonzero((traces['active_steps'][:length, world] == 1) & active)
                stopped = np.flatnonzero(~active)
                last_end = int(stopped[0] - 1) if len(stopped) else int(length - 1)
                ends = np.concatenate([starts[1:] - 1, [last_end]])
                for start, end in zip(starts, ends):
                    cur = {key: value[start, world] for key, value in traces.items()}
                    state = {key: dataset[key][offset + start] for key in self.state_arrays}
                    if 'dynamics_info' in dataset:
                        state['dynamics_info'] = dataset['dynamics_info'][offset + start]
                    info = self._restore_state(state)
                    name = 'mistake' if cur['is_mistake'] else 'goal'
                    if self.family == 'block':
                        subtask = 'cube'
                        subgoal = dict(
                            block=int(cur['target_block']),
                            target_pos=cur['target_pos'].astype(np.float64),
                            target_yaw=float(cur['target_yaw']),
                            name=name,
                        )
                    elif self.family == 'hanoi':
                        subtask = 'hanoi'
                        pegs = state['dynamics_info'].reshape(-1, 4)[:, :2]
                        subgoal = dict(
                            disk=int(cur['target_disk']),
                            source=int(cur['target_source']),
                            target=int(np.argmin(np.linalg.norm(pegs - cur['target_pos'][:2], axis=1))),
                            target_pos=cur['target_pos'].astype(np.float64),
                            target_yaw=float(cur['target_yaw']),
                            grasp=1.0,
                            name=name,
                        )
                    elif self.family == 'switch':
                        subtask = 'button'
                        subgoal = dict(
                            button=int(cur['target_button']), target_state=int(cur['target_state']), name=name
                        )
                    else:
                        subtask = ('cube', 'button', 'drawer', 'window')[int(cur['active_task'])]
                        subgoal = dict(task=subtask, name=name)
                        if subtask == 'cube':
                            pos = cur['target_pos'].astype(np.float64)
                            if cur['settle_after_cube']:
                                pos[2] -= 0.09
                                subgoal['name'] = 'put_cube_in_drawer'
                            subgoal.update(block=int(cur['target_block']), target_pos=pos)
                        elif subtask == 'button':
                            button = int(cur['target_button'])
                            subgoal.update(button=button, target_state=1 - int(state['button_states'][button]))
                        else:
                            subgoal['target_pos'] = float(cur['target_slide_pos'])
                    target = self.cpu_controller._target_info(subgoal, info)
                    if end + 1 < length:
                        end_state = {key: dataset[key][offset + end + 1] for key in self.state_arrays}
                    else:
                        end_state = {key: value[world] for key, value in final_state.items()}
                    if 'dynamics_info' in state:
                        end_state['dynamics_info'] = state['dynamics_info'].copy()
                        if self.family == 'chamber':
                            end_state['dynamics_info'][: self.env._num_buttons] = end_state['button_states']
                    end_info = self._restore_state(end_state)
                    episode['segments'].append(
                        dict(
                            segment_idx=len(episode['segments']),
                            subtask_name=subtask,
                            start=int(start),
                            end=int(end),
                            is_potential_high_level_mistake=bool(cur['is_mistake']),
                            subtask_success=self.cpu_controller._subtask_success(subgoal, end_info),
                            subtask_return=None,
                            target=target,
                        )
                    )
            episodes.append(episode)
            offset += length
        return episodes
