from typing import Any

import warp as wp

from ocbench.mjwarp.primitives.primitive_kernels import (
    _mat_transpose,
    _matmul,
    _rotation_log,
)


@wp.func
def _reset_randf(seed: int, counter: int) -> float:
    return wp.randf(wp.rand_init(seed + counter * 9176))


@wp.func
def _reset_rand_uniform(seed: int, counter: int, low: float, high: float) -> float:
    return low + (high - low) * _reset_randf(seed, counter)


@wp.func
def _reset_rand_int(seed: int, counter: int, high: int) -> int:
    value = int(wp.floor(_reset_rand_uniform(seed, counter, 0.0, float(high))))
    if value >= high:
        value = high - 1
    return value


@wp.func
def _dist2_xy(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


@wp.func
def _quat_yaw(q0: float, q1: float, q2: float, q3: float) -> float:
    return wp.atan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))


@wp.func
def _jac_elem(jacp: wp.array3d[float], jacr: wp.array3d[float], world_id: int, row: int, dof: int) -> float:
    if row < 3:
        return jacp[world_id, row, dof]
    return jacr[world_id, row - 3, dof]


@wp.kernel
def set_jac_points(
    site_xpos: wp.array2d[wp.vec3],
    point: wp.array(dtype=wp.vec3),
    body: wp.array(dtype=int),
    attach_site_id: int,
    attach_body_id: int,
):
    world_id = wp.tid()
    point[world_id] = site_xpos[world_id, attach_site_id]
    body[world_id] = attach_body_id


@wp.kernel
def ik_update(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    jacp: wp.array3d[float],
    jacr: wp.array3d[float],
    target_pos: wp.array(dtype=wp.vec3),
    target_xmat: wp.array(dtype=wp.mat33),
    arm_qpos_ids: wp.array(dtype=int),
    arm_dof_ids: wp.array(dtype=int),
    attach_site_id: int,
    damping: float,
    max_angle_change: float,
):
    world_id = wp.tid()

    current_pos = site_xpos[world_id, attach_site_id]
    current_xmat = site_xmat[world_id, attach_site_id]
    target = target_xmat[world_id]

    err = wp.zeros(6, dtype=float)
    err[0] = target_pos[world_id][0] - current_pos[0]
    err[1] = target_pos[world_id][1] - current_pos[1]
    err[2] = target_pos[world_id][2] - current_pos[2]

    rot_error = _rotation_log(_matmul(target, _mat_transpose(current_xmat)))
    err[3] = rot_error[0]
    err[4] = rot_error[1]
    err[5] = rot_error[2]

    h = wp.zeros(36, dtype=float)
    for row in range(6):
        for col in range(6):
            value = float(0.0)
            for joint in range(6):
                dof = arm_dof_ids[joint]
                value += _jac_elem(jacp, jacr, world_id, row, dof) * _jac_elem(jacp, jacr, world_id, col, dof)
            if row == col:
                value += damping
            h[row * 6 + col] = value

    chol = wp.zeros(36, dtype=float)
    for row in range(6):
        for col in range(row + 1):
            value = h[row * 6 + col]
            for k in range(col):
                value -= chol[row * 6 + k] * chol[col * 6 + k]
            if row == col:
                chol[row * 6 + col] = wp.sqrt(wp.max(value, 1.0e-12))
            else:
                chol[row * 6 + col] = value / chol[col * 6 + col]

    y = wp.zeros(6, dtype=float)
    for row in range(6):
        value = err[row]
        for col in range(row):
            value -= chol[row * 6 + col] * y[col]
        y[row] = value / chol[row * 6 + row]

    sol = wp.zeros(6, dtype=float)
    for row_rev in range(6):
        row = 5 - row_rev
        value = y[row]
        for col in range(row + 1, 6):
            value -= chol[col * 6 + row] * sol[col]
        sol[row] = value / chol[row * 6 + row]

    update = wp.zeros(6, dtype=float)
    update_max = float(0.0)
    for joint in range(6):
        dof = arm_dof_ids[joint]
        value = float(0.0)
        for row in range(6):
            value += _jac_elem(jacp, jacr, world_id, row, dof) * sol[row]
        update[joint] = value
        update_max = wp.max(update_max, wp.abs(value))

    update_scale = float(1.0)
    if update_max > max_angle_change:
        update_scale = max_angle_change / update_max
    for joint in range(6):
        qpos[world_id, arm_qpos_ids[joint]] += update[joint] * update_scale


@wp.kernel
def ik_action(
    ik_qpos: wp.array2d[float],
    sim_qpos: wp.array2d[float],
    target_gripper: wp.array(dtype=float),
    action: wp.array2d[float],
    arm_qpos_ids: wp.array(dtype=int),
    gripper_qpos_id: int,
    joint_action_delta: wp.array(dtype=float),
):
    world_id = wp.tid()
    for joint in range(6):
        qpos_id = arm_qpos_ids[joint]
        value = (ik_qpos[world_id, qpos_id] - sim_qpos[world_id, qpos_id]) / joint_action_delta[joint]
        action[world_id, joint] = wp.clamp(value, -1.0, 1.0)

    gripper_opening = sim_qpos[world_id, gripper_qpos_id] / 0.8
    gripper_delta = (wp.clamp(target_gripper[world_id], 0.0, 1.0) - gripper_opening) / joint_action_delta[6]
    action[world_id, 6] = wp.clamp(gripper_delta, -1.0, 1.0)


@wp.kernel
def hold_current_attach_targets(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    target_pos: wp.array(dtype=wp.vec3),
    target_xmat: wp.array(dtype=wp.mat33),
    target_gripper: wp.array(dtype=float),
    attach_site_id: int,
    gripper_qpos_id: int,
):
    world_id = wp.tid()
    target_pos[world_id] = site_xpos[world_id, attach_site_id]
    target_xmat[world_id] = site_xmat[world_id, attach_site_id]
    target_gripper[world_id] = wp.clamp(qpos[world_id, gripper_qpos_id] / 0.8, 0.0, 1.0)


@wp.kernel
def set_control_from_action(
    qpos: wp.array2d[float],
    ctrl: wp.array2d[float],
    action: wp.array2d[float],
    done: wp.array(dtype=int),
    arm_qpos_ids: wp.array(dtype=int),
    arm_actuator_ids: wp.array(dtype=int),
    gripper_actuator_ids: wp.array(dtype=int),
    arm_ctrl_low: wp.array(dtype=float),
    arm_ctrl_high: wp.array(dtype=float),
    joint_action_delta: wp.array(dtype=float),
    gripper_qpos_id: int,
    gripper_actuator_count: int,
):
    world_id = wp.tid()
    is_done = done[world_id] != 0

    for joint in range(6):
        value = float(0.0)
        if not is_done:
            value = wp.clamp(action[world_id, joint], -1.0, 1.0) * joint_action_delta[joint]
        target = qpos[world_id, arm_qpos_ids[joint]] + value
        ctrl[world_id, arm_actuator_ids[joint]] = wp.clamp(target, arm_ctrl_low[joint], arm_ctrl_high[joint])

    gripper_delta = float(0.0)
    if not is_done:
        gripper_delta = wp.clamp(action[world_id, 6], -1.0, 1.0) * joint_action_delta[6]
    gripper_opening = qpos[world_id, gripper_qpos_id] / 0.8
    target_gripper = wp.clamp(gripper_opening + gripper_delta, 0.0, 1.0)
    for actuator_idx in range(gripper_actuator_count):
        ctrl[world_id, gripper_actuator_ids[actuator_idx]] = 255.0 * target_gripper


@wp.kernel
def record_transition(
    action: wp.array2d[float],
    success: wp.array(dtype=int),
    healthy: wp.array(dtype=int),
    done: wp.array(dtype=int),
    episode_length: wp.array(dtype=int),
    actions: wp.array2d[float],
    rewards: wp.array(dtype=float),
    masks: wp.array(dtype=float),
    terminals: wp.array(dtype=int),
    step: int,
    nworld: int,
    max_steps: int,
    action_dim: int,
):
    world_id = wp.tid()
    if done[world_id] != 0:
        return
    row = step * nworld + world_id
    for dim in range(7):
        if dim < action_dim:
            actions[row, dim] = action[world_id, dim]

    is_success = success[world_id] != 0
    is_healthy = healthy[world_id] != 0
    is_terminated = is_success or not is_healthy
    is_truncated = episode_length[world_id] + 1 >= max_steps
    is_done = is_terminated or is_truncated

    reward = float(0.0)
    if is_success and is_healthy:
        reward = 1.0
    rewards[row] = reward
    masks[row] = 1.0
    if is_terminated:
        masks[row] = 0.0
    terminals[row] = int(is_done)


@wp.kernel
def compact_transitions(
    actions: wp.array2d[float],
    rewards: wp.array(dtype=float),
    masks: wp.array(dtype=float),
    terminals: wp.array(dtype=int),
    lengths: wp.array(dtype=int),
    offsets: wp.array(dtype=int),
    compact_actions: wp.array2d[float],
    compact_rewards: wp.array(dtype=float),
    compact_masks: wp.array(dtype=float),
    compact_terminals: wp.array(dtype=int),
    world_start: int,
    world_count: int,
    nworld: int,
    action_dim: int,
):
    tid = wp.tid()
    local_world_id = tid % world_count
    step = tid // world_count
    world_id = world_start + local_world_id
    if step >= lengths[world_id]:
        return

    src = step * nworld + world_id
    dst = offsets[local_world_id] + step
    for dim in range(action_dim):
        compact_actions[dst, dim] = actions[src, dim]
    compact_rewards[dst] = rewards[src]
    compact_masks[dst] = masks[src]
    compact_terminals[dst] = terminals[src]


@wp.kernel
def compact_field(
    values: wp.array2d(dtype=Any),
    lengths: wp.array(dtype=int),
    offsets: wp.array(dtype=int),
    output: wp.array2d[float],
    world_start: int,
    world_count: int,
    nworld: int,
):
    tid, col = wp.tid()
    local_world_id = tid % world_count
    step = tid // world_count
    world_id = world_start + local_world_id
    if step < lengths[world_id]:
        output[offsets[local_world_id] + step, col] = float(values[step * nworld + world_id, col])
