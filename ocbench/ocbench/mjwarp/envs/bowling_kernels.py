import warp as wp

from ocbench.mjwarp.envs.manipulation_kernels import _reset_rand_uniform
from ocbench.mjwarp.primitives.primitive_kernels import (
    _mat_vec,
    _matmul,
)

MAX_BOWLING_PINS = 10


@wp.func
def _pin_knocked(q1: float, q2: float) -> int:
    return int(1.0 - 2.0 * (q1 * q1 + q2 * q2) < 0.75)


@wp.kernel
def park_done_bowling_worlds(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    ctrl: wp.array2d[float],
    done: wp.array(dtype=int),
    ball_qpos_addr: int,
    pin_qpos_addrs: wp.array(dtype=int),
    arm_qpos_ids: wp.array(dtype=int),
    arm_actuator_ids: wp.array(dtype=int),
    gripper_actuator_ids: wp.array(dtype=int),
    gripper_qpos_id: int,
    num_pins: int,
    nv: int,
    gripper_actuator_count: int,
):
    world_id = wp.tid()
    if done[world_id] == 0:
        return

    for dof in range(nv):
        qvel[world_id, dof] = 0.0

    for joint in range(6):
        ctrl[world_id, arm_actuator_ids[joint]] = qpos[world_id, arm_qpos_ids[joint]]
    gripper_opening = wp.clamp(qpos[world_id, gripper_qpos_id] / 0.8, 0.0, 1.0)
    for actuator_idx in range(gripper_actuator_count):
        ctrl[world_id, gripper_actuator_ids[actuator_idx]] = 255.0 * gripper_opening

    qpos[world_id, ball_qpos_addr + 0] = 5.0
    qpos[world_id, ball_qpos_addr + 1] = 5.0
    qpos[world_id, ball_qpos_addr + 2] = 5.0
    qpos[world_id, ball_qpos_addr + 3] = 1.0
    qpos[world_id, ball_qpos_addr + 4] = 0.0
    qpos[world_id, ball_qpos_addr + 5] = 0.0
    qpos[world_id, ball_qpos_addr + 6] = 0.0

    for pin in range(MAX_BOWLING_PINS):
        if pin < num_pins:
            addr = pin_qpos_addrs[pin]
            qpos[world_id, addr + 0] = 5.0 + 0.25 * float(pin)
            qpos[world_id, addr + 1] = 5.0
            qpos[world_id, addr + 2] = 5.0
            qpos[world_id, addr + 3] = 1.0
            qpos[world_id, addr + 4] = 0.0
            qpos[world_id, addr + 5] = 0.0
            qpos[world_id, addr + 6] = 0.0


@wp.kernel
def reset_bowling_worlds(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    ctrl: wp.array2d[float],
    time: wp.array(dtype=float),
    world_ids: wp.array(dtype=int),
    seeds: wp.array(dtype=int),
    target_attach_pos: wp.array(dtype=wp.vec3),
    target_attach_xmat: wp.array(dtype=wp.mat33),
    target_gripper: wp.array(dtype=float),
    arm_qpos_ids: wp.array(dtype=int),
    ball_qpos_addr: int,
    pin_qpos_addrs: wp.array(dtype=int),
    pin_knocked: wp.array2d[int],
    prev_num_knocked_pins: wp.array(dtype=int),
    num_knocked_pins: wp.array(dtype=int),
    home_qpos: wp.array(dtype=float),
    pin_init_pos: wp.array(dtype=wp.vec3),
    down_xmat: wp.array(dtype=wp.mat33),
    t_pa_rot: wp.array(dtype=wp.mat33),
    t_pa_translation: wp.array(dtype=wp.vec3),
    nq: int,
    nv: int,
    nu: int,
    num_pins: int,
    ball_init_pos: wp.vec3,
    ball_xy_lo: wp.vec3,
    ball_xy_hi: wp.vec3,
    pin_rack_xy_lo: wp.vec3,
    pin_rack_xy_hi: wp.vec3,
    arm_lo: wp.vec3,
    arm_hi: wp.vec3,
):
    reset_idx = wp.tid()
    world_id = world_ids[reset_idx]
    seed = seeds[reset_idx]

    for q in range(nq):
        qpos[world_id, q] = 0.0
    for v in range(nv):
        qvel[world_id, v] = 0.0
    for u in range(nu):
        ctrl[world_id, u] = 0.0
    time[world_id] = 0.0

    for joint in range(6):
        qpos[world_id, arm_qpos_ids[joint]] = home_qpos[joint]

    counter = int(0)
    eff_pos = wp.vec3(
        _reset_rand_uniform(seed, counter, arm_lo[0], arm_hi[0]),
        _reset_rand_uniform(seed, counter + 1, arm_lo[1], arm_hi[1]),
        _reset_rand_uniform(seed, counter + 2, arm_lo[2], arm_hi[2]),
    )
    counter += 3
    eff_yaw = _reset_rand_uniform(seed, counter, -3.141592653589793, 3.141592653589793)
    counter += 1
    c = wp.cos(eff_yaw)
    s = wp.sin(eff_yaw)
    yaw_xmat = wp.mat33(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
    eff_xmat = _matmul(yaw_xmat, down_xmat[0])
    target_attach_pos[world_id] = eff_pos + _mat_vec(eff_xmat, t_pa_translation[0])
    target_attach_xmat[world_id] = _matmul(eff_xmat, t_pa_rot[0])
    target_gripper[world_id] = 0.0

    qpos[world_id, ball_qpos_addr + 0] = ball_init_pos[0] + _reset_rand_uniform(seed, counter, ball_xy_lo[0], ball_xy_hi[0])
    qpos[world_id, ball_qpos_addr + 1] = ball_init_pos[1] + _reset_rand_uniform(seed, counter + 1, ball_xy_lo[1], ball_xy_hi[1])
    qpos[world_id, ball_qpos_addr + 2] = ball_init_pos[2]
    qpos[world_id, ball_qpos_addr + 3] = 1.0
    qpos[world_id, ball_qpos_addr + 4] = 0.0
    qpos[world_id, ball_qpos_addr + 5] = 0.0
    qpos[world_id, ball_qpos_addr + 6] = 0.0
    counter += 2

    rack_dx = _reset_rand_uniform(seed, counter, pin_rack_xy_lo[0], pin_rack_xy_hi[0])
    rack_dy = _reset_rand_uniform(seed, counter + 1, pin_rack_xy_lo[1], pin_rack_xy_hi[1])
    for pin in range(MAX_BOWLING_PINS):
        if pin < num_pins:
            addr = pin_qpos_addrs[pin]
            base = pin_init_pos[pin]
            qpos[world_id, addr + 0] = base[0] + rack_dx
            qpos[world_id, addr + 1] = base[1] + rack_dy
            qpos[world_id, addr + 2] = base[2]
            qpos[world_id, addr + 3] = 1.0
            qpos[world_id, addr + 4] = 0.0
            qpos[world_id, addr + 5] = 0.0
            qpos[world_id, addr + 6] = 0.0
            pin_knocked[world_id, pin] = 0

    prev_num_knocked_pins[world_id] = 0
    num_knocked_pins[world_id] = 0


@wp.kernel
def save_bowling_prev_num_knocked(
    prev_num_knocked_pins: wp.array(dtype=int),
    num_knocked_pins: wp.array(dtype=int),
):
    world_id = wp.tid()
    prev_num_knocked_pins[world_id] = num_knocked_pins[world_id]


@wp.kernel
def bowling_score(
    qpos: wp.array2d[float],
    ball_qpos_addr: int,
    pin_qpos_addrs: wp.array(dtype=int),
    pin_knocked: wp.array2d[int],
    num_knocked_pins: wp.array(dtype=int),
    success: wp.array(dtype=int),
    healthy: wp.array(dtype=int),
    num_pins: int,
    floor_half_size: float,
    lower_z: float,
    upper_z: float,
    margin: float,
):
    world_id = wp.tid()
    lower_x = -floor_half_size - margin
    lower_y = -floor_half_size - margin
    lower_z_margin = lower_z - margin
    upper_x = floor_half_size + margin
    upper_y = floor_half_size + margin
    upper_z_margin = upper_z + margin

    is_healthy = int(1)
    ball_x = qpos[world_id, ball_qpos_addr + 0]
    ball_y = qpos[world_id, ball_qpos_addr + 1]
    ball_z = qpos[world_id, ball_qpos_addr + 2]
    if ball_x <= lower_x or ball_x >= upper_x or ball_y <= lower_y or ball_y >= upper_y or ball_z <= lower_z_margin or ball_z >= upper_z_margin:
        is_healthy = 0

    count = int(0)
    for pin in range(MAX_BOWLING_PINS):
        if pin < num_pins:
            addr = pin_qpos_addrs[pin]
            pin_x = qpos[world_id, addr + 0]
            pin_y = qpos[world_id, addr + 1]
            pin_z = qpos[world_id, addr + 2]
            if pin_x <= lower_x or pin_x >= upper_x or pin_y <= lower_y or pin_y >= upper_y or pin_z <= lower_z_margin or pin_z >= upper_z_margin:
                is_healthy = 0
            if _pin_knocked(
                qpos[world_id, addr + 4],
                qpos[world_id, addr + 5],
            ) != 0:
                pin_knocked[world_id, pin] = 1
            if pin_knocked[world_id, pin] != 0:
                count += 1

    healthy[world_id] = is_healthy
    success[world_id] = 0
    num_knocked_pins[world_id] = count


@wp.func
def _write_bowling_observation(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    cfrc_ext: wp.array2d[wp.spatial_vector],
    output: wp.array2d[float],
    arm_qpos_ids: wp.array(dtype=int),
    ball_qpos_addr: int,
    ball_qvel_addr: int,
    pin_qpos_addrs: wp.array(dtype=int),
    world_id: int,
    row: int,
    num_pins: int,
    pinch_site_id: int,
    gripper_qpos_id: int,
    right_pad_body_id: int,
):
    col = int(0)
    for joint in range(6):
        qpos_id = arm_qpos_ids[joint]
        output[row, col] = qpos[world_id, qpos_id]
        col += 1
    for joint in range(6):
        qpos_id = arm_qpos_ids[joint]
        output[row, col] = qvel[world_id, qpos_id]
        col += 1

    pos = site_xpos[world_id, pinch_site_id]
    output[row, col] = (pos[0] - 0.425) * 10.0
    col += 1
    output[row, col] = pos[1] * 10.0
    col += 1
    output[row, col] = pos[2] * 10.0
    col += 1

    xmat = site_xmat[world_id, pinch_site_id]
    yaw = wp.atan2(xmat[1, 0], xmat[0, 0])
    output[row, col] = wp.cos(yaw)
    col += 1
    output[row, col] = wp.sin(yaw)
    col += 1

    output[row, col] = wp.clamp(qpos[world_id, gripper_qpos_id] / 0.8, 0.0, 1.0) * 3.0
    col += 1
    contact = cfrc_ext[world_id, right_pad_body_id]
    contact_norm = wp.sqrt(
        contact[0] * contact[0]
        + contact[1] * contact[1]
        + contact[2] * contact[2]
        + contact[3] * contact[3]
        + contact[4] * contact[4]
        + contact[5] * contact[5]
    )
    output[row, col] = wp.clamp(contact_norm / 50.0, 0.0, 1.0)
    col += 1

    output[row, col] = (qpos[world_id, ball_qpos_addr + 0] - 0.425) * 10.0
    col += 1
    output[row, col] = qpos[world_id, ball_qpos_addr + 1] * 10.0
    col += 1
    output[row, col] = qpos[world_id, ball_qpos_addr + 2] * 10.0
    col += 1
    for k in range(4):
        output[row, col] = qpos[world_id, ball_qpos_addr + 3 + k]
        col += 1
    for k in range(3):
        output[row, col] = qvel[world_id, ball_qvel_addr + k] * 10.0
        col += 1
    for k in range(3):
        output[row, col] = qvel[world_id, ball_qvel_addr + 3 + k]
        col += 1

    for pin in range(MAX_BOWLING_PINS):
        if pin < num_pins:
            addr = pin_qpos_addrs[pin]
            output[row, col] = (qpos[world_id, addr + 0] - 0.425) * 10.0
            col += 1
            output[row, col] = qpos[world_id, addr + 1] * 10.0
            col += 1
            output[row, col] = qpos[world_id, addr + 2] * 10.0
            col += 1
            for k in range(4):
                output[row, col] = qpos[world_id, addr + 3 + k]
                col += 1


@wp.kernel
def record_observations(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    cfrc_ext: wp.array2d[wp.spatial_vector],
    done: wp.array(dtype=int),
    output: wp.array2d[float],
    arm_qpos_ids: wp.array(dtype=int),
    ball_qpos_addr: int,
    ball_qvel_addr: int,
    pin_qpos_addrs: wp.array(dtype=int),
    step: int,
    nworld: int,
    num_pins: int,
    pinch_site_id: int,
    gripper_qpos_id: int,
    right_pad_body_id: int,
):
    world_id = wp.tid()
    if done[world_id] != 0:
        return
    row = step * nworld + world_id
    _write_bowling_observation(
        qpos,
        qvel,
        site_xpos,
        site_xmat,
        cfrc_ext,
        output,
        arm_qpos_ids,
        ball_qpos_addr,
        ball_qvel_addr,
        pin_qpos_addrs,
        world_id,
        row,
        num_pins,
        pinch_site_id,
        gripper_qpos_id,
        right_pad_body_id,
    )


@wp.kernel
def record_transition_bowling(
    action: wp.array2d[float],
    healthy: wp.array(dtype=int),
    done: wp.array(dtype=int),
    episode_length: wp.array(dtype=int),
    prev_num_knocked_pins: wp.array(dtype=int),
    num_knocked_pins: wp.array(dtype=int),
    actions: wp.array2d[float],
    rewards: wp.array(dtype=float),
    masks: wp.array(dtype=float),
    terminals: wp.array(dtype=int),
    step: int,
    nworld: int,
    max_steps: int,
):
    world_id = wp.tid()
    if done[world_id] != 0:
        return
    row = step * nworld + world_id
    for action_dim in range(7):
        actions[row, action_dim] = action[world_id, action_dim]

    is_healthy = healthy[world_id] != 0
    is_terminated = (not is_healthy) or episode_length[world_id] + 1 >= max_steps
    newly_knocked = num_knocked_pins[world_id] - prev_num_knocked_pins[world_id]
    if newly_knocked < 0:
        newly_knocked = 0
    rewards[row] = 0.1 * float(newly_knocked)
    masks[row] = 1.0
    if is_terminated:
        masks[row] = 0.0
    terminals[row] = int(is_terminated)
