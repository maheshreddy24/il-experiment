import warp as wp

from ocbench.mjwarp.envs.manipulation_kernels import (
    _reset_rand_int,
    _reset_rand_uniform,
)
from ocbench.mjwarp.primitives.primitive_kernels import (
    _mat_vec,
    _matmul,
)

MAX_SWITCH_BUTTONS = 25


@wp.func
def _toggle_switch_neighbors(
    button_states: wp.array2d[int],
    world_id: int,
    button: int,
    num_rows: int,
    num_cols: int,
    num_buttons: int,
    dropped_state: int,
    num_colors: int,
):
    if button_states[world_id, button] == dropped_state:
        return
    x = button // num_cols
    y = button - x * num_cols
    for idx in range(MAX_SWITCH_BUTTONS):
        if idx < num_buttons and button_states[world_id, idx] != dropped_state:
            nx = idx // num_cols
            ny = idx - nx * num_cols
            dx = nx - x
            dy = ny - y
            if dx * dx + dy * dy <= 1:
                button_states[world_id, idx] = (button_states[world_id, idx] + 1) % num_colors


@wp.func
def _switch_is_monochrome(
    button_states: wp.array2d[int],
    world_id: int,
    num_buttons: int,
    dropped_state: int,
) -> int:
    all_zero = int(1)
    all_one = int(1)
    for button in range(MAX_SWITCH_BUTTONS):
        if button < num_buttons and button_states[world_id, button] != dropped_state:
            state = button_states[world_id, button]
            if state != 0:
                all_zero = 0
            if state != 1:
                all_one = 0
    return int(all_zero != 0 or all_one != 0)


@wp.kernel
def reset_switch_worlds(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    ctrl: wp.array2d[float],
    time: wp.array(dtype=float),
    mocap_pos: wp.array2d[wp.vec3],
    mocap_quat: wp.array2d[wp.quat],
    world_ids: wp.array(dtype=int),
    seeds: wp.array(dtype=int),
    target_attach_pos: wp.array(dtype=wp.vec3),
    target_attach_xmat: wp.array(dtype=wp.mat33),
    target_gripper: wp.array(dtype=float),
    arm_qpos_ids: wp.array(dtype=int),
    button_qpos_addrs: wp.array(dtype=int),
    button_base_mocap_ids: wp.array(dtype=int),
    home_qpos: wp.array(dtype=float),
    nominal_button_pos: wp.array(dtype=wp.vec3),
    nominal_button_quat: wp.array(dtype=wp.quat),
    down_xmat: wp.array(dtype=wp.mat33),
    t_pa_rot: wp.array(dtype=wp.mat33),
    t_pa_translation: wp.array(dtype=wp.vec3),
    button_states: wp.array2d[int],
    prev_button_states: wp.array2d[int],
    prev_button_qpos: wp.array2d[float],
    nq: int,
    nv: int,
    nu: int,
    num_rows: int,
    num_cols: int,
    num_buttons: int,
    max_dropped_buttons: int,
    dropped_state: int,
    num_colors: int,
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

    dropped = wp.zeros(MAX_SWITCH_BUTTONS, dtype=int)
    if max_dropped_buttons == 1:
        choice = _reset_rand_int(seed, counter, 1 + num_buttons)
        counter += 1
        if choice > 0:
            dropped[choice - 1] = 1
    elif max_dropped_buttons == 2:
        num_pairs = num_buttons * (num_buttons - 1) // 2
        choice = _reset_rand_int(seed, counter, 1 + num_buttons + num_pairs)
        counter += 1
        if choice > 0 and choice <= num_buttons:
            dropped[choice - 1] = 1
        elif choice > num_buttons:
            pair_idx = choice - num_buttons - 1
            for first in range(MAX_SWITCH_BUTTONS):
                if first < num_buttons:
                    for second in range(MAX_SWITCH_BUTTONS):
                        if second > first and second < num_buttons:
                            if pair_idx == 0:
                                dropped[first] = 1
                                dropped[second] = 1
                            pair_idx -= 1

    for button in range(MAX_SWITCH_BUTTONS):
        if button < num_buttons:
            qpos[world_id, button_qpos_addrs[button]] = 0.0
            prev_button_qpos[world_id, button] = 0.0
            mocap_id = button_base_mocap_ids[button]
            if dropped[button] != 0:
                mocap_pos[world_id, mocap_id] = wp.vec3(5.0 + 0.1 * float(button), 5.0, 5.0)
            else:
                mocap_pos[world_id, mocap_id] = nominal_button_pos[button]
            mocap_quat[world_id, mocap_id] = nominal_button_quat[button]

    for attempt in range(100):
        base_state = _reset_rand_int(seed, counter, num_colors)
        counter += 1
        for button in range(MAX_SWITCH_BUTTONS):
            if button < num_buttons:
                if dropped[button] != 0:
                    button_states[world_id, button] = dropped_state
                else:
                    button_states[world_id, button] = base_state

        for button in range(MAX_SWITCH_BUTTONS):
            if button < num_buttons and dropped[button] == 0:
                pressed = _reset_rand_int(seed, counter, num_colors)
                counter += 1
                if pressed != 0:
                    _toggle_switch_neighbors(
                        button_states,
                        world_id,
                        button,
                        num_rows,
                        num_cols,
                        num_buttons,
                        dropped_state,
                        num_colors,
                    )

        if _switch_is_monochrome(button_states, world_id, num_buttons, dropped_state) == 0:
            break

    for button in range(MAX_SWITCH_BUTTONS):
        if button < num_buttons:
            prev_button_states[world_id, button] = button_states[world_id, button]


@wp.kernel
def store_button_qpos_and_states(
    qpos: wp.array2d[float],
    button_qpos_addrs: wp.array(dtype=int),
    button_states: wp.array2d[int],
    prev_button_states: wp.array2d[int],
    prev_button_qpos: wp.array2d[float],
    num_buttons: int,
):
    world_id = wp.tid()
    for button in range(MAX_SWITCH_BUTTONS):
        if button < num_buttons:
            prev_button_qpos[world_id, button] = qpos[world_id, button_qpos_addrs[button]]
            prev_button_states[world_id, button] = button_states[world_id, button]


@wp.kernel
def update_switch_button_states(
    qpos: wp.array2d[float],
    button_qpos_addrs: wp.array(dtype=int),
    prev_button_qpos: wp.array2d[float],
    button_states: wp.array2d[int],
    num_rows: int,
    num_cols: int,
    num_buttons: int,
    dropped_state: int,
    num_colors: int,
):
    world_id = wp.tid()
    for button in range(MAX_SWITCH_BUTTONS):
        if button < num_buttons and button_states[world_id, button] != dropped_state:
            prev_pos = prev_button_qpos[world_id, button]
            cur_pos = qpos[world_id, button_qpos_addrs[button]]
            if prev_pos > -0.02 and cur_pos <= -0.02:
                _toggle_switch_neighbors(
                    button_states,
                    world_id,
                    button,
                    num_rows,
                    num_cols,
                    num_buttons,
                    dropped_state,
                    num_colors,
                )


@wp.kernel
def switch_success(
    button_states: wp.array2d[int],
    success: wp.array(dtype=int),
    healthy: wp.array(dtype=int),
    num_buttons: int,
    dropped_state: int,
):
    world_id = wp.tid()
    healthy[world_id] = 1
    success[world_id] = _switch_is_monochrome(button_states, world_id, num_buttons, dropped_state)


@wp.kernel
def park_done_switch_worlds(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    ctrl: wp.array2d[float],
    done: wp.array(dtype=int),
    arm_qpos_ids: wp.array(dtype=int),
    arm_actuator_ids: wp.array(dtype=int),
    gripper_actuator_ids: wp.array(dtype=int),
    gripper_qpos_id: int,
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


@wp.func
def _write_switch_observation(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    cfrc_ext: wp.array2d[wp.spatial_vector],
    button_states: wp.array2d[int],
    output: wp.array2d[float],
    arm_qpos_ids: wp.array(dtype=int),
    button_qpos_addrs: wp.array(dtype=int),
    button_dof_addrs: wp.array(dtype=int),
    world_id: int,
    row: int,
    num_buttons: int,
    num_button_states: int,
    dropped_state: int,
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

    for button in range(MAX_SWITCH_BUTTONS):
        if button < num_buttons:
            state = button_states[world_id, button]
            for state_idx in range(3):
                value = float(0.0)
                if state_idx < num_button_states and state == state_idx:
                    value = 1.0
                output[row, col] = value
                col += 1
            button_pos = qpos[world_id, button_qpos_addrs[button]]
            button_vel = qvel[world_id, button_dof_addrs[button]]
            if state == dropped_state:
                button_pos = 0.0
                button_vel = 0.0
            output[row, col] = button_pos * 120.0
            col += 1
            output[row, col] = button_vel
            col += 1


@wp.kernel
def record_observations(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    cfrc_ext: wp.array2d[wp.spatial_vector],
    button_states: wp.array2d[int],
    done: wp.array(dtype=int),
    output: wp.array2d[float],
    arm_qpos_ids: wp.array(dtype=int),
    button_qpos_addrs: wp.array(dtype=int),
    button_dof_addrs: wp.array(dtype=int),
    step: int,
    nworld: int,
    num_buttons: int,
    num_button_states: int,
    dropped_state: int,
    pinch_site_id: int,
    gripper_qpos_id: int,
    right_pad_body_id: int,
):
    world_id = wp.tid()
    if done[world_id] != 0:
        return
    row = step * nworld + world_id
    _write_switch_observation(
        qpos,
        qvel,
        site_xpos,
        site_xmat,
        cfrc_ext,
        button_states,
        output,
        arm_qpos_ids,
        button_qpos_addrs,
        button_dof_addrs,
        world_id,
        row,
        num_buttons,
        num_button_states,
        dropped_state,
        pinch_site_id,
        gripper_qpos_id,
        right_pad_body_id,
    )
