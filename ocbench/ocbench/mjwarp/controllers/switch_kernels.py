import warp as wp

from ocbench.mjwarp.controllers.controller_kernels import _hold_current_target
from ocbench.mjwarp.envs.switch_kernels import MAX_SWITCH_BUTTONS, _switch_is_monochrome
from ocbench.mjwarp.primitives.button_kernels import _generate_button_plan
from ocbench.mjwarp.primitives.primitive_kernels import (
    _clamp_vec3,
    _evaluate_plan,
    _evaluate_plan_quat,
    _mat_transpose,
    _mat_vec,
    _matmul,
    _norm3,
    _quat_to_mat,
    _rand_int,
    _rand_uniform,
    _randf,
    _rotation_log,
    _rotvec_to_mat,
    _sample_plan_time,
)

MAX_SWITCH_SOLVE_COMBOS = 64
MAX_SWITCH_AUG_COLS = 26
MAX_SWITCH_MAT_SIZE = 650
MAX_SWITCH_BASIS_SIZE = 625


@wp.func
def _switch_toggle_value(
    button: int,
    target: int,
    button_states: wp.array2d[int],
    world_id: int,
    num_cols: int,
    dropped_state: int,
) -> int:
    if button_states[world_id, button] == dropped_state or button_states[world_id, target] == dropped_state:
        return 0
    bx = button // num_cols
    by = button - bx * num_cols
    tx = target // num_cols
    ty = target - tx * num_cols
    dx = bx - tx
    dy = by - ty
    return int(dx * dx + dy * dy <= 1)


@wp.func
def _pending_contains(
    pending_order: wp.array2d[int],
    pending_count: wp.array(dtype=int),
    world_id: int,
    button: int,
) -> int:
    out = int(0)
    for idx in range(MAX_SWITCH_BUTTONS):
        if idx < pending_count[world_id] and pending_order[world_id, idx] == button:
            out = 1
    return out


@wp.func
def _shuffle_pending(
    pending_order: wp.array2d[int],
    pending_count: wp.array(dtype=int),
    rng_counter: wp.array(dtype=int),
    world_id: int,
    seeds: wp.array(dtype=int),
):
    count = pending_count[world_id]
    for rev in range(MAX_SWITCH_BUTTONS):
        i = count - 1 - rev
        if i > 0:
            j = _rand_int(world_id, seeds, rng_counter, i + 1)
            tmp = pending_order[world_id, i]
            pending_order[world_id, i] = pending_order[world_id, j]
            pending_order[world_id, j] = tmp


@wp.func
def _solve_pending(
    button_states: wp.array2d[int],
    pending_order: wp.array2d[int],
    pending_count: wp.array(dtype=int),
    world_id: int,
    target_color: int,
    num_buttons: int,
    num_cols: int,
    dropped_state: int,
):
    active_buttons = wp.zeros(MAX_SWITCH_BUTTONS, dtype=int)
    active_count = int(0)
    for button in range(MAX_SWITCH_BUTTONS):
        if button < num_buttons and button_states[world_id, button] != dropped_state:
            active_buttons[active_count] = button
            active_count += 1

    stride = MAX_SWITCH_AUG_COLS
    mat = wp.zeros(MAX_SWITCH_MAT_SIZE, dtype=int)
    for row in range(MAX_SWITCH_BUTTONS):
        if row < active_count:
            target_button = active_buttons[row]
            for col in range(MAX_SWITCH_BUTTONS):
                if col < active_count:
                    press_button = active_buttons[col]
                    mat[row * stride + col] = _switch_toggle_value(
                        press_button,
                        target_button,
                        button_states,
                        world_id,
                        num_cols,
                        dropped_state,
                    )
            mat[row * stride + active_count] = (button_states[world_id, target_button] + target_color) % 2

    rank = int(0)
    pivots = wp.zeros(MAX_SWITCH_BUTTONS, dtype=int)
    for col in range(MAX_SWITCH_BUTTONS):
        if col < active_count:
            pivot = int(-1)
            for row in range(MAX_SWITCH_BUTTONS):
                if row >= rank and row < active_count and pivot < 0 and mat[row * stride + col] != 0:
                    pivot = row
            if pivot >= 0:
                if pivot != rank:
                    for swap_col in range(MAX_SWITCH_AUG_COLS):
                        if swap_col <= active_count:
                            tmp = mat[rank * stride + swap_col]
                            mat[rank * stride + swap_col] = mat[pivot * stride + swap_col]
                            mat[pivot * stride + swap_col] = tmp
                for row in range(MAX_SWITCH_BUTTONS):
                    if row < active_count and row != rank and mat[row * stride + col] != 0:
                        for elim_col in range(MAX_SWITCH_AUG_COLS):
                            if elim_col <= active_count:
                                mat[row * stride + elim_col] = (mat[row * stride + elim_col] + mat[rank * stride + elim_col]) % 2
                pivots[rank] = col
                rank += 1

    solution = wp.zeros(MAX_SWITCH_BUTTONS, dtype=int)
    for row in range(MAX_SWITCH_BUTTONS):
        if row < rank:
            col = pivots[row]
            solution[col] = mat[row * stride + active_count]

    free_cols = wp.zeros(MAX_SWITCH_BUTTONS, dtype=int)
    free_count = int(0)
    for col in range(MAX_SWITCH_BUTTONS):
        if col < active_count:
            is_pivot = int(0)
            for row in range(MAX_SWITCH_BUTTONS):
                if row < rank and pivots[row] == col:
                    is_pivot = 1
            if is_pivot == 0:
                free_cols[free_count] = col
                free_count += 1

    basis = wp.zeros(MAX_SWITCH_BASIS_SIZE, dtype=int)
    for free_idx in range(MAX_SWITCH_BUTTONS):
        if free_idx < free_count:
            free_col = free_cols[free_idx]
            basis[free_idx * MAX_SWITCH_BUTTONS + free_col] = 1
            for row in range(MAX_SWITCH_BUTTONS):
                if row < rank:
                    basis[free_idx * MAX_SWITCH_BUTTONS + pivots[row]] = mat[row * stride + free_col]

    best = wp.zeros(MAX_SWITCH_BUTTONS, dtype=int)
    best_sum = int(0)
    for col in range(MAX_SWITCH_BUTTONS):
        if col < active_count:
            best[col] = solution[col]
            best_sum += solution[col]

    combo_limit = int(1)
    for free_idx in range(MAX_SWITCH_BUTTONS):
        if free_idx < free_count:
            combo_limit = combo_limit * 2
    for mask in range(MAX_SWITCH_SOLVE_COMBOS):
        if mask > 0 and mask < combo_limit:
            candidate_sum = int(0)
            candidate = wp.zeros(MAX_SWITCH_BUTTONS, dtype=int)
            for col in range(MAX_SWITCH_BUTTONS):
                if col < active_count:
                    value = solution[col]
                    bit_scale = int(1)
                    for free_idx in range(MAX_SWITCH_BUTTONS):
                        if free_idx < free_count:
                            if ((mask // bit_scale) % 2) != 0:
                                value = (value + basis[free_idx * MAX_SWITCH_BUTTONS + col]) % 2
                            bit_scale = bit_scale * 2
                    candidate[col] = value
                    candidate_sum += value
            if candidate_sum < best_sum:
                best_sum = candidate_sum
                for col in range(MAX_SWITCH_BUTTONS):
                    if col < active_count:
                        best[col] = candidate[col]

    count = int(0)
    for button in range(MAX_SWITCH_BUTTONS):
        pending_order[world_id, button] = 0
    for col in range(MAX_SWITCH_BUTTONS):
        if col < active_count and best[col] != 0:
            pending_order[world_id, count] = active_buttons[col]
            count += 1
    pending_count[world_id] = count


@wp.func
def _toggle_pending(
    pending_order: wp.array2d[int],
    pending_count: wp.array(dtype=int),
    rng_counter: wp.array(dtype=int),
    world_id: int,
    seeds: wp.array(dtype=int),
    button: int,
):
    count = pending_count[world_id]
    found = int(-1)
    for idx in range(MAX_SWITCH_BUTTONS):
        if idx < count and pending_order[world_id, idx] == button:
            found = idx
    if found >= 0:
        for idx in range(MAX_SWITCH_BUTTONS - 1):
            if idx >= found and idx < count - 1:
                pending_order[world_id, idx] = pending_order[world_id, idx + 1]
        pending_count[world_id] = count - 1
    else:
        insert_idx = _rand_int(world_id, seeds, rng_counter, count + 1)
        for rev in range(MAX_SWITCH_BUTTONS):
            idx = count - rev
            if idx > insert_idx and idx < MAX_SWITCH_BUTTONS:
                pending_order[world_id, idx] = pending_order[world_id, idx - 1]
        pending_order[world_id, insert_idx] = button
        pending_count[world_id] = count + 1


@wp.func
def _pressed_button(
    last_button_states: wp.array2d[int],
    button_states: wp.array2d[int],
    world_id: int,
    num_buttons: int,
    num_cols: int,
    dropped_state: int,
) -> int:
    dropped_changed = int(0)
    any_diff = int(0)
    diff = wp.zeros(MAX_SWITCH_BUTTONS, dtype=int)
    for button in range(MAX_SWITCH_BUTTONS):
        if button < num_buttons:
            last_dropped = int(last_button_states[world_id, button] == dropped_state)
            cur_dropped = int(button_states[world_id, button] == dropped_state)
            if last_dropped != cur_dropped:
                dropped_changed = 1
            if cur_dropped == 0:
                value = (last_button_states[world_id, button] + button_states[world_id, button]) % 2
                diff[button] = value
                if value != 0:
                    any_diff = 1
    if dropped_changed != 0 or any_diff == 0:
        return -1

    match_count = int(0)
    match_button = int(-1)
    for button in range(MAX_SWITCH_BUTTONS):
        if button < num_buttons and button_states[world_id, button] != dropped_state:
            match = int(1)
            for target in range(MAX_SWITCH_BUTTONS):
                if target < num_buttons:
                    expected = _switch_toggle_value(button, target, button_states, world_id, num_cols, dropped_state)
                    if expected != diff[target]:
                        match = 0
            if match != 0:
                match_count += 1
                match_button = button
    if match_count == 1:
        return match_button
    return -1


@wp.func
def _update_plan_from_states(
    button_states: wp.array2d[int],
    last_button_states: wp.array2d[int],
    pending_order: wp.array2d[int],
    pending_count: wp.array(dtype=int),
    target_color: wp.array(dtype=int),
    rng_counter: wp.array(dtype=int),
    world_id: int,
    seeds: wp.array(dtype=int),
    num_buttons: int,
    num_cols: int,
    dropped_state: int,
):
    pressed = _pressed_button(last_button_states, button_states, world_id, num_buttons, num_cols, dropped_state)
    changed = int(0)
    for button in range(MAX_SWITCH_BUTTONS):
        if button < num_buttons and last_button_states[world_id, button] != button_states[world_id, button]:
            changed = 1
    if pressed >= 0:
        _toggle_pending(pending_order, pending_count, rng_counter, world_id, seeds, pressed)
    elif changed != 0:
        _solve_pending(
            button_states,
            pending_order,
            pending_count,
            world_id,
            target_color[world_id],
            num_buttons,
            num_cols,
            dropped_state,
        )
        _shuffle_pending(pending_order, pending_count, rng_counter, world_id, seeds)
    for button in range(MAX_SWITCH_BUTTONS):
        if button < num_buttons:
            last_button_states[world_id, button] = button_states[world_id, button]


@wp.func
def _choose_switch_subgoal(
    button_states: wp.array2d[int],
    pending_order: wp.array2d[int],
    pending_count: wp.array(dtype=int),
    target_button: wp.array(dtype=int),
    target_state: wp.array(dtype=int),
    agent_done: wp.array(dtype=int),
    rng_counter: wp.array(dtype=int),
    is_mistake: wp.array(dtype=int),
    p_mistake: wp.array(dtype=float),
    world_id: int,
    seeds: wp.array(dtype=int),
    num_buttons: int,
    dropped_state: int,
) -> int:
    if _switch_is_monochrome(button_states, world_id, num_buttons, dropped_state) != 0:
        agent_done[world_id] = 1
        return 0
    if pending_count[world_id] == 0:
        agent_done[world_id] = 1
        return 0

    planned = pending_order[world_id, 0]
    is_mistake[world_id] = 0
    chosen = planned
    if _randf(world_id, seeds, rng_counter) < p_mistake[world_id]:
        candidates = wp.zeros(MAX_SWITCH_BUTTONS, dtype=int)
        candidate_count = int(0)
        for button in range(MAX_SWITCH_BUTTONS):
            if button < num_buttons and button_states[world_id, button] != dropped_state:
                if _pending_contains(pending_order, pending_count, world_id, button) == 0:
                    candidates[candidate_count] = button
                    candidate_count += 1
        if candidate_count == 0:
            for button in range(MAX_SWITCH_BUTTONS):
                if button < num_buttons and button_states[world_id, button] != dropped_state and button != planned:
                    candidates[candidate_count] = button
                    candidate_count += 1
        if candidate_count > 0:
            is_mistake[world_id] = 1
            chosen = candidates[_rand_int(world_id, seeds, rng_counter, candidate_count)]

    target_button[world_id] = chosen
    target_state[world_id] = 1 - button_states[world_id, chosen]
    agent_done[world_id] = 0
    return 1


@wp.func
def _start_switch_plan(
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    button_site_ids: wp.array(dtype=int),
    rng_counter: wp.array(dtype=int),
    target_button: wp.array(dtype=int),
    key_count: wp.array(dtype=int),
    key_time: wp.array2d[float],
    key_xyz: wp.array2d[wp.vec3],
    key_quat: wp.array2d[wp.quat],
    key_grasp: wp.array2d[float],
    key_stop: wp.array2d[int],
    key_tangent: wp.array2d[wp.vec3],
    gate_time: wp.array2d[float],
    gate_xyz: wp.array2d[wp.vec3],
    gate_passed: wp.array2d[int],
    plan_time: wp.array(dtype=wp.float64),
    last_time: wp.array(dtype=float),
    agent_done: wp.array(dtype=int),
    world_id: int,
    seeds: wp.array(dtype=int),
    pinch_site_id: int,
    speed_dt: wp.array(dtype=float),
    arm_lo: wp.vec3,
    arm_hi: wp.vec3,
    tilt_randomization: int,
):
    _generate_button_plan(
        site_xpos,
        site_xmat,
        rng_counter,
        key_count,
        key_time,
        key_xyz,
        key_quat,
        key_grasp,
        key_stop,
        key_tangent,
        gate_time,
        gate_xyz,
        gate_passed,
        plan_time,
        last_time,
        agent_done,
        world_id,
        seeds,
        pinch_site_id,
        button_site_ids[target_button[world_id]],
        speed_dt[world_id],
        arm_lo,
        arm_hi,
        1,
        tilt_randomization,
    )


@wp.kernel
def reset_controller(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    button_states: wp.array2d[int],
    rng_counter: wp.array(dtype=int),
    is_mistake: wp.array(dtype=int),
    done: wp.array(dtype=int),
    episode_length: wp.array(dtype=int),
    episode_success: wp.array(dtype=int),
    controller_done: wp.array(dtype=int),
    agent_done: wp.array(dtype=int),
    active_steps: wp.array(dtype=int),
    speed_dt: wp.array(dtype=float),
    p_mistake: wp.array(dtype=float),
    target_color: wp.array(dtype=int),
    pending_order: wp.array2d[int],
    pending_count: wp.array(dtype=int),
    last_button_states: wp.array2d[int],
    target_button: wp.array(dtype=int),
    target_state: wp.array(dtype=int),
    key_count: wp.array(dtype=int),
    key_time: wp.array2d[float],
    key_xyz: wp.array2d[wp.vec3],
    key_quat: wp.array2d[wp.quat],
    key_grasp: wp.array2d[float],
    key_stop: wp.array2d[int],
    key_tangent: wp.array2d[wp.vec3],
    gate_time: wp.array2d[float],
    gate_xyz: wp.array2d[wp.vec3],
    gate_passed: wp.array2d[int],
    plan_time: wp.array(dtype=wp.float64),
    last_time: wp.array(dtype=float),
    seeds: wp.array(dtype=int),
    button_site_ids: wp.array(dtype=int),
    num_buttons: int,
    num_cols: int,
    dropped_state: int,
    pinch_site_id: int,
    arm_lo: wp.vec3,
    arm_hi: wp.vec3,
    p_mistake_low: float,
    p_mistake_high: float,
    segment_dt_scale: float,
    tilt_randomization: int,
):
    world_id = wp.tid()
    done[world_id] = 0
    episode_length[world_id] = 0
    episode_success[world_id] = 0
    controller_done[world_id] = 0
    agent_done[world_id] = 0
    active_steps[world_id] = 0
    key_count[world_id] = 0
    rng_counter[world_id] = 0
    speed_dt[world_id] = _rand_uniform(world_id, seeds, rng_counter, 0.7, 1.5) * segment_dt_scale
    p_mistake[world_id] = _rand_uniform(world_id, seeds, rng_counter, p_mistake_low, p_mistake_high)
    target_color[world_id] = _rand_int(world_id, seeds, rng_counter, 2)
    for button in range(MAX_SWITCH_BUTTONS):
        if button < num_buttons:
            last_button_states[world_id, button] = button_states[world_id, button]

    _solve_pending(button_states, pending_order, pending_count, world_id, target_color[world_id], num_buttons, num_cols, dropped_state)
    _shuffle_pending(pending_order, pending_count, rng_counter, world_id, seeds)
    ok = _choose_switch_subgoal(
        button_states,
        pending_order,
        pending_count,
        target_button,
        target_state,
        agent_done,
        rng_counter,
        is_mistake,
        p_mistake,
        world_id,
        seeds,
        num_buttons,
        dropped_state,
    )
    if ok == 0:
        controller_done[world_id] = 1
        done[world_id] = 1
    else:
        _start_switch_plan(
            site_xpos,
            site_xmat,
            button_site_ids,
            rng_counter,
            target_button,
            key_count,
            key_time,
            key_xyz,
            key_quat,
            key_grasp,
            key_stop,
            key_tangent,
            gate_time,
            gate_xyz,
            gate_passed,
            plan_time,
            last_time,
            agent_done,
            world_id,
            seeds,
            pinch_site_id,
            speed_dt,
            arm_lo,
            arm_hi,
            tilt_randomization,
        )


@wp.kernel
def make_targets(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    button_states: wp.array2d[int],
    rng_counter: wp.array(dtype=int),
    is_mistake: wp.array(dtype=int),
    done: wp.array(dtype=int),
    controller_done: wp.array(dtype=int),
    agent_done: wp.array(dtype=int),
    active_steps: wp.array(dtype=int),
    speed_dt: wp.array(dtype=float),
    p_mistake: wp.array(dtype=float),
    target_color: wp.array(dtype=int),
    pending_order: wp.array2d[int],
    pending_count: wp.array(dtype=int),
    last_button_states: wp.array2d[int],
    target_button: wp.array(dtype=int),
    target_state: wp.array(dtype=int),
    key_count: wp.array(dtype=int),
    key_time: wp.array2d[float],
    key_xyz: wp.array2d[wp.vec3],
    key_quat: wp.array2d[wp.quat],
    key_grasp: wp.array2d[float],
    key_stop: wp.array2d[int],
    key_tangent: wp.array2d[wp.vec3],
    gate_time: wp.array2d[float],
    gate_xyz: wp.array2d[wp.vec3],
    gate_passed: wp.array2d[int],
    plan_time: wp.array(dtype=wp.float64),
    last_time: wp.array(dtype=float),
    target_attach_pos: wp.array(dtype=wp.vec3),
    target_attach_xmat: wp.array(dtype=wp.mat33),
    target_gripper: wp.array(dtype=float),
    down_xmat_arr: wp.array(dtype=wp.mat33),
    down_xmat_inv_arr: wp.array(dtype=wp.mat33),
    t_pa_rot_arr: wp.array(dtype=wp.mat33),
    t_pa_translation: wp.array(dtype=wp.vec3),
    seeds: wp.array(dtype=int),
    button_site_ids: wp.array(dtype=int),
    num_buttons: int,
    num_cols: int,
    dropped_state: int,
    pinch_site_id: int,
    gripper_qpos_id: int,
    arm_lo: wp.vec3,
    arm_hi: wp.vec3,
    workspace_lo: wp.vec3,
    workspace_hi: wp.vec3,
    ee_low: wp.array(dtype=float),
    ee_high: wp.array(dtype=float),
    env_dt: wp.float64,
    max_subgoal_steps: int,
    tilt_randomization: int,
):
    world_id = wp.tid()
    current_xmat = site_xmat[world_id, pinch_site_id]
    current_pos = site_xpos[world_id, pinch_site_id]
    gripper_opening = wp.clamp(qpos[world_id, gripper_qpos_id] / 0.8, 0.0, 1.0)
    if done[world_id] != 0 or controller_done[world_id] != 0:
        _hold_current_target(site_xpos, site_xmat, qpos, target_attach_pos, target_attach_xmat, target_gripper, t_pa_rot_arr, t_pa_translation, world_id, pinch_site_id, gripper_qpos_id)
        return

    _update_plan_from_states(
        button_states,
        last_button_states,
        pending_order,
        pending_count,
        target_color,
        rng_counter,
        world_id,
        seeds,
        num_buttons,
        num_cols,
        dropped_state,
    )
    if agent_done[world_id] != 0 or active_steps[world_id] >= max_subgoal_steps:
        ok = _choose_switch_subgoal(
            button_states,
            pending_order,
            pending_count,
            target_button,
            target_state,
            agent_done,
            rng_counter,
            is_mistake,
            p_mistake,
            world_id,
            seeds,
            num_buttons,
            dropped_state,
        )
        if ok == 0:
            controller_done[world_id] = 1
            _hold_current_target(site_xpos, site_xmat, qpos, target_attach_pos, target_attach_xmat, target_gripper, t_pa_rot_arr, t_pa_translation, world_id, pinch_site_id, gripper_qpos_id)
            return
        active_steps[world_id] = 0
        _start_switch_plan(
            site_xpos,
            site_xmat,
            button_site_ids,
            rng_counter,
            target_button,
            key_count,
            key_time,
            key_xyz,
            key_quat,
            key_grasp,
            key_stop,
            key_tangent,
            gate_time,
            gate_xyz,
            gate_passed,
            plan_time,
            last_time,
            agent_done,
            world_id,
            seeds,
            pinch_site_id,
            speed_dt,
            arm_lo,
            arm_hi,
            tilt_randomization,
        )

    if key_count[world_id] == 0:
        _hold_current_target(site_xpos, site_xmat, qpos, target_attach_pos, target_attach_xmat, target_gripper, t_pa_rot_arr, t_pa_translation, world_id, pinch_site_id, gripper_qpos_id)
        return
    active_steps[world_id] += 1

    if last_time[world_id] >= 0.0:
        plan_time[world_id] += env_dt
    last_time[world_id] = 0.0

    blocked_gate = int(-1)
    for gate in range(2):
        if gate_passed[world_id, gate] == 0 and plan_time[world_id] >= wp.float64(gate_time[world_id, gate]):
            diff = current_pos - gate_xyz[world_id, gate]
            if wp.dot(diff, diff) > 0.04 * 0.04:
                plan_time[world_id] = wp.float64(gate_time[world_id, gate])
                blocked_gate = gate
            else:
                gate_passed[world_id, gate] = 1
            break

    query_time, plan_done = _sample_plan_time(
        plan_time[world_id], key_time[world_id, key_count[world_id] - 1], env_dt,
    )
    if plan_done != 0:
        agent_done[world_id] = 1

    eval = _evaluate_plan(key_count, key_time, key_xyz, key_quat, key_grasp, key_tangent, world_id, query_time)
    ab_xyz = wp.vec3(eval[0], eval[1], eval[2])
    ab_grasp = eval[3]
    plan_quat = _evaluate_plan_quat(key_count, key_time, key_quat, world_id, query_time)
    if blocked_gate >= 0:
        ab_xyz = gate_xyz[world_id, blocked_gate]

    ee_delta = wp.vec3(
        wp.clamp(ab_xyz[0] - current_pos[0], ee_low[0], ee_high[0]),
        wp.clamp(ab_xyz[1] - current_pos[1], ee_low[1], ee_high[1]),
        wp.clamp(ab_xyz[2] - current_pos[2], ee_low[2], ee_high[2]),
    )
    target_eff_pos = _clamp_vec3(current_pos + ee_delta, workspace_lo, workspace_hi)
    gripper_delta = wp.clamp(ab_grasp - gripper_opening, ee_low[4], ee_high[4])
    target_gripper[world_id] = wp.clamp(gripper_opening + gripper_delta, 0.0, 1.0)

    down_xmat = down_xmat_arr[0]
    down_inv = down_xmat_inv_arr[0]
    current_eff_rot = _matmul(current_xmat, down_inv)
    target_eff_rot = _quat_to_mat(plan_quat)
    rel = _matmul(_mat_transpose(current_eff_rot), target_eff_rot)
    rotvec = _rotation_log(rel)
    rot_norm = _norm3(rotvec)
    max_rot = ee_high[3]
    if rot_norm > max_rot:
        rotvec = rotvec * (max_rot / rot_norm)
    limited_eff_rot = _matmul(current_eff_rot, _rotvec_to_mat(rotvec))
    target_eff_xmat = _matmul(limited_eff_rot, down_xmat)
    target_attach_pos[world_id] = target_eff_pos + _mat_vec(target_eff_xmat, t_pa_translation[0])
    target_attach_xmat[world_id] = _matmul(target_eff_xmat, t_pa_rot_arr[0])
