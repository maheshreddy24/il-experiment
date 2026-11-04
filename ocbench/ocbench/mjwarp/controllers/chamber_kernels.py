import warp as wp

from ocbench.mjwarp.controllers.controller_kernels import _hold_current_target
from ocbench.mjwarp.envs.chamber_kernels import _chamber_in_drawer
from ocbench.mjwarp.primitives.button_kernels import _generate_button_plan
from ocbench.mjwarp.primitives.cube_kernels import (
    MAX_CUBES,
    MAX_PHASES,
    _block_pos,
    _generate_cube_plan,
)
from ocbench.mjwarp.primitives.drawer_kernels import _generate_drawer_plan
from ocbench.mjwarp.primitives.primitive_kernels import (
    _clamp_vec3,
    _evaluate_plan,
    _evaluate_plan_quat,
    _mat_transpose,
    _mat_vec,
    _matmul,
    _norm2,
    _norm3,
    _quat_from_mat,
    _quat_to_mat,
    _rand_int,
    _rand_uniform,
    _randf,
    _rotation_log,
    _rotvec_to_mat,
    _sample_plan_time,
)
from ocbench.mjwarp.primitives.window_kernels import _generate_window_plan

TASK_CUBE = 0
TASK_BUTTON = 1
TASK_DRAWER = 2
TASK_WINDOW = 3


@wp.func
def _all_cubes_in_drawer(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    mocap_pos: wp.array2d[wp.vec3],
    object_qpos_addrs: wp.array(dtype=int),
    world_id: int,
    num_cubes: int,
    drawer_site_id: int,
    drawer_base_mocap_id: int,
    mirrored: int,
) -> int:
    drawer_base_pos = mocap_pos[world_id, drawer_base_mocap_id]
    drawer_handle_pos = site_xpos[world_id, drawer_site_id]
    ok = int(1)
    for cube in range(MAX_CUBES):
        if cube < num_cubes:
            pos = _block_pos(qpos, object_qpos_addrs, world_id, cube)
            if _chamber_in_drawer(pos, drawer_base_pos, drawer_handle_pos, mirrored) == 0:
                ok = 0
    return ok


@wp.func
def _chamber_stack_anywhere_success(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    mocap_pos: wp.array2d[wp.vec3],
    object_qpos_addrs: wp.array(dtype=int),
    world_id: int,
    num_cubes: int,
    cube_size: float,
    drawer_site_id: int,
    drawer_base_mocap_id: int,
    mirrored: int,
) -> int:
    xs = wp.zeros(MAX_CUBES, dtype=float)
    ys = wp.zeros(MAX_CUBES, dtype=float)
    zs = wp.zeros(MAX_CUBES, dtype=float)
    order = wp.zeros(MAX_CUBES, dtype=int)
    used = wp.zeros(MAX_CUBES, dtype=int)
    drawer_base_pos = mocap_pos[world_id, drawer_base_mocap_id]
    drawer_handle_pos = site_xpos[world_id, drawer_site_id]
    for cube in range(MAX_CUBES):
        if cube < num_cubes:
            pos = _block_pos(qpos, object_qpos_addrs, world_id, cube)
            if _chamber_in_drawer(pos, drawer_base_pos, drawer_handle_pos, mirrored) != 0:
                return 0
            xs[cube] = pos[0]
            ys[cube] = pos[1]
            zs[cube] = pos[2]
    for rank in range(MAX_CUBES):
        if rank < num_cubes:
            best = int(0)
            best_z = 1.0e9
            for cube in range(MAX_CUBES):
                if cube < num_cubes and used[cube] == 0 and zs[cube] < best_z:
                    best = cube
                    best_z = zs[cube]
            order[rank] = best
            used[best] = 1
    base = order[0]
    for cube in range(MAX_CUBES):
        if cube < num_cubes:
            if _norm2(xs[cube] - xs[base], ys[cube] - ys[base]) > 0.04:
                return 0
    for rank in range(MAX_CUBES):
        if rank < num_cubes:
            cube = order[rank]
            if wp.abs(zs[cube] - cube_size * (2.0 * float(rank) + 1.0)) > 0.03:
                return 0
    return 1


@wp.func
def _sample_chamber_goal_order(
    rng_counter: wp.array(dtype=int),
    goal_order: wp.array2d[int],
    world_id: int,
    seeds: wp.array(dtype=int),
    num_cubes: int,
):
    for cube in range(MAX_CUBES):
        goal_order[world_id, cube] = cube
    for rank in range(MAX_CUBES):
        if rank < num_cubes:
            swap_rank = rank + _rand_int(world_id, seeds, rng_counter, num_cubes - rank)
            tmp = goal_order[world_id, rank]
            goal_order[world_id, rank] = goal_order[world_id, swap_rank]
            goal_order[world_id, swap_rank] = tmp


@wp.func
def _sample_chamber_stack_goal(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    mocap_pos: wp.array2d[wp.vec3],
    object_qpos_addrs: wp.array(dtype=int),
    rng_counter: wp.array(dtype=int),
    goal_order: wp.array2d[int],
    stack_goal_xyz: wp.array2d[wp.vec3],
    world_id: int,
    seeds: wp.array(dtype=int),
    num_cubes: int,
    cube_size: float,
    drawer_site_id: int,
    drawer_base_mocap_id: int,
    mirrored: int,
    target_lo: wp.vec3,
    target_hi: wp.vec3,
):
    floor_count = int(0)
    floor_blocks = wp.zeros(MAX_CUBES, dtype=int)
    drawer_base_pos = mocap_pos[world_id, drawer_base_mocap_id]
    drawer_handle_pos = site_xpos[world_id, drawer_site_id]
    for cube in range(MAX_CUBES):
        if cube < num_cubes:
            pos = _block_pos(qpos, object_qpos_addrs, world_id, cube)
            if wp.abs(pos[2] - cube_size) <= 0.03 and _chamber_in_drawer(pos, drawer_base_pos, drawer_handle_pos, mirrored) == 0:
                floor_blocks[floor_count] = cube
                floor_count += 1

    floor_block = int(-1)
    xy = wp.vec3(
        _rand_uniform(world_id, seeds, rng_counter, target_lo[0], target_hi[0]),
        _rand_uniform(world_id, seeds, rng_counter, target_lo[1], target_hi[1]),
        0.0,
    )
    if mirrored != 0:
        xy = wp.vec3(xy[0], -xy[1], 0.0)
    if floor_count > 0:
        floor_block = floor_blocks[_rand_int(world_id, seeds, rng_counter, floor_count)]
        floor_pos = _block_pos(qpos, object_qpos_addrs, world_id, floor_block)
        xy = wp.vec3(floor_pos[0], floor_pos[1], 0.0)

    level = int(0)
    if floor_block >= 0:
        stack_goal_xyz[world_id, floor_block] = wp.vec3(xy[0], xy[1], cube_size)
        level = 1
    for rank in range(MAX_CUBES):
        if rank < num_cubes:
            block = goal_order[world_id, rank]
            if block != floor_block:
                stack_goal_xyz[world_id, block] = wp.vec3(
                    xy[0],
                    xy[1],
                    cube_size * (2.0 * float(level) + 1.0),
                )
                level += 1


@wp.func
def _first_unsuccessful_in_drawer_block(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    mocap_pos: wp.array2d[wp.vec3],
    object_qpos_addrs: wp.array(dtype=int),
    goal_order: wp.array2d[int],
    world_id: int,
    num_cubes: int,
    drawer_site_id: int,
    drawer_base_mocap_id: int,
    mirrored: int,
) -> int:
    drawer_base_pos = mocap_pos[world_id, drawer_base_mocap_id]
    drawer_handle_pos = site_xpos[world_id, drawer_site_id]
    out = int(0)
    found = int(0)
    for rank in range(MAX_CUBES):
        if rank < num_cubes and found == 0:
            block = goal_order[world_id, rank]
            pos = _block_pos(qpos, object_qpos_addrs, world_id, block)
            if _chamber_in_drawer(pos, drawer_base_pos, drawer_handle_pos, mirrored) == 0:
                out = block
                found = 1
    return out


@wp.func
def _first_unsuccessful_stack_block(
    qpos: wp.array2d[float],
    object_qpos_addrs: wp.array(dtype=int),
    stack_goal_xyz: wp.array2d[wp.vec3],
    world_id: int,
    num_cubes: int,
) -> int:
    out = int(0)
    best_z = 1.0e9
    for cube in range(MAX_CUBES):
        if cube < num_cubes:
            pos = _block_pos(qpos, object_qpos_addrs, world_id, cube)
            goal = stack_goal_xyz[world_id, cube]
            diff = pos - goal
            success = int(_norm3(diff) <= 0.04)
            if success == 0 and goal[2] < best_z:
                best_z = goal[2]
                out = cube
    return out


@wp.func
def _in_drawer_cube_plan_target(
    block: int,
    num_cubes: int,
    drawer_base_pos: wp.vec3,
    nominal_drawer_pos: wp.vec3,
    drawer_pos: float,
    drawer_goal_pos: float,
    mirrored: int,
) -> wp.vec3:
    goal_x = 0.33
    if num_cubes == 2:
        goal_x = 0.27
        if block == 1:
            goal_x = 0.39
    elif num_cubes >= 3:
        goal_x = 0.26
        if block == 1:
            goal_x = 0.33
        elif block >= 2:
            goal_x = 0.40

    goal_y = -0.356
    nominal_y = nominal_drawer_pos[1]
    slide_sign = -1.0
    if mirrored != 0:
        goal_y = -goal_y
        nominal_y = -nominal_y
        slide_sign = 1.0
    return wp.vec3(
        drawer_base_pos[0] + (goal_x - nominal_drawer_pos[0]),
        drawer_base_pos[1] + (goal_y - nominal_y) + slide_sign * (drawer_pos - drawer_goal_pos),
        0.166,
    )


@wp.func
def _choose_chamber_subgoal(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    mocap_pos: wp.array2d[wp.vec3],
    object_qpos_addrs: wp.array(dtype=int),
    button_states: wp.array2d[int],
    target_button_states: wp.array2d[int],
    mirrored_arr: wp.array(dtype=int),
    rng_counter: wp.array(dtype=int),
    is_mistake: wp.array(dtype=int),
    active_task: wp.array(dtype=int),
    goal_order: wp.array2d[int],
    stack_goal_xyz: wp.array2d[wp.vec3],
    target_button: wp.array(dtype=int),
    target_slide_pos: wp.array(dtype=float),
    target_block: wp.array(dtype=int),
    target_pos_arr: wp.array(dtype=wp.vec3),
    target_yaw_arr: wp.array(dtype=float),
    agent_done: wp.array(dtype=int),
    active_settle_steps: wp.array(dtype=int),
    settle_after_cube: wp.array(dtype=int),
    world_id: int,
    seeds: wp.array(dtype=int),
    num_cubes: int,
    task_mode: int,
    cube_size: float,
    drawer_qpos_addr: int,
    window_qpos_addr: int,
    drawer_site_id: int,
    window_site_id: int,
    button0_site_id: int,
    button1_site_id: int,
    drawer_base_mocap_id: int,
    drawer_open_threshold: float,
    drawer_slide_min: float,
    drawer_slide_max: float,
    window_slide_min: float,
    window_slide_max: float,
    p_mistake: float,
    nominal_drawer_pos: wp.vec3,
    target_lo: wp.vec3,
    target_hi: wp.vec3,
) -> int:
    mirrored = mirrored_arr[world_id]
    drawer_pos = qpos[world_id, drawer_qpos_addr]
    window_pos = qpos[world_id, window_qpos_addr]
    drawer_base_pos = mocap_pos[world_id, drawer_base_mocap_id]
    drawer_handle_pos = site_xpos[world_id, drawer_site_id]

    in_drawer_success = _all_cubes_in_drawer(
        qpos,
        site_xpos,
        mocap_pos,
        object_qpos_addrs,
        world_id,
        num_cubes,
        drawer_site_id,
        drawer_base_mocap_id,
        mirrored,
    )
    stack_success = _chamber_stack_anywhere_success(
        qpos,
        site_xpos,
        mocap_pos,
        object_qpos_addrs,
        world_id,
        num_cubes,
        cube_size,
        drawer_site_id,
        drawer_base_mocap_id,
        mirrored,
    )
    cube_success = in_drawer_success
    drawer_goal = drawer_slide_max
    window_goal = window_slide_max
    if task_mode == 1:
        cube_success = stack_success
        window_goal = window_slide_min
    elif task_mode == 2:
        cube_success = 1
        drawer_goal = drawer_slide_min
    drawer_success = int(wp.abs(drawer_pos - drawer_goal) <= 0.04)
    window_success = int(wp.abs(window_pos - window_goal) <= 0.04)
    button0_success = int(button_states[world_id, 0] == target_button_states[world_id, 0])
    button1_success = int(button_states[world_id, 1] == target_button_states[world_id, 1])
    active_settle_steps[world_id] = 0
    settle_after_cube[world_id] = 0

    drawer_candidate = int(-1)
    drawer_target_pos = drawer_goal
    window_candidate = int(-1)
    cube_block = int(0)
    cube_target = wp.vec3(0.0, 0.0, cube_size)
    if cube_success == 0:
        if task_mode == 0:
            if drawer_pos > drawer_open_threshold:
                if button_states[world_id, 0] == 0:
                    drawer_candidate = TASK_BUTTON
                else:
                    drawer_candidate = TASK_DRAWER
                    drawer_target_pos = drawer_slide_min
            else:
                drawer_candidate = TASK_CUBE
                cube_block = _first_unsuccessful_in_drawer_block(
                    qpos,
                    site_xpos,
                    mocap_pos,
                    object_qpos_addrs,
                    goal_order,
                    world_id,
                    num_cubes,
                    drawer_site_id,
                    drawer_base_mocap_id,
                    mirrored,
                )
                cube_target = _in_drawer_cube_plan_target(
                    cube_block,
                    num_cubes,
                    drawer_base_pos,
                    nominal_drawer_pos,
                    drawer_pos,
                    drawer_slide_max,
                    mirrored,
                )
        elif task_mode == 1:
            cube_block = _first_unsuccessful_stack_block(qpos, object_qpos_addrs, stack_goal_xyz, world_id, num_cubes)
            cube_pos = _block_pos(qpos, object_qpos_addrs, world_id, cube_block)
            if _chamber_in_drawer(cube_pos, drawer_base_pos, drawer_handle_pos, mirrored) != 0 and drawer_pos > drawer_open_threshold:
                if button_states[world_id, 0] == 0:
                    drawer_candidate = TASK_BUTTON
                else:
                    drawer_candidate = TASK_DRAWER
                    drawer_target_pos = drawer_slide_min
            else:
                drawer_candidate = TASK_CUBE
                cube_target = stack_goal_xyz[world_id, cube_block]
    elif drawer_success == 0:
        if button_states[world_id, 0] == 0:
            drawer_candidate = TASK_BUTTON
        else:
            drawer_candidate = TASK_DRAWER
    elif button0_success == 0:
        drawer_candidate = TASK_BUTTON

    if window_success == 0:
        if button_states[world_id, 1] == 0:
            window_candidate = TASK_BUTTON
        else:
            window_candidate = TASK_WINDOW
    elif button1_success == 0:
        window_candidate = TASK_BUTTON

    if drawer_candidate < 0 and window_candidate < 0:
        agent_done[world_id] = 1
        return 0

    mistake_count = int(0)
    mistake_codes = wp.zeros(5, dtype=int)
    mistake_codes[mistake_count] = 0
    mistake_count += 1
    mistake_codes[mistake_count] = 1
    mistake_count += 1
    if button_states[world_id, 0] == 1:
        mistake_codes[mistake_count] = 2
        mistake_count += 1
    if button_states[world_id, 1] == 1:
        mistake_codes[mistake_count] = 3
        mistake_count += 1
    if (task_mode == 0 and cube_success == 0) or (task_mode == 1 and cube_success != 0 and drawer_pos <= drawer_open_threshold):
        mistake_codes[mistake_count] = 4
        mistake_count += 1

    is_mistake[world_id] = 0
    if mistake_count > 0 and _randf(world_id, seeds, rng_counter) < p_mistake:
        is_mistake[world_id] = 1
        mistake = mistake_codes[_rand_int(world_id, seeds, rng_counter, mistake_count)]
        if mistake <= 1:
            active_task[world_id] = TASK_BUTTON
            target_button[world_id] = mistake
        elif mistake == 2:
            active_task[world_id] = TASK_DRAWER
            drawer_mid = (drawer_slide_min + drawer_slide_max) * 0.5
            target_slide_pos[world_id] = drawer_slide_max
            if drawer_pos >= drawer_mid:
                target_slide_pos[world_id] = drawer_slide_min
        elif mistake == 3:
            active_task[world_id] = TASK_WINDOW
            window_mid = (window_slide_min + window_slide_max) * 0.5
            target_slide_pos[world_id] = window_slide_min
            if window_pos <= window_mid:
                target_slide_pos[world_id] = window_slide_max
        else:
            active_task[world_id] = TASK_CUBE
            if task_mode == 0:
                success_count = int(0)
                success_blocks = wp.zeros(MAX_CUBES, dtype=int)
                for cube in range(MAX_CUBES):
                    if cube < num_cubes:
                        pos = _block_pos(qpos, object_qpos_addrs, world_id, cube)
                        if _chamber_in_drawer(pos, drawer_base_pos, drawer_handle_pos, mirrored) != 0:
                            success_blocks[success_count] = cube
                            success_count += 1
                if success_count > 0:
                    target_block[world_id] = success_blocks[_rand_int(world_id, seeds, rng_counter, success_count)]
                else:
                    target_block[world_id] = _rand_int(world_id, seeds, rng_counter, num_cubes)
                y = _rand_uniform(world_id, seeds, rng_counter, target_lo[1], target_hi[1])
                if mirrored != 0:
                    y = -y
                target_pos_arr[world_id] = wp.vec3(
                    _rand_uniform(world_id, seeds, rng_counter, target_lo[0], target_hi[0]),
                    y,
                    cube_size,
                )
            else:
                slide_sign = -1.0
                if mirrored != 0:
                    slide_sign = 1.0
                target_block[world_id] = 0
                target_pos_arr[world_id] = wp.vec3(
                    drawer_base_pos[0] + _rand_uniform(world_id, seeds, rng_counter, -0.06, 0.06),
                    drawer_handle_pos[1] + slide_sign * _rand_uniform(world_id, seeds, rng_counter, 0.16, 0.18),
                    0.076,
                )
            target_yaw_arr[world_id] = 0.0
        agent_done[world_id] = 0
        return 1

    chosen = int(-1)
    use_window = int(0)
    if drawer_candidate >= 0 and window_candidate >= 0:
        if _randf(world_id, seeds, rng_counter) < 0.5:
            use_window = 1
    elif window_candidate >= 0:
        use_window = 1
    if use_window != 0:
        chosen = window_candidate
    else:
        chosen = drawer_candidate

    active_task[world_id] = chosen
    if chosen == TASK_BUTTON:
        button = int(0)
        if use_window != 0:
            button = 1
        target_button[world_id] = button
    elif chosen == TASK_DRAWER:
        target_slide_pos[world_id] = drawer_target_pos
    elif chosen == TASK_WINDOW:
        target_slide_pos[world_id] = window_goal
    else:
        target_block[world_id] = cube_block
        target_pos_arr[world_id] = cube_target
        target_yaw_arr[world_id] = 0.0
        if task_mode == 0:
            settle_after_cube[world_id] = 1
    agent_done[world_id] = 0
    return 1


@wp.func
def _start_chamber_plan(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    object_qpos_addrs: wp.array(dtype=int),
    mirrored_arr: wp.array(dtype=int),
    rng_counter: wp.array(dtype=int),
    active_task: wp.array(dtype=int),
    target_button: wp.array(dtype=int),
    target_slide_pos: wp.array(dtype=float),
    target_block: wp.array(dtype=int),
    target_pos_arr: wp.array(dtype=wp.vec3),
    target_yaw_arr: wp.array(dtype=float),
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
    phase_start: wp.array2d[float],
    phase_end: wp.array2d[float],
    phase_c1: wp.array2d[wp.vec3],
    phase_c2: wp.array2d[wp.vec3],
    pick_check_time: wp.array(dtype=float),
    pick_source_z: wp.array(dtype=float),
    plan_time: wp.array(dtype=wp.float64),
    last_time: wp.array(dtype=float),
    agent_done: wp.array(dtype=int),
    pick_checked: wp.array(dtype=int),
    num_pick_retries: wp.array(dtype=int),
    active_block: wp.array(dtype=int),
    world_id: int,
    seeds: wp.array(dtype=int),
    pinch_site_id: int,
    button0_site_id: int,
    button1_site_id: int,
    drawer_site_id: int,
    window_site_id: int,
    drawer_qpos_addr: int,
    window_qpos_addr: int,
    segment_dt: float,
    arm_lo: wp.vec3,
    arm_hi: wp.vec3,
    tilt_randomization: int,
):
    task = active_task[world_id]
    if task == TASK_CUBE:
        _generate_cube_plan(
            qpos,
            site_xpos,
            site_xmat,
            object_qpos_addrs,
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
            phase_start,
            phase_end,
            phase_c1,
            phase_c2,
            pick_check_time,
            pick_source_z,
            plan_time,
            last_time,
            agent_done,
            pick_checked,
            num_pick_retries,
            active_block,
            target_block,
            target_pos_arr,
            target_yaw_arr,
            world_id,
            seeds,
            pinch_site_id,
            segment_dt,
            arm_lo,
            arm_hi,
            tilt_randomization,
        )
    elif task == TASK_BUTTON:
        button_site = button0_site_id
        if target_button[world_id] == 1:
            button_site = button1_site_id
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
            button_site,
            segment_dt,
            arm_lo,
            arm_hi,
            0,
            tilt_randomization,
        )
    elif task == TASK_DRAWER:
        _generate_drawer_plan(
            qpos,
            site_xpos,
            site_xmat,
            mirrored_arr,
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
            drawer_site_id,
            drawer_qpos_addr,
            target_slide_pos,
            segment_dt,
            arm_lo,
            arm_hi,
            tilt_randomization,
        )
    else:
        _generate_window_plan(
            qpos,
            site_xpos,
            site_xmat,
            mirrored_arr,
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
            window_site_id,
            window_qpos_addr,
            target_slide_pos,
            segment_dt,
            arm_lo,
            arm_hi,
            tilt_randomization,
        )


@wp.kernel
def start_chamber_plans(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    mocap_pos: wp.array2d[wp.vec3],
    object_qpos_addrs: wp.array(dtype=int),
    button_states: wp.array2d[int],
    target_button_states: wp.array2d[int],
    mirrored_arr: wp.array(dtype=int),
    rng_counter: wp.array(dtype=int),
    is_mistake: wp.array(dtype=int),
    done: wp.array(dtype=int),
    controller_done: wp.array(dtype=int),
    agent_done: wp.array(dtype=int),
    active_steps: wp.array(dtype=int),
    active_settle_steps: wp.array(dtype=int),
    settle_after_cube: wp.array(dtype=int),
    active_task: wp.array(dtype=int),
    goal_order: wp.array2d[int],
    stack_goal_xyz: wp.array2d[wp.vec3],
    target_button: wp.array(dtype=int),
    target_slide_pos: wp.array(dtype=float),
    active_block: wp.array(dtype=int),
    target_block: wp.array(dtype=int),
    target_pos_arr: wp.array(dtype=wp.vec3),
    target_yaw_arr: wp.array(dtype=float),
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
    phase_start: wp.array2d[float],
    phase_end: wp.array2d[float],
    phase_c1: wp.array2d[wp.vec3],
    phase_c2: wp.array2d[wp.vec3],
    pick_check_time: wp.array(dtype=float),
    pick_source_z: wp.array(dtype=float),
    plan_time: wp.array(dtype=wp.float64),
    last_time: wp.array(dtype=float),
    pick_checked: wp.array(dtype=int),
    num_pick_retries: wp.array(dtype=int),
    seeds: wp.array(dtype=int),
    num_cubes: int,
    task_mode: int,
    cube_size: float,
    drawer_qpos_addr: int,
    window_qpos_addr: int,
    drawer_site_id: int,
    window_site_id: int,
    button0_site_id: int,
    button1_site_id: int,
    drawer_base_mocap_id: int,
    drawer_open_threshold: float,
    drawer_slide_min: float,
    drawer_slide_max: float,
    window_slide_min: float,
    window_slide_max: float,
    p_mistake: float,
    cube_settle_steps: int,
    nominal_drawer_pos: wp.vec3,
    target_lo: wp.vec3,
    target_hi: wp.vec3,
    pinch_site_id: int,
    speed_dt: wp.array(dtype=float),
    arm_lo: wp.vec3,
    arm_hi: wp.vec3,
    max_subgoal_steps: int,
    tilt_randomization: int,
):
    world_id = wp.tid()
    if done[world_id] != 0 or controller_done[world_id] != 0:
        return

    if (
        agent_done[world_id] != 0
        and active_task[world_id] == TASK_CUBE
        and settle_after_cube[world_id] != 0
        and active_settle_steps[world_id] < cube_settle_steps
        and active_steps[world_id] < max_subgoal_steps
    ):
        block = target_block[world_id]
        block_pos = _block_pos(qpos, object_qpos_addrs, world_id, block)
        if _chamber_in_drawer(
            block_pos,
            mocap_pos[world_id, drawer_base_mocap_id],
            site_xpos[world_id, drawer_site_id],
            mirrored_arr[world_id],
        ) == 0:
            return

    should_start = key_count[world_id] == 0
    if agent_done[world_id] != 0 or active_steps[world_id] >= max_subgoal_steps:
        ok = _choose_chamber_subgoal(
            qpos,
            site_xpos,
            site_xmat,
            mocap_pos,
            object_qpos_addrs,
            button_states,
            target_button_states,
            mirrored_arr,
            rng_counter,
            is_mistake,
            active_task,
            goal_order,
            stack_goal_xyz,
            target_button,
            target_slide_pos,
            target_block,
            target_pos_arr,
            target_yaw_arr,
            agent_done,
            active_settle_steps,
            settle_after_cube,
            world_id,
            seeds,
            num_cubes,
            task_mode,
            cube_size,
            drawer_qpos_addr,
            window_qpos_addr,
            drawer_site_id,
            window_site_id,
            button0_site_id,
            button1_site_id,
            drawer_base_mocap_id,
            drawer_open_threshold,
            drawer_slide_min,
            drawer_slide_max,
            window_slide_min,
            window_slide_max,
            p_mistake,
            nominal_drawer_pos,
            target_lo,
            target_hi,
        )
        if ok == 0:
            controller_done[world_id] = 1
            return
        active_steps[world_id] = 0
        num_pick_retries[world_id] = 0
        should_start = True

    if should_start:
        _start_chamber_plan(
            qpos,
            site_xpos,
            site_xmat,
            object_qpos_addrs,
            mirrored_arr,
            rng_counter,
            active_task,
            target_button,
            target_slide_pos,
            target_block,
            target_pos_arr,
            target_yaw_arr,
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
            phase_start,
            phase_end,
            phase_c1,
            phase_c2,
            pick_check_time,
            pick_source_z,
            plan_time,
            last_time,
            agent_done,
            pick_checked,
            num_pick_retries,
            active_block,
            world_id,
            seeds,
            pinch_site_id,
            button0_site_id,
            button1_site_id,
            drawer_site_id,
            window_site_id,
            drawer_qpos_addr,
            window_qpos_addr,
            speed_dt[world_id],
            arm_lo,
            arm_hi,
            tilt_randomization,
        )


@wp.kernel
def reset_controller(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    mocap_pos: wp.array2d[wp.vec3],
    object_qpos_addrs: wp.array(dtype=int),
    button_states: wp.array2d[int],
    target_button_states: wp.array2d[int],
    mirrored_arr: wp.array(dtype=int),
    rng_counter: wp.array(dtype=int),
    is_mistake: wp.array(dtype=int),
    done: wp.array(dtype=int),
    episode_length: wp.array(dtype=int),
    episode_success: wp.array(dtype=int),
    controller_done: wp.array(dtype=int),
    agent_done: wp.array(dtype=int),
    active_steps: wp.array(dtype=int),
    active_settle_steps: wp.array(dtype=int),
    settle_after_cube: wp.array(dtype=int),
    speed_dt: wp.array(dtype=float),
    goal_order: wp.array2d[int],
    stack_goal_xyz: wp.array2d[wp.vec3],
    active_task: wp.array(dtype=int),
    target_button: wp.array(dtype=int),
    target_slide_pos: wp.array(dtype=float),
    active_block: wp.array(dtype=int),
    target_block: wp.array(dtype=int),
    target_pos_arr: wp.array(dtype=wp.vec3),
    target_yaw_arr: wp.array(dtype=float),
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
    phase_start: wp.array2d[float],
    phase_end: wp.array2d[float],
    phase_c1: wp.array2d[wp.vec3],
    phase_c2: wp.array2d[wp.vec3],
    pick_check_time: wp.array(dtype=float),
    pick_source_z: wp.array(dtype=float),
    plan_time: wp.array(dtype=wp.float64),
    last_time: wp.array(dtype=float),
    pick_checked: wp.array(dtype=int),
    num_pick_retries: wp.array(dtype=int),
    seeds: wp.array(dtype=int),
    num_cubes: int,
    task_mode: int,
    cube_size: float,
    drawer_qpos_addr: int,
    window_qpos_addr: int,
    drawer_site_id: int,
    window_site_id: int,
    button0_site_id: int,
    button1_site_id: int,
    drawer_base_mocap_id: int,
    drawer_open_threshold: float,
    drawer_slide_min: float,
    drawer_slide_max: float,
    window_slide_min: float,
    window_slide_max: float,
    p_mistake: float,
    cube_settle_steps: int,
    nominal_drawer_pos: wp.vec3,
    target_lo: wp.vec3,
    target_hi: wp.vec3,
    pinch_site_id: int,
    arm_lo: wp.vec3,
    arm_hi: wp.vec3,
    segment_dt_scale: float,
):
    world_id = wp.tid()
    done[world_id] = 0
    episode_length[world_id] = 0
    episode_success[world_id] = 0
    controller_done[world_id] = 0
    agent_done[world_id] = 0
    active_steps[world_id] = 0
    active_settle_steps[world_id] = 0
    settle_after_cube[world_id] = 0
    num_pick_retries[world_id] = 0
    key_count[world_id] = 0
    rng_counter[world_id] = 0
    speed_dt[world_id] = _rand_uniform(world_id, seeds, rng_counter, 0.7, 1.5) * segment_dt_scale
    _sample_chamber_goal_order(rng_counter, goal_order, world_id, seeds, num_cubes)
    if task_mode == 1:
        _sample_chamber_stack_goal(
            qpos,
            site_xpos,
            mocap_pos,
            object_qpos_addrs,
            rng_counter,
            goal_order,
            stack_goal_xyz,
            world_id,
            seeds,
            num_cubes,
            cube_size,
            drawer_site_id,
            drawer_base_mocap_id,
            mirrored_arr[world_id],
            target_lo,
            target_hi,
        )
    ok = _choose_chamber_subgoal(
        qpos,
        site_xpos,
        site_xmat,
        mocap_pos,
        object_qpos_addrs,
        button_states,
        target_button_states,
        mirrored_arr,
        rng_counter,
        is_mistake,
        active_task,
        goal_order,
        stack_goal_xyz,
        target_button,
        target_slide_pos,
        target_block,
        target_pos_arr,
        target_yaw_arr,
        agent_done,
        active_settle_steps,
        settle_after_cube,
        world_id,
        seeds,
        num_cubes,
        task_mode,
        cube_size,
        drawer_qpos_addr,
        window_qpos_addr,
        drawer_site_id,
        window_site_id,
        button0_site_id,
        button1_site_id,
        drawer_base_mocap_id,
        drawer_open_threshold,
        drawer_slide_min,
        drawer_slide_max,
        window_slide_min,
        window_slide_max,
        p_mistake,
        nominal_drawer_pos,
        target_lo,
        target_hi,
    )
    if ok == 0:
        controller_done[world_id] = 1
        done[world_id] = 1


@wp.kernel
def make_targets(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    mocap_pos: wp.array2d[wp.vec3],
    object_qpos_addrs: wp.array(dtype=int),
    button_states: wp.array2d[int],
    target_button_states: wp.array2d[int],
    mirrored_arr: wp.array(dtype=int),
    rng_counter: wp.array(dtype=int),
    is_mistake: wp.array(dtype=int),
    done: wp.array(dtype=int),
    controller_done: wp.array(dtype=int),
    agent_done: wp.array(dtype=int),
    active_steps: wp.array(dtype=int),
    active_settle_steps: wp.array(dtype=int),
    settle_after_cube: wp.array(dtype=int),
    speed_dt: wp.array(dtype=float),
    goal_order: wp.array2d[int],
    stack_goal_xyz: wp.array2d[wp.vec3],
    active_task: wp.array(dtype=int),
    target_button: wp.array(dtype=int),
    target_slide_pos: wp.array(dtype=float),
    active_block: wp.array(dtype=int),
    target_block: wp.array(dtype=int),
    target_pos_arr: wp.array(dtype=wp.vec3),
    target_yaw_arr: wp.array(dtype=float),
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
    phase_start: wp.array2d[float],
    phase_end: wp.array2d[float],
    phase_c1: wp.array2d[wp.vec3],
    phase_c2: wp.array2d[wp.vec3],
    pick_check_time: wp.array(dtype=float),
    pick_source_z: wp.array(dtype=float),
    plan_time: wp.array(dtype=wp.float64),
    last_time: wp.array(dtype=float),
    pick_checked: wp.array(dtype=int),
    num_pick_retries: wp.array(dtype=int),
    target_attach_pos: wp.array(dtype=wp.vec3),
    target_attach_xmat: wp.array(dtype=wp.mat33),
    target_gripper: wp.array(dtype=float),
    down_xmat_arr: wp.array(dtype=wp.mat33),
    down_xmat_inv_arr: wp.array(dtype=wp.mat33),
    t_pa_rot_arr: wp.array(dtype=wp.mat33),
    t_pa_translation: wp.array(dtype=wp.vec3),
    seeds: wp.array(dtype=int),
    num_cubes: int,
    task_mode: int,
    cube_size: float,
    drawer_qpos_addr: int,
    window_qpos_addr: int,
    drawer_site_id: int,
    window_site_id: int,
    button0_site_id: int,
    button1_site_id: int,
    drawer_base_mocap_id: int,
    drawer_open_threshold: float,
    drawer_slide_min: float,
    drawer_slide_max: float,
    window_slide_min: float,
    window_slide_max: float,
    p_mistake: float,
    cube_settle_steps: int,
    nominal_drawer_pos: wp.vec3,
    target_lo: wp.vec3,
    target_hi: wp.vec3,
    arm_lo: wp.vec3,
    arm_hi: wp.vec3,
    workspace_lo: wp.vec3,
    workspace_hi: wp.vec3,
    ee_low: wp.array(dtype=float),
    ee_high: wp.array(dtype=float),
    pinch_site_id: int,
    gripper_qpos_id: int,
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

    if (
        agent_done[world_id] != 0
        and active_task[world_id] == TASK_CUBE
        and settle_after_cube[world_id] != 0
        and active_settle_steps[world_id] < cube_settle_steps
        and active_steps[world_id] < max_subgoal_steps
    ):
        block = target_block[world_id]
        block_pos = _block_pos(qpos, object_qpos_addrs, world_id, block)
        if _chamber_in_drawer(
            block_pos,
            mocap_pos[world_id, drawer_base_mocap_id],
            site_xpos[world_id, drawer_site_id],
            mirrored_arr[world_id],
        ) == 0:
            active_settle_steps[world_id] += 1
            active_steps[world_id] += 1
            _hold_current_target(site_xpos, site_xmat, qpos, target_attach_pos, target_attach_xmat, target_gripper, t_pa_rot_arr, t_pa_translation, world_id, pinch_site_id, gripper_qpos_id)
            return

    if key_count[world_id] == 0:
        _hold_current_target(site_xpos, site_xmat, qpos, target_attach_pos, target_attach_xmat, target_gripper, t_pa_rot_arr, t_pa_translation, world_id, pinch_site_id, gripper_qpos_id)
        return
    active_steps[world_id] += 1

    if last_time[world_id] >= 0.0:
        plan_time[world_id] += env_dt
    last_time[world_id] = 0.0

    blocked_gate = int(-1)
    task = active_task[world_id]
    if task == TASK_CUBE or task == TASK_BUTTON or task == TASK_DRAWER or task == TASK_WINDOW:
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

    if task == TASK_CUBE and pick_checked[world_id] == 0 and plan_time[world_id] >= wp.float64(pick_check_time[world_id]) and num_pick_retries[world_id] < 2:
        pick_checked[world_id] = 1
        block = active_block[world_id]
        block_pos = _block_pos(qpos, object_qpos_addrs, world_id, block)
        if block_pos[2] < pick_source_z[world_id] + 0.025:
            num_pick_retries[world_id] += 1
            _generate_cube_plan(
                qpos,
                site_xpos,
                site_xmat,
                object_qpos_addrs,
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
                phase_start,
                phase_end,
                phase_c1,
                phase_c2,
                pick_check_time,
                pick_source_z,
                plan_time,
                last_time,
                agent_done,
                pick_checked,
                num_pick_retries,
                active_block,
                target_block,
                target_pos_arr,
                target_yaw_arr,
                world_id,
                seeds,
                pinch_site_id,
                speed_dt[world_id],
                arm_lo,
                arm_hi,
                tilt_randomization,
            )
            key_quat[world_id, 0] = _quat_from_mat(_matmul(current_xmat, down_xmat_inv_arr[0]))
            query_time = 0.0
            blocked_gate = -1
            last_time[world_id] = 0.0

    eval = _evaluate_plan(key_count, key_time, key_xyz, key_quat, key_grasp, key_tangent, world_id, query_time)
    ab_xyz = wp.vec3(eval[0], eval[1], eval[2])
    ab_grasp = eval[3]
    plan_quat = _evaluate_plan_quat(key_count, key_time, key_quat, world_id, query_time)
    if blocked_gate >= 0:
        ab_xyz = gate_xyz[world_id, blocked_gate]

    if task == TASK_CUBE:
        for phase in range(MAX_PHASES):
            if plan_time[world_id] >= wp.float64(phase_start[world_id, phase]) and plan_time[world_id] <= wp.float64(phase_end[world_id, phase]):
                u = wp.clamp(
                    float(
                        (plan_time[world_id] - wp.float64(phase_start[world_id, phase]))
                        / wp.float64(phase_end[world_id, phase] - phase_start[world_id, phase])
                    ),
                    0.0,
                    1.0,
                )
                c1 = phase_c1[world_id, phase]
                c2 = phase_c2[world_id, phase]
                curve = 16.0 * u * u * (1.0 - u) * (1.0 - u) * ((1.0 - u) * c1 + u * c2)
                ab_xyz = wp.vec3(ab_xyz[0] + curve[0], ab_xyz[1] + curve[1], ab_xyz[2])
                break

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
