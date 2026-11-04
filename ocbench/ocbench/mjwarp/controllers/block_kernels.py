import warp as wp

from ocbench.mjwarp.primitives.cube_kernels import (
    MAX_CUBES,
    MAX_PHASES,
    _block_pos,
    _generate_cube_plan,
    _is_top_block,
    _sample_clear_pos,
    _stack_anywhere_success,
    _target_pos,
    _target_success,
    _target_yaw,
)
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


@wp.func
def _choose_subgoal(
    qpos: wp.array2d[float],
    mocap_pos: wp.array2d[wp.vec3],
    mocap_quat: wp.array2d[wp.quat],
    stack_goal_xyz: wp.array2d[wp.vec3],
    object_qpos_addrs: wp.array(dtype=int),
    cube_target_mocap_ids: wp.array(dtype=int),
    goal_order: wp.array2d[int],
    rng_counter: wp.array(dtype=int),
    is_mistake: wp.array(dtype=int),
    target_block: wp.array(dtype=int),
    target_pos_arr: wp.array(dtype=wp.vec3),
    target_yaw_arr: wp.array(dtype=float),
    agent_done: wp.array(dtype=int),
    world_id: int,
    seeds: wp.array(dtype=int),
    num_cubes: int,
    task_mode: int,
    stack_anywhere: int,
    cube_size: float,
    min_dist: float,
    target_lo: wp.vec3,
    target_hi: wp.vec3,
) -> int:
    successes = wp.zeros(MAX_CUBES, dtype=int)
    all_success = int(1)
    if stack_anywhere != 0:
        all_success = _stack_anywhere_success(qpos, object_qpos_addrs, world_id, num_cubes, cube_size)
    else:
        for block in range(MAX_CUBES):
            if block < num_cubes:
                successes[block] = _target_success(
                    qpos, mocap_pos, stack_goal_xyz, object_qpos_addrs, cube_target_mocap_ids, world_id, block, 0
                )
                if successes[block] == 0:
                    all_success = 0
    if all_success != 0:
        agent_done[world_id] = 1
        return 0

    if stack_anywhere != 0:
        for block in range(MAX_CUBES):
            if block < num_cubes:
                successes[block] = _target_success(
                    qpos, mocap_pos, stack_goal_xyz, object_qpos_addrs, cube_target_mocap_ids, world_id, block, 1
                )

    goal_block = int(-1)
    if task_mode == 0:
        if successes[0] == 0:
            goal_block = 0
    elif task_mode == 1:
        for order_idx in range(MAX_CUBES):
            if order_idx < num_cubes and goal_block < 0:
                block = goal_order[world_id, order_idx]
                if successes[block] == 0:
                    goal_block = block
    else:
        best_z = 1.0e9
        for block in range(MAX_CUBES):
            if block < num_cubes and successes[block] == 0:
                target = _target_pos(mocap_pos, stack_goal_xyz, cube_target_mocap_ids, world_id, block, stack_anywhere)
                if target[2] < best_z:
                    best_z = target[2]
                    goal_block = block
    if goal_block < 0:
        agent_done[world_id] = 1
        return 0

    sub_block = goal_block
    sub_pos = _target_pos(mocap_pos, stack_goal_xyz, cube_target_mocap_ids, world_id, goal_block, stack_anywhere)
    sub_yaw = _target_yaw(mocap_quat, cube_target_mocap_ids, world_id, goal_block, stack_anywhere)
    allow_clear = stack_anywhere == 0
    if allow_clear != 0:
        for block in range(MAX_CUBES):
            if block < num_cubes and block != goal_block and successes[block] == 0:
                if _is_top_block(qpos, object_qpos_addrs, world_id, num_cubes, cube_size, block) != 0:
                    block_pos = _block_pos(qpos, object_qpos_addrs, world_id, block)
                    xy_overlap = _norm2(block_pos[0] - sub_pos[0], block_pos[1] - sub_pos[1]) <= cube_size
                    z_overlap = wp.abs(block_pos[2] - sub_pos[2]) <= cube_size
                    if xy_overlap and z_overlap:
                        sub_block = block
                        sub_pos = _sample_clear_pos(
                            qpos,
                            mocap_pos,
                            stack_goal_xyz,
                            object_qpos_addrs,
                            cube_target_mocap_ids,
                            rng_counter,
                            world_id,
                            seeds,
                            num_cubes,
                            stack_anywhere,
                            target_lo,
                            target_hi,
                            min_dist,
                            cube_size,
                        )
                        sub_yaw = _rand_uniform(world_id, seeds, rng_counter, 0.0, 6.283185307179586)
                        break

    top_count = int(0)
    top_blocks = wp.zeros(MAX_CUBES, dtype=int)
    for block in range(MAX_CUBES):
        if block < num_cubes and _is_top_block(qpos, object_qpos_addrs, world_id, num_cubes, cube_size, block) != 0:
            top_blocks[top_count] = block
            top_count += 1
    is_mistake[world_id] = 0
    if top_count > 0 and _randf(world_id, seeds, rng_counter) < 0.1:
        is_mistake[world_id] = 1
        mistake_idx = _rand_int(world_id, seeds, rng_counter, top_count)
        mistake_block = top_blocks[mistake_idx]
        sub_block = mistake_block
        if top_count >= 2 and _randf(world_id, seeds, rng_counter) < 0.5:
            support_rank = _rand_int(world_id, seeds, rng_counter, top_count - 1)
            support_block = top_blocks[support_rank]
            if support_block == mistake_block:
                support_block = top_blocks[top_count - 1]
            support_pos = _block_pos(qpos, object_qpos_addrs, world_id, support_block)
            sub_pos = wp.vec3(support_pos[0], support_pos[1], support_pos[2] + 2.0 * cube_size)
        else:
            sub_pos = wp.vec3(
                _rand_uniform(world_id, seeds, rng_counter, target_lo[0], target_hi[0]),
                _rand_uniform(world_id, seeds, rng_counter, target_lo[1], target_hi[1]),
                cube_size,
            )
        sub_yaw = _rand_uniform(world_id, seeds, rng_counter, 0.0, 6.283185307179586)

    target_block[world_id] = sub_block
    target_pos_arr[world_id] = sub_pos
    target_yaw_arr[world_id] = sub_yaw
    agent_done[world_id] = 0
    return 1


@wp.kernel
def reset_oracle(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    mocap_pos: wp.array2d[wp.vec3],
    mocap_quat: wp.array2d[wp.quat],
    object_qpos_addrs: wp.array(dtype=int),
    cube_target_mocap_ids: wp.array(dtype=int),
    rng_counter: wp.array(dtype=int),
    is_mistake: wp.array(dtype=int),
    done: wp.array(dtype=int),
    episode_length: wp.array(dtype=int),
    episode_success: wp.array(dtype=int),
    controller_done: wp.array(dtype=int),
    agent_done: wp.array(dtype=int),
    active_steps: wp.array(dtype=int),
    speed_dt: wp.array(dtype=float),
    goal_order: wp.array2d[int],
    stack_goal_xyz: wp.array2d[wp.vec3],
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
    stack_anywhere: int,
    cube_size: float,
    min_dist: float,
    target_lo: wp.vec3,
    target_hi: wp.vec3,
    arm_lo: wp.vec3,
    arm_hi: wp.vec3,
    pinch_site_id: int,
    tilt_randomization: int,
    segment_dt_scale: float,
):
    world_id = wp.tid()
    done[world_id] = 0
    episode_length[world_id] = 0
    episode_success[world_id] = 0
    controller_done[world_id] = 0
    agent_done[world_id] = 0
    active_steps[world_id] = 0
    num_pick_retries[world_id] = 0
    rng_counter[world_id] = 0
    speed_dt[world_id] = _rand_uniform(world_id, seeds, rng_counter, 0.7, 1.5) * segment_dt_scale

    for i in range(MAX_CUBES):
        goal_order[world_id, i] = i
    for i_rev in range(MAX_CUBES):
        i = num_cubes - 1 - i_rev
        if i > 0:
            j = _rand_int(world_id, seeds, rng_counter, i + 1)
            a = goal_order[world_id, i]
            goal_order[world_id, i] = goal_order[world_id, j]
            goal_order[world_id, j] = a

    if stack_anywhere != 0:
        floor_count = int(0)
        floor_blocks = wp.zeros(MAX_CUBES, dtype=int)
        for block in range(MAX_CUBES):
            if block < num_cubes:
                pos = _block_pos(qpos, object_qpos_addrs, world_id, block)
                if wp.abs(pos[2] - cube_size) <= 0.03:
                    floor_blocks[floor_count] = block
                    floor_count += 1
        xy = wp.vec3(
            _rand_uniform(world_id, seeds, rng_counter, target_lo[0], target_hi[0]),
            _rand_uniform(world_id, seeds, rng_counter, target_lo[1], target_hi[1]),
            0.0,
        )
        floor_block = int(-1)
        if floor_count > 0 and _randf(world_id, seeds, rng_counter) < 0.2:
            floor_block = floor_blocks[_rand_int(world_id, seeds, rng_counter, floor_count)]
            floor_pos = _block_pos(qpos, object_qpos_addrs, world_id, floor_block)
            xy = wp.vec3(floor_pos[0], floor_pos[1], 0.0)
        if floor_block >= 0:
            old_order = wp.zeros(MAX_CUBES, dtype=int)
            for i in range(MAX_CUBES):
                if i < num_cubes:
                    old_order[i] = goal_order[world_id, i]
            goal_order[world_id, 0] = floor_block
            write_idx = int(1)
            for i in range(MAX_CUBES):
                if i < num_cubes:
                    candidate = old_order[i]
                    if candidate != floor_block:
                        goal_order[world_id, write_idx] = candidate
                        write_idx += 1
        for level in range(MAX_CUBES):
            if level < num_cubes:
                block = goal_order[world_id, level]
                stack_goal_xyz[world_id, block] = wp.vec3(xy[0], xy[1], cube_size * (2.0 * float(level) + 1.0))

    ok = _choose_subgoal(
        qpos,
        mocap_pos,
        mocap_quat,
        stack_goal_xyz,
        object_qpos_addrs,
        cube_target_mocap_ids,
        goal_order,
        rng_counter,
        is_mistake,
        target_block,
        target_pos_arr,
        target_yaw_arr,
        agent_done,
        world_id,
        seeds,
        num_cubes,
        task_mode,
        stack_anywhere,
        cube_size,
        min_dist,
        target_lo,
        target_hi,
    )
    if ok == 0:
        controller_done[world_id] = 1
        done[world_id] = 1
    else:
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


@wp.kernel
def make_targets(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    mocap_pos: wp.array2d[wp.vec3],
    mocap_quat: wp.array2d[wp.quat],
    object_qpos_addrs: wp.array(dtype=int),
    cube_target_mocap_ids: wp.array(dtype=int),
    rng_counter: wp.array(dtype=int),
    is_mistake: wp.array(dtype=int),
    done: wp.array(dtype=int),
    controller_done: wp.array(dtype=int),
    agent_done: wp.array(dtype=int),
    active_steps: wp.array(dtype=int),
    speed_dt: wp.array(dtype=float),
    goal_order: wp.array2d[int],
    stack_goal_xyz: wp.array2d[wp.vec3],
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
    stack_anywhere: int,
    cube_size: float,
    min_dist: float,
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
        target_eff_xmat_hold = current_xmat
        target_attach_pos[world_id] = current_pos + _mat_vec(target_eff_xmat_hold, t_pa_translation[0])
        target_attach_xmat[world_id] = _matmul(target_eff_xmat_hold, t_pa_rot_arr[0])
        target_gripper[world_id] = gripper_opening
        return

    if agent_done[world_id] != 0 or active_steps[world_id] >= max_subgoal_steps:
        ok = _choose_subgoal(
            qpos,
            mocap_pos,
            mocap_quat,
            stack_goal_xyz,
            object_qpos_addrs,
            cube_target_mocap_ids,
            goal_order,
            rng_counter,
            is_mistake,
            target_block,
            target_pos_arr,
            target_yaw_arr,
            agent_done,
            world_id,
            seeds,
            num_cubes,
            task_mode,
            stack_anywhere,
            cube_size,
            min_dist,
            target_lo,
            target_hi,
        )
        if ok == 0:
            controller_done[world_id] = 1
            target_attach_pos[world_id] = current_pos + _mat_vec(current_xmat, t_pa_translation[0])
            target_attach_xmat[world_id] = _matmul(current_xmat, t_pa_rot_arr[0])
            target_gripper[world_id] = gripper_opening
            return
        active_steps[world_id] = 0
        num_pick_retries[world_id] = 0
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

    if pick_checked[world_id] == 0 and plan_time[world_id] >= wp.float64(pick_check_time[world_id]) and num_pick_retries[world_id] < 2:
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
