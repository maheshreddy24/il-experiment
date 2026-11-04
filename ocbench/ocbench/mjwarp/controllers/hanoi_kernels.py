import warp as wp

from ocbench.mjwarp import MAX_PLAN_KEYS
from ocbench.mjwarp.envs.hanoi_kernels import (
    MAX_HANOI_DISKS,
    MAX_HANOI_PEGS,
    MAX_HANOI_STACK_SLOTS,
    _disk_pos,
    _disk_yaw_for_peg,
)
from ocbench.mjwarp.envs.manipulation_kernels import (
    _dist2_xy,
    _quat_yaw,
)
from ocbench.mjwarp.primitives.primitive_kernels import (
    _clamp_vec3,
    _compute_tangents,
    _evaluate_plan,
    _evaluate_plan_quat,
    _mat_transpose,
    _mat_vec,
    _matmul,
    _norm2,
    _norm3,
    _quat_from_rpy,
    _quat_to_mat,
    _rand_int,
    _rand_uniform,
    _randf,
    _rotation_log,
    _rotvec_to_mat,
    _sample_plan_time,
    _shortest_yaw_n2,
    _smooth_times,
    _write_key,
    _yaw_error,
)


@wp.func
def _other_peg(source: int, target: int) -> int:
    out = int(0)
    for peg in range(MAX_HANOI_PEGS):
        if peg != source and peg != target:
            out = peg
    return out


@wp.func
def _encode_move(disk: int, source: int, target: int) -> int:
    return disk * 9 + source * 3 + target


@wp.func
def _move_disk(code: int) -> int:
    return code // 9


@wp.func
def _move_source(code: int) -> int:
    return (code - (code // 9) * 9) // 3


@wp.func
def _move_target(code: int) -> int:
    return code - (code // 3) * 3


@wp.func
def _first_move_1(target: int, positions: wp.array(dtype=int)) -> int:
    if positions[0] != target:
        return _encode_move(0, positions[0], target)
    return -1


@wp.func
def _first_move_2(target: int, positions: wp.array(dtype=int)) -> int:
    if positions[1] != target:
        source = positions[1]
        aux = _other_peg(source, target)
        move = _first_move_1(aux, positions)
        if move >= 0:
            return move
        return _encode_move(1, source, target)
    return _first_move_1(target, positions)


@wp.func
def _first_move_3(target: int, positions: wp.array(dtype=int)) -> int:
    if positions[2] != target:
        source = positions[2]
        aux = _other_peg(source, target)
        move = _first_move_2(aux, positions)
        if move >= 0:
            return move
        return _encode_move(2, source, target)
    return _first_move_2(target, positions)


@wp.func
def _first_move(num_disks: int, target: int, positions: wp.array(dtype=int)) -> int:
    if num_disks == 1:
        return _first_move_1(target, positions)
    if num_disks == 2:
        return _first_move_2(target, positions)
    return _first_move_3(target, positions)


@wp.func
def _infer_stacks(
    qpos: wp.array2d[float],
    disk_qpos_addrs: wp.array(dtype=int),
    peg_pos: wp.array2d[wp.vec3],
    stacks: wp.array(dtype=int),
    stack_counts: wp.array(dtype=int),
    positions: wp.array(dtype=int),
    disk_zs: wp.array(dtype=float),
    world_id: int,
    num_disks: int,
    num_pegs: int,
):
    for peg in range(MAX_HANOI_PEGS):
        stack_counts[peg] = 0
    for disk in range(MAX_HANOI_DISKS):
        if disk < num_disks:
            pos = _disk_pos(qpos, disk_qpos_addrs, world_id, disk)
            disk_zs[disk] = pos[2]
            best_peg = int(0)
            best_dist = float(1.0e9)
            for peg in range(MAX_HANOI_PEGS):
                if peg < num_pegs:
                    target = peg_pos[world_id, peg]
                    dist = _dist2_xy(pos[0], pos[1], target[0], target[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_peg = peg
            positions[disk] = best_peg
            count = stack_counts[best_peg]
            stacks[best_peg * MAX_HANOI_DISKS + count] = disk
            stack_counts[best_peg] = count + 1

    for peg in range(MAX_HANOI_PEGS):
        if peg < num_pegs:
            count = stack_counts[peg]
            for i in range(MAX_HANOI_DISKS):
                if i < count:
                    best_i = i
                    best_z = disk_zs[stacks[peg * MAX_HANOI_DISKS + i]]
                    for j in range(MAX_HANOI_DISKS):
                        if j >= i + 1 and j < count:
                            disk_j = stacks[peg * MAX_HANOI_DISKS + j]
                            if disk_zs[disk_j] < best_z:
                                best_i = j
                                best_z = disk_zs[disk_j]
                    if best_i != i:
                        tmp = stacks[peg * MAX_HANOI_DISKS + i]
                        stacks[peg * MAX_HANOI_DISKS + i] = stacks[peg * MAX_HANOI_DISKS + best_i]
                        stacks[peg * MAX_HANOI_DISKS + best_i] = tmp


@wp.func
def _tower_success(stacks: wp.array(dtype=int), stack_counts: wp.array(dtype=int), num_disks: int, goal_peg0: int, goal_peg1: int) -> int:
    solved = int(0)
    for goal_idx in range(2):
        peg = goal_peg0
        if goal_idx == 1:
            peg = goal_peg1
        ok = int(stack_counts[peg] == num_disks)
        for level in range(MAX_HANOI_DISKS):
            if level < num_disks:
                if stacks[peg * MAX_HANOI_DISKS + level] != num_disks - 1 - level:
                    ok = 0
        if ok != 0:
            solved = 1
    return solved


@wp.func
def _choose_subgoal(
    qpos: wp.array2d[float],
    disk_qpos_addrs: wp.array(dtype=int),
    peg_pos: wp.array2d[wp.vec3],
    peg_yaw: wp.array2d[float],
    rng_counter: wp.array(dtype=int),
    is_mistake: wp.array(dtype=int),
    target_peg: wp.array(dtype=int),
    target_disk: wp.array(dtype=int),
    target_source: wp.array(dtype=int),
    target_pos_arr: wp.array(dtype=wp.vec3),
    target_yaw_arr: wp.array(dtype=float),
    agent_done: wp.array(dtype=int),
    world_id: int,
    seeds: wp.array(dtype=int),
    num_disks: int,
    num_pegs: int,
    goal_peg0: int,
    goal_peg1: int,
    disk_half_height: float,
    disk_gap: float,
    p_mistake: float,
) -> int:
    stacks = wp.zeros(MAX_HANOI_STACK_SLOTS, dtype=int)
    stack_counts = wp.zeros(MAX_HANOI_PEGS, dtype=int)
    positions = wp.zeros(MAX_HANOI_DISKS, dtype=int)
    disk_zs = wp.zeros(MAX_HANOI_DISKS, dtype=float)
    _infer_stacks(qpos, disk_qpos_addrs, peg_pos, stacks, stack_counts, positions, disk_zs, world_id, num_disks, num_pegs)

    if _tower_success(stacks, stack_counts, num_disks, goal_peg0, goal_peg1) != 0:
        agent_done[world_id] = 1
        return 0

    planned = _first_move(num_disks, target_peg[world_id], positions)
    if planned < 0:
        agent_done[world_id] = 1
        return 0

    is_mistake[world_id] = 0
    move = planned
    if _randf(world_id, seeds, rng_counter) < p_mistake:
        candidates = wp.zeros(6, dtype=int)
        candidate_count = int(0)
        for source in range(MAX_HANOI_PEGS):
            if source < num_pegs and stack_counts[source] > 0:
                disk = stacks[source * MAX_HANOI_DISKS + stack_counts[source] - 1]
                for target in range(MAX_HANOI_PEGS):
                    if target < num_pegs and target != source:
                        code = _encode_move(disk, source, target)
                        if code != planned:
                            candidates[candidate_count] = code
                            candidate_count += 1
        if candidate_count > 0:
            is_mistake[world_id] = 1
            move = candidates[_rand_int(world_id, seeds, rng_counter, candidate_count)]

    disk = _move_disk(move)
    source = _move_source(move)
    target = _move_target(move)
    target_level = stack_counts[target]
    target_xy = peg_pos[world_id, target]

    target_disk[world_id] = disk
    target_source[world_id] = source
    target_pos_arr[world_id] = wp.vec3(
        target_xy[0],
        target_xy[1],
        disk_half_height + float(target_level) * (2.0 * disk_half_height + disk_gap),
    )
    target_yaw_arr[world_id] = _disk_yaw_for_peg(peg_yaw[world_id, target])
    agent_done[world_id] = 0
    return 1


@wp.func
def _rotated_grasp_offset(disk_yaw: float, sign: float, grasp_y: float) -> wp.vec3:
    c = wp.cos(disk_yaw)
    s = wp.sin(disk_yaw)
    local_y = sign * grasp_y
    return wp.vec3(-s * local_y, c * local_y, 0.0)


@wp.func
def _generate_hanoi_plan(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    disk_qpos_addrs: wp.array(dtype=int),
    rng_counter: wp.array(dtype=int),
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
    vertical_start: wp.array2d[float],
    vertical_end: wp.array2d[float],
    vertical_xy: wp.array2d[wp.vec3],
    vertical_yaw: wp.array2d[float],
    vertical_c1: wp.array2d[wp.vec3],
    vertical_c2: wp.array2d[wp.vec3],
    plan_time: wp.array(dtype=wp.float64),
    last_time: wp.array(dtype=float),
    agent_done: wp.array(dtype=int),
    active_disk: wp.array(dtype=int),
    target_disk: wp.array(dtype=int),
    target_pos_arr: wp.array(dtype=wp.vec3),
    target_yaw_arr: wp.array(dtype=float),
    world_id: int,
    seeds: wp.array(dtype=int),
    pinch_site_id: int,
    segment_dt: float,
    arm_lo: wp.vec3,
    arm_hi: wp.vec3,
    peg_half_height: float,
    disk_half_height: float,
    disk_hole_half_size: float,
    disk_min_long_half_size: float,
    disk_long_size_step: float,
    tilt_randomization: int,
):
    disk = target_disk[world_id]
    active_disk[world_id] = disk
    agent_done[world_id] = 0
    plan_time[world_id] = wp.float64(0.0)
    last_time[world_id] = -1.0
    for gate in range(6):
        gate_passed[world_id, gate] = 0

    eff_pos = site_xpos[world_id, pinch_site_id]
    eff_yaw = wp.atan2(site_xmat[world_id, pinch_site_id][1, 0], site_xmat[world_id, pinch_site_id][0, 0])
    eff_initial_quat = _quat_from_rpy(0.0, 0.0, eff_yaw)

    disk_center = _disk_pos(qpos, disk_qpos_addrs, world_id, disk)
    addr = disk_qpos_addrs[disk]
    disk_yaw = _quat_yaw(
        qpos[world_id, addr + 3],
        qpos[world_id, addr + 4],
        qpos[world_id, addr + 5],
        qpos[world_id, addr + 6],
    )
    grasp_sign = float(1.0)
    if _randf(world_id, seeds, rng_counter) < 0.5:
        grasp_sign = -1.0
    grasp_fraction = _rand_uniform(world_id, seeds, rng_counter, 0.7, 0.85)
    grasp_y = disk_hole_half_size + grasp_fraction * (
        disk_min_long_half_size + disk_long_size_step * float(disk) - disk_hole_half_size
    )

    disk_initial_pos = disk_center + _rotated_grasp_offset(disk_yaw, grasp_sign, grasp_y)
    disk_initial_pos = wp.vec3(disk_initial_pos[0], disk_initial_pos[1], disk_initial_pos[2] + disk_half_height * 0.8)
    disk_initial_yaw = _shortest_yaw_n2(eff_yaw, disk_yaw + 1.5707963267948966)
    disk_initial_yaw = disk_initial_yaw + _rand_uniform(world_id, seeds, rng_counter, -0.02, 0.02)

    target_center = target_pos_arr[world_id]
    target_yaw = target_yaw_arr[world_id]
    disk_goal_pos = target_center + _rotated_grasp_offset(target_yaw, grasp_sign, grasp_y)
    disk_goal_pos = wp.vec3(disk_goal_pos[0], disk_goal_pos[1], disk_goal_pos[2] + disk_half_height * 0.8)
    disk_goal_yaw = _shortest_yaw_n2(disk_initial_yaw, target_yaw + 1.5707963267948966)
    eff_goal_pos = wp.vec3(
        _rand_uniform(world_id, seeds, rng_counter, arm_lo[0], arm_hi[0]),
        _rand_uniform(world_id, seeds, rng_counter, arm_lo[1], arm_hi[1]),
        _rand_uniform(world_id, seeds, rng_counter, arm_lo[2], arm_hi[2]),
    )
    eff_goal_quat = _quat_from_rpy(0.0, 0.0, _rand_uniform(world_id, seeds, rng_counter, -3.141592653589793, 3.141592653589793))

    clearance_z = wp.max(
        2.0 * peg_half_height + 0.12,
        wp.max(disk_initial_pos[2] + 0.08, disk_goal_pos[2] + 0.08),
    )
    extract_z = wp.min(
        clearance_z,
        wp.max(
            2.0 * peg_half_height + disk_half_height * 1.8 + 0.025,
            wp.max(disk_initial_pos[2] + 0.06, disk_goal_pos[2] + 0.06),
        ),
    )

    lift_pos = wp.vec3(eff_pos[0], eff_pos[1], clearance_z)
    pick_pos = wp.vec3(disk_initial_pos[0], disk_initial_pos[1], clearance_z)
    extract_pos = wp.vec3(disk_initial_pos[0], disk_initial_pos[1], extract_z)
    postpick_pos = pick_pos
    carry_end = wp.vec3(disk_goal_pos[0], disk_goal_pos[1], clearance_z)
    place_pos = carry_end
    insert_pos = wp.vec3(disk_goal_pos[0], disk_goal_pos[1], extract_z)
    release_z = disk_goal_pos[2]
    peg_top_z = 2.0 * peg_half_height
    if release_z < peg_top_z:
        release_z = _rand_uniform(world_id, seeds, rng_counter, release_z, peg_top_z)
    place_start_pos = wp.vec3(disk_goal_pos[0], disk_goal_pos[1], release_z)
    postplace_pos = insert_pos

    grasp_quat = _quat_from_rpy(0.0, 0.0, disk_initial_yaw)
    target_quat = _quat_from_rpy(0.0, 0.0, disk_goal_yaw)
    lift_quat = _quat_from_rpy(0.0, 0.0, eff_yaw)

    num_midpoints = 1 + _rand_int(world_id, seeds, rng_counter, 2)
    carry_delta = carry_end - postpick_pos
    xy_norm = _norm2(carry_delta[0], carry_delta[1])
    side_dir = wp.vec3(-wp.sin(disk_initial_yaw), wp.cos(disk_initial_yaw), 0.0)
    if xy_norm > 1.0e-6:
        side_dir = wp.vec3(-carry_delta[1] / xy_norm, carry_delta[0] / xy_norm, 0.0)
    yaw_delta = _yaw_error(disk_goal_yaw, disk_initial_yaw)
    alpha0 = 0.5 + _rand_uniform(world_id, seeds, rng_counter, -0.08, 0.08)
    alpha1 = 0.0
    if num_midpoints == 2:
        alpha0 = 0.3333333333333333 + _rand_uniform(world_id, seeds, rng_counter, -0.08, 0.08)
        alpha1 = 0.6666666666666666 + _rand_uniform(world_id, seeds, rng_counter, -0.08, 0.08)
        alpha0 = wp.clamp(alpha0, 0.2, 0.8)
        alpha1 = wp.clamp(alpha1, 0.2, 0.8)
        if alpha1 < alpha0:
            tmp = alpha0
            alpha0 = alpha1
            alpha1 = tmp
    else:
        alpha0 = wp.clamp(alpha0, 0.2, 0.8)

    idx = int(0)
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, 0.0, eff_pos, eff_initial_quat, 0.0, 1)
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, segment_dt * 0.8, lift_pos, lift_quat, 0.0, 0)
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt, pick_pos, grasp_quat, 0.0, 0)
    pick_idx = idx
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt * 1.4, disk_initial_pos, grasp_quat, 0.0, 1)
    pick_start_idx = idx
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt * 0.9, disk_initial_pos, grasp_quat, 1.0, 1)
    pick_end_idx = idx
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt, extract_pos, grasp_quat, 1.0, 1)
    extract_idx = idx
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt * 0.6, postpick_pos, grasp_quat, 1.0, 0)
    idx += 1

    for midpoint in range(2):
        if midpoint < num_midpoints:
            alpha = alpha0
            if midpoint == 1:
                alpha = alpha1
            midpoint_xy = postpick_pos + alpha * carry_delta
            midpoint_xy = midpoint_xy + side_dir * _rand_uniform(world_id, seeds, rng_counter, -0.05, 0.05)
            midpoint_pos = wp.vec3(midpoint_xy[0], midpoint_xy[1], clearance_z + _rand_uniform(world_id, seeds, rng_counter, 0.0, 0.04))
            midpoint_yaw = disk_initial_yaw + alpha * yaw_delta + _rand_uniform(world_id, seeds, rng_counter, -0.1, 0.1)
            midpoint_roll = 0.0
            midpoint_pitch = 0.0
            if tilt_randomization != 0:
                midpoint_roll = _rand_uniform(world_id, seeds, rng_counter, -0.03, 0.03)
                midpoint_pitch = _rand_uniform(world_id, seeds, rng_counter, -0.03, 0.03)
            midpoint_quat = _quat_from_rpy(midpoint_roll, midpoint_pitch, midpoint_yaw)
            _write_key(
                key_time,
                key_xyz,
                key_quat,
                key_grasp,
                key_stop,
                world_id,
                idx,
                key_time[world_id, 6] + segment_dt * 2.0 * float(midpoint + 1) / float(num_midpoints + 1),
                midpoint_pos,
                midpoint_quat,
                1.0,
                0,
            )
            idx += 1

    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, 6] + segment_dt * 2.0, place_pos, target_quat, 1.0, 0)
    place_idx = idx
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt * 0.8, insert_pos, target_quat, 1.0, 1)
    insert_idx = idx
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt * 1.2, place_start_pos, target_quat, 1.0, 1)
    place_start_idx = idx
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt * 0.9, place_start_pos, target_quat, 0.0, 1)
    place_end_idx = idx
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt * 0.8, postplace_pos, target_quat, 0.0, 0)
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt, eff_goal_pos, eff_goal_quat, 0.0, 1)
    idx += 1
    key_count[world_id] = idx

    for key in range(MAX_PLAN_KEYS):
        if key < idx and key != 0:
            key_time[world_id, key] = key_time[world_id, key] + _rand_uniform(world_id, seeds, rng_counter, -1.0, 1.0) * segment_dt * 0.05
        if key < idx:
            grasp = float(0.0)
            if key >= pick_end_idx and key < place_end_idx:
                grasp = 1.0
            key_grasp[world_id, key] = grasp

    _smooth_times(key_count, key_time, key_xyz, world_id, segment_dt)
    _compute_tangents(key_count, key_time, key_xyz, key_stop, key_tangent, world_id)

    gate_time[world_id, 0] = key_time[world_id, pick_idx]
    gate_xyz[world_id, 0] = key_xyz[world_id, pick_idx]
    gate_time[world_id, 1] = key_time[world_id, pick_start_idx]
    gate_xyz[world_id, 1] = key_xyz[world_id, pick_start_idx]
    gate_time[world_id, 2] = key_time[world_id, extract_idx]
    gate_xyz[world_id, 2] = key_xyz[world_id, extract_idx]
    gate_time[world_id, 3] = key_time[world_id, place_idx]
    gate_xyz[world_id, 3] = key_xyz[world_id, place_idx]
    gate_time[world_id, 4] = key_time[world_id, insert_idx]
    gate_xyz[world_id, 4] = key_xyz[world_id, insert_idx]
    gate_time[world_id, 5] = key_time[world_id, place_start_idx]
    gate_xyz[world_id, 5] = key_xyz[world_id, place_start_idx]

    vertical_start[world_id, 0] = key_time[world_id, pick_end_idx]
    vertical_end[world_id, 0] = key_time[world_id, extract_idx]
    vertical_xy[world_id, 0] = wp.vec3(extract_pos[0], extract_pos[1], 0.0)
    vertical_yaw[world_id, 0] = disk_initial_yaw
    vertical_start[world_id, 1] = key_time[world_id, insert_idx]
    vertical_end[world_id, 1] = key_time[world_id, place_start_idx]
    vertical_xy[world_id, 1] = wp.vec3(insert_pos[0], insert_pos[1], 0.0)
    vertical_yaw[world_id, 1] = disk_goal_yaw
    for phase in range(2):
        vertical_c1[world_id, phase] = wp.vec3(
            _rand_uniform(world_id, seeds, rng_counter, -0.01, 0.01),
            _rand_uniform(world_id, seeds, rng_counter, -0.01, 0.01),
            0.0,
        )
        vertical_c2[world_id, phase] = wp.vec3(
            _rand_uniform(world_id, seeds, rng_counter, -0.01, 0.01),
            _rand_uniform(world_id, seeds, rng_counter, -0.01, 0.01),
            0.0,
        )


@wp.kernel
def reset_controller(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    disk_qpos_addrs: wp.array(dtype=int),
    peg_pos: wp.array2d[wp.vec3],
    peg_yaw: wp.array2d[float],
    rng_counter: wp.array(dtype=int),
    is_mistake: wp.array(dtype=int),
    done: wp.array(dtype=int),
    episode_length: wp.array(dtype=int),
    episode_success: wp.array(dtype=int),
    controller_done: wp.array(dtype=int),
    agent_done: wp.array(dtype=int),
    active_steps: wp.array(dtype=int),
    speed_dt: wp.array(dtype=float),
    target_peg: wp.array(dtype=int),
    active_disk: wp.array(dtype=int),
    target_disk: wp.array(dtype=int),
    target_source: wp.array(dtype=int),
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
    vertical_start: wp.array2d[float],
    vertical_end: wp.array2d[float],
    vertical_xy: wp.array2d[wp.vec3],
    vertical_yaw: wp.array2d[float],
    vertical_c1: wp.array2d[wp.vec3],
    vertical_c2: wp.array2d[wp.vec3],
    plan_time: wp.array(dtype=wp.float64),
    last_time: wp.array(dtype=float),
    seeds: wp.array(dtype=int),
    num_disks: int,
    num_pegs: int,
    goal_peg0: int,
    goal_peg1: int,
    disk_half_height: float,
    disk_gap: float,
    peg_half_height: float,
    disk_hole_half_size: float,
    disk_min_long_half_size: float,
    disk_long_size_step: float,
    p_mistake: float,
    pinch_site_id: int,
    arm_lo: wp.vec3,
    arm_hi: wp.vec3,
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
    rng_counter[world_id] = 0
    speed_dt[world_id] = _rand_uniform(world_id, seeds, rng_counter, 0.7, 1.5) * segment_dt_scale
    if _rand_int(world_id, seeds, rng_counter, 2) == 0:
        target_peg[world_id] = goal_peg0
    else:
        target_peg[world_id] = goal_peg1

    ok = _choose_subgoal(
        qpos,
        disk_qpos_addrs,
        peg_pos,
        peg_yaw,
        rng_counter,
        is_mistake,
        target_peg,
        target_disk,
        target_source,
        target_pos_arr,
        target_yaw_arr,
        agent_done,
        world_id,
        seeds,
        num_disks,
        num_pegs,
        goal_peg0,
        goal_peg1,
        disk_half_height,
        disk_gap,
        p_mistake,
    )
    if ok == 0:
        controller_done[world_id] = 1
        done[world_id] = 1
    else:
        _generate_hanoi_plan(
            qpos,
            site_xpos,
            site_xmat,
            disk_qpos_addrs,
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
            vertical_start,
            vertical_end,
            vertical_xy,
            vertical_yaw,
            vertical_c1,
            vertical_c2,
            plan_time,
            last_time,
            agent_done,
            active_disk,
            target_disk,
            target_pos_arr,
            target_yaw_arr,
            world_id,
            seeds,
            pinch_site_id,
            speed_dt[world_id],
            arm_lo,
            arm_hi,
            peg_half_height,
            disk_half_height,
            disk_hole_half_size,
            disk_min_long_half_size,
            disk_long_size_step,
            tilt_randomization,
        )


@wp.kernel
def make_targets(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    disk_qpos_addrs: wp.array(dtype=int),
    peg_pos: wp.array2d[wp.vec3],
    peg_yaw: wp.array2d[float],
    rng_counter: wp.array(dtype=int),
    is_mistake: wp.array(dtype=int),
    done: wp.array(dtype=int),
    controller_done: wp.array(dtype=int),
    agent_done: wp.array(dtype=int),
    active_steps: wp.array(dtype=int),
    speed_dt: wp.array(dtype=float),
    target_peg: wp.array(dtype=int),
    active_disk: wp.array(dtype=int),
    target_disk: wp.array(dtype=int),
    target_source: wp.array(dtype=int),
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
    vertical_start: wp.array2d[float],
    vertical_end: wp.array2d[float],
    vertical_xy: wp.array2d[wp.vec3],
    vertical_yaw: wp.array2d[float],
    vertical_c1: wp.array2d[wp.vec3],
    vertical_c2: wp.array2d[wp.vec3],
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
    num_disks: int,
    num_pegs: int,
    goal_peg0: int,
    goal_peg1: int,
    disk_half_height: float,
    disk_gap: float,
    peg_half_height: float,
    disk_hole_half_size: float,
    disk_min_long_half_size: float,
    disk_long_size_step: float,
    p_mistake: float,
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
        target_attach_pos[world_id] = current_pos + _mat_vec(current_xmat, t_pa_translation[0])
        target_attach_xmat[world_id] = _matmul(current_xmat, t_pa_rot_arr[0])
        target_gripper[world_id] = gripper_opening
        return

    if agent_done[world_id] != 0 or active_steps[world_id] >= max_subgoal_steps:
        ok = _choose_subgoal(
            qpos,
            disk_qpos_addrs,
            peg_pos,
            peg_yaw,
            rng_counter,
            is_mistake,
            target_peg,
            target_disk,
            target_source,
            target_pos_arr,
            target_yaw_arr,
            agent_done,
            world_id,
            seeds,
            num_disks,
            num_pegs,
            goal_peg0,
            goal_peg1,
            disk_half_height,
            disk_gap,
            p_mistake,
        )
        if ok == 0:
            controller_done[world_id] = 1
            target_attach_pos[world_id] = current_pos + _mat_vec(current_xmat, t_pa_translation[0])
            target_attach_xmat[world_id] = _matmul(current_xmat, t_pa_rot_arr[0])
            target_gripper[world_id] = gripper_opening
            return
        active_steps[world_id] = 0
        _generate_hanoi_plan(
            qpos,
            site_xpos,
            site_xmat,
            disk_qpos_addrs,
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
            vertical_start,
            vertical_end,
            vertical_xy,
            vertical_yaw,
            vertical_c1,
            vertical_c2,
            plan_time,
            last_time,
            agent_done,
            active_disk,
            target_disk,
            target_pos_arr,
            target_yaw_arr,
            world_id,
            seeds,
            pinch_site_id,
            speed_dt[world_id],
            arm_lo,
            arm_hi,
            peg_half_height,
            disk_half_height,
            disk_hole_half_size,
            disk_min_long_half_size,
            disk_long_size_step,
            tilt_randomization,
        )
    active_steps[world_id] += 1

    if last_time[world_id] >= 0.0:
        plan_time[world_id] += env_dt
    last_time[world_id] = 0.0

    blocked_gate = int(-1)
    for gate in range(6):
        if gate_passed[world_id, gate] == 0 and plan_time[world_id] >= wp.float64(gate_time[world_id, gate]):
            diff = current_pos - gate_xyz[world_id, gate]
            if wp.dot(diff, diff) > 0.012 * 0.012:
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

    vertical_active = int(0)
    for phase in range(2):
        if plan_time[world_id] >= wp.float64(vertical_start[world_id, phase]) and plan_time[world_id] <= wp.float64(vertical_end[world_id, phase]):
            vertical_active = 1
            u = wp.clamp(
                float(
                    (plan_time[world_id] - wp.float64(vertical_start[world_id, phase]))
                    / wp.float64(vertical_end[world_id, phase] - vertical_start[world_id, phase])
                ),
                0.0,
                1.0,
            )
            c1 = vertical_c1[world_id, phase]
            c2 = vertical_c2[world_id, phase]
            xy_offset = 3.0 * (1.0 - u) * (1.0 - u) * u * c1 + 3.0 * (1.0 - u) * u * u * c2
            xy = vertical_xy[world_id, phase]
            ab_xyz = wp.vec3(xy[0] + xy_offset[0], xy[1] + xy_offset[1], ab_xyz[2])
            break

    ee_delta = ab_xyz - current_pos
    if vertical_active != 0:
        max_xyz_delta = 0.018
        dz = wp.clamp(ee_delta[2], -0.65 * max_xyz_delta, 0.65 * max_xyz_delta)
        xy_norm = _norm2(ee_delta[0], ee_delta[1])
        xy_budget = wp.sqrt(wp.max(max_xyz_delta * max_xyz_delta - dz * dz, 0.0))
        if xy_norm > xy_budget and xy_norm > 1.0e-8:
            ee_delta = wp.vec3(ee_delta[0] / xy_norm * xy_budget, ee_delta[1] / xy_norm * xy_budget, dz)
        else:
            ee_delta = wp.vec3(ee_delta[0], ee_delta[1], dz)
    else:
        max_xyz_delta = 0.018
        delta_norm = _norm3(ee_delta)
        if delta_norm > max_xyz_delta and delta_norm > 1.0e-8:
            ee_delta = ee_delta * (max_xyz_delta / delta_norm)
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
