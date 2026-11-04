import warp as wp

from ocbench.mjwarp import MAX_PLAN_KEYS
from ocbench.mjwarp.primitives.primitive_kernels import (
    _above,
    _clamp_vec3,
    _compute_tangents,
    _norm2,
    _quat_apply,
    _quat_from_rpy,
    _quat_yaw,
    _rand_sign,
    _rand_uniform,
    _randf,
    _smooth_times,
    _write_key,
    _yaw_error,
)

MAX_CUBES = 4
MAX_PHASES = 4


@wp.func
def _block_pos(qpos: wp.array2d[float], object_qpos_addrs: wp.array(dtype=int), world_id: int, block: int) -> wp.vec3:
    addr = object_qpos_addrs[block]
    return wp.vec3(qpos[world_id, addr + 0], qpos[world_id, addr + 1], qpos[world_id, addr + 2])


@wp.func
def _block_yaw(qpos: wp.array2d[float], object_qpos_addrs: wp.array(dtype=int), world_id: int, block: int) -> float:
    addr = object_qpos_addrs[block]
    q = wp.quat(qpos[world_id, addr + 3], qpos[world_id, addr + 4], qpos[world_id, addr + 5], qpos[world_id, addr + 6])
    return _quat_yaw(q)


@wp.func
def _target_pos(
    mocap_pos: wp.array2d[wp.vec3],
    stack_goal_xyz: wp.array2d[wp.vec3],
    cube_target_mocap_ids: wp.array(dtype=int),
    world_id: int,
    block: int,
    stack_anywhere: int,
) -> wp.vec3:
    if stack_anywhere != 0:
        return stack_goal_xyz[world_id, block]
    return mocap_pos[world_id, cube_target_mocap_ids[block]]


@wp.func
def _target_yaw(
    mocap_quat: wp.array2d[wp.quat],
    cube_target_mocap_ids: wp.array(dtype=int),
    world_id: int,
    block: int,
    stack_anywhere: int,
) -> float:
    if stack_anywhere != 0:
        return 0.0
    return _quat_yaw(mocap_quat[world_id, cube_target_mocap_ids[block]])


@wp.func
def _target_success(
    qpos: wp.array2d[float],
    mocap_pos: wp.array2d[wp.vec3],
    stack_goal_xyz: wp.array2d[wp.vec3],
    object_qpos_addrs: wp.array(dtype=int),
    cube_target_mocap_ids: wp.array(dtype=int),
    world_id: int,
    block: int,
    stack_anywhere: int,
) -> int:
    pos = _block_pos(qpos, object_qpos_addrs, world_id, block)
    target = _target_pos(mocap_pos, stack_goal_xyz, cube_target_mocap_ids, world_id, block, stack_anywhere)
    diff = pos - target
    if wp.dot(diff, diff) <= 0.04 * 0.04:
        return 1
    return 0


@wp.func
def _stack_anywhere_success(
    qpos: wp.array2d[float],
    object_qpos_addrs: wp.array(dtype=int),
    world_id: int,
    num_cubes: int,
    cube_size: float,
) -> int:
    xs = wp.zeros(MAX_CUBES, dtype=float)
    ys = wp.zeros(MAX_CUBES, dtype=float)
    zs = wp.zeros(MAX_CUBES, dtype=float)
    order = wp.zeros(MAX_CUBES, dtype=int)
    used = wp.zeros(MAX_CUBES, dtype=int)
    for cube in range(MAX_CUBES):
        if cube < num_cubes:
            pos = _block_pos(qpos, object_qpos_addrs, world_id, cube)
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
    ok = int(1)
    base = order[0]
    for cube in range(MAX_CUBES):
        if cube < num_cubes:
            if _norm2(xs[cube] - xs[base], ys[cube] - ys[base]) > 0.04:
                ok = 0
    for rank in range(MAX_CUBES):
        if rank < num_cubes:
            cube = order[rank]
            if wp.abs(zs[cube] - cube_size * (2.0 * float(rank) + 1.0)) > 0.03:
                ok = 0
    return ok


@wp.func
def _is_top_block(
    qpos: wp.array2d[float],
    object_qpos_addrs: wp.array(dtype=int),
    world_id: int,
    num_cubes: int,
    cube_size: float,
    block: int,
) -> int:
    pos = _block_pos(qpos, object_qpos_addrs, world_id, block)
    top = int(1)
    for other in range(MAX_CUBES):
        if other < num_cubes and other != block:
            other_pos = _block_pos(qpos, object_qpos_addrs, world_id, other)
            if other_pos[2] > pos[2] and _norm2(pos[0] - other_pos[0], pos[1] - other_pos[1]) < cube_size:
                top = 0
    return top


@wp.func
def _sample_clear_pos(
    qpos: wp.array2d[float],
    mocap_pos: wp.array2d[wp.vec3],
    stack_goal_xyz: wp.array2d[wp.vec3],
    object_qpos_addrs: wp.array(dtype=int),
    cube_target_mocap_ids: wp.array(dtype=int),
    rng_counter: wp.array(dtype=int),
    world_id: int,
    seeds: wp.array(dtype=int),
    num_cubes: int,
    stack_anywhere: int,
    target_lo: wp.vec3,
    target_hi: wp.vec3,
    min_dist: float,
    cube_size: float,
) -> wp.vec3:
    xy = wp.vec3(target_lo[0], target_lo[1], cube_size)
    for attempt in range(100):
        candidate = wp.vec3(
            _rand_uniform(world_id, seeds, rng_counter, target_lo[0], target_hi[0]),
            _rand_uniform(world_id, seeds, rng_counter, target_lo[1], target_hi[1]),
            cube_size,
        )
        ok = int(1)
        for block in range(MAX_CUBES):
            if block < num_cubes:
                pos = _block_pos(qpos, object_qpos_addrs, world_id, block)
                if _norm2(candidate[0] - pos[0], candidate[1] - pos[1]) < min_dist:
                    ok = 0
                target = _target_pos(mocap_pos, stack_goal_xyz, cube_target_mocap_ids, world_id, block, stack_anywhere)
                if _norm2(candidate[0] - target[0], candidate[1] - target[1]) < min_dist:
                    ok = 0
        xy = candidate
        if ok != 0:
            break
    return xy


@wp.func
def _shortest_yaw(eff_yaw: float, obj_yaw: float) -> float:
    best_yaw = obj_yaw
    best_abs = wp.abs(eff_yaw - obj_yaw)
    for i in range(9):
        k = i - 4
        yaw = obj_yaw + float(k) * 1.5707963267948966
        err = wp.abs(eff_yaw - yaw)
        if err < best_abs:
            best_abs = err
            best_yaw = yaw
    return best_yaw


@wp.func
def _generate_cube_plan(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    object_qpos_addrs: wp.array(dtype=int),
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
    target_block: wp.array(dtype=int),
    target_pos_arr: wp.array(dtype=wp.vec3),
    target_yaw_arr: wp.array(dtype=float),
    world_id: int,
    seeds: wp.array(dtype=int),
    pinch_site_id: int,
    segment_dt: float,
    arm_lo: wp.vec3,
    arm_hi: wp.vec3,
    tilt_randomization: int,
):
    block = target_block[world_id]
    target_pos = target_pos_arr[world_id]
    target_yaw = target_yaw_arr[world_id]
    active_block[world_id] = block
    pick_source_z[world_id] = _block_pos(qpos, object_qpos_addrs, world_id, block)[2]
    pick_checked[world_id] = 0
    agent_done[world_id] = 0
    plan_time[world_id] = wp.float64(0.0)
    last_time[world_id] = -1.0
    gate_passed[world_id, 0] = 0
    gate_passed[world_id, 1] = 0

    eff_pos = site_xpos[world_id, pinch_site_id]
    eff_yaw = wp.atan2(site_xmat[world_id, pinch_site_id][1, 0], site_xmat[world_id, pinch_site_id][0, 0])
    block_pos = _block_pos(qpos, object_qpos_addrs, world_id, block)
    block_yaw = _block_yaw(qpos, object_qpos_addrs, world_id, block)
    eff_goal_pos = wp.vec3(
        _rand_uniform(world_id, seeds, rng_counter, arm_lo[0], arm_hi[0]),
        _rand_uniform(world_id, seeds, rng_counter, arm_lo[1], arm_hi[1]),
        _rand_uniform(world_id, seeds, rng_counter, arm_lo[2], arm_hi[2]),
    )
    eff_goal_quat = _quat_from_rpy(0.0, 0.0, _rand_uniform(world_id, seeds, rng_counter, -3.141592653589793, 3.141592653589793))

    noise_scale = wp.pow(0.75, float(num_pick_retries[world_id]))
    use_pick_hold = _randf(world_id, seeds, rng_counter) < 0.4 * noise_scale

    base_yaw = _shortest_yaw(eff_yaw, block_yaw)
    yaw_mode_rand = _randf(world_id, seeds, rng_counter)
    yaw_mode = 0.0
    if yaw_mode_rand < 0.15:
        yaw_mode = -1.5707963267948966
    elif yaw_mode_rand > 0.85:
        yaw_mode = 1.5707963267948966
    yaw_offset = _yaw_error(base_yaw, block_yaw) + yaw_mode + _rand_uniform(world_id, seeds, rng_counter, -0.04, 0.04) * noise_scale

    pick_tilt = wp.vec3(0.0, 0.0, 0.0)
    place_tilt = wp.vec3(0.0, 0.0, 0.0)
    if tilt_randomization != 0:
        pick_tilt_angle = _rand_uniform(world_id, seeds, rng_counter, -3.141592653589793, 3.141592653589793)
        pick_tilt_mag = _rand_uniform(world_id, seeds, rng_counter, 0.0, 25.0 * 3.141592653589793 / 180.0) * noise_scale
        pick_tilt = wp.vec3(wp.cos(pick_tilt_angle) * pick_tilt_mag, wp.sin(pick_tilt_angle) * pick_tilt_mag, 0.0)
        place_tilt = wp.vec3(
            0.2 * pick_tilt[0] + _rand_uniform(world_id, seeds, rng_counter, -0.06, 0.06) * noise_scale,
            0.2 * pick_tilt[1] + _rand_uniform(world_id, seeds, rng_counter, -0.06, 0.06) * noise_scale,
            0.0,
        )

    block_initial_quat = _quat_from_rpy(pick_tilt[0], pick_tilt[1], block_yaw + yaw_offset)
    pick_pos = _above(block_pos, block_initial_quat, 0.1 + _rand_uniform(world_id, seeds, rng_counter, 0.0, 0.1))
    block_goal_quat = _quat_from_rpy(place_tilt[0], place_tilt[1], target_yaw + yaw_offset)
    place_pos = _above(target_pos, block_goal_quat, 0.1 + _rand_uniform(world_id, seeds, rng_counter, 0.0, 0.1))
    eff_initial_quat = _quat_from_rpy(0.0, 0.0, eff_yaw)

    pick_start_pos = block_pos
    place_start_pos = target_pos
    place_yaw = target_yaw + yaw_offset
    place_forward = wp.vec3(wp.cos(place_yaw), wp.sin(place_yaw), 0.0)
    side_sign = _rand_sign(world_id, seeds, rng_counter)
    place_dir = wp.vec3(-wp.sin(place_yaw) * side_sign, wp.cos(place_yaw) * side_sign, 0.0)
    descent_xy = _rand_uniform(world_id, seeds, rng_counter, 0.016, 0.06) * noise_scale
    place_offset = descent_xy * place_dir + _rand_uniform(world_id, seeds, rng_counter, -0.02, 0.02) * noise_scale * place_forward

    pick_axis = _quat_apply(block_initial_quat, wp.vec3(0.0, 0.0, 1.0))
    contact_angle = _rand_uniform(world_id, seeds, rng_counter, -3.141592653589793, 3.141592653589793)
    contact_dir = _quat_apply(block_initial_quat, wp.vec3(wp.cos(contact_angle), wp.sin(contact_angle), 0.0))
    contact_shift = _rand_uniform(world_id, seeds, rng_counter, 0.0, 0.012) * noise_scale
    start_lift = _rand_uniform(world_id, seeds, rng_counter, 0.0, 0.012) * noise_scale
    end_depth = _rand_uniform(world_id, seeds, rng_counter, -0.004, 0.006) * noise_scale
    pick_start_pos = block_pos + start_lift * pick_axis - 0.5 * contact_shift * contact_dir
    pick_end_pos = block_pos + end_depth * pick_axis + 0.5 * contact_shift * contact_dir
    pick_end_tilt = pick_tilt
    if tilt_randomization != 0:
        pick_end_tilt = wp.vec3(
            pick_tilt[0] + _rand_uniform(world_id, seeds, rng_counter, -0.035, 0.035) * noise_scale,
            pick_tilt[1] + _rand_uniform(world_id, seeds, rng_counter, -0.035, 0.035) * noise_scale,
            0.0,
        )
    pick_end_yaw = block_yaw + yaw_offset + _rand_uniform(world_id, seeds, rng_counter, -0.06, 0.06) * noise_scale
    pick_start_quat = block_initial_quat
    pick_end_quat = _quat_from_rpy(pick_end_tilt[0], pick_end_tilt[1], pick_end_yaw)

    pick_pos = pick_start_pos + _rand_uniform(world_id, seeds, rng_counter, 0.06, 0.22) * pick_axis
    pick_z = pick_pos[2]
    place_z = place_start_pos[2] + _rand_uniform(world_id, seeds, rng_counter, 0.06, 0.22)
    place_pos = wp.vec3(place_start_pos[0] + place_offset[0], place_start_pos[1] + place_offset[1], place_z)

    diff = place_pos - pick_pos
    xy_norm = _norm2(diff[0], diff[1])
    side = wp.vec3(0.0, 1.0, 0.0)
    if xy_norm > 1.0e-6:
        side = wp.vec3(-diff[1] / xy_norm, diff[0] / xy_norm, 0.0)
    else:
        yaw = _rand_uniform(world_id, seeds, rng_counter, -3.141592653589793, 3.141592653589793)
        side = wp.vec3(-wp.sin(yaw), wp.cos(yaw), 0.0)
    alpha = 0.5
    if _randf(world_id, seeds, rng_counter) < 0.35:
        alpha = _rand_uniform(world_id, seeds, rng_counter, 1.15, 1.35) * noise_scale + 1.0 * (1.0 - noise_scale)
    else:
        alpha = wp.clamp(0.5 + _rand_uniform(world_id, seeds, rng_counter, -0.08, 0.08) * noise_scale, 0.15, 0.85)
    carry_side = _rand_sign(world_id, seeds, rng_counter)
    side_offset = _rand_uniform(world_id, seeds, rng_counter, 0.04, 0.14) * noise_scale
    yaw_delta = _yaw_error(target_yaw + yaw_offset, block_yaw + yaw_offset)
    midpoint_xy = pick_pos + alpha * diff + carry_side * side_offset * side
    midpoint_xy = midpoint_xy + wp.vec3(
        _rand_uniform(world_id, seeds, rng_counter, -0.02, 0.02) * noise_scale,
        _rand_uniform(world_id, seeds, rng_counter, -0.02, 0.02) * noise_scale,
        0.0,
    )
    midpoint_z = wp.max(pick_z, place_z) + _rand_uniform(world_id, seeds, rng_counter, 0.03, 0.12)
    midpoint_pos = _clamp_vec3(wp.vec3(midpoint_xy[0], midpoint_xy[1], midpoint_z), arm_lo, arm_hi)
    midpoint_yaw = block_yaw + yaw_offset + alpha * yaw_delta + _rand_uniform(world_id, seeds, rng_counter, -0.4, 0.4) * noise_scale
    midpoint_tilt = wp.vec3((1.0 - alpha) * pick_tilt[0] + alpha * place_tilt[0], (1.0 - alpha) * pick_tilt[1] + alpha * place_tilt[1], 0.0)
    if tilt_randomization != 0:
        midpoint_tilt = wp.vec3(
            midpoint_tilt[0] + _rand_uniform(world_id, seeds, rng_counter, -0.05, 0.05) * noise_scale,
            midpoint_tilt[1] + _rand_uniform(world_id, seeds, rng_counter, -0.05, 0.05) * noise_scale,
            0.0,
        )
    midpoint_quat = _quat_from_rpy(midpoint_tilt[0], midpoint_tilt[1], midpoint_yaw)

    idx = int(0)
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, 0.0, eff_pos, eff_initial_quat, 0.0, 1)
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, segment_dt, pick_pos, block_initial_quat, 0.0, 0)
    pick_idx = idx
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt * 1.8, pick_start_pos, pick_start_quat, 0.0, 0)
    pick_start_idx = idx
    idx += 1
    _write_key(
        key_time,
        key_xyz,
        key_quat,
        key_grasp,
        key_stop,
        world_id,
        idx,
        key_time[world_id, idx - 1] + segment_dt * _rand_uniform(world_id, seeds, rng_counter, 0.75, 1.25),
        pick_end_pos,
        pick_end_quat,
        0.0,
        0,
    )
    pick_end_idx = idx
    idx += 1
    pick_hold_idx = -1
    if use_pick_hold:
        _write_key(
            key_time,
            key_xyz,
            key_quat,
            key_grasp,
            key_stop,
            world_id,
            idx,
            key_time[world_id, idx - 1] + segment_dt * _rand_uniform(world_id, seeds, rng_counter, 0.1, 0.22),
            pick_end_pos,
            pick_end_quat,
            0.0,
            1,
        )
        pick_hold_idx = idx
        idx += 1
    postpick_time = key_time[world_id, idx - 1] + segment_dt * _rand_uniform(world_id, seeds, rng_counter, 0.75, 1.05)
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, postpick_time, pick_pos, block_initial_quat, 1.0, 0)
    postpick_idx = idx
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt, midpoint_pos, midpoint_quat, 1.0, 0)
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt, place_pos, block_goal_quat, 1.0, 0)
    place_idx = idx
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt * 1.5, place_start_pos, block_goal_quat, 1.0, 1)
    place_start_idx = idx
    idx += 1
    _write_key(
        key_time,
        key_xyz,
        key_quat,
        key_grasp,
        key_stop,
        world_id,
        idx,
        key_time[world_id, idx - 1] + segment_dt * _rand_uniform(world_id, seeds, rng_counter, 0.75, 1.1),
        place_start_pos,
        block_goal_quat,
        0.0,
        1,
    )
    place_end_idx = idx
    idx += 1
    _write_key(
        key_time,
        key_xyz,
        key_quat,
        key_grasp,
        key_stop,
        world_id,
        idx,
        key_time[world_id, idx - 1] + segment_dt * _rand_uniform(world_id, seeds, rng_counter, 0.8, 1.15),
        place_pos,
        block_goal_quat,
        0.0,
        0,
    )
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt, eff_goal_pos, eff_goal_quat, 0.0, 1)
    idx += 1
    key_count[world_id] = idx

    for key in range(MAX_PLAN_KEYS):
        if key < idx and key != 0:
            key_time[world_id, key] = key_time[world_id, key] + _rand_uniform(world_id, seeds, rng_counter, -1.0, 1.0) * segment_dt * 0.03

    close_draw = _randf(world_id, seeds, rng_counter)
    close_idx = pick_end_idx
    if use_pick_hold:
        if close_draw >= 0.65:
            close_idx = pick_hold_idx
    else:
        if close_draw < 0.1:
            close_idx = pick_start_idx
    grasp = 0.0
    for key in range(MAX_PLAN_KEYS):
        if key < idx:
            if key == close_idx:
                grasp = 1.0
            elif key == place_end_idx:
                grasp = 0.0
            key_grasp[world_id, key] = grasp

    _smooth_times(key_count, key_time, key_xyz, world_id, segment_dt)
    _compute_tangents(key_count, key_time, key_xyz, key_stop, key_tangent, world_id)

    gate_time[world_id, 0] = key_time[world_id, pick_idx]
    gate_xyz[world_id, 0] = key_xyz[world_id, pick_idx]
    gate_time[world_id, 1] = key_time[world_id, place_idx]
    gate_xyz[world_id, 1] = key_xyz[world_id, place_idx]
    pick_check_time[world_id] = key_time[world_id, postpick_idx]

    retreat_idx = pick_end_idx
    if use_pick_hold:
        retreat_idx = pick_hold_idx
    phase_start[world_id, 0] = key_time[world_id, pick_idx]
    phase_end[world_id, 0] = key_time[world_id, pick_start_idx]
    phase_start[world_id, 1] = key_time[world_id, retreat_idx]
    phase_end[world_id, 1] = key_time[world_id, postpick_idx]
    phase_start[world_id, 2] = key_time[world_id, place_idx]
    phase_end[world_id, 2] = key_time[world_id, place_start_idx]
    phase_start[world_id, 3] = key_time[world_id, place_end_idx]
    phase_end[world_id, 3] = key_time[world_id, place_end_idx + 1]
    for phase in range(MAX_PHASES):
        phase_c1[world_id, phase] = wp.vec3(
            _rand_uniform(world_id, seeds, rng_counter, -0.025, 0.025) * noise_scale,
            _rand_uniform(world_id, seeds, rng_counter, -0.025, 0.025) * noise_scale,
            0.0,
        )
        phase_c2[world_id, phase] = wp.vec3(
            _rand_uniform(world_id, seeds, rng_counter, -0.025, 0.025) * noise_scale,
            _rand_uniform(world_id, seeds, rng_counter, -0.025, 0.025) * noise_scale,
            0.0,
        )
