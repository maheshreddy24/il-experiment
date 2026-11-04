import warp as wp

from ocbench.mjwarp import MAX_PLAN_KEYS
from ocbench.mjwarp.primitives.primitive_kernels import (
    _compute_tangents,
    _current_yaw,
    _norm2,
    _positive_yaw,
    _quat_from_rpy,
    _rand_int,
    _rand_sign,
    _rand_uniform,
    _randf,
    _shortest_yaw_n2,
    _smooth_times,
    _write_key,
    _yaw_error,
)


@wp.func
def _generate_drawer_plan(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    mirrored_arr: wp.array(dtype=int),
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
    plan_time: wp.array(dtype=wp.float64),
    last_time: wp.array(dtype=float),
    agent_done: wp.array(dtype=int),
    world_id: int,
    seeds: wp.array(dtype=int),
    pinch_site_id: int,
    drawer_site_id: int,
    drawer_qpos_addr: int,
    target_slide_pos: wp.array(dtype=float),
    segment_dt: float,
    arm_lo: wp.vec3,
    arm_hi: wp.vec3,
    tilt_randomization: int,
):
    eff = site_xpos[world_id, pinch_site_id]
    eff_yaw = _positive_yaw(_current_yaw(site_xmat, world_id, pinch_site_id))
    handle_pos = site_xpos[world_id, drawer_site_id]
    handle_yaw = _shortest_yaw_n2(eff_yaw, _positive_yaw(_current_yaw(site_xmat, world_id, drawer_site_id)))
    final = wp.vec3(
        _rand_uniform(world_id, seeds, rng_counter, arm_lo[0], arm_hi[0]),
        _rand_uniform(world_id, seeds, rng_counter, arm_lo[1], arm_hi[1]),
        _rand_uniform(world_id, seeds, rng_counter, arm_lo[2], arm_hi[2]),
    )
    final_quat = _quat_from_rpy(0.0, 0.0, _rand_uniform(world_id, seeds, rng_counter, -3.141592653589793, 3.141592653589793))

    current_slide = qpos[world_id, drawer_qpos_addr]
    drawer_sign = -1.0
    if mirrored_arr[world_id] != 0:
        drawer_sign = 1.0
    goal_pos = handle_pos + wp.vec3(0.0, drawer_sign * (target_slide_pos[world_id] - current_slide), 0.0)

    yaw_offset = 0.0
    if _randf(world_id, seeds, rng_counter) >= 0.85:
        yaw_offset = 3.141592653589793
    yaw_offset += _rand_uniform(world_id, seeds, rng_counter, -0.08, 0.08)
    grasp_roll = 0.0
    grasp_pitch = 0.0
    move_roll = 0.0
    move_pitch = 0.0
    if tilt_randomization != 0:
        grasp_roll = _rand_uniform(world_id, seeds, rng_counter, -0.06, 0.06)
        grasp_pitch = _rand_uniform(world_id, seeds, rng_counter, -0.06, 0.06)
        move_roll = 0.7 * grasp_roll + _rand_uniform(world_id, seeds, rng_counter, -0.04, 0.04)
        move_pitch = 0.7 * grasp_pitch + _rand_uniform(world_id, seeds, rng_counter, -0.04, 0.04)
    grasp_yaw = handle_yaw + yaw_offset
    move_yaw = handle_yaw + yaw_offset + _rand_uniform(world_id, seeds, rng_counter, -0.04, 0.04)

    move_delta = goal_pos - handle_pos
    move_dist = _norm2(move_delta[0], move_delta[1])
    slide_dir = wp.vec3(wp.cos(grasp_yaw), wp.sin(grasp_yaw), 0.0)
    if move_dist > 1.0e-6:
        slide_dir = wp.vec3(move_delta[0] / move_dist, move_delta[1] / move_dist, 0.0)
    side_dir = wp.vec3(-slide_dir[1], slide_dir[0], 0.0)

    side_sign = _rand_sign(world_id, seeds, rng_counter)
    approach_offset = (
        side_sign * _rand_uniform(world_id, seeds, rng_counter, 0.015, 0.04) * side_dir
        + _rand_uniform(world_id, seeds, rng_counter, -0.015, 0.015) * slide_dir
    )
    clearance_offset = (
        side_sign * _rand_uniform(world_id, seeds, rng_counter, 0.015, 0.04) * side_dir
        + _rand_uniform(world_id, seeds, rng_counter, -0.015, 0.015) * slide_dir
    )

    grasp_quat = _quat_from_rpy(grasp_roll, grasp_pitch, grasp_yaw)
    move_quat = _quat_from_rpy(move_roll, move_pitch, move_yaw)
    idx = int(0)
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, 0.0, eff, _quat_from_rpy(0.0, 0.0, eff_yaw), 0.0, 1)
    idx += 1
    approach = handle_pos + approach_offset + wp.vec3(0.0, 0.0, _rand_uniform(world_id, seeds, rng_counter, 0.09, 0.14))
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, segment_dt, approach, grasp_quat, 0.0, 0)
    approach_idx = idx
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt * 0.5, handle_pos, grasp_quat, 0.0, 1)
    grasp_start_idx = idx
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt * 0.5, handle_pos, grasp_quat, 1.0, 1)
    idx += 1

    num_midpoints = _rand_int(world_id, seeds, rng_counter, 2) + 1
    alpha0 = 0.0
    alpha1 = 0.0
    alpha2 = 0.0
    if num_midpoints == 1:
        alpha0 = wp.clamp(0.5 + _rand_uniform(world_id, seeds, rng_counter, -0.08, 0.08), 0.2, 0.8)
    else:
        alpha1 = wp.clamp(0.3333333333333333 + _rand_uniform(world_id, seeds, rng_counter, -0.08, 0.08), 0.2, 0.8)
        alpha2 = wp.clamp(0.6666666666666666 + _rand_uniform(world_id, seeds, rng_counter, -0.08, 0.08), 0.2, 0.8)
    for midpoint in range(2):
        if midpoint < num_midpoints:
            alpha = alpha0
            if num_midpoints == 2:
                alpha = alpha1
                if midpoint == 1:
                    alpha = alpha2
            pos = handle_pos + alpha * move_delta
            pos = pos + _rand_uniform(world_id, seeds, rng_counter, -0.005, 0.005) * side_dir
            pos = pos + wp.vec3(0.0, 0.0, _rand_uniform(world_id, seeds, rng_counter, -0.003, 0.006))
            yaw = grasp_yaw + alpha * _yaw_error(move_yaw, grasp_yaw) + _rand_uniform(world_id, seeds, rng_counter, -0.04, 0.04)
            roll = 0.0
            pitch = 0.0
            if tilt_randomization != 0:
                roll = (1.0 - alpha) * grasp_roll + alpha * move_roll + _rand_uniform(world_id, seeds, rng_counter, -0.02, 0.02)
                pitch = (1.0 - alpha) * grasp_pitch + alpha * move_pitch + _rand_uniform(world_id, seeds, rng_counter, -0.02, 0.02)
            _write_key(
                key_time,
                key_xyz,
                key_quat,
                key_grasp,
                key_stop,
                world_id,
                idx,
                key_time[world_id, idx - 1] + segment_dt * 0.5,
                pos,
                _quat_from_rpy(roll, pitch, yaw),
                1.0,
                0,
            )
            idx += 1

    clearance = goal_pos + clearance_offset + wp.vec3(0.0, 0.0, _rand_uniform(world_id, seeds, rng_counter, 0.09, 0.14))
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt * 0.5, goal_pos, move_quat, 1.0, 1)
    idx += 1
    _write_key(
        key_time,
        key_xyz,
        key_quat,
        key_grasp,
        key_stop,
        world_id,
        idx,
        key_time[world_id, idx - 1] + segment_dt * _rand_uniform(world_id, seeds, rng_counter, 0.3, 0.7),
        goal_pos,
        move_quat,
        0.0,
        1,
    )
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt * 0.5, clearance, move_quat, 0.0, 0)
    idx += 1
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, key_time[world_id, idx - 1] + segment_dt, final, final_quat, 0.0, 1)
    idx += 1
    key_count[world_id] = idx

    for key in range(MAX_PLAN_KEYS):
        if key < idx and key != 0:
            key_time[world_id, key] = key_time[world_id, key] + _rand_uniform(world_id, seeds, rng_counter, -1.0, 1.0) * segment_dt * 0.1

    _smooth_times(key_count, key_time, key_xyz, world_id, segment_dt)
    _compute_tangents(key_count, key_time, key_xyz, key_stop, key_tangent, world_id)
    gate_time[world_id, 0] = key_time[world_id, approach_idx]
    gate_xyz[world_id, 0] = key_xyz[world_id, approach_idx]
    gate_time[world_id, 1] = key_time[world_id, grasp_start_idx]
    gate_xyz[world_id, 1] = key_xyz[world_id, grasp_start_idx]
    gate_passed[world_id, 0] = 0
    gate_passed[world_id, 1] = 0
    plan_time[world_id] = wp.float64(0.0)
    last_time[world_id] = -1.0
    agent_done[world_id] = 0
