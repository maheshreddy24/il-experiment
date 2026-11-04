import warp as wp

from ocbench.mjwarp.primitives.primitive_kernels import (
    _compute_tangents,
    _current_yaw,
    _norm3,
    _positive_yaw,
    _quat_from_rpy,
    _rand_int,
    _rand_uniform,
    _randf,
    _smooth_times,
    _write_key,
    _yaw_error,
)


@wp.func
def _generate_button_plan(
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
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
    button_site_id: int,
    segment_dt: float,
    arm_lo: wp.vec3,
    arm_hi: wp.vec3,
    gripper_always_closed: int,
    tilt_randomization: int,
):
    eff = site_xpos[world_id, pinch_site_id]
    button = site_xpos[world_id, button_site_id]
    final = wp.vec3(
        _rand_uniform(world_id, seeds, rng_counter, arm_lo[0], arm_hi[0]),
        _rand_uniform(world_id, seeds, rng_counter, arm_lo[1], arm_hi[1]),
        _rand_uniform(world_id, seeds, rng_counter, arm_lo[2], arm_hi[2]),
    )
    final_quat = _quat_from_rpy(0.0, 0.0, _rand_uniform(world_id, seeds, rng_counter, -3.141592653589793, 3.141592653589793))

    button_yaw = _positive_yaw(_current_yaw(site_xmat, world_id, button_site_id))
    yaw_sample = _randf(world_id, seeds, rng_counter)
    yaw_offset = 0.0
    if yaw_sample < 0.15:
        yaw_offset = -1.5707963267948966
    elif yaw_sample > 0.85:
        yaw_offset = 1.5707963267948966
    yaw_offset += _rand_uniform(world_id, seeds, rng_counter, -0.08, 0.08)
    press_yaw = button_yaw + yaw_offset

    press_tilt = wp.vec3(0.0, 0.0, 0.0)
    exit_tilt = wp.vec3(0.0, 0.0, 0.0)
    if tilt_randomization != 0:
        press_tilt = wp.vec3(
            _rand_uniform(world_id, seeds, rng_counter, -0.015, 0.015),
            _rand_uniform(world_id, seeds, rng_counter, -0.015, 0.015),
            0.0,
        )
        exit_tilt = wp.vec3(
            0.7 * press_tilt[0] + _rand_uniform(world_id, seeds, rng_counter, -0.01, 0.01),
            0.7 * press_tilt[1] + _rand_uniform(world_id, seeds, rng_counter, -0.01, 0.01),
            0.0,
        )
    press_xy_x = _rand_uniform(world_id, seeds, rng_counter, -0.0015, 0.0015)
    press_xy_y = _rand_uniform(world_id, seeds, rng_counter, -0.0015, 0.0015)
    press_height = _rand_uniform(world_id, seeds, rng_counter, -0.026, -0.024)
    start_height = _rand_uniform(world_id, seeds, rng_counter, 0.05, 0.08)
    press_start = button + wp.vec3(press_xy_x, press_xy_y, start_height)
    press = button + wp.vec3(press_xy_x, press_xy_y, press_height)
    press_end = button + wp.vec3(
        press_xy_x + _rand_uniform(world_id, seeds, rng_counter, -0.001, 0.001),
        press_xy_y + _rand_uniform(world_id, seeds, rng_counter, -0.001, 0.001),
        start_height,
    )

    num_midpoints = _rand_int(world_id, seeds, rng_counter, 2)
    midpoint = wp.vec3(0.0, 0.0, 0.0)
    midpoint_quat = _quat_from_rpy(0.0, 0.0, press_yaw)
    if num_midpoints != 0:
        alpha = _rand_uniform(world_id, seeds, rng_counter, 0.35, 0.65)
        midpoint = (1.0 - alpha) * eff + alpha * press_start
        midpoint = midpoint + wp.vec3(
            _rand_uniform(world_id, seeds, rng_counter, -0.015, 0.015),
            _rand_uniform(world_id, seeds, rng_counter, -0.015, 0.015),
            0.0,
        )
        midpoint_z = wp.max(midpoint[2], button[2] + _rand_uniform(world_id, seeds, rng_counter, 0.08, 0.12))
        midpoint = wp.vec3(midpoint[0], midpoint[1], midpoint_z)
        initial_yaw = _positive_yaw(_current_yaw(site_xmat, world_id, pinch_site_id))
        midpoint_yaw = initial_yaw + alpha * _yaw_error(press_yaw, initial_yaw)
        midpoint_tilt = wp.vec3(0.0, 0.0, 0.0)
        if tilt_randomization != 0:
            midpoint_tilt = wp.vec3(
                alpha * press_tilt[0] + _rand_uniform(world_id, seeds, rng_counter, -0.02, 0.02),
                alpha * press_tilt[1] + _rand_uniform(world_id, seeds, rng_counter, -0.02, 0.02),
                0.0,
            )
        midpoint_quat = _quat_from_rpy(midpoint_tilt[0], midpoint_tilt[1], midpoint_yaw)

    press_start_quat = _quat_from_rpy(press_tilt[0], press_tilt[1], press_yaw)
    press_quat = _quat_from_rpy(press_tilt[0], press_tilt[1], press_yaw + _rand_uniform(world_id, seeds, rng_counter, -0.01, 0.01))
    press_end_quat = _quat_from_rpy(exit_tilt[0], exit_tilt[1], press_yaw + _rand_uniform(world_id, seeds, rng_counter, -0.015, 0.015))
    initial_quat = _quat_from_rpy(0.0, 0.0, _positive_yaw(_current_yaw(site_xmat, world_id, pinch_site_id)))

    idx = int(0)
    open_grasp = 0.0
    if gripper_always_closed != 0:
        open_grasp = 1.0
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, 0.0, eff, initial_quat, open_grasp, 1)
    idx += 1
    if num_midpoints != 0:
        _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, segment_dt * 0.5, midpoint, midpoint_quat, open_grasp, 0)
        idx += 1

    distance = _norm3(eff - press_start)
    press_start_time = key_time[world_id, idx - 1] + segment_dt * (0.5 + distance * 4.0)
    _write_key(key_time, key_xyz, key_quat, key_grasp, key_stop, world_id, idx, press_start_time, press_start, press_start_quat, 1.0, 1)
    press_start_idx = idx
    idx += 1
    _write_key(
        key_time,
        key_xyz,
        key_quat,
        key_grasp,
        key_stop,
        world_id,
        idx,
        key_time[world_id, idx - 1] + segment_dt * _rand_uniform(world_id, seeds, rng_counter, 0.7, 0.9),
        press,
        press_quat,
        1.0,
        1,
    )
    idx += 1
    _write_key(
        key_time,
        key_xyz,
        key_quat,
        key_grasp,
        key_stop,
        world_id,
        idx,
        key_time[world_id, idx - 1] + segment_dt * _rand_uniform(world_id, seeds, rng_counter, 0.7, 0.9),
        press_end,
        press_end_quat,
        1.0,
        1,
    )
    idx += 1
    _write_key(
        key_time,
        key_xyz,
        key_quat,
        key_grasp,
        key_stop,
        world_id,
        idx,
        key_time[world_id, idx - 1] + segment_dt * _rand_uniform(world_id, seeds, rng_counter, 1.1, 1.4),
        final,
        final_quat,
        open_grasp,
        1,
    )
    idx += 1
    key_count[world_id] = idx

    for key in range(6):
        if key < idx and key != 0:
            key_time[world_id, key] = key_time[world_id, key] + _rand_uniform(world_id, seeds, rng_counter, -1.0, 1.0) * segment_dt * 0.1

    _smooth_times(key_count, key_time, key_xyz, world_id, segment_dt)
    _compute_tangents(key_count, key_time, key_xyz, key_stop, key_tangent, world_id)
    gate_time[world_id, 0] = key_time[world_id, press_start_idx]
    gate_xyz[world_id, 0] = key_xyz[world_id, press_start_idx]
    gate_passed[world_id, 0] = 0
    gate_passed[world_id, 1] = 1
    plan_time[world_id] = wp.float64(0.0)
    last_time[world_id] = -1.0
    agent_done[world_id] = 0
