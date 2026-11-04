import warp as wp

from ocbench.mjwarp.primitives.primitive_kernels import (
    _clamp_vec3,
    _mat_transpose,
    _mat_vec,
    _matmul,
    _norm3,
    _quat_from_rpy,
    _quat_to_mat,
    _rand_uniform,
    _rotation_log,
    _rotvec_to_mat,
    _yaw_error,
)

PHASE_PREPUSH = 0
PHASE_LOWER = 1
PHASE_PRELOAD = 2
PHASE_PUSH = 3
PHASE_SETTLE = 4


@wp.func
def _rand_int_range(world_id: int, seeds: wp.array(dtype=int), rng_counter: wp.array(dtype=int), low: int, high_exclusive: int) -> int:
    value = int(wp.floor(_rand_uniform(world_id, seeds, rng_counter, float(low), float(high_exclusive))))
    if value < low:
        value = low
    if value >= high_exclusive:
        value = high_exclusive - 1
    return value


@wp.func
def _ready(current_pos: wp.vec3, current_yaw: float, gripper_opening: float, target_pos: wp.vec3, target_yaw: float, target_gripper: float) -> int:
    diff = current_pos - target_pos
    return int(_norm3(diff) <= 0.012 and wp.abs(_yaw_error(target_yaw, current_yaw)) <= 0.08 and wp.abs(target_gripper - gripper_opening) <= 0.08)


@wp.func
def _push_poses(ball_start_pos: wp.vec3, prepush_dist: float, side_offset: float, push_z: float, push_dir: wp.vec3, side_dir: wp.vec3) -> wp.vec3:
    xy = ball_start_pos - prepush_dist * push_dir + side_offset * side_dir
    return wp.vec3(xy[0], xy[1], push_z)


@wp.func
def _target_rotation(phase_value: int, phase_step_count: int, push_step_count: int, yaw: float, tilt: wp.vec3, tilt_drift: wp.vec3) -> wp.mat33:
    cur_tilt = tilt
    if phase_value == PHASE_PUSH:
        u = wp.clamp(float(phase_step_count + 1) / wp.max(float(push_step_count), 1.0), 0.0, 1.0)
        u = u * u * (3.0 - 2.0 * u)
        cur_tilt = tilt + u * tilt_drift
    return _quat_to_mat(_quat_from_rpy(cur_tilt[0], cur_tilt[1], yaw))


@wp.kernel
def reset_controller(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    ball_qpos_addr: int,
    pin_qpos_addrs: wp.array(dtype=int),
    rng_counter: wp.array(dtype=int),
    done: wp.array(dtype=int),
    episode_length: wp.array(dtype=int),
    episode_score: wp.array(dtype=int),
    controller_done: wp.array(dtype=int),
    active_steps: wp.array(dtype=int),
    speed_dt: wp.array(dtype=float),
    phase: wp.array(dtype=int),
    phase_steps: wp.array(dtype=int),
    ball_start_pos: wp.array(dtype=wp.vec3),
    push_speed: wp.array(dtype=float),
    push_steps: wp.array(dtype=int),
    push_ramp_steps: wp.array(dtype=int),
    prepush_dist: wp.array(dtype=float),
    push_z: wp.array(dtype=float),
    gripper: wp.array(dtype=float),
    side_offset: wp.array(dtype=float),
    push_dir: wp.array(dtype=wp.vec3),
    side_dir: wp.array(dtype=wp.vec3),
    yaw: wp.array(dtype=float),
    tilt: wp.array(dtype=wp.vec3),
    tilt_drift: wp.array(dtype=wp.vec3),
    seeds: wp.array(dtype=int),
    pinch_site_id: int,
    num_pins: int,
):
    world_id = wp.tid()
    done[world_id] = 0
    episode_length[world_id] = 0
    episode_score[world_id] = 0
    controller_done[world_id] = 0
    active_steps[world_id] = 0
    rng_counter[world_id] = 0
    speed_dt[world_id] = 1.0
    phase[world_id] = PHASE_PREPUSH
    phase_steps[world_id] = 0

    push_speed[world_id] = _rand_uniform(world_id, seeds, rng_counter, 0.03, 0.056)
    push_steps[world_id] = _rand_int_range(world_id, seeds, rng_counter, 130, 221)
    push_ramp_steps[world_id] = _rand_int_range(world_id, seeds, rng_counter, 30, 66)
    prepush_dist[world_id] = _rand_uniform(world_id, seeds, rng_counter, 0.10, 0.14)
    push_z[world_id] = _rand_uniform(world_id, seeds, rng_counter, 0.048, 0.064)
    gripper[world_id] = _rand_uniform(world_id, seeds, rng_counter, 0.06, 0.20)
    side_offset[world_id] = _rand_uniform(world_id, seeds, rng_counter, -0.025, 0.025)
    angle_noise = _rand_uniform(world_id, seeds, rng_counter, -0.10, 0.10)
    yaw_noise = _rand_uniform(world_id, seeds, rng_counter, -0.08, 0.08)
    target_x_offset = _rand_uniform(world_id, seeds, rng_counter, -0.18, 0.18)

    ball = wp.vec3(qpos[world_id, ball_qpos_addr + 0], qpos[world_id, ball_qpos_addr + 1], qpos[world_id, ball_qpos_addr + 2])
    ball_start_pos[world_id] = ball
    pin_head = wp.vec3(qpos[world_id, pin_qpos_addrs[0] + 0], qpos[world_id, pin_qpos_addrs[0] + 1], qpos[world_id, pin_qpos_addrs[0] + 2])
    if num_pins <= 0:
        pin_head = site_xpos[world_id, pinch_site_id]

    target_x = pin_head[0] + target_x_offset
    angle = wp.atan2(target_x - ball[0], pin_head[1] - ball[1]) + angle_noise
    angle = wp.clamp(angle, -0.45, 0.45)
    push_dir[world_id] = wp.vec3(wp.sin(angle), wp.cos(angle), 0.0)
    side_dir[world_id] = wp.vec3(wp.cos(angle), -wp.sin(angle), 0.0)
    yaw[world_id] = 1.5707963267948966 + angle + yaw_noise
    tilt[world_id] = wp.vec3(
        _rand_uniform(world_id, seeds, rng_counter, -0.08, 0.08),
        _rand_uniform(world_id, seeds, rng_counter, -0.08, 0.08),
        0.0,
    )
    tilt_drift[world_id] = wp.vec3(
        _rand_uniform(world_id, seeds, rng_counter, -0.04, 0.04),
        _rand_uniform(world_id, seeds, rng_counter, -0.04, 0.04),
        0.0,
    )


@wp.kernel
def make_targets(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    done: wp.array(dtype=int),
    controller_done: wp.array(dtype=int),
    active_steps: wp.array(dtype=int),
    phase: wp.array(dtype=int),
    phase_steps: wp.array(dtype=int),
    ball_start_pos: wp.array(dtype=wp.vec3),
    push_speed: wp.array(dtype=float),
    push_steps: wp.array(dtype=int),
    push_ramp_steps: wp.array(dtype=int),
    prepush_dist: wp.array(dtype=float),
    push_z: wp.array(dtype=float),
    gripper: wp.array(dtype=float),
    side_offset: wp.array(dtype=float),
    push_dir: wp.array(dtype=wp.vec3),
    side_dir: wp.array(dtype=wp.vec3),
    yaw: wp.array(dtype=float),
    tilt: wp.array(dtype=wp.vec3),
    tilt_drift: wp.array(dtype=wp.vec3),
    target_attach_pos: wp.array(dtype=wp.vec3),
    target_attach_xmat: wp.array(dtype=wp.mat33),
    target_gripper: wp.array(dtype=float),
    down_xmat_arr: wp.array(dtype=wp.mat33),
    down_xmat_inv_arr: wp.array(dtype=wp.mat33),
    t_pa_rot_arr: wp.array(dtype=wp.mat33),
    t_pa_translation: wp.array(dtype=wp.vec3),
    workspace_lo: wp.vec3,
    workspace_hi: wp.vec3,
    ee_high: wp.array(dtype=float),
    pinch_site_id: int,
    gripper_qpos_id: int,
    max_subgoal_steps: int,
):
    world_id = wp.tid()
    current_xmat = site_xmat[world_id, pinch_site_id]
    current_pos = site_xpos[world_id, pinch_site_id]
    current_yaw = wp.atan2(current_xmat[1, 0], current_xmat[0, 0])
    gripper_opening = wp.clamp(qpos[world_id, gripper_qpos_id] / 0.8, 0.0, 1.0)
    if done[world_id] != 0 or controller_done[world_id] != 0:
        target_attach_pos[world_id] = current_pos + _mat_vec(current_xmat, t_pa_translation[0])
        target_attach_xmat[world_id] = _matmul(current_xmat, t_pa_rot_arr[0])
        target_gripper[world_id] = gripper_opening
        return

    if active_steps[world_id] >= max_subgoal_steps:
        controller_done[world_id] = 1
        target_attach_pos[world_id] = current_pos + _mat_vec(current_xmat, t_pa_translation[0])
        target_attach_xmat[world_id] = _matmul(current_xmat, t_pa_rot_arr[0])
        target_gripper[world_id] = gripper_opening
        return

    phase_value = phase[world_id]
    phase_step_count = phase_steps[world_id]
    prepush_xy = ball_start_pos[world_id] - prepush_dist[world_id] * push_dir[world_id] + side_offset[world_id] * side_dir[world_id]
    prepush_pos = wp.vec3(prepush_xy[0], prepush_xy[1], 0.18)
    push_pos = _push_poses(ball_start_pos[world_id], prepush_dist[world_id], side_offset[world_id], push_z[world_id], push_dir[world_id], side_dir[world_id])

    raw_delta = wp.vec3(0.0, 0.0, 0.0)
    max_xyz_delta = 0.018
    if phase_value == PHASE_PREPUSH:
        if _ready(current_pos, current_yaw, gripper_opening, prepush_pos, yaw[world_id], gripper[world_id]) != 0:
            phase[world_id] = PHASE_LOWER
            phase_steps[world_id] = 0
            phase_value = PHASE_LOWER
            phase_step_count = 0
        target_pos = wp.vec3(prepush_pos[0], wp.min(prepush_pos[1], -0.015), prepush_pos[2])
        raw_delta = target_pos - current_pos
    elif phase_value == PHASE_LOWER:
        if _ready(current_pos, current_yaw, gripper_opening, push_pos, yaw[world_id], gripper[world_id]) != 0:
            phase[world_id] = PHASE_PRELOAD
            phase_steps[world_id] = 0
            phase_value = PHASE_PRELOAD
            phase_step_count = 0
        target_pos = wp.vec3(push_pos[0], wp.min(push_pos[1], -0.015), push_pos[2])
        raw_delta = target_pos - current_pos
        max_xyz_delta = 0.012
    elif phase_value == PHASE_PRELOAD:
        if phase_step_count >= 25:
            phase[world_id] = PHASE_PUSH
            phase_steps[world_id] = 0
            phase_value = PHASE_PUSH
            phase_step_count = 0
        raw_delta = wp.vec3(0.014 * push_dir[world_id][0], 0.014 * push_dir[world_id][1], 0.0)
        raw_delta = wp.vec3(raw_delta[0], wp.min(raw_delta[1], -0.015 - current_pos[1]), 0.0)
        max_xyz_delta = 0.045
    elif phase_value == PHASE_PUSH:
        if phase_step_count >= push_steps[world_id] or current_pos[1] >= -0.015 - 1.0e-3:
            phase[world_id] = PHASE_SETTLE
            phase_steps[world_id] = 0
            phase_value = PHASE_SETTLE
            phase_step_count = 0
        push_scale = wp.min(1.0, float(phase_step_count + 1) / float(push_ramp_steps[world_id]))
        speed = push_speed[world_id] * push_scale
        raw_delta = wp.vec3(speed * push_dir[world_id][0], speed * push_dir[world_id][1], 0.0)
        raw_delta = wp.vec3(raw_delta[0], wp.min(raw_delta[1], -0.015 - current_pos[1]), 0.0)
        max_xyz_delta = 0.045
    elif phase_value == PHASE_SETTLE:
        if phase_step_count >= 220:
            controller_done[world_id] = 1
        raw_delta = wp.vec3(0.0, 0.0, 0.0)
        max_xyz_delta = 0.01

    delta_norm = _norm3(raw_delta)
    if delta_norm > max_xyz_delta and delta_norm > 1.0e-8:
        raw_delta = raw_delta * (max_xyz_delta / delta_norm)
    target_eff_pos = _clamp_vec3(current_pos + raw_delta, workspace_lo, workspace_hi)
    target_eff_rot = _target_rotation(phase_value, phase_step_count, push_steps[world_id], yaw[world_id], tilt[world_id], tilt_drift[world_id])
    current_eff_rot = _matmul(current_xmat, down_xmat_inv_arr[0])
    rel = _matmul(_mat_transpose(current_eff_rot), target_eff_rot)
    rotvec = _rotation_log(rel)
    rot_norm = _norm3(rotvec)
    max_rot = ee_high[3]
    if rot_norm > max_rot:
        rotvec = rotvec * (max_rot / rot_norm)
    limited_eff_rot = _matmul(current_eff_rot, _rotvec_to_mat(rotvec))
    target_eff_xmat = _matmul(limited_eff_rot, down_xmat_arr[0])

    target_attach_pos[world_id] = target_eff_pos + _mat_vec(target_eff_xmat, t_pa_translation[0])
    target_attach_xmat[world_id] = _matmul(target_eff_xmat, t_pa_rot_arr[0])
    target_gripper[world_id] = wp.clamp(gripper[world_id], 0.0, 1.0)

    active_steps[world_id] += 1
    phase_steps[world_id] += 1


@wp.kernel
def update_done(
    healthy: wp.array(dtype=int),
    num_knocked_pins: wp.array(dtype=int),
    done: wp.array(dtype=int),
    episode_length: wp.array(dtype=int),
    episode_score: wp.array(dtype=int),
    max_steps: int,
):
    world_id = wp.tid()
    if done[world_id] != 0:
        return
    episode_length[world_id] += 1
    episode_score[world_id] = num_knocked_pins[world_id]
    if healthy[world_id] == 0 or episode_length[world_id] >= max_steps:
        done[world_id] = 1
