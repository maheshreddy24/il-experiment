import warp as wp

from ocbench.mjwarp import MAX_PLAN_KEYS


@wp.func
def _sample_plan_time(plan_time: wp.float64, final_time: float, env_dt: wp.float64):
    sample = wp.floor((plan_time + wp.float64(1.0e-7)) / env_dt)
    final_sample = wp.ceil(wp.float64(final_time) / env_dt) - wp.float64(1.0)
    return float(wp.min(sample, final_sample) * env_dt), int(sample >= final_sample)


@wp.func
def _wrap_pi(x: float) -> float:
    two_pi = 6.283185307179586
    return x - two_pi * wp.floor((x + 3.141592653589793) / two_pi)


@wp.func
def _yaw_error(target_yaw: float, yaw: float) -> float:
    return _wrap_pi(target_yaw - yaw)


@wp.func
def _positive_yaw(yaw: float) -> float:
    out = yaw
    if out < 0.0:
        out = out + 6.283185307179586
    return out


@wp.func
def _shortest_yaw_n2(eff_yaw: float, obj_yaw: float) -> float:
    best_yaw = obj_yaw
    best_abs = wp.abs(eff_yaw - obj_yaw)
    for i in range(5):
        k = i - 2
        yaw = obj_yaw + float(k) * 3.141592653589793
        err = wp.abs(eff_yaw - yaw)
        if err < best_abs:
            best_abs = err
            best_yaw = yaw
    return best_yaw


@wp.func
def _randf(world_id: int, seeds: wp.array(dtype=int), rng_counter: wp.array(dtype=int)) -> float:
    counter = rng_counter[world_id]
    rng_counter[world_id] = counter + 1
    return wp.randf(wp.rand_init(seeds[world_id] + counter * 9176))


@wp.func
def _rand_uniform(world_id: int, seeds: wp.array(dtype=int), rng_counter: wp.array(dtype=int), low: float, high: float) -> float:
    return low + (high - low) * _randf(world_id, seeds, rng_counter)


@wp.func
def _rand_sign(world_id: int, seeds: wp.array(dtype=int), rng_counter: wp.array(dtype=int)) -> float:
    if _randf(world_id, seeds, rng_counter) < 0.5:
        return -1.0
    return 1.0


@wp.func
def _rand_int(world_id: int, seeds: wp.array(dtype=int), rng_counter: wp.array(dtype=int), high: int) -> int:
    value = int(wp.floor(_randf(world_id, seeds, rng_counter) * float(high)))
    if value >= high:
        value = high - 1
    return value


@wp.func
def _norm3(v: wp.vec3) -> float:
    return wp.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


@wp.func
def _norm2(x: float, y: float) -> float:
    return wp.sqrt(x * x + y * y)


@wp.func
def _clamp_vec3(v: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> wp.vec3:
    return wp.vec3(
        wp.clamp(v[0], lo[0], hi[0]),
        wp.clamp(v[1], lo[1], hi[1]),
        wp.clamp(v[2], lo[2], hi[2]),
    )


@wp.func
def _quat_normalize(q: wp.quat) -> wp.quat:
    n = wp.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return wp.quat(q[0] / n, q[1] / n, q[2] / n, q[3] / n)


@wp.func
def _quat_mul(a: wp.quat, b: wp.quat) -> wp.quat:
    return wp.quat(
        -a[1] * b[1] - a[2] * b[2] - a[3] * b[3] + a[0] * b[0],
        a[1] * b[0] + a[2] * b[3] - a[3] * b[2] + a[0] * b[1],
        -a[1] * b[3] + a[2] * b[0] + a[3] * b[1] + a[0] * b[2],
        a[1] * b[2] - a[2] * b[1] + a[3] * b[0] + a[0] * b[3],
    )


@wp.func
def _quat_inv(q: wp.quat) -> wp.quat:
    return wp.quat(q[0], -q[1], -q[2], -q[3])


@wp.func
def _quat_from_axis_angle(x: float, y: float, z: float, theta: float) -> wp.quat:
    half = 0.5 * theta
    s = wp.sin(half)
    return wp.quat(wp.cos(half), x * s, y * s, z * s)


@wp.func
def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> wp.quat:
    qx = _quat_from_axis_angle(1.0, 0.0, 0.0, roll)
    qy = _quat_from_axis_angle(0.0, 1.0, 0.0, pitch)
    qz = _quat_from_axis_angle(0.0, 0.0, 1.0, yaw)
    return _quat_mul(qz, _quat_mul(qy, qx))


@wp.func
def _quat_yaw(q: wp.quat) -> float:
    return wp.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))


@wp.func
def _quat_apply(q: wp.quat, v: wp.vec3) -> wp.vec3:
    p = wp.quat(0.0, v[0], v[1], v[2])
    out = _quat_mul(_quat_mul(q, p), _quat_inv(q))
    return wp.vec3(out[1], out[2], out[3])


@wp.func
def _quat_to_mat(q: wp.quat) -> wp.mat33:
    qn = _quat_normalize(q)
    w = qn[0]
    x = qn[1]
    y = qn[2]
    z = qn[3]
    return wp.mat33(
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
    )


@wp.func
def _quat_from_mat(m: wp.mat33) -> wp.quat:
    xyzw = wp.quat_from_matrix(m)
    return wp.quat(xyzw[3], xyzw[0], xyzw[1], xyzw[2])


@wp.func
def _quat_slerp(q0: wp.quat, q1_in: wp.quat, u: float) -> wp.quat:
    q1 = q1_in
    dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]
    if dot < 0.0:
        dot = -dot
        q1 = wp.quat(-q1[0], -q1[1], -q1[2], -q1[3])
    if dot > 0.9995:
        return _quat_normalize(
            wp.quat(
                q0[0] + u * (q1[0] - q0[0]),
                q0[1] + u * (q1[1] - q0[1]),
                q0[2] + u * (q1[2] - q0[2]),
                q0[3] + u * (q1[3] - q0[3]),
            )
        )
    theta_0 = wp.acos(wp.clamp(dot, -1.0, 1.0))
    theta = theta_0 * u
    sin_theta = wp.sin(theta)
    sin_theta_0 = wp.sin(theta_0)
    s0 = wp.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _quat_normalize(
        wp.quat(
            s0 * q0[0] + s1 * q1[0],
            s0 * q0[1] + s1 * q1[1],
            s0 * q0[2] + s1 * q1[2],
            s0 * q0[3] + s1 * q1[3],
        )
    )


@wp.func
def _matmul(a: wp.mat33, b: wp.mat33) -> wp.mat33:
    return wp.mat33(
        a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] + a[0, 2] * b[2, 0],
        a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1] + a[0, 2] * b[2, 1],
        a[0, 0] * b[0, 2] + a[0, 1] * b[1, 2] + a[0, 2] * b[2, 2],
        a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0] + a[1, 2] * b[2, 0],
        a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1] + a[1, 2] * b[2, 1],
        a[1, 0] * b[0, 2] + a[1, 1] * b[1, 2] + a[1, 2] * b[2, 2],
        a[2, 0] * b[0, 0] + a[2, 1] * b[1, 0] + a[2, 2] * b[2, 0],
        a[2, 0] * b[0, 1] + a[2, 1] * b[1, 1] + a[2, 2] * b[2, 1],
        a[2, 0] * b[0, 2] + a[2, 1] * b[1, 2] + a[2, 2] * b[2, 2],
    )


@wp.func
def _mat_transpose(a: wp.mat33) -> wp.mat33:
    return wp.mat33(a[0, 0], a[1, 0], a[2, 0], a[0, 1], a[1, 1], a[2, 1], a[0, 2], a[1, 2], a[2, 2])


@wp.func
def _mat_vec(a: wp.mat33, v: wp.vec3) -> wp.vec3:
    return wp.vec3(
        a[0, 0] * v[0] + a[0, 1] * v[1] + a[0, 2] * v[2],
        a[1, 0] * v[0] + a[1, 1] * v[1] + a[1, 2] * v[2],
        a[2, 0] * v[0] + a[2, 1] * v[1] + a[2, 2] * v[2],
    )


@wp.func
def _rotvec_to_mat(v: wp.vec3) -> wp.mat33:
    angle = _norm3(v)
    if angle <= 1.0e-8:
        return wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    axis = v / angle
    x = axis[0]
    y = axis[1]
    z = axis[2]
    c = wp.cos(angle)
    s = wp.sin(angle)
    one_c = 1.0 - c
    return wp.mat33(
        c + x * x * one_c,
        x * y * one_c - z * s,
        x * z * one_c + y * s,
        y * x * one_c + z * s,
        c + y * y * one_c,
        y * z * one_c - x * s,
        z * x * one_c - y * s,
        z * y * one_c + x * s,
        c + z * z * one_c,
    )


@wp.func
def _rotation_log(err_xmat: wp.mat33) -> wp.vec3:
    # Warp quaternions use xyzw; choose the shortest rotation.
    quat = wp.quat_from_matrix(err_xmat)
    if quat[3] < 0.0:
        quat = -quat
    axis = wp.vec3(quat[0], quat[1], quat[2])
    sin_half_angle = wp.length(axis)
    if sin_half_angle < 1.0e-8:
        return 2.0 * axis
    return (2.0 * wp.atan2(sin_half_angle, quat[3]) / sin_half_angle) * axis


@wp.func
def _write_key(
    key_time: wp.array2d[float],
    key_xyz: wp.array2d[wp.vec3],
    key_quat: wp.array2d[wp.quat],
    key_grasp: wp.array2d[float],
    key_stop: wp.array2d[int],
    world_id: int,
    idx: int,
    t: float,
    xyz: wp.vec3,
    quat: wp.quat,
    grasp: float,
    stop: int,
):
    key_time[world_id, idx] = t
    key_xyz[world_id, idx] = xyz
    key_quat[world_id, idx] = quat
    key_grasp[world_id, idx] = grasp
    key_stop[world_id, idx] = stop


@wp.func
def _above(pos: wp.vec3, quat: wp.quat, z: float) -> wp.vec3:
    return pos + _quat_apply(quat, wp.vec3(0.0, 0.0, z))


@wp.func
def _compute_tangents(
    key_count: wp.array(dtype=int),
    key_time: wp.array2d[float],
    key_xyz: wp.array2d[wp.vec3],
    key_stop: wp.array2d[int],
    key_tangent: wp.array2d[wp.vec3],
    world_id: int,
):
    count = key_count[world_id]
    for i in range(MAX_PLAN_KEYS):
        tangent = wp.vec3(0.0, 0.0, 0.0)
        if i > 0 and i < count - 1 and key_stop[world_id, i] == 0:
            xyz_prev = key_xyz[world_id, i - 1]
            xyz_cur = key_xyz[world_id, i]
            xyz_next = key_xyz[world_id, i + 1]
            t_prev = key_time[world_id, i - 1]
            t_cur = key_time[world_id, i]
            t_next = key_time[world_id, i + 1]
            tangent = (xyz_next - xyz_prev) / (t_next - t_prev)
            prev_speed = _norm3((xyz_cur - xyz_prev) / (t_cur - t_prev))
            next_speed = _norm3((xyz_next - xyz_cur) / (t_next - t_cur))
            max_speed = 1.25 * wp.max(prev_speed, next_speed)
            speed = _norm3(tangent)
            if speed > max_speed:
                tangent = tangent * (max_speed / speed)
        key_tangent[world_id, i] = tangent


@wp.func
def _smooth_times(
    key_count: wp.array(dtype=int),
    key_time: wp.array2d[float],
    key_xyz: wp.array2d[wp.vec3],
    world_id: int,
    segment_dt: float,
):
    count = key_count[world_id]
    prev_old = float(key_time[world_id, 0])
    prev_new = float(0.0)
    key_time[world_id, 0] = 0.0
    peak_xyz_speed = float(0.3 / segment_dt)
    for i in range(MAX_PLAN_KEYS - 1):
        if i < count - 1:
            old_next = float(key_time[world_id, i + 1])
            dt = float(old_next - prev_old)
            dist = float(_norm3(key_xyz[world_id, i + 1] - key_xyz[world_id, i]))
            min_dt = float(1.875 * dist / peak_xyz_speed)
            new_next = float(prev_new + wp.max(dt, min_dt))
            key_time[world_id, i + 1] = new_next
            prev_old = old_next
            prev_new = new_next


@wp.func
def _evaluate_plan(
    key_count: wp.array(dtype=int),
    key_time: wp.array2d[float],
    key_xyz: wp.array2d[wp.vec3],
    key_quat: wp.array2d[wp.quat],
    key_grasp: wp.array2d[float],
    key_tangent: wp.array2d[wp.vec3],
    world_id: int,
    query_time: float,
) -> wp.vec4:
    count = key_count[world_id]
    seg = int(0)
    for i in range(MAX_PLAN_KEYS - 1):
        if i < count - 1 and query_time >= key_time[world_id, i + 1]:
            seg = i + 1
    if seg >= count - 1:
        seg = count - 2
    t0 = key_time[world_id, seg]
    t1 = key_time[world_id, seg + 1]
    dt = t1 - t0
    u = wp.clamp((query_time - t0) / dt, 0.0, 1.0)
    p0 = key_xyz[world_id, seg]
    p1 = key_xyz[world_id, seg + 1]
    m0 = key_tangent[world_id, seg]
    m1 = key_tangent[world_id, seg + 1]
    u2 = u * u
    u3 = u2 * u
    xyz = (2.0 * u3 - 3.0 * u2 + 1.0) * p0 + (u3 - 2.0 * u2 + u) * dt * m0
    xyz = xyz + (-2.0 * u3 + 3.0 * u2) * p1 + (u3 - u2) * dt * m1
    rot_u = u3 * (10.0 - 15.0 * u + 6.0 * u2)
    rot_u = wp.clamp(rot_u, 0.0, 1.0)
    grasp = (1.0 - rot_u) * key_grasp[world_id, seg] + rot_u * key_grasp[world_id, seg + 1]
    return wp.vec4(xyz[0], xyz[1], xyz[2], grasp)


@wp.func
def _evaluate_plan_quat(
    key_count: wp.array(dtype=int),
    key_time: wp.array2d[float],
    key_quat: wp.array2d[wp.quat],
    world_id: int,
    query_time: float,
) -> wp.quat:
    count = key_count[world_id]
    seg = int(0)
    for i in range(MAX_PLAN_KEYS - 1):
        if i < count - 1 and query_time >= key_time[world_id, i + 1]:
            seg = i + 1
    if seg >= count - 1:
        seg = count - 2
    t0 = key_time[world_id, seg]
    t1 = key_time[world_id, seg + 1]
    u = wp.clamp((query_time - t0) / (t1 - t0), 0.0, 1.0)
    u2 = u * u
    u3 = u2 * u
    rot_u = wp.clamp(u3 * (10.0 - 15.0 * u + 6.0 * u2), 0.0, 1.0)
    return _quat_slerp(key_quat[world_id, seg], key_quat[world_id, seg + 1], rot_u)


@wp.func
def _current_yaw(site_xmat: wp.array2d[wp.mat33], world_id: int, site_id: int) -> float:
    xmat = site_xmat[world_id, site_id]
    return wp.atan2(xmat[1, 0], xmat[0, 0])
