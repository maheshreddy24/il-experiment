import warp as wp

from ocbench.mjwarp.envs.manipulation_kernels import (
    _dist2_xy,
    _quat_yaw,
    _reset_rand_uniform,
)
from ocbench.mjwarp.primitives.cube_kernels import MAX_CUBES
from ocbench.mjwarp.primitives.primitive_kernels import (
    _mat_vec,
    _matmul,
)


@wp.kernel
def park_done_worlds(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    ctrl: wp.array2d[float],
    mocap_pos: wp.array2d[wp.vec3],
    mocap_quat: wp.array2d[wp.quat],
    done: wp.array(dtype=int),
    object_qpos_addrs: wp.array(dtype=int),
    cube_target_mocap_ids: wp.array(dtype=int),
    arm_qpos_ids: wp.array(dtype=int),
    arm_actuator_ids: wp.array(dtype=int),
    gripper_actuator_ids: wp.array(dtype=int),
    gripper_qpos_id: int,
    num_cubes: int,
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

    for cube in range(num_cubes):
        addr = object_qpos_addrs[cube]
        pos = wp.vec3(5.0 + 0.25 * float(cube), 5.0, 5.0)
        qpos[world_id, addr + 0] = pos[0]
        qpos[world_id, addr + 1] = pos[1]
        qpos[world_id, addr + 2] = pos[2]
        qpos[world_id, addr + 3] = 1.0
        qpos[world_id, addr + 4] = 0.0
        qpos[world_id, addr + 5] = 0.0
        qpos[world_id, addr + 6] = 0.0

        mocap_id = cube_target_mocap_ids[cube]
        mocap_pos[world_id, mocap_id] = pos
        mocap_quat[world_id, mocap_id] = wp.quat(1.0, 0.0, 0.0, 0.0)


@wp.kernel
def reset_block_worlds(
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
    object_qpos_addrs: wp.array(dtype=int),
    cube_target_mocap_ids: wp.array(dtype=int),
    home_qpos: wp.array(dtype=float),
    goal_xyzs: wp.array(dtype=wp.vec3),
    down_xmat: wp.array(dtype=wp.mat33),
    t_pa_rot: wp.array(dtype=wp.mat33),
    t_pa_translation: wp.array(dtype=wp.vec3),
    nq: int,
    nv: int,
    nu: int,
    num_cubes: int,
    has_fixed_targets: int,
    cube_size: float,
    min_object_dist: float,
    object_lo: wp.vec2,
    object_hi: wp.vec2,
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

    xs = wp.zeros(MAX_CUBES, dtype=float)
    ys = wp.zeros(MAX_CUBES, dtype=float)
    zs = wp.zeros(MAX_CUBES, dtype=float)
    for episode_attempt in range(100):
        for cube in range(MAX_CUBES):
            if cube < num_cubes:
                x = object_lo[0]
                y = object_lo[1]
                for sample_attempt in range(100):
                    x = _reset_rand_uniform(seed, counter, object_lo[0], object_hi[0])
                    y = _reset_rand_uniform(seed, counter + 1, object_lo[1], object_hi[1])
                    counter += 2
                    ok = int(1)
                    for other in range(MAX_CUBES):
                        if other < cube:
                            if _dist2_xy(x, y, xs[other], ys[other]) < min_object_dist * min_object_dist:
                                ok = 0
                    if ok != 0:
                        break
                xs[cube] = x
                ys[cube] = y
                zs[cube] = cube_size
                yaw = _reset_rand_uniform(seed, counter, 0.0, 6.283185307179586)
                counter += 1
                addr = object_qpos_addrs[cube]
                qpos[world_id, addr + 0] = x
                qpos[world_id, addr + 1] = y
                qpos[world_id, addr + 2] = cube_size
                qpos[world_id, addr + 3] = wp.cos(0.5 * yaw)
                qpos[world_id, addr + 4] = 0.0
                qpos[world_id, addr + 5] = 0.0
                qpos[world_id, addr + 6] = wp.sin(0.5 * yaw)

                mocap_id = cube_target_mocap_ids[cube]
                if has_fixed_targets != 0:
                    mocap_pos[world_id, mocap_id] = goal_xyzs[cube]
                else:
                    mocap_pos[world_id, mocap_id] = wp.vec3(x, y, cube_size)
                mocap_quat[world_id, mocap_id] = wp.quat(1.0, 0.0, 0.0, 0.0)

        success = int(1)
        if has_fixed_targets != 0:
            for cube in range(MAX_CUBES):
                if cube < num_cubes:
                    goal = goal_xyzs[cube]
                    dx = xs[cube] - goal[0]
                    dy = ys[cube] - goal[1]
                    dz = zs[cube] - goal[2]
                    if dx * dx + dy * dy + dz * dz > 0.04 * 0.04:
                        success = 0
        else:
            order = wp.zeros(MAX_CUBES, dtype=int)
            used = wp.zeros(MAX_CUBES, dtype=int)
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
                    if _dist2_xy(xs[cube], ys[cube], xs[base], ys[base]) > 0.04 * 0.04:
                        success = 0
            for rank in range(MAX_CUBES):
                if rank < num_cubes:
                    cube = order[rank]
                    if wp.abs(zs[cube] - cube_size * (2.0 * float(rank) + 1.0)) > 0.03:
                        success = 0

        if success == 0:
            break


@wp.kernel
def block_success(
    qpos: wp.array2d[float],
    mocap_pos: wp.array2d[wp.vec3],
    object_qpos_addrs: wp.array(dtype=int),
    cube_target_mocap_ids: wp.array(dtype=int),
    success: wp.array(dtype=int),
    healthy: wp.array(dtype=int),
    num_cubes: int,
    stack_anywhere: int,
    cube_size: float,
    lower: wp.vec3,
    upper: wp.vec3,
):
    world_id = wp.tid()

    xs = wp.zeros(MAX_CUBES, dtype=float)
    ys = wp.zeros(MAX_CUBES, dtype=float)
    zs = wp.zeros(MAX_CUBES, dtype=float)

    is_healthy = int(1)
    for cube in range(num_cubes):
        addr = object_qpos_addrs[cube]
        x = qpos[world_id, addr + 0]
        y = qpos[world_id, addr + 1]
        z = qpos[world_id, addr + 2]
        xs[cube] = x
        ys[cube] = y
        zs[cube] = z
        if x <= lower[0] or x >= upper[0] or y <= lower[1] or y >= upper[1] or z <= lower[2] or z >= upper[2]:
            is_healthy = 0

    ok = is_healthy
    if ok != 0:
        if stack_anywhere != 0:
            order = wp.zeros(MAX_CUBES, dtype=int)
            used = wp.zeros(MAX_CUBES, dtype=int)
            for rank in range(num_cubes):
                best = int(0)
                best_z = float(1.0e9)
                for cube in range(num_cubes):
                    if used[cube] == 0 and zs[cube] < best_z:
                        best = cube
                        best_z = zs[cube]
                order[rank] = best
                used[best] = 1

            base = order[0]
            for cube in range(num_cubes):
                if _dist2_xy(xs[cube], ys[cube], xs[base], ys[base]) > 0.04 * 0.04:
                    ok = 0
            for rank in range(num_cubes):
                cube = order[rank]
                target_z = cube_size * (2.0 * float(rank) + 1.0)
                if wp.abs(zs[cube] - target_z) > 0.03:
                    ok = 0
        else:
            for cube in range(num_cubes):
                addr = object_qpos_addrs[cube]
                target = mocap_pos[world_id, cube_target_mocap_ids[cube]]
                dx = qpos[world_id, addr + 0] - target[0]
                dy = qpos[world_id, addr + 1] - target[1]
                dz = qpos[world_id, addr + 2] - target[2]
                if dx * dx + dy * dy + dz * dz > 0.04 * 0.04:
                    ok = 0

    healthy[world_id] = is_healthy
    success[world_id] = ok


@wp.func
def _write_observation(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    cfrc_ext: wp.array2d[wp.spatial_vector],
    output: wp.array2d[float],
    arm_qpos_ids: wp.array(dtype=int),
    object_qpos_addrs: wp.array(dtype=int),
    world_id: int,
    row: int,
    num_cubes: int,
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

    for cube in range(num_cubes):
        addr = object_qpos_addrs[cube]
        output[row, col] = (qpos[world_id, addr + 0] - 0.425) * 10.0
        col += 1
        output[row, col] = qpos[world_id, addr + 1] * 10.0
        col += 1
        output[row, col] = qpos[world_id, addr + 2] * 10.0
        col += 1
        q0 = qpos[world_id, addr + 3]
        q1 = qpos[world_id, addr + 4]
        q2 = qpos[world_id, addr + 5]
        q3 = qpos[world_id, addr + 6]
        output[row, col] = q0
        col += 1
        output[row, col] = q1
        col += 1
        output[row, col] = q2
        col += 1
        output[row, col] = q3
        col += 1
        block_yaw = _quat_yaw(q0, q1, q2, q3)
        output[row, col] = wp.cos(block_yaw)
        col += 1
        output[row, col] = wp.sin(block_yaw)
        col += 1


@wp.kernel
def record_observations(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    cfrc_ext: wp.array2d[wp.spatial_vector],
    done: wp.array(dtype=int),
    output: wp.array2d[float],
    arm_qpos_ids: wp.array(dtype=int),
    object_qpos_addrs: wp.array(dtype=int),
    step: int,
    nworld: int,
    num_cubes: int,
    pinch_site_id: int,
    gripper_qpos_id: int,
    right_pad_body_id: int,
):
    world_id = wp.tid()
    if done[world_id] != 0:
        return
    row = step * nworld + world_id
    _write_observation(
        qpos,
        qvel,
        site_xpos,
        site_xmat,
        cfrc_ext,
        output,
        arm_qpos_ids,
        object_qpos_addrs,
        world_id,
        row,
        num_cubes,
        pinch_site_id,
        gripper_qpos_id,
        right_pad_body_id,
    )
