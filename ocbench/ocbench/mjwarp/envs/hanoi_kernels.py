import warp as wp

from ocbench.mjwarp.envs.manipulation_kernels import (
    _dist2_xy,
    _quat_yaw,
    _reset_rand_uniform,
)
from ocbench.mjwarp.primitives.primitive_kernels import (
    _mat_vec,
    _matmul,
)

MAX_HANOI_DISKS = 3
MAX_HANOI_PEGS = 3
MAX_HANOI_STACK_SLOTS = 9
MAX_HANOI_DISK_GEOMS = 8


@wp.func
def _disk_yaw_for_peg(peg_yaw: float) -> float:
    half_pi = 1.5707963267948966
    quarter_pi = 0.7853981633974483
    value = peg_yaw + quarter_pi
    return value - half_pi * wp.floor(value / half_pi) - quarter_pi


@wp.func
def _disk_pos(qpos: wp.array2d[float], disk_qpos_addrs: wp.array(dtype=int), world_id: int, disk: int) -> wp.vec3:
    addr = disk_qpos_addrs[disk]
    return wp.vec3(qpos[world_id, addr], qpos[world_id, addr + 1], qpos[world_id, addr + 2])


@wp.func
def _stack_disk_z(disk: int, num_disks: int, disk_half_height: float, disk_gap: float) -> float:
    level = num_disks - 1 - disk
    return disk_half_height + float(level) * (2.0 * disk_half_height + disk_gap)


@wp.func
def _is_through_peg(pos: wp.vec3, peg_pos: wp.array2d[wp.vec3], world_id: int, num_pegs: int) -> int:
    through = int(0)
    for peg in range(MAX_HANOI_PEGS):
        if peg < num_pegs:
            target = peg_pos[world_id, peg]
            if _dist2_xy(pos[0], pos[1], target[0], target[1]) <= 0.04 * 0.04:
                through = 1
    return through


@wp.func
def _is_disk_geom(disk_geom_ids: wp.array2d[int], disk_geom_counts: wp.array(dtype=int), disk: int, geom: int) -> int:
    out = int(0)
    for geom_idx in range(MAX_HANOI_DISK_GEOMS):
        if geom_idx < disk_geom_counts[disk] and disk_geom_ids[disk, geom_idx] == geom:
            out = 1
    return out


@wp.kernel
def clear_int_flags(flags: wp.array(dtype=int)):
    flags[wp.tid()] = 0


@wp.kernel
def reset_hanoi_worlds(
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
    disk_qpos_addrs: wp.array(dtype=int),
    peg_mocap_ids: wp.array(dtype=int),
    peg_pos: wp.array2d[wp.vec3],
    peg_yaw: wp.array2d[float],
    home_qpos: wp.array(dtype=float),
    base_peg_pos: wp.array(dtype=wp.vec3),
    down_xmat: wp.array(dtype=wp.mat33),
    t_pa_rot: wp.array(dtype=wp.mat33),
    t_pa_translation: wp.array(dtype=wp.vec3),
    nq: int,
    nv: int,
    nu: int,
    num_disks: int,
    num_pegs: int,
    init_peg: int,
    randomize_pegs: int,
    disk_half_height: float,
    disk_gap: float,
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

    for peg in range(MAX_HANOI_PEGS):
        if peg < num_pegs:
            pos = base_peg_pos[peg]
            yaw = float(0.0)
            if randomize_pegs != 0:
                pos = wp.vec3(
                    pos[0] + _reset_rand_uniform(seed, counter, -0.05, 0.05),
                    pos[1] + _reset_rand_uniform(seed, counter + 1, -0.025, 0.025),
                    pos[2],
                )
                counter += 2
                yaw = _reset_rand_uniform(seed, counter, 0.0, 1.5707963267948966)
                counter += 1
            peg_pos[world_id, peg] = pos
            peg_yaw[world_id, peg] = yaw
            mocap_id = peg_mocap_ids[peg]
            mocap_pos[world_id, mocap_id] = pos
            mocap_quat[world_id, mocap_id] = wp.quat(wp.cos(0.5 * yaw), 0.0, 0.0, wp.sin(0.5 * yaw))

    yaw = _disk_yaw_for_peg(peg_yaw[world_id, init_peg])
    stack_xy = peg_pos[world_id, init_peg]
    for disk in range(MAX_HANOI_DISKS):
        if disk < num_disks:
            addr = disk_qpos_addrs[disk]
            qpos[world_id, addr + 0] = stack_xy[0]
            qpos[world_id, addr + 1] = stack_xy[1]
            qpos[world_id, addr + 2] = _stack_disk_z(disk, num_disks, disk_half_height, disk_gap)
            qpos[world_id, addr + 3] = wp.cos(0.5 * yaw)
            qpos[world_id, addr + 4] = 0.0
            qpos[world_id, addr + 5] = 0.0
            qpos[world_id, addr + 6] = wp.sin(0.5 * yaw)


@wp.kernel
def hanoi_bad_floor_contacts(
    qpos: wp.array2d[float],
    contact_geom: wp.array(dtype=wp.vec2i),
    contact_dim: wp.array(dtype=int),
    contact_worldid: wp.array(dtype=int),
    nacon: wp.array(dtype=int),
    disk_qpos_addrs: wp.array(dtype=int),
    disk_geom_ids: wp.array2d[int],
    disk_geom_counts: wp.array(dtype=int),
    peg_pos: wp.array2d[wp.vec3],
    bad_floor_contact: wp.array(dtype=int),
    num_disks: int,
    num_pegs: int,
    floor_geom_id: int,
    naconmax: int,
):
    con = wp.tid()
    if con >= nacon[0] or con >= naconmax or contact_dim[con] <= 0:
        return

    world_id = contact_worldid[con]
    geom_pair = contact_geom[con]
    for disk in range(MAX_HANOI_DISKS):
        if disk < num_disks:
            disk_touching_floor = int(0)
            if geom_pair[0] == floor_geom_id and _is_disk_geom(disk_geom_ids, disk_geom_counts, disk, geom_pair[1]) != 0:
                disk_touching_floor = 1
            if geom_pair[1] == floor_geom_id and _is_disk_geom(disk_geom_ids, disk_geom_counts, disk, geom_pair[0]) != 0:
                disk_touching_floor = 1
            if disk_touching_floor != 0:
                pos = _disk_pos(qpos, disk_qpos_addrs, world_id, disk)
                if _is_through_peg(pos, peg_pos, world_id, num_pegs) == 0:
                    bad_floor_contact[world_id] = 1


@wp.kernel
def hanoi_success(
    qpos: wp.array2d[float],
    bad_floor_contact: wp.array(dtype=int),
    disk_qpos_addrs: wp.array(dtype=int),
    peg_pos: wp.array2d[wp.vec3],
    success: wp.array(dtype=int),
    healthy: wp.array(dtype=int),
    failure: wp.array(dtype=int),
    alive: wp.array(dtype=int),
    num_disks: int,
    num_pegs: int,
    goal_peg0: int,
    goal_peg1: int,
    disk_half_height: float,
    disk_gap: float,
    lower: wp.vec3,
    upper: wp.vec3,
):
    world_id = wp.tid()

    is_healthy = int(1)
    stacks = wp.zeros(MAX_HANOI_STACK_SLOTS, dtype=int)
    stack_counts = wp.zeros(MAX_HANOI_PEGS, dtype=int)
    disk_zs = wp.zeros(MAX_HANOI_DISKS, dtype=float)

    for disk in range(MAX_HANOI_DISKS):
        if disk < num_disks:
            pos = _disk_pos(qpos, disk_qpos_addrs, world_id, disk)
            disk_zs[disk] = pos[2]
            if pos[0] <= lower[0] or pos[0] >= upper[0] or pos[1] <= lower[1] or pos[1] >= upper[1] or pos[2] <= lower[2] or pos[2] >= upper[2]:
                is_healthy = 0

            best_peg = int(0)
            best_dist = float(1.0e9)
            for peg in range(MAX_HANOI_PEGS):
                if peg < num_pegs:
                    target = peg_pos[world_id, peg]
                    dist = _dist2_xy(pos[0], pos[1], target[0], target[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_peg = peg
            if best_dist <= 0.04 * 0.04:
                count = stack_counts[best_peg]
                stacks[best_peg * MAX_HANOI_DISKS + count] = disk
                stack_counts[best_peg] = count + 1

    illegal_stack = int(0)
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
            for i in range(MAX_HANOI_DISKS - 1):
                if i < count - 1:
                    lower_disk = stacks[peg * MAX_HANOI_DISKS + i]
                    upper_disk = stacks[peg * MAX_HANOI_DISKS + i + 1]
                    z_gap = disk_zs[upper_disk] - disk_zs[lower_disk]
                    if lower_disk < upper_disk and z_gap <= 2.0 * disk_half_height + 0.04:
                        illegal_stack = 1

    solved = int(0)
    for goal_idx in range(2):
        peg = goal_peg0
        if goal_idx == 1:
            peg = goal_peg1
        ok = int(1)
        target_xy = peg_pos[world_id, peg]
        for disk in range(MAX_HANOI_DISKS):
            if disk < num_disks:
                pos = _disk_pos(qpos, disk_qpos_addrs, world_id, disk)
                target_z = _stack_disk_z(disk, num_disks, disk_half_height, disk_gap)
                if _dist2_xy(pos[0], pos[1], target_xy[0], target_xy[1]) > 0.04 * 0.04:
                    ok = 0
                if wp.abs(pos[2] - target_z) > 0.03:
                    ok = 0
        if ok != 0:
            solved = 1

    is_alive = int(is_healthy != 0 and bad_floor_contact[world_id] == 0 and illegal_stack == 0)
    healthy[world_id] = is_healthy
    failure[world_id] = int(is_healthy != 0 and (bad_floor_contact[world_id] != 0 or illegal_stack != 0))
    alive[world_id] = is_alive
    success[world_id] = int(is_alive != 0 and solved != 0)


@wp.kernel
def park_done_worlds(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    ctrl: wp.array2d[float],
    done: wp.array(dtype=int),
    disk_qpos_addrs: wp.array(dtype=int),
    arm_qpos_ids: wp.array(dtype=int),
    arm_actuator_ids: wp.array(dtype=int),
    gripper_actuator_ids: wp.array(dtype=int),
    gripper_qpos_id: int,
    num_disks: int,
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

    for disk in range(MAX_HANOI_DISKS):
        if disk < num_disks:
            addr = disk_qpos_addrs[disk]
            qpos[world_id, addr + 0] = 5.0 + 0.25 * float(disk)
            qpos[world_id, addr + 1] = 5.0
            qpos[world_id, addr + 2] = 5.0
            qpos[world_id, addr + 3] = 1.0
            qpos[world_id, addr + 4] = 0.0
            qpos[world_id, addr + 5] = 0.0
            qpos[world_id, addr + 6] = 0.0


@wp.func
def _write_observation(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    cfrc_ext: wp.array2d[wp.spatial_vector],
    output: wp.array2d[float],
    arm_qpos_ids: wp.array(dtype=int),
    disk_qpos_addrs: wp.array(dtype=int),
    peg_pos: wp.array2d[wp.vec3],
    peg_yaw: wp.array2d[float],
    world_id: int,
    row: int,
    num_disks: int,
    num_pegs: int,
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

    for disk in range(MAX_HANOI_DISKS):
        if disk < num_disks:
            addr = disk_qpos_addrs[disk]
            output[row, col] = (qpos[world_id, addr + 0] - 0.425) * 10.0
            col += 1
            output[row, col] = qpos[world_id, addr + 1] * 10.0
            col += 1
            output[row, col] = qpos[world_id, addr + 2] * 10.0
            col += 1
            disk_yaw = _quat_yaw(
                qpos[world_id, addr + 3],
                qpos[world_id, addr + 4],
                qpos[world_id, addr + 5],
                qpos[world_id, addr + 6],
            )
            output[row, col] = wp.cos(disk_yaw)
            col += 1
            output[row, col] = wp.sin(disk_yaw)
            col += 1

    for peg in range(MAX_HANOI_PEGS):
        if peg < num_pegs:
            pos = peg_pos[world_id, peg]
            output[row, col] = (pos[0] - 0.425) * 10.0
            col += 1
            output[row, col] = pos[1] * 10.0
            col += 1
            output[row, col] = pos[2] * 10.0
            col += 1
            yaw = peg_yaw[world_id, peg]
            output[row, col] = wp.cos(yaw)
            col += 1
            output[row, col] = wp.sin(yaw)
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
    disk_qpos_addrs: wp.array(dtype=int),
    peg_pos: wp.array2d[wp.vec3],
    peg_yaw: wp.array2d[float],
    step: int,
    nworld: int,
    num_disks: int,
    num_pegs: int,
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
        disk_qpos_addrs,
        peg_pos,
        peg_yaw,
        world_id,
        row,
        num_disks,
        num_pegs,
        pinch_site_id,
        gripper_qpos_id,
        right_pad_body_id,
    )
