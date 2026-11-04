import warp as wp

from ocbench.mjwarp.envs.manipulation_kernels import (
    _dist2_xy,
    _quat_yaw,
    _reset_rand_int,
    _reset_rand_uniform,
)
from ocbench.mjwarp.primitives.cube_kernels import MAX_CUBES
from ocbench.mjwarp.primitives.primitive_kernels import (
    _mat_vec,
    _matmul,
    _quat_from_rpy,
)


@wp.func
def _mirror_y(pos: wp.vec3) -> wp.vec3:
    return wp.vec3(pos[0], -pos[1], pos[2])


@wp.kernel
def reset_chamber_worlds(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    ctrl: wp.array2d[float],
    time: wp.array(dtype=float),
    mocap_pos: wp.array2d[wp.vec3],
    mocap_quat: wp.array2d[wp.quat],
    eq_active: wp.array2d[bool],
    world_ids: wp.array(dtype=int),
    seeds: wp.array(dtype=int),
    target_attach_pos: wp.array(dtype=wp.vec3),
    target_attach_xmat: wp.array(dtype=wp.mat33),
    target_gripper: wp.array(dtype=float),
    arm_qpos_ids: wp.array(dtype=int),
    object_qpos_addrs: wp.array(dtype=int),
    button_qpos_addrs: wp.array(dtype=int),
    drawer_qpos_addr: int,
    window_qpos_addr: int,
    button_base_mocap_ids: wp.array(dtype=int),
    drawer_base_mocap_id: int,
    window_base_mocap_id: int,
    drawer_lock_eq_id: int,
    window_lock_eq_id: int,
    home_qpos: wp.array(dtype=float),
    nominal_button_pos: wp.array(dtype=wp.vec3),
    nominal_button_quat: wp.array(dtype=wp.quat),
    nominal_drawer_pos: wp.vec3,
    nominal_window_pos: wp.vec3,
    drawer_quat: wp.quat,
    window_quat: wp.quat,
    mirrored_drawer_quat: wp.quat,
    mirrored_window_quat: wp.quat,
    down_xmat: wp.array(dtype=wp.mat33),
    t_pa_rot: wp.array(dtype=wp.mat33),
    t_pa_translation: wp.array(dtype=wp.vec3),
    button_states: wp.array2d[int],
    target_button_states: wp.array2d[int],
    mirrored_arr: wp.array(dtype=int),
    target_drawer_pos: wp.array(dtype=float),
    target_window_pos: wp.array(dtype=float),
    nq: int,
    nv: int,
    nu: int,
    num_cubes: int,
    task_mode: int,
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
    mirrored = int(0)
    if _reset_rand_uniform(seed, counter, 0.0, 1.0) < 0.5:
        mirrored = 1
    counter += 1
    mirrored_arr[world_id] = mirrored

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

    for button in range(2):
        pos = nominal_button_pos[button]
        if mirrored != 0:
            pos = _mirror_y(pos)
        mocap_id = button_base_mocap_ids[button]
        mocap_pos[world_id, mocap_id] = pos
        mocap_quat[world_id, mocap_id] = nominal_button_quat[button]
        state = int(wp.floor(_reset_rand_uniform(seed, counter, 0.0, 2.0)))
        if state > 1:
            state = 1
        button_states[world_id, button] = state
        target_button_states[world_id, button] = 0
        qpos[world_id, button_qpos_addrs[button]] = 0.0
        counter += 1

    drawer_state = int(0)
    button_states[world_id, 0] = drawer_state

    drawer_pos = nominal_drawer_pos + wp.vec3(
        _reset_rand_uniform(seed, counter, -0.05, 0.05),
        _reset_rand_uniform(seed, counter + 1, -0.15, 0.0),
        0.0,
    )
    counter += 2
    window_pos = nominal_window_pos + wp.vec3(
        _reset_rand_uniform(seed, counter, 0.0, 0.05),
        _reset_rand_uniform(seed, counter + 1, 0.0, 0.15),
        0.0,
    )
    counter += 2
    if mirrored != 0:
        drawer_pos = _mirror_y(drawer_pos)
        window_pos = _mirror_y(window_pos)
    mocap_pos[world_id, drawer_base_mocap_id] = drawer_pos
    mocap_quat[world_id, drawer_base_mocap_id] = drawer_quat
    if mirrored != 0:
        mocap_quat[world_id, drawer_base_mocap_id] = mirrored_drawer_quat
    mocap_pos[world_id, window_base_mocap_id] = window_pos
    mocap_quat[world_id, window_base_mocap_id] = window_quat
    if mirrored != 0:
        mocap_quat[world_id, window_base_mocap_id] = mirrored_window_quat

    init_drawer_pos = _reset_rand_uniform(seed, counter, -0.06, 0.0)
    counter += 1
    init_window_pos = _reset_rand_uniform(seed, counter, 0.0, 0.2)
    counter += 1
    qpos[world_id, drawer_qpos_addr] = init_drawer_pos
    qpos[world_id, window_qpos_addr] = init_window_pos
    target_drawer_pos[world_id] = 0.0
    target_window_pos[world_id] = 0.2
    if task_mode == 1:
        target_window_pos[world_id] = 0.0
    elif task_mode == 2:
        target_drawer_pos[world_id] = -0.16

    xs = wp.zeros(MAX_CUBES, dtype=float)
    ys = wp.zeros(MAX_CUBES, dtype=float)
    for cube in range(MAX_CUBES):
        if cube < num_cubes:
            x = object_lo[0]
            y = object_lo[1]
            for attempt in range(100):
                x = _reset_rand_uniform(seed, counter, object_lo[0], object_hi[0])
                y = _reset_rand_uniform(seed, counter + 1, object_lo[1], object_hi[1])
                counter += 2
                if mirrored != 0:
                    y = -y
                ok = int(1)
                for other in range(MAX_CUBES):
                    if other < cube:
                        if _dist2_xy(x, y, xs[other], ys[other]) < min_object_dist * min_object_dist:
                            ok = 0
                if ok != 0:
                    break
            xs[cube] = x
            ys[cube] = y
            yaw = _reset_rand_uniform(seed, counter, 0.0, 6.283185307179586)
            counter += 1
            quat = _quat_from_rpy(0.0, 0.0, yaw)
            addr = object_qpos_addrs[cube]
            qpos[world_id, addr + 0] = x
            qpos[world_id, addr + 1] = y
            qpos[world_id, addr + 2] = cube_size
            qpos[world_id, addr + 3] = quat[0]
            qpos[world_id, addr + 4] = quat[1]
            qpos[world_id, addr + 5] = quat[2]
            qpos[world_id, addr + 6] = quat[3]

    eq_active[world_id, drawer_lock_eq_id] = False
    eq_active[world_id, window_lock_eq_id] = False


@wp.kernel
def place_chamber_objects(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    mocap_pos: wp.array2d[wp.vec3],
    world_ids: wp.array(dtype=int),
    seeds: wp.array(dtype=int),
    object_qpos_addrs: wp.array(dtype=int),
    drawer_site_id: int,
    drawer_base_mocap_id: int,
    mirrored_arr: wp.array(dtype=int),
    num_cubes: int,
    task_mode: int,
    cube_size: float,
    min_object_dist: float,
    object_lo: wp.vec2,
    object_hi: wp.vec2,
):
    reset_idx = wp.tid()
    world_id = world_ids[reset_idx]
    seed = seeds[reset_idx]
    mirrored = mirrored_arr[world_id]
    counter = int(1000)

    order = wp.zeros(MAX_CUBES, dtype=int)
    drawer_blocks = wp.zeros(MAX_CUBES, dtype=int)
    drawer_top_blocks = wp.zeros(MAX_CUBES, dtype=int)
    for cube in range(MAX_CUBES):
        order[cube] = cube
        drawer_blocks[cube] = 0
        drawer_top_blocks[cube] = 0

    for rank in range(MAX_CUBES):
        if rank < num_cubes:
            swap_rank = rank + _reset_rand_int(seed, counter, num_cubes - rank)
            counter += 1
            tmp = order[rank]
            order[rank] = order[swap_rank]
            order[swap_rank] = tmp

    num_drawer_blocks = int(0)
    num_drawer_top_blocks = int(0)
    if task_mode == 0:
        if num_cubes > 1:
            num_drawer_top_blocks = 1 + _reset_rand_int(seed, counter, 2)
            counter += 1
            if num_drawer_top_blocks > num_cubes:
                num_drawer_top_blocks = num_cubes
    elif task_mode == 1:
        if num_cubes == 1:
            num_drawer_blocks = 1
        elif num_cubes == 2:
            choice = _reset_rand_int(seed, counter, 2)
            counter += 1
            if choice == 0:
                num_drawer_blocks = 1
                num_drawer_top_blocks = 1
            else:
                num_drawer_blocks = 2
        else:
            choice = _reset_rand_int(seed, counter, 5)
            counter += 1
            if choice == 0:
                num_drawer_blocks = 1
                num_drawer_top_blocks = 0
            elif choice == 1:
                num_drawer_blocks = 1
                num_drawer_top_blocks = 1
            elif choice == 2:
                num_drawer_blocks = 1
                num_drawer_top_blocks = 2
            elif choice == 3:
                num_drawer_blocks = 2
                num_drawer_top_blocks = 0
            else:
                num_drawer_blocks = 2
                num_drawer_top_blocks = 1

    for rank in range(MAX_CUBES):
        if rank < num_cubes:
            block = order[rank]
            if rank < num_drawer_blocks:
                drawer_blocks[block] = 1
            elif rank < num_drawer_blocks + num_drawer_top_blocks:
                drawer_top_blocks[block] = 1

    drawer_offset0 = -0.05 + _reset_rand_uniform(seed, counter, -0.005, 0.005)
    counter += 1
    drawer_offset1 = 0.05 + _reset_rand_uniform(seed, counter, -0.005, 0.005)
    counter += 1
    if _reset_rand_uniform(seed, counter, 0.0, 1.0) < 0.5:
        tmp_offset = drawer_offset0
        drawer_offset0 = drawer_offset1
        drawer_offset1 = tmp_offset
    counter += 1

    top_offset0 = -0.05 + _reset_rand_uniform(seed, counter, -0.005, 0.005)
    counter += 1
    top_offset1 = 0.05 + _reset_rand_uniform(seed, counter, -0.005, 0.005)
    counter += 1
    if _reset_rand_uniform(seed, counter, 0.0, 1.0) < 0.5:
        tmp_top_offset = top_offset0
        top_offset0 = top_offset1
        top_offset1 = tmp_top_offset
    counter += 1

    drawer_base_pos = mocap_pos[world_id, drawer_base_mocap_id]
    drawer_handle_pos = site_xpos[world_id, drawer_site_id]
    slide_sign = -1.0
    if mirrored != 0:
        slide_sign = 1.0

    floor_xs = wp.zeros(MAX_CUBES, dtype=float)
    floor_ys = wp.zeros(MAX_CUBES, dtype=float)
    floor_count = int(0)
    drawer_idx = int(0)
    top_idx = int(0)
    for cube in range(MAX_CUBES):
        if cube < num_cubes:
            x = object_lo[0]
            y = object_lo[1]
            z = cube_size
            if drawer_blocks[cube] != 0:
                if num_drawer_blocks == 2:
                    x = drawer_base_pos[0] + drawer_offset0
                    if drawer_idx == 1:
                        x = drawer_base_pos[0] + drawer_offset1
                    drawer_idx += 1
                else:
                    x = drawer_base_pos[0] + _reset_rand_uniform(seed, counter, -0.06, 0.06)
                    counter += 1
                y = drawer_handle_pos[1] + slide_sign * _reset_rand_uniform(seed, counter, 0.16, 0.18)
                counter += 1
                z = 0.076
            elif drawer_top_blocks[cube] != 0:
                if num_drawer_top_blocks == 2:
                    x = drawer_base_pos[0] + top_offset0
                    if top_idx == 1:
                        x = drawer_base_pos[0] + top_offset1
                    top_idx += 1
                else:
                    x = drawer_base_pos[0] + _reset_rand_uniform(seed, counter, -0.06, 0.06)
                    counter += 1
                y = drawer_base_pos[1] - slide_sign * _reset_rand_uniform(seed, counter, 0.07, 0.1)
                counter += 1
                z = drawer_base_pos[2] + 0.108
            else:
                for attempt in range(100):
                    x = _reset_rand_uniform(seed, counter, object_lo[0], object_hi[0])
                    y = _reset_rand_uniform(seed, counter + 1, object_lo[1], object_hi[1])
                    counter += 2
                    if mirrored != 0:
                        y = -y
                    ok = int(1)
                    for other in range(MAX_CUBES):
                        if other < floor_count:
                            if _dist2_xy(x, y, floor_xs[other], floor_ys[other]) < min_object_dist * min_object_dist:
                                ok = 0
                    if ok != 0:
                        break
                floor_xs[floor_count] = x
                floor_ys[floor_count] = y
                floor_count += 1

            yaw = _reset_rand_uniform(seed, counter, 0.0, 6.283185307179586)
            counter += 1
            quat = _quat_from_rpy(0.0, 0.0, yaw)
            addr = object_qpos_addrs[cube]
            qpos[world_id, addr + 0] = x
            qpos[world_id, addr + 1] = y
            qpos[world_id, addr + 2] = z
            qpos[world_id, addr + 3] = quat[0]
            qpos[world_id, addr + 4] = quat[1]
            qpos[world_id, addr + 5] = quat[2]
            qpos[world_id, addr + 6] = quat[3]


@wp.kernel
def capture_chamber_locks(
    button_states: wp.array2d[int],
    eq_active: wp.array2d[bool],
    mocap_pos: wp.array2d[wp.vec3],
    mocap_quat: wp.array2d[wp.quat],
    xpos: wp.array2d[wp.vec3],
    xquat: wp.array2d[wp.quat],
    drawer_lock_eq_id: int,
    window_lock_eq_id: int,
    drawer_lock_mocap_id: int,
    window_lock_mocap_id: int,
    drawer_link_body_id: int,
    window_link_body_id: int,
    force: int,
):
    world_id = wp.tid()
    drawer_active = int(button_states[world_id, 0] == 0)
    window_active = int(button_states[world_id, 1] == 0)
    if drawer_active != 0 and (force != 0 or eq_active[world_id, drawer_lock_eq_id] == False):
        mocap_pos[world_id, drawer_lock_mocap_id] = xpos[world_id, drawer_link_body_id]
        mocap_quat[world_id, drawer_lock_mocap_id] = xquat[world_id, drawer_link_body_id]
    if window_active != 0 and (force != 0 or eq_active[world_id, window_lock_eq_id] == False):
        mocap_pos[world_id, window_lock_mocap_id] = xpos[world_id, window_link_body_id]
        mocap_quat[world_id, window_lock_mocap_id] = xquat[world_id, window_link_body_id]
    eq_active[world_id, drawer_lock_eq_id] = drawer_active != 0
    eq_active[world_id, window_lock_eq_id] = window_active != 0


@wp.kernel
def store_button_qpos(
    qpos: wp.array2d[float],
    button_qpos_addrs: wp.array(dtype=int),
    prev_button_qpos: wp.array2d[float],
):
    world_id = wp.tid()
    for button in range(2):
        prev_button_qpos[world_id, button] = qpos[world_id, button_qpos_addrs[button]]


@wp.kernel
def update_button_states(
    qpos: wp.array2d[float],
    button_qpos_addrs: wp.array(dtype=int),
    prev_button_qpos: wp.array2d[float],
    button_states: wp.array2d[int],
):
    world_id = wp.tid()
    for button in range(2):
        prev_pos = prev_button_qpos[world_id, button]
        cur_pos = qpos[world_id, button_qpos_addrs[button]]
        if prev_pos > -0.02 and cur_pos <= -0.02:
            button_states[world_id, button] = (button_states[world_id, button] + 1) % 2


@wp.func
def _chamber_in_drawer(
    obj_pos: wp.vec3,
    drawer_base_pos: wp.vec3,
    drawer_handle_pos: wp.vec3,
    mirrored: int,
) -> int:
    drawer_low_y = drawer_handle_pos[1] - 0.31
    drawer_high_y = drawer_handle_pos[1] - 0.11
    if mirrored != 0:
        drawer_low_y = drawer_handle_pos[1] + 0.11
        drawer_high_y = drawer_handle_pos[1] + 0.31
    if obj_pos[0] < drawer_base_pos[0] - 0.15 or obj_pos[0] > drawer_base_pos[0] + 0.15:
        return 0
    if obj_pos[1] < drawer_low_y or obj_pos[1] > drawer_high_y:
        return 0
    if obj_pos[2] < drawer_base_pos[2] - 0.044 or obj_pos[2] > drawer_base_pos[2] + 0.046:
        return 0
    return 1


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
            addr = object_qpos_addrs[cube]
            pos = wp.vec3(qpos[world_id, addr], qpos[world_id, addr + 1], qpos[world_id, addr + 2])
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
            if wp.sqrt((xs[cube] - xs[base]) * (xs[cube] - xs[base]) + (ys[cube] - ys[base]) * (ys[cube] - ys[base])) > 0.04:
                return 0
    for rank in range(MAX_CUBES):
        if rank < num_cubes:
            cube = order[rank]
            if wp.abs(zs[cube] - cube_size * (2.0 * float(rank) + 1.0)) > 0.03:
                return 0
    return 1


@wp.func
def _chamber_on_floor_success(
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
    drawer_base_pos = mocap_pos[world_id, drawer_base_mocap_id]
    drawer_handle_pos = site_xpos[world_id, drawer_site_id]
    for cube in range(MAX_CUBES):
        if cube < num_cubes:
            addr = object_qpos_addrs[cube]
            pos = wp.vec3(qpos[world_id, addr], qpos[world_id, addr + 1], qpos[world_id, addr + 2])
            if wp.abs(pos[2] - cube_size) > 0.03:
                return 0
            if _chamber_in_drawer(pos, drawer_base_pos, drawer_handle_pos, mirrored) != 0:
                return 0
    return 1


@wp.kernel
def chamber_success(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    mocap_pos: wp.array2d[wp.vec3],
    object_qpos_addrs: wp.array(dtype=int),
    button_states: wp.array2d[int],
    target_button_states: wp.array2d[int],
    target_drawer_pos: wp.array(dtype=float),
    target_window_pos: wp.array(dtype=float),
    mirrored_arr: wp.array(dtype=int),
    success: wp.array(dtype=int),
    healthy: wp.array(dtype=int),
    num_cubes: int,
    cube_success_type: int,
    cube_size: float,
    lower: wp.vec3,
    upper: wp.vec3,
    drawer_qpos_addr: int,
    window_qpos_addr: int,
    drawer_site_id: int,
    drawer_base_mocap_id: int,
):
    world_id = wp.tid()
    is_healthy = int(1)
    cube_ok = int(1)
    drawer_base_pos = mocap_pos[world_id, drawer_base_mocap_id]
    drawer_handle_pos = site_xpos[world_id, drawer_site_id]
    mirrored = mirrored_arr[world_id]
    for cube in range(MAX_CUBES):
        if cube < num_cubes:
            addr = object_qpos_addrs[cube]
            pos = wp.vec3(qpos[world_id, addr], qpos[world_id, addr + 1], qpos[world_id, addr + 2])
            if pos[0] <= lower[0] or pos[0] >= upper[0] or pos[1] <= lower[1] or pos[1] >= upper[1] or pos[2] <= lower[2] or pos[2] >= upper[2]:
                is_healthy = 0
            if cube_success_type == 0:
                if _chamber_in_drawer(pos, drawer_base_pos, drawer_handle_pos, mirrored) == 0:
                    cube_ok = 0
    if cube_success_type == 1:
        cube_ok = _chamber_stack_anywhere_success(
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
    elif cube_success_type == 2:
        cube_ok = _chamber_on_floor_success(
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
    buttons_ok = int(1)
    for button in range(2):
        if button_states[world_id, button] != target_button_states[world_id, button]:
            buttons_ok = 0
    drawer_ok = int(wp.abs(qpos[world_id, drawer_qpos_addr] - target_drawer_pos[world_id]) <= 0.04)
    window_ok = int(wp.abs(qpos[world_id, window_qpos_addr] - target_window_pos[world_id]) <= 0.04)
    healthy[world_id] = is_healthy
    success[world_id] = int(is_healthy != 0 and cube_ok != 0 and buttons_ok != 0 and drawer_ok != 0 and window_ok != 0)


@wp.kernel
def park_done_chamber_worlds(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    ctrl: wp.array2d[float],
    done: wp.array(dtype=int),
    object_qpos_addrs: wp.array(dtype=int),
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
    for cube in range(MAX_CUBES):
        if cube < num_cubes:
            addr = object_qpos_addrs[cube]
            qpos[world_id, addr + 0] = 5.0 + 0.25 * float(cube)
            qpos[world_id, addr + 1] = 5.0
            qpos[world_id, addr + 2] = 5.0
            qpos[world_id, addr + 3] = 1.0
            qpos[world_id, addr + 4] = 0.0
            qpos[world_id, addr + 5] = 0.0
            qpos[world_id, addr + 6] = 0.0


@wp.func
def _write_chamber_observation(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    cfrc_ext: wp.array2d[wp.spatial_vector],
    mocap_pos: wp.array2d[wp.vec3],
    button_states: wp.array2d[int],
    mirrored: wp.array(dtype=int),
    output: wp.array2d[float],
    arm_qpos_ids: wp.array(dtype=int),
    object_qpos_addrs: wp.array(dtype=int),
    button_qpos_addrs: wp.array(dtype=int),
    button_dof_addrs: wp.array(dtype=int),
    button_base_mocap_ids: wp.array(dtype=int),
    world_id: int,
    row: int,
    num_cubes: int,
    num_buttons: int,
    num_button_states: int,
    pinch_site_id: int,
    gripper_qpos_id: int,
    right_pad_body_id: int,
    drawer_qpos_addr: int,
    drawer_dof_addr: int,
    drawer_base_mocap_id: int,
    window_qpos_addr: int,
    window_dof_addr: int,
    window_base_mocap_id: int,
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

    for cube in range(MAX_CUBES):
        if cube < num_cubes:
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

    for button in range(2):
        if button < num_buttons:
            state = button_states[world_id, button]
            for state_idx in range(2):
                if state_idx < num_button_states and state == state_idx:
                    output[row, col] = 1.0
                else:
                    output[row, col] = 0.0
                col += 1

            base_pos = mocap_pos[world_id, button_base_mocap_ids[button]]
            output[row, col] = (base_pos[0] - 0.425) * 10.0
            col += 1
            output[row, col] = base_pos[1] * 10.0
            col += 1
            output[row, col] = base_pos[2] * 10.0
            col += 1
            output[row, col] = qpos[world_id, button_qpos_addrs[button]] * 120.0
            col += 1
            output[row, col] = qvel[world_id, button_dof_addrs[button]]
            col += 1

    output[row, col] = float(mirrored[world_id])
    col += 1
    output[row, col] = qpos[world_id, drawer_qpos_addr] * 18.0
    col += 1
    output[row, col] = qvel[world_id, drawer_dof_addr]
    col += 1
    drawer_base_pos = mocap_pos[world_id, drawer_base_mocap_id]
    output[row, col] = (drawer_base_pos[0] - 0.425) * 10.0
    col += 1
    output[row, col] = drawer_base_pos[1] * 10.0
    col += 1
    output[row, col] = drawer_base_pos[2] * 10.0
    col += 1
    output[row, col] = qpos[world_id, window_qpos_addr] * 15.0
    col += 1
    output[row, col] = qvel[world_id, window_dof_addr]
    col += 1
    window_base_pos = mocap_pos[world_id, window_base_mocap_id]
    output[row, col] = (window_base_pos[0] - 0.425) * 10.0
    col += 1
    output[row, col] = window_base_pos[1] * 10.0
    col += 1
    output[row, col] = window_base_pos[2] * 10.0


@wp.kernel
def record_observations(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    cfrc_ext: wp.array2d[wp.spatial_vector],
    mocap_pos: wp.array2d[wp.vec3],
    button_states: wp.array2d[int],
    mirrored: wp.array(dtype=int),
    done: wp.array(dtype=int),
    output: wp.array2d[float],
    arm_qpos_ids: wp.array(dtype=int),
    object_qpos_addrs: wp.array(dtype=int),
    button_qpos_addrs: wp.array(dtype=int),
    button_dof_addrs: wp.array(dtype=int),
    button_base_mocap_ids: wp.array(dtype=int),
    step: int,
    nworld: int,
    num_cubes: int,
    num_buttons: int,
    num_button_states: int,
    pinch_site_id: int,
    gripper_qpos_id: int,
    right_pad_body_id: int,
    drawer_qpos_addr: int,
    drawer_dof_addr: int,
    drawer_base_mocap_id: int,
    window_qpos_addr: int,
    window_dof_addr: int,
    window_base_mocap_id: int,
):
    world_id = wp.tid()
    if done[world_id] != 0:
        return
    row = step * nworld + world_id
    _write_chamber_observation(
        qpos,
        qvel,
        site_xpos,
        site_xmat,
        cfrc_ext,
        mocap_pos,
        button_states,
        mirrored,
        output,
        arm_qpos_ids,
        object_qpos_addrs,
        button_qpos_addrs,
        button_dof_addrs,
        button_base_mocap_ids,
        world_id,
        row,
        num_cubes,
        num_buttons,
        num_button_states,
        pinch_site_id,
        gripper_qpos_id,
        right_pad_body_id,
        drawer_qpos_addr,
        drawer_dof_addr,
        drawer_base_mocap_id,
        window_qpos_addr,
        window_dof_addr,
        window_base_mocap_id,
    )
