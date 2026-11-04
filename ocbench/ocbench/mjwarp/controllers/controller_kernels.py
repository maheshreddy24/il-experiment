import warp as wp

from ocbench.mjwarp.primitives.primitive_kernels import (
    _mat_vec,
    _matmul,
)


@wp.kernel
def update_done(
    success: wp.array(dtype=int),
    healthy: wp.array(dtype=int),
    done: wp.array(dtype=int),
    episode_length: wp.array(dtype=int),
    episode_success: wp.array(dtype=int),
    max_steps: int,
):
    world_id = wp.tid()
    if done[world_id] == 0:
        episode_length[world_id] += 1
        if success[world_id] != 0:
            episode_success[world_id] = 1
        if success[world_id] != 0 or healthy[world_id] == 0 or episode_length[world_id] >= max_steps:
            done[world_id] = 1


@wp.func
def _hold_current_target(
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    qpos: wp.array2d[float],
    target_attach_pos: wp.array(dtype=wp.vec3),
    target_attach_xmat: wp.array(dtype=wp.mat33),
    target_gripper: wp.array(dtype=float),
    t_pa_rot_arr: wp.array(dtype=wp.mat33),
    t_pa_translation: wp.array(dtype=wp.vec3),
    world_id: int,
    pinch_site_id: int,
    gripper_qpos_id: int,
):
    current_xmat = site_xmat[world_id, pinch_site_id]
    current_pos = site_xpos[world_id, pinch_site_id]
    target_attach_pos[world_id] = current_pos + _mat_vec(current_xmat, t_pa_translation[0])
    target_attach_xmat[world_id] = _matmul(current_xmat, t_pa_rot_arr[0])
    target_gripper[world_id] = wp.clamp(qpos[world_id, gripper_qpos_id] / 0.8, 0.0, 1.0)
