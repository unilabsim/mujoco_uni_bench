from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .model_utils import mujoco


# ===================================================================
# Python for-loop baseline
# ===================================================================
def python_loop_step(model, states, nstep):
    """Step all envs sequentially with a Python for-loop."""
    nbatch = states.shape[0]
    d = mujoco.MjData(model)
    nstate = states.shape[1]
    out = np.zeros_like(states)
    for i in range(nbatch):
        mujoco.mj_setState(model, d, states[i], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        for _ in range(nstep):
            mujoco.mj_step(model, d)
        mujoco.mj_getState(model, d, out[i], mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return out


def python_loop_forward(model, states):
    """Forward all envs sequentially."""
    nbatch = states.shape[0]
    d = mujoco.MjData(model)
    sensors = np.zeros((nbatch, model.nsensordata), dtype=np.float64)
    for i in range(nbatch):
        mujoco.mj_setState(model, d, states[i], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(model, d)
        sensors[i] = d.sensordata.copy()
    return sensors


def python_loop_reset(model, states, env_ids):
    """Reset a subset of envs sequentially."""
    d = mujoco.MjData(model)
    nstate = states.shape[1]
    out = np.zeros((len(env_ids), nstate), dtype=np.float64)
    for idx, eid in enumerate(env_ids):
        mujoco.mj_resetData(model, d)
        mujoco.mj_setState(model, d, states[idx], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(model, d)
        mujoco.mj_getState(model, d, out[idx], mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return out


def python_loop_jacobian(model, states, site_id):
    """Compute site Jacobians sequentially."""
    nbatch = states.shape[0]
    nv = model.nv
    jacp = np.zeros((nbatch, 3, nv), dtype=np.float64)
    jacr = np.zeros((nbatch, 3, nv), dtype=np.float64)
    d = mujoco.MjData(model)
    for i in range(nbatch):
        mujoco.mj_setState(model, d, states[i], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        mujoco.mj_jacSite(model, d, jacp[i], jacr[i], site_id)
    return jacp, jacr


def python_loop_hfield(model, d_template, states, hfield_geom_id, offsets, frame_body_id):
    """Sample hfield heights sequentially (bilinear interpolation)."""
    nbatch = states.shape[0]
    npoint = offsets.shape[0]
    heights = np.zeros((nbatch, npoint), dtype=np.float64)
    d = mujoco.MjData(model)

    hfield_id = model.geom_dataid[hfield_geom_id]
    nrow = model.hfield_nrow[hfield_id]
    ncol = model.hfield_ncol[hfield_id]
    hfield_size = model.hfield_size[hfield_id]  # (x_half, y_half, z_max, z_min)

    for i in range(nbatch):
        mujoco.mj_setState(model, d, states[i], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(model, d)

        # Get frame body position and yaw
        body_pos = d.xpos[frame_body_id].copy()

        for j in range(npoint):
            # Convert offset to world coordinates (simplified: yaw-aligned)
            wx = body_pos[0] + offsets[j, 0]
            wy = body_pos[1] + offsets[j, 1]

            # Map world coords to hfield grid
            # hfield spans [-size[0], size[0]] x [-size[1], size[1]]
            fx = (wx / hfield_size[0] + 1.0) * 0.5 * (ncol - 1)
            fy = (wy / hfield_size[1] + 1.0) * 0.5 * (nrow - 1)

            # Clamp to valid range
            fx = max(0, min(ncol - 1.001, fx))
            fy = max(0, min(nrow - 1.001, fy))

            # Bilinear interpolation
            ix, iy = int(fx), int(fy)
            sx, sy = fx - ix, fy - iy

            h00 = model.hfield_data[hfield_id * nrow * ncol + iy * ncol + ix]
            h10 = model.hfield_data[hfield_id * nrow * ncol + iy * ncol + min(ix+1, ncol-1)]
            h01 = model.hfield_data[hfield_id * nrow * ncol + min(iy+1, nrow-1) * ncol + ix]
            h11 = model.hfield_data[hfield_id * nrow * ncol + min(iy+1, nrow-1) * ncol + min(ix+1, ncol-1)]

            h = (1-sx)*(1-sy)*h00 + sx*(1-sy)*h10 + (1-sx)*sy*h01 + sx*sy*h11
            heights[i, j] = h * hfield_size[2]  # scale by z_max

    return heights


# ===================================================================
# Python multiprocessing baseline
# ===================================================================
def _mp_step_worker(args):
    model_path, state, nstep = args
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    for _ in range(nstep):
        mujoco.mj_step(m, d)
    out = np.zeros(state.shape, dtype=np.float64)
    mujoco.mj_getState(m, d, out, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return out


def _mp_forward_worker(args):
    model_path, state = args
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(m, d)
    return d.sensordata.copy()


def _mp_reset_worker(args):
    model_path, state = args
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_setState(m, d, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(m, d)
    out = np.zeros(state.shape, dtype=np.float64)
    mujoco.mj_getState(m, d, out, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return out


def _mp_jacobian_worker(args):
    model_path, state, site_id = args
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    jacp = np.zeros((3, m.nv), dtype=np.float64)
    jacr = np.zeros((3, m.nv), dtype=np.float64)
    mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
    return jacp, jacr


def _mp_hfield_worker(args):
    model_path, state, hfield_geom_id, offsets_flat, npoint, frame_body_id = args
    offsets = offsets_flat.reshape(npoint, 2)
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(m, d)
    hfield_id = m.geom_dataid[hfield_geom_id]
    nrow = m.hfield_nrow[hfield_id]
    ncol = m.hfield_ncol[hfield_id]
    hfield_size = m.hfield_size[hfield_id]
    body_pos = d.xpos[frame_body_id].copy()
    heights = np.zeros(npoint, dtype=np.float64)
    for j in range(npoint):
        wx = body_pos[0] + offsets[j, 0]
        wy = body_pos[1] + offsets[j, 1]
        fx = (wx / hfield_size[0] + 1.0) * 0.5 * (ncol - 1)
        fy = (wy / hfield_size[1] + 1.0) * 0.5 * (nrow - 1)
        fx = max(0, min(ncol - 1.001, fx))
        fy = max(0, min(nrow - 1.001, fy))
        ix, iy = int(fx), int(fy)
        sx, sy = fx - ix, fy - iy
        h00 = m.hfield_data[hfield_id * nrow * ncol + iy * ncol + ix]
        h10 = m.hfield_data[hfield_id * nrow * ncol + iy * ncol + min(ix+1, ncol-1)]
        h01 = m.hfield_data[hfield_id * nrow * ncol + min(iy+1, nrow-1) * ncol + ix]
        h11 = m.hfield_data[hfield_id * nrow * ncol + min(iy+1, nrow-1) * ncol + min(ix+1, ncol-1)]
        h = (1-sx)*(1-sy)*h00 + sx*(1-sy)*h10 + (1-sx)*sy*h01 + sx*sy*h11
        heights[j] = h * hfield_size[2]
    return heights


def python_mp_hfield(model_path, states, hfield_geom_id, offsets, frame_body_id, nworkers):
    offsets_flat = offsets.flatten()
    npoint = offsets.shape[0]
    with ProcessPoolExecutor(max_workers=nworkers) as pool:
        args = [(model_path, states[i], hfield_geom_id, offsets_flat, npoint, frame_body_id)
                for i in range(states.shape[0])]
        results = list(pool.map(_mp_hfield_worker, args))
    return np.array(results)


def python_mp_step(model_path, states, nstep, nworkers):
    with ProcessPoolExecutor(max_workers=nworkers) as pool:
        args = [(model_path, states[i], nstep) for i in range(states.shape[0])]
        results = list(pool.map(_mp_step_worker, args))
    return np.array(results)


def python_mp_forward(model_path, states, nworkers):
    with ProcessPoolExecutor(max_workers=nworkers) as pool:
        args = [(model_path, states[i]) for i in range(states.shape[0])]
        results = list(pool.map(_mp_forward_worker, args))
    return np.array(results)


def python_mp_reset(model_path, states, env_ids, nworkers):
    with ProcessPoolExecutor(max_workers=nworkers) as pool:
        args = [(model_path, states[idx]) for idx in range(len(env_ids))]
        results = list(pool.map(_mp_reset_worker, args))
    return np.array(results)


def python_mp_jacobian(model_path, states, site_id, nworkers):
    with ProcessPoolExecutor(max_workers=nworkers) as pool:
        args = [(model_path, states[i], site_id) for i in range(states.shape[0])]
        results = list(pool.map(_mp_jacobian_worker, args))
    jacp = np.array([r[0] for r in results])
    jacr = np.array([r[1] for r in results])
    return jacp, jacr


# ===================================================================
# Stock-mjbatch baselines: a forward() call plus NumPy assembly over the
# bound derived fields — what mjbatch without the native query ops can do.
# ===================================================================
def mjbatch_numpy_jacobian(model, site_id, cdof, subtree_com, site_xpos):
    """Site Jacobians assembled in NumPy from mjbatch-bound derived fields.

    cdof, subtree_com and site_xpos are arrays bound BEFORE the last physics
    call, so they are current. Mirrors mj_jacSite: for each dof on the site's
    kinematic chain, jacp = cdof_lin + cross(cdof_ang, site_xpos -
    subtree_com[root]) and jacr = cdof_ang.
    """
    nbatch, nv = cdof.shape[0], model.nv
    cdof = cdof.reshape(nbatch, nv, 6)
    body = int(model.site_bodyid[site_id])
    root = int(model.body_rootid[body])
    weld = int(model.body_weldid[body])
    chain = []
    i = int(model.body_dofadr[weld]) + int(model.body_dofnum[weld]) - 1
    while i >= 0:
        chain.append(i)
        i = int(model.dof_parentid[i])

    offset = site_xpos[:, site_id] - subtree_com[:, root]
    ang = cdof[:, chain, 0:3]
    lin = cdof[:, chain, 3:6]
    jacp = np.zeros((nbatch, 3, nv), dtype=np.float64)
    jacr = np.zeros((nbatch, 3, nv), dtype=np.float64)
    jacp[:, :, chain] = (lin + np.cross(ang, offset[:, None, :])).transpose(0, 2, 1)
    jacr[:, :, chain] = ang.transpose(0, 2, 1)
    return jacp, jacr


def mjbatch_numpy_hfield(model, hfield_geom_id, offsets, frame_body_id,
                         geom_xpos, geom_xmat, xpos):
    """Bilinear hfield heights assembled in NumPy from mjbatch-bound fields.

    Same math as python_loop_hfield, vectorized across the batch and over the
    sample points. geom_xpos, geom_xmat and xpos are arrays bound BEFORE the
    last physics call, so they are current.
    """
    hfield_id = int(model.geom_dataid[hfield_geom_id])
    nrow = int(model.hfield_nrow[hfield_id])
    ncol = int(model.hfield_ncol[hfield_id])
    hsize = model.hfield_size[hfield_id]
    grid = model.hfield_data[hfield_id * nrow * ncol:(hfield_id + 1) * nrow * ncol]
    grid = grid.reshape(nrow, ncol)

    gpos = geom_xpos[:, hfield_geom_id]                                 # (N, 3)
    gmat = geom_xmat[:, hfield_geom_id].reshape(-1, 3, 3)
    bpos = xpos[:, frame_body_id]                                       # (N, 3)

    wx = bpos[:, None, 0] + offsets[None, :, 0]
    wy = bpos[:, None, 1] + offsets[None, :, 1]
    wz = np.broadcast_to(gpos[:, None, 2], wx.shape)
    rel = np.stack([wx, wy, wz], axis=-1) - gpos[:, None, :]            # (N, P, 3)
    lp = np.einsum("nji,npj->npi", gmat, rel)                           # gmat^T @ rel

    fx = np.clip((lp[..., 0] / hsize[0] + 1.0) * 0.5 * (ncol - 1), 0.0, ncol - 1.001)
    fy = np.clip((lp[..., 1] / hsize[1] + 1.0) * 0.5 * (nrow - 1), 0.0, nrow - 1.001)
    ix, iy = fx.astype(np.int64), fy.astype(np.int64)
    sx, sy = fx - ix, fy - iy
    ix1, iy1 = np.minimum(ix + 1, ncol - 1), np.minimum(iy + 1, nrow - 1)
    h = ((1 - sx) * (1 - sy) * grid[iy, ix] + sx * (1 - sy) * grid[iy, ix1]
         + (1 - sx) * sy * grid[iy1, ix] + sx * sy * grid[iy1, ix1])
    return h * hsize[2]
