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
