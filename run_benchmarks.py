#!/usr/bin/env python3
"""
Benchmark runner for the MuJoCoUni technical report.

Runs all benchmarks across representative robot models and saves results
to JSON files that plot_benchmarks.py can consume.

Requirements:
    pip install mujoco-uni numpy

Usage:
    python scripts/run_benchmarks.py                    # run all benchmarks
    python scripts/run_benchmarks.py --bench 1          # run only benchmark 1
    python scripts/run_benchmarks.py --bench 1 3        # run benchmarks 1 and 3
    python scripts/run_benchmarks.py --nthread 16       # override thread count

Output:  scripts/benchmark_results.json
"""

import argparse
import json
import os
import sys
import time
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

# ---------------------------------------------------------------------------
# Attempt to import mujoco-uni; fall back to standard mujoco for baselines
# ---------------------------------------------------------------------------
try:
    import mujoco
    from mujoco.batch_env import BatchEnvPool
    HAS_BATCH_ENV = True
    print(f"mujoco version: {mujoco.__version__} (BatchEnvPool available)")
except ImportError:
    try:
        import mujoco
        HAS_BATCH_ENV = False
        print(f"mujoco version: {mujoco.__version__} (standard mujoco, no BatchEnvPool)")
    except ImportError:
        print("ERROR: mujoco is not installed")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Paths (relative to this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "Go1":      os.path.join(SCRIPT_DIR, "models", "unitree_go1", "go1.xml"),
    "Allegro":  os.path.join(SCRIPT_DIR, "models", "wonik_allegro", "right_hand.xml"),
    "Franka":   os.path.join(SCRIPT_DIR, "models", "franka_emika_panda", "panda.xml"),
    "Humanoid": os.path.join(SCRIPT_DIR, "models", "humanoid", "humanoid_CMU_V2020.xml"),
    "Terrain":  os.path.join(SCRIPT_DIR, "models", "terrain", "stairs_terrain.xml"),
}

RESULTS_PATH = os.path.join(SCRIPT_DIR, "benchmark_results.json")

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
NUM_ENVS_LIST = [32, 64, 128, 256, 512, 1024, 2048, 4096]
NSTEP = 50
WARMUP = 2
REPEAT = 10
REPEAT_FAST = 50    # C++ fast path
REPEAT_SLOW = 3     # Python baselines (100-1000x slower)
WARMUP_FAST = 5
WARMUP_SLOW = 2
RESET_FRACTIONS = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
PARTIAL_RESET_NUM_ENVS = 4096


def get_nthread(args):
    return args.nthread if args.nthread else min(cpu_count(), 16)


def _enable_discardvisual(root):
    compiler_tag = root.find("compiler")
    if compiler_tag is None:
        compiler_tag = ET.Element("compiler")
        root.insert(0, compiler_tag)
    compiler_tag.set("discardvisual", "true")


def _create_discardvisual_xml(model_file):
    tree = ET.parse(model_file)
    _enable_discardvisual(tree.getroot())
    fd, output_path = tempfile.mkstemp(
        suffix=".xml", dir=os.path.dirname(os.path.abspath(model_file))
    )
    os.close(fd)
    tree.write(output_path)
    return output_path


_MODEL_CACHE = {}

def load_model(name):
    path = MODEL_PATHS[name]
    if path in _MODEL_CACHE:
        return _MODEL_CACHE[path]
    tmp = _create_discardvisual_xml(path)
    model = mujoco.MjModel.from_xml_path(tmp)
    os.unlink(tmp)
    _MODEL_CACHE[path] = model
    return model


_MP_PATH_CACHE = {}

def _get_mp_path(name):
    """Cached discardvisual temp XML for multiprocessing workers."""
    if name not in _MP_PATH_CACHE:
        _MP_PATH_CACHE[name] = _create_discardvisual_xml(MODEL_PATHS[name])
    return _MP_PATH_CACHE[name]


def get_random_state(model, nbatch, rng=None):
    """Generate a random but valid initial state for the model."""
    if rng is None:
        rng = np.random.default_rng(0)
    d = mujoco.MjData(model)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    states = np.zeros((nbatch, nstate), dtype=np.float64)
    for i in range(nbatch):
        mujoco.mj_resetData(model, d)
        # Small random perturbation to qpos
        d.qpos[:] += rng.normal(0, 0.01, size=d.qpos.shape)
        mujoco.mj_forward(model, d)
        mujoco.mj_getState(model, d, states[i], mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return states


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
# Timing helper
# ===================================================================
def bench_time(func, warmup=WARMUP, repeat=REPEAT):
    """Time a function, returning mean time over repeat runs."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.mean(times))


# ===================================================================
# Benchmark 1: Step / Forward throughput
# ===================================================================
def bench_step_forward(args):
    print("\n" + "=" * 60)
    print("Benchmark 1: Step / Forward throughput")
    print("=" * 60)

    nthread = get_nthread(args)
    robots = ["Go1", "Allegro", "Franka", "Humanoid"]
    results = {"step": {}, "forward": {}, "num_envs": NUM_ENVS_LIST}

    for robot in robots:
        print(f"\n--- {robot} ---")
        model = load_model(robot)
        step_data = []
        fwd_data = []

        for nenv in NUM_ENVS_LIST:
            states = get_random_state(model, nenv)

            if HAS_BATCH_ENV:
                pool = BatchEnvPool(model, nbatch=nenv, nthread=nthread)

                # Step benchmark
                nstep = args.nstep
                t = bench_time(lambda: pool.step(states, nstep=nstep),
                               warmup=WARMUP_FAST, repeat=REPEAT_FAST)
                steps_per_sec = nenv * nstep / t
                step_data.append(steps_per_sec)

                # Forward benchmark
                fwd_cs = args.fwd_chunk_size
                t = bench_time(
                    lambda: pool.forward(states, chunk_size=fwd_cs),
                    warmup=WARMUP_FAST, repeat=REPEAT_FAST)
                fwd_per_sec = nenv / t
                fwd_data.append(fwd_per_sec)

                del pool
            else:
                # Fallback: python loop
                nstep = args.nstep
                t = bench_time(lambda: python_loop_step(model, states, nstep),
                               warmup=args.warmup, repeat=args.repeat)
                step_data.append(nenv * nstep / t)
                t = bench_time(lambda: python_loop_forward(model, states),
                               warmup=args.warmup, repeat=args.repeat)
                fwd_data.append(nenv / t)

            print(f"  nenv={nenv:5d}  step={step_data[-1]:.0f} steps/s  fwd={fwd_data[-1]:.0f} fwd/s")

        results["step"][robot] = step_data
        results["forward"][robot] = fwd_data

    return results


# ===================================================================
# Benchmark 2: Multi-model comparison
# ===================================================================
def bench_multimodel(args):
    print("\n" + "=" * 60)
    print("Benchmark 2: Multi-model comparison")
    print("=" * 60)

    if not HAS_BATCH_ENV:
        print("  SKIPPED: BatchEnvPool not available")
        return {}

    nthread = get_nthread(args)
    robots = ["Go1", "Allegro"]
    # Multi-model uses full env range (32-4096)
    multimodel_envs = [n for n in NUM_ENVS_LIST if n >= 32]
    results = {"num_envs": multimodel_envs}

    for robot in robots:
        print(f"\n--- {robot} ---")
        model = load_model(robot)
        single_data = []
        multi_data = []

        for nenv in multimodel_envs:
            states = get_random_state(model, nenv)

            nstep = args.nstep

            # Single shared model
            pool_single = BatchEnvPool(model, nbatch=nenv, nthread=nthread)
            t = bench_time(lambda: pool_single.step(states, nstep=nstep),
                           warmup=WARMUP_FAST, repeat=REPEAT_FAST)
            single_data.append(nenv * nstep / t)

            # Multi-model (nbatch independent copies)
            model_seq = [mujoco.MjModel.from_xml_path(_get_mp_path(robot))
                         for _ in range(nenv)]
            pool_multi = BatchEnvPool(model_seq, nbatch=nenv, nthread=nthread)
            t = bench_time(lambda: pool_multi.step(states, nstep=nstep),
                           warmup=WARMUP_FAST, repeat=REPEAT_FAST)
            multi_data.append(nenv * nstep / t)

            del pool_single, pool_multi
            print(f"  nenv={nenv:5d}  single={single_data[-1]:.0f}  multi={multi_data[-1]:.0f} steps/s")

        results[robot] = {"single": single_data, "multi": multi_data}

    return results


# ===================================================================
# Benchmark 3: Reset (full + partial)
# ===================================================================
def bench_reset(args):
    print("\n" + "=" * 60)
    print("Benchmark 3: Reset performance")
    print("=" * 60)

    nthread = get_nthread(args)
    robots = ["Go1"]
    results = {
        "full": {"num_envs": NUM_ENVS_LIST},
        "partial": {
            "num_envs": PARTIAL_RESET_NUM_ENVS,
            "fractions": RESET_FRACTIONS,
        },
    }

    for robot in robots:
        print(f"\n--- {robot} (Full Reset) ---")
        model = load_model(robot)
        model_path = _get_mp_path(robot)
        loop_data, mp_data, cpp_data = [], [], []

        for nenv in NUM_ENVS_LIST:
            states = get_random_state(model, nenv)
            env_ids = np.arange(nenv, dtype=np.int32)

            # Python for-loop
            t = bench_time(lambda: python_loop_reset(model, states, env_ids),
                           warmup=WARMUP_SLOW, repeat=REPEAT_SLOW)
            loop_data.append(t)

            # Python multiprocessing
            nw = min(nenv, get_nthread(args))
            t = bench_time(lambda: python_mp_reset(model_path, states, env_ids, nw),
                           warmup=WARMUP_SLOW, repeat=REPEAT_SLOW)
            mp_data.append(t)

            # MuJoCoUni C++
            if HAS_BATCH_ENV:
                pool = BatchEnvPool(model, nbatch=nenv, nthread=nthread)
                t = bench_time(lambda: pool.reset(env_ids, states),
                               warmup=WARMUP_FAST, repeat=REPEAT_FAST)
                cpp_data.append(t)
                del pool
            else:
                cpp_data.append(None)

            print(f"  nenv={nenv:5d}  loop={loop_data[-1]:.4f}s  mp={mp_data[-1]:.4f}s  cpp={cpp_data[-1]}")

        results["full"][robot] = {
            "python-loop": loop_data,
            "python-mp": mp_data,
            "mujocouni-cpp": cpp_data,
        }

    # Partial reset
    for robot in robots:
        print(f"\n--- {robot} (Partial Reset) ---")
        model = load_model(robot)
        model_path = _get_mp_path(robot)
        nenv = PARTIAL_RESET_NUM_ENVS
        states = get_random_state(model, nenv)
        loop_data, mp_data, cpp_data = [], [], []

        if HAS_BATCH_ENV:
            pool = BatchEnvPool(model, nbatch=nenv, nthread=nthread)

        for frac in RESET_FRACTIONS:
            n_reset = int(nenv * frac)
            env_ids = np.arange(n_reset, dtype=np.int32)
            reset_states = states[:n_reset]

            # Python for-loop
            t = bench_time(lambda: python_loop_reset(model, reset_states, env_ids),
                           warmup=WARMUP_SLOW, repeat=REPEAT_SLOW)
            loop_data.append(t)

            # Python multiprocessing
            nw = min(n_reset, get_nthread(args))
            t = bench_time(lambda: python_mp_reset(model_path, reset_states, env_ids, nw),
                           warmup=WARMUP_SLOW, repeat=REPEAT_SLOW)
            mp_data.append(t)

            # MuJoCoUni C++
            if HAS_BATCH_ENV:
                t = bench_time(lambda: pool.reset(env_ids, reset_states),
                               warmup=WARMUP_FAST, repeat=REPEAT_FAST)
                cpp_data.append(t)
            else:
                cpp_data.append(None)

            print(f"  frac={frac:.0%}  loop={loop_data[-1]:.4f}s  mp={mp_data[-1]:.4f}s  cpp={cpp_data[-1]}")

        if HAS_BATCH_ENV:
            del pool

        results["partial"][robot] = {
            "python-loop": loop_data,
            "python-mp": mp_data,
            "mujocouni-cpp": cpp_data,
        }

    return results


# ===================================================================
# Benchmark 4: compute_site_jacobians (Franka)
# ===================================================================
def bench_jacobian(args):
    print("\n" + "=" * 60)
    print("Benchmark 4: compute_site_jacobians (Franka)")
    print("=" * 60)

    nthread = get_nthread(args)
    model = load_model("Franka")
    model_path = _get_mp_path("Franka")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    print(f"  site_id={site_id}, nv={model.nv}")

    results = {"num_envs": NUM_ENVS_LIST, "Franka": {}}
    loop_data, mp_data, cpp_data = [], [], []

    for nenv in NUM_ENVS_LIST:
        states = get_random_state(model, nenv)

        # Python for-loop
        t = bench_time(lambda: python_loop_jacobian(model, states, site_id),
                       warmup=WARMUP_SLOW, repeat=REPEAT_SLOW)
        loop_data.append(t)

        # Python multiprocessing
        nw = min(nenv, get_nthread(args))
        t = bench_time(lambda: python_mp_jacobian(model_path, states, site_id, nw),
                       warmup=WARMUP_SLOW, repeat=REPEAT_SLOW)
        mp_data.append(t)

        # MuJoCoUni C++
        if HAS_BATCH_ENV:
            pool = BatchEnvPool(model, nbatch=nenv, nthread=nthread)
            t = bench_time(lambda: pool.compute_site_jacobians(states, site_id),
                           warmup=WARMUP_FAST, repeat=REPEAT_FAST)
            cpp_data.append(t)
            del pool
        else:
            cpp_data.append(None)

        print(f"  nenv={nenv:5d}  loop={loop_data[-1]:.4f}s  mp={mp_data[-1]:.4f}s  cpp={cpp_data[-1]}")

    results["Franka"] = {
        "python-loop": loop_data,
        "python-mp": mp_data,
        "mujocouni-cpp": cpp_data,
    }
    return results


# ===================================================================
# Benchmark 5: sample_hfield_height
# ===================================================================
def bench_hfield(args):
    print("\n" + "=" * 60)
    print("Benchmark 5: sample_hfield_height")
    print("=" * 60)

    nthread = get_nthread(args)
    model = load_model("Terrain")
    model_path = _get_mp_path("Terrain")

    # Find hfield geom and frame body
    hfield_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    frame_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_base")
    print(f"  hfield_geom_id={hfield_geom_id}, frame_body_id={frame_body_id}")

    # Sample offsets: 4x4 grid around robot
    offsets = np.array([[x, y] for x in np.linspace(-1, 1, 4)
                                for y in np.linspace(-1, 1, 4)], dtype=np.float64)
    print(f"  npoint={offsets.shape[0]}")

    results = {"num_envs": NUM_ENVS_LIST, "Hfield": {}}
    loop_data, mp_data, cpp_data = [], [], []
    d_template = mujoco.MjData(model)

    for nenv in NUM_ENVS_LIST:
        states = get_random_state(model, nenv)

        # Python for-loop
        t = bench_time(
            lambda: python_loop_hfield(model, d_template, states, hfield_geom_id, offsets, frame_body_id),
            warmup=WARMUP_SLOW, repeat=REPEAT_SLOW)
        loop_data.append(t)

        # Python multiprocessing (real per-env parallelism)
        nw = min(nenv, get_nthread(args))
        t = bench_time(
            lambda: python_mp_hfield(model_path, states, hfield_geom_id, offsets, frame_body_id, nw),
            warmup=WARMUP_SLOW, repeat=REPEAT_SLOW)
        mp_data.append(t)

        # MuJoCoUni C++
        if HAS_BATCH_ENV:
            pool = BatchEnvPool(model, nbatch=nenv, nthread=nthread)
            t = bench_time(
                lambda: pool.sample_hfield_height(
                    states, hfield_geom_id, offsets, frame_body_id),
                warmup=WARMUP_FAST, repeat=REPEAT_FAST)
            cpp_data.append(t)
            del pool
        else:
            cpp_data.append(None)

        print(f"  nenv={nenv:5d}  loop={loop_data[-1]:.4f}s  mp={mp_data[-1]:.4f}s  cpp={cpp_data[-1]}")

    results["Hfield"] = {
        "python-loop": loop_data,
        "python-mp": mp_data,
        "mujocouni-cpp": cpp_data,
    }
    return results


# ===================================================================
# Main
# ===================================================================
BENCH_MAP = {
    1: ("Step / Forward throughput",       bench_step_forward),
    2: ("Multi-model comparison",          bench_multimodel),
    3: ("Reset (full + partial)",          bench_reset),
    4: ("compute_site_jacobians (Franka)", bench_jacobian),
    5: ("sample_hfield_height",            bench_hfield),
}


def main():
    parser = argparse.ArgumentParser(description="MuJoCoUni benchmark runner")
    parser.add_argument("--bench", nargs="*", type=int, default=None,
                        help="Benchmark numbers to run (default: all)")
    parser.add_argument("--nthread", type=int, default=None,
                        help=f"Thread count (default: min(cpu_count, 16) = {min(cpu_count(), 16)})")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--repeat", type=int, default=REPEAT)
    parser.add_argument("--nstep", type=int, default=NSTEP)
    parser.add_argument("--fwd-chunk-size", type=int, default=4,
                        help="Thread-pool chunk size for forward() (default: 4)")
    parser.add_argument("--output", type=str, default=RESULTS_PATH)
    args = parser.parse_args()

    # Allow overriding nstep from command line

    # Print hardware info
    print("=" * 60)
    print("Hardware & Software")
    print("=" * 60)
    import platform
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"  NumPy:     {np.__version__}")
    print(f"  MuJoCo:    {mujoco.__version__}")
    print(f"  BatchEnv:  {'yes' if HAS_BATCH_ENV else 'no'}")
    print(f"  CPU:       {platform.processor() or platform.machine()}")
    print(f"  Cores:     {cpu_count()}")
    print(f"  nthread:   {get_nthread(args)}")
    print(f"  Platform:  {platform.platform()}")
    print(f"  nstep:     {args.nstep}")
    print(f"  warmup:    {args.warmup}")
    print(f"  repeat:    {args.repeat}")

    # Verify models
    print("\nVerifying models...")
    for name, path in MODEL_PATHS.items():
        try:
            m = mujoco.MjModel.from_xml_path(path)
            print(f"  {name:10s}  nq={m.nq:3d}  nv={m.nv:3d}  nu={m.nu:3d}  OK")
        except Exception as e:
            print(f"  {name:10s}  FAIL: {e}")

    # Run benchmarks
    benches = args.bench if args.bench else sorted(BENCH_MAP.keys())
    all_results = {
        "meta": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "batch_env": HAS_BATCH_ENV,
            "cpu": platform.processor() or platform.machine(),
            "cores": cpu_count(),
            "nthread": get_nthread(args),
            "platform": platform.platform(),
            "nstep": args.nstep,
            "fwd_chunk_size": args.fwd_chunk_size,
            "warmup": args.warmup,
            "repeat": args.repeat,
        }
    }

    for i in benches:
        name, func = BENCH_MAP[i]
        print(f"\n{'='*60}\nRunning benchmark {i}: {name}\n{'='*60}")
        all_results[f"bench{i}"] = func(args)

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
