import numpy as np

from ..constants import (
    NUM_ENVS_LIST, WARMUP_FAST, REPEAT_FAST, WARMUP_SLOW, REPEAT_SLOW, get_nthread,
)
from ..model_utils import mujoco, HAS_BATCH_ENV, BatchEnvPool, load_model, _get_mp_path, get_random_state
from ..timing import bench_time
from ..baselines import python_loop_hfield, python_mp_hfield


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
