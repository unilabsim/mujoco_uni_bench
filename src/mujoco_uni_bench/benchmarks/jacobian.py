from ..constants import (
    NUM_ENVS_LIST, WARMUP_FAST, REPEAT_FAST, WARMUP_SLOW, REPEAT_SLOW, get_nthread,
)
from ..model_utils import mujoco, HAS_BATCH_ENV, BatchEnvPool, load_model, _get_mp_path, get_random_state
from ..timing import bench_time
from ..baselines import python_loop_jacobian, python_mp_jacobian


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
