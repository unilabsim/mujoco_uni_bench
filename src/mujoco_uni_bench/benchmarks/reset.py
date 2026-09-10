import numpy as np

from ..constants import (
    NUM_ENVS_LIST, RESET_FRACTIONS, PARTIAL_RESET_NUM_ENVS,
    WARMUP_FAST, REPEAT_FAST, WARMUP_SLOW, REPEAT_SLOW, get_nthread,
)
from ..model_utils import HAS_BATCH_ENV, BatchEnvPool, load_model, _get_mp_path, get_random_state
from ..timing import bench_time
from ..baselines import python_loop_reset, python_mp_reset


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
