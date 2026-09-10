import numpy as np

from ..constants import (
    NUM_ENVS_LIST, RESET_FRACTIONS, PARTIAL_RESET_NUM_ENVS,
    WARMUP_FAST, REPEAT_FAST, WARMUP_SLOW, REPEAT_SLOW, get_nthread,
)
from ..model_utils import (
    HAS_BATCH_ENV, HAS_MJBATCH, BatchEnvPool, mjbatch,
    load_model, _get_mp_path, get_random_state, mjbatch_state_loader,
)
from ..timing import bench_time
from ..baselines import python_loop_reset, python_mp_reset

# Arm keys in the results dict, grouped by the --impl selector that enables
# them. Default (no --impl) runs all arms, matching the historical behavior.
_IMPL_ARMS = {
    "python": ["python-loop", "python-mp"],
    "batch_env": ["mujocouni-cpp"],
    "mjbatch": ["mjbatch"],
}

_IMPL_AVAILABLE = {
    "python": lambda: True,
    "batch_env": lambda: HAS_BATCH_ENV,
    "mjbatch": lambda: HAS_MJBATCH,
}


def _selected_arms(args):
    impls = args.impl if args.impl else ["python", "batch_env"]
    arms = []
    for impl in impls:
        if not _IMPL_AVAILABLE[impl]():
            raise RuntimeError(f"implementation {impl!r} requested but not available")
        arms.extend(_IMPL_ARMS[impl])
    return arms


def _mjbatch_reset_latency(batch, load, env_ids):
    ids = np.ascontiguousarray(env_ids, dtype=np.int64)

    def reset_once():
        load(ids)
        batch.reset(ids)

    return bench_time(reset_once, warmup=WARMUP_FAST, repeat=REPEAT_FAST)


# ===================================================================
# Benchmark 3: Reset (full + partial)
# ===================================================================
def bench_reset(args):
    print("\n" + "=" * 60)
    print("Benchmark 3: Reset performance")
    print("=" * 60)

    nthread = get_nthread(args)
    robots = ["Go1"]
    arms = _selected_arms(args)
    results = {
        "arms": arms,
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
        data = {arm: [] for arm in arms}

        for nenv in NUM_ENVS_LIST:
            states = get_random_state(model, nenv)
            env_ids = np.arange(nenv, dtype=np.int32)

            if "python-loop" in arms:
                t = bench_time(lambda: python_loop_reset(model, states, env_ids),
                               warmup=WARMUP_SLOW, repeat=REPEAT_SLOW)
                data["python-loop"].append(t)

            if "python-mp" in arms:
                nw = min(nenv, get_nthread(args))
                t = bench_time(lambda: python_mp_reset(model_path, states, env_ids, nw),
                               warmup=WARMUP_SLOW, repeat=REPEAT_SLOW)
                data["python-mp"].append(t)

            if "mujocouni-cpp" in arms:
                pool = BatchEnvPool(model, nbatch=nenv, nthread=nthread)
                t = bench_time(lambda: pool.reset(env_ids, states),
                               warmup=WARMUP_FAST, repeat=REPEAT_FAST)
                data["mujocouni-cpp"].append(t)
                del pool

            if "mjbatch" in arms:
                batch = mjbatch.Batch(model, nenv, num_threads=nthread)
                load = mjbatch_state_loader(batch, model, states)
                data["mjbatch"].append(_mjbatch_reset_latency(batch, load, env_ids))
                del batch

            line = f"  nenv={nenv:5d}"
            for arm in arms:
                line += f"  {arm}={data[arm][-1]:.4f}s"
            print(line)

        results["full"][robot] = data

    # Partial reset
    for robot in robots:
        print(f"\n--- {robot} (Partial Reset) ---")
        model = load_model(robot)
        model_path = _get_mp_path(robot)
        nenv = PARTIAL_RESET_NUM_ENVS
        states = get_random_state(model, nenv)
        data = {arm: [] for arm in arms}

        pool = BatchEnvPool(model, nbatch=nenv, nthread=nthread) \
            if "mujocouni-cpp" in arms else None
        if "mjbatch" in arms:
            batch = mjbatch.Batch(model, nenv, num_threads=nthread)
            load = mjbatch_state_loader(batch, model, states)

        for frac in RESET_FRACTIONS:
            n_reset = int(nenv * frac)
            env_ids = np.arange(n_reset, dtype=np.int32)
            reset_states = states[:n_reset]

            if "python-loop" in arms:
                t = bench_time(lambda: python_loop_reset(model, reset_states, env_ids),
                               warmup=WARMUP_SLOW, repeat=REPEAT_SLOW)
                data["python-loop"].append(t)

            if "python-mp" in arms:
                nw = min(n_reset, get_nthread(args))
                t = bench_time(lambda: python_mp_reset(model_path, reset_states, env_ids, nw),
                               warmup=WARMUP_SLOW, repeat=REPEAT_SLOW)
                data["python-mp"].append(t)

            if pool is not None:
                t = bench_time(lambda: pool.reset(env_ids, reset_states),
                               warmup=WARMUP_FAST, repeat=REPEAT_FAST)
                data["mujocouni-cpp"].append(t)

            if "mjbatch" in arms:
                data["mjbatch"].append(_mjbatch_reset_latency(batch, load, env_ids))

            line = f"  frac={frac:.0%}"
            for arm in arms:
                line += f"  {arm}={data[arm][-1]:.4f}s"
            print(line)

        del pool
        if "mjbatch" in arms:
            del batch

        results["partial"][robot] = data

    return results
