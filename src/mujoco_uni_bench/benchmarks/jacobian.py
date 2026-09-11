from ..constants import (
    NUM_ENVS_LIST, WARMUP_FAST, REPEAT_FAST, WARMUP_SLOW, REPEAT_SLOW, get_nthread,
)
from ..model_utils import (
    mujoco, HAS_BATCH_ENV, HAS_MJBATCH, BatchEnvPool, mjbatch,
    load_model, _get_mp_path, get_random_state, mjbatch_state_loader,
)
from ..timing import bench_time
from ..baselines import python_loop_jacobian, python_mp_jacobian, mjbatch_numpy_jacobian


# Arm keys in the results dict, grouped by the --impl selector that enables
# them. "mjbatch" is the native jac_site op; "mjbatch-numpy" is what stock
# mjbatch can do: a forward() call plus NumPy assembly over bound fields.
_IMPL_ARMS = {
    "python": ["python-loop", "python-mp"],
    "batch_env": ["mujocouni-cpp"],
    "mjbatch": ["mjbatch", "mjbatch-numpy"],
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
    site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
    arms = _selected_arms(args)
    print(f"  site_id={site_id}, nv={model.nv}, arms={arms}")

    results = {"num_envs": NUM_ENVS_LIST, "arms": arms, "Franka": {}}
    data = {arm: [] for arm in arms}

    for nenv in NUM_ENVS_LIST:
        states = get_random_state(model, nenv)

        if "python-loop" in arms:
            t = bench_time(lambda: python_loop_jacobian(model, states, site_id),
                           warmup=WARMUP_SLOW, repeat=REPEAT_SLOW)
            data["python-loop"].append(t)

        if "python-mp" in arms:
            nw = min(nenv, get_nthread(args))
            t = bench_time(lambda: python_mp_jacobian(model_path, states, site_id, nw),
                           warmup=WARMUP_SLOW, repeat=REPEAT_SLOW)
            data["python-mp"].append(t)

        if "mujocouni-cpp" in arms:
            pool = BatchEnvPool(model, nbatch=nenv, nthread=nthread)
            t = bench_time(lambda: pool.compute_site_jacobians(states, site_id),
                           warmup=WARMUP_FAST, repeat=REPEAT_FAST)
            data["mujocouni-cpp"].append(t)
            del pool

        if "mjbatch" in arms:
            batch = mjbatch.Batch(model, nenv, num_threads=nthread)
            load = mjbatch_state_loader(batch, model, states)

            def native_once():
                load()
                batch.jac_site(site_name)

            data["mjbatch"].append(
                bench_time(native_once, warmup=WARMUP_FAST, repeat=REPEAT_FAST))
            del batch

        if "mjbatch-numpy" in arms:
            batch = mjbatch.Batch(model, nenv, num_threads=nthread)
            load = mjbatch_state_loader(batch, model, states)
            # Bind before any call: bound derived fields are copied out after
            # each call, so binding inside the timed region would read stale
            # values and add per-call binding overhead.
            cdof = batch.bind("cdof")
            subtree_com = batch.bind("subtree_com")
            site_xpos = batch.bind("site_xpos")

            def numpy_once():
                load()
                batch.forward()
                mjbatch_numpy_jacobian(model, site_id, cdof, subtree_com, site_xpos)

            data["mjbatch-numpy"].append(
                bench_time(numpy_once, warmup=WARMUP_FAST, repeat=REPEAT_FAST))
            del batch

        line = f"  nenv={nenv:5d}"
        for arm in arms:
            line += f"  {arm}={data[arm][-1]:.4f}s"
        print(line)

    results["Franka"] = data
    return results
