import numpy as np

from ..constants import (
    NUM_ENVS_LIST, WARMUP_FAST, REPEAT_FAST, WARMUP_SLOW, REPEAT_SLOW, get_nthread,
)
from ..model_utils import (
    mujoco, HAS_BATCH_ENV, HAS_MJBATCH, BatchEnvPool, mjbatch,
    load_model, _get_mp_path, get_random_state, mjbatch_state_loader,
)
from ..timing import bench_time
from ..baselines import python_loop_hfield, python_mp_hfield, mjbatch_numpy_hfield


# Arm keys in the results dict, grouped by the --impl selector that enables
# them. "mjbatch" is the native sample_hfield op; "mjbatch-numpy" is what
# stock mjbatch can do: a forward() call plus NumPy bilinear interpolation
# over bound fields.
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
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, hfield_geom_id)
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, frame_body_id)
    arms = _selected_arms(args)
    print(f"  hfield_geom_id={hfield_geom_id}, frame_body_id={frame_body_id}, arms={arms}")

    # Sample offsets: 4x4 grid around robot
    offsets = np.array([[x, y] for x in np.linspace(-1, 1, 4)
                                for y in np.linspace(-1, 1, 4)], dtype=np.float64)
    print(f"  npoint={offsets.shape[0]}")

    results = {"num_envs": NUM_ENVS_LIST, "arms": arms, "Hfield": {}}
    data = {arm: [] for arm in arms}
    d_template = mujoco.MjData(model)

    for nenv in NUM_ENVS_LIST:
        states = get_random_state(model, nenv)

        if "python-loop" in arms:
            t = bench_time(
                lambda: python_loop_hfield(model, d_template, states, hfield_geom_id, offsets, frame_body_id),
                warmup=WARMUP_SLOW, repeat=REPEAT_SLOW)
            data["python-loop"].append(t)

        if "python-mp" in arms:
            nw = min(nenv, get_nthread(args))
            t = bench_time(
                lambda: python_mp_hfield(model_path, states, hfield_geom_id, offsets, frame_body_id, nw),
                warmup=WARMUP_SLOW, repeat=REPEAT_SLOW)
            data["python-mp"].append(t)

        if "mujocouni-cpp" in arms:
            pool = BatchEnvPool(model, nbatch=nenv, nthread=nthread)
            t = bench_time(
                lambda: pool.sample_hfield_height(
                    states, hfield_geom_id, offsets, frame_body_id),
                warmup=WARMUP_FAST, repeat=REPEAT_FAST)
            data["mujocouni-cpp"].append(t)
            del pool

        if "mjbatch" in arms:
            batch = mjbatch.Batch(model, nenv, num_threads=nthread)
            load = mjbatch_state_loader(batch, model, states)

            def native_once():
                load()
                batch.sample_hfield(geom_name, body_name, offsets)

            data["mjbatch"].append(
                bench_time(native_once, warmup=WARMUP_FAST, repeat=REPEAT_FAST))
            del batch

        if "mjbatch-numpy" in arms:
            batch = mjbatch.Batch(model, nenv, num_threads=nthread)
            load = mjbatch_state_loader(batch, model, states)
            # Bind before any call: bound derived fields are copied out after
            # each call, so binding inside the timed region would read stale
            # values and add per-call binding overhead.
            geom_xpos = batch.bind("geom_xpos")
            geom_xmat = batch.bind("geom_xmat")
            xpos = batch.bind("xpos")

            def numpy_once():
                load()
                batch.forward()
                mjbatch_numpy_hfield(model, hfield_geom_id, offsets, frame_body_id,
                                     geom_xpos, geom_xmat, xpos)

            data["mjbatch-numpy"].append(
                bench_time(numpy_once, warmup=WARMUP_FAST, repeat=REPEAT_FAST))
            del batch

        line = f"  nenv={nenv:5d}"
        for arm in arms:
            line += f"  {arm}={data[arm][-1]:.4f}s"
        print(line)

    results["Hfield"] = data
    return results
