from ..constants import NUM_ENVS_LIST, WARMUP_FAST, REPEAT_FAST, get_nthread
from ..model_utils import (
    HAS_BATCH_ENV,
    HAS_MJBATCH,
    BatchEnvPool,
    mjbatch,
    load_model,
    get_random_state,
    mjbatch_state_loader,
)
from ..timing import bench_time
from ..baselines import python_loop_step, python_loop_forward


# ===================================================================
# Per-implementation runners: (steps_per_sec, forwards_per_sec)
# ===================================================================
def _run_batch_env(model, states, nenv, nstep, nthread, fwd_chunk_size, args):
    pool = BatchEnvPool(model, nbatch=nenv, nthread=nthread)

    t = bench_time(lambda: pool.step(states, nstep=nstep),
                   warmup=WARMUP_FAST, repeat=REPEAT_FAST)
    steps_per_sec = nenv * nstep / t

    t = bench_time(lambda: pool.forward(states, chunk_size=fwd_chunk_size),
                   warmup=WARMUP_FAST, repeat=REPEAT_FAST)
    fwd_per_sec = nenv / t

    del pool
    return steps_per_sec, fwd_per_sec


def _run_mjbatch(model, states, nenv, nstep, nthread, fwd_chunk_size, args):
    batch = mjbatch.Batch(model, nenv, num_threads=nthread)
    load_states = mjbatch_state_loader(batch, model, states)

    def step_once():
        load_states()
        batch.step(nstep=nstep)

    def forward_once():
        load_states()
        batch.forward()

    t = bench_time(step_once, warmup=WARMUP_FAST, repeat=REPEAT_FAST)
    steps_per_sec = nenv * nstep / t

    t = bench_time(forward_once, warmup=WARMUP_FAST, repeat=REPEAT_FAST)
    fwd_per_sec = nenv / t

    return steps_per_sec, fwd_per_sec


def _run_python(model, states, nenv, nstep, nthread, fwd_chunk_size, args):
    t = bench_time(lambda: python_loop_step(model, states, nstep),
                   warmup=args.warmup, repeat=args.repeat)
    steps_per_sec = nenv * nstep / t
    t = bench_time(lambda: python_loop_forward(model, states),
                   warmup=args.warmup, repeat=args.repeat)
    fwd_per_sec = nenv / t
    return steps_per_sec, fwd_per_sec


_IMPL_RUNNERS = {
    "batch_env": _run_batch_env,
    "mjbatch": _run_mjbatch,
    "python": _run_python,
}

_IMPL_AVAILABLE = {
    "batch_env": lambda: HAS_BATCH_ENV,
    "mjbatch": lambda: HAS_MJBATCH,
    "python": lambda: True,
}


# ===================================================================
# Benchmark 1: Step / Forward throughput
# ===================================================================
def bench_step_forward(args):
    print("\n" + "=" * 60)
    print("Benchmark 1: Step / Forward throughput")
    print("=" * 60)

    nthread = get_nthread(args)
    robots = ["Go1", "Allegro", "Franka", "Humanoid"]
    impls = args.impl if args.impl else (["batch_env"] if HAS_BATCH_ENV else ["python"])
    for impl in impls:
        if not _IMPL_AVAILABLE[impl]():
            raise RuntimeError(f"implementation {impl!r} requested but not available")

    results = {"num_envs": NUM_ENVS_LIST, "impls": impls}
    for impl in impls:
        suffix = "" if impl == "batch_env" else f"_{impl}"
        results[f"step{suffix}"] = {}
        results[f"forward{suffix}"] = {}

    for robot in robots:
        print(f"\n--- {robot} ---")
        model = load_model(robot)

        for nenv in NUM_ENVS_LIST:
            states = get_random_state(model, nenv)
            line = f"  nenv={nenv:5d}"

            for impl in impls:
                runner = _IMPL_RUNNERS[impl]
                steps_per_sec, fwd_per_sec = runner(
                    model, states, nenv, args.nstep, nthread, args.fwd_chunk_size, args
                )
                suffix = "" if impl == "batch_env" else f"_{impl}"
                results[f"step{suffix}"][robot] = (
                    results[f"step{suffix}"].get(robot, []) + [steps_per_sec]
                )
                results[f"forward{suffix}"][robot] = (
                    results[f"forward{suffix}"].get(robot, []) + [fwd_per_sec]
                )
                line += f"  [{impl}] step={steps_per_sec:.0f} steps/s fwd={fwd_per_sec:.0f} fwd/s"

            print(line)

    return results
