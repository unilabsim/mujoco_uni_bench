from ..constants import NUM_ENVS_LIST, WARMUP_FAST, REPEAT_FAST, get_nthread
from ..model_utils import HAS_BATCH_ENV, BatchEnvPool, load_model, get_random_state
from ..timing import bench_time
from ..baselines import python_loop_step, python_loop_forward


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
