from ..constants import NUM_ENVS_LIST, WARMUP_FAST, REPEAT_FAST, get_nthread
from ..model_utils import mujoco, HAS_BATCH_ENV, BatchEnvPool, load_model, _get_mp_path, get_random_state
from ..timing import bench_time


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
