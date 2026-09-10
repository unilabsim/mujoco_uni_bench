from multiprocessing import cpu_count

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
