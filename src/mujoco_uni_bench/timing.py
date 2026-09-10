import time

import numpy as np

from .constants import WARMUP, REPEAT


# ===================================================================
# Timing helper
# ===================================================================
def bench_time(func, warmup=WARMUP, repeat=REPEAT):
    """Time a function, returning mean time over repeat runs."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.mean(times))
