"""
Benchmark runner for the MuJoCoUni technical report.

Runs all benchmarks across representative robot models and saves results
to JSON files that the mujoco-uni-plot command can consume.

Requirements:
    pip install mujoco-uni numpy

Usage:
    mujoco-uni-bench                    # run all benchmarks
    mujoco-uni-bench --bench 1          # run only benchmark 1
    mujoco-uni-bench --bench 1 3        # run benchmarks 1 and 3
    mujoco-uni-bench --nthread 16       # override thread count

Output:  benchmark_results.json (in the current working directory)
"""

import argparse
import json
import sys
from multiprocessing import cpu_count

import numpy as np

from .constants import NSTEP, WARMUP, REPEAT, get_nthread
from .model_utils import mujoco, HAS_BATCH_ENV, MODEL_PATHS
from .benchmarks import BENCH_MAP


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="MuJoCoUni benchmark runner")
    parser.add_argument("--bench", nargs="*", type=int, default=None,
                        help="Benchmark numbers to run (default: all)")
    parser.add_argument("--nthread", type=int, default=None,
                        help=f"Thread count (default: min(cpu_count, 16) = {min(cpu_count(), 16)})")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--repeat", type=int, default=REPEAT)
    parser.add_argument("--nstep", type=int, default=NSTEP)
    parser.add_argument("--fwd-chunk-size", type=int, default=4,
                        help="Thread-pool chunk size for forward() (default: 4)")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    if mujoco is None:
        print("ERROR: mujoco is not installed")
        sys.exit(1)

    # Allow overriding nstep from command line

    # Print hardware info
    print("=" * 60)
    print("Hardware & Software")
    print("=" * 60)
    import platform
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"  NumPy:     {np.__version__}")
    print(f"  MuJoCo:    {mujoco.__version__}")
    print(f"  BatchEnv:  {'yes' if HAS_BATCH_ENV else 'no'}")
    print(f"  CPU:       {platform.processor() or platform.machine()}")
    print(f"  Cores:     {cpu_count()}")
    print(f"  nthread:   {get_nthread(args)}")
    print(f"  Platform:  {platform.platform()}")
    print(f"  nstep:     {args.nstep}")
    print(f"  warmup:    {args.warmup}")
    print(f"  repeat:    {args.repeat}")

    # Verify models
    print("\nVerifying models...")
    for name, path in MODEL_PATHS.items():
        try:
            m = mujoco.MjModel.from_xml_path(path)
            print(f"  {name:10s}  nq={m.nq:3d}  nv={m.nv:3d}  nu={m.nu:3d}  OK")
        except Exception as e:
            print(f"  {name:10s}  FAIL: {e}")

    # Run benchmarks
    benches = args.bench if args.bench else sorted(BENCH_MAP.keys())
    all_results = {
        "meta": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "batch_env": HAS_BATCH_ENV,
            "cpu": platform.processor() or platform.machine(),
            "cores": cpu_count(),
            "nthread": get_nthread(args),
            "platform": platform.platform(),
            "nstep": args.nstep,
            "fwd_chunk_size": args.fwd_chunk_size,
            "warmup": args.warmup,
            "repeat": args.repeat,
        }
    }

    for i in benches:
        name, func = BENCH_MAP[i]
        print(f"\n{'='*60}\nRunning benchmark {i}: {name}\n{'='*60}")
        all_results[f"bench{i}"] = func(args)

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
