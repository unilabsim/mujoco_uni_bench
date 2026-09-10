# MuJoCoUni Benchmark Suite

Benchmark suite comparing **MuJoCoUni** (`BatchEnvPool` C++ thread pool) against Python for-loop and Python multiprocessing baselines across five benchmark categories.

完整基准测试报告（含结果与图）见 [doc/benchmark_report.md](doc/benchmark_report.md)。

## Benchmarks

| # | Benchmark | Models | Metric |
|---|-----------|--------|--------|
| 1 | Step / Forward throughput | Go1, Allegro, Franka, Humanoid | steps/s, forwards/s |
| 2 | Multi-model comparison | Go1, Allegro | single vs model-variant overhead |
| 3 | Reset (full + partial) | Go1 | reset latency (ms) |
| 4 | Site Jacobian computation | Franka | jacobian time (ms) |
| 5 | Height-field sampling | Terrain (stairs) | hfield sample time (ms) |

## Quick Start

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install the package

```bash
# Install with the MuJoCoUni runtime (BatchEnvPool C++ fast path)
uv sync --extra mujoco-uni-runtime
# or: pip install -e ".[mujoco-uni-runtime]"

# Option B: standard mujoco (Python baselines only, no BatchEnvPool)
# uv sync --extra mujoco
# or: pip install -e ".[mujoco]"
```

### 3. Run benchmarks

```bash
# Run all benchmarks (takes ~15-20 minutes due to slow Python baselines)
mujoco-uni-bench

# Run specific benchmarks
mujoco-uni-bench --bench 1        # Step/Forward only
mujoco-uni-bench --bench 1 4      # Step/Forward + Jacobian

# Customize parameters
mujoco-uni-bench --repeat 50 --warmup 5 --nthread 16
```

### 4. Generate figures

```bash
# Generate all figures (requires benchmark_results.json from step 3)
mujoco-uni-plot

# Generate specific figures
mujoco-uni-plot --fig 1         # Step/Forward figure only
mujoco-uni-plot --fig 1 2 3 4 5 # All figures

# Custom data / output directory
mujoco-uni-plot --data path/to/results.json --outdir ./my_figures
```

Output figures are saved to `figures/` by default.

## File Structure

```
mujoco_uni_bench/
├── pyproject.toml          # Project metadata, build system, entry points
├── README.md               # This file
└── src/
    └── mujoco_uni_bench/
        ├── __init__.py     # Package version
        ├── constants.py    # Default benchmark parameters
        ├── model_utils.py  # mujoco import guard, model loading helpers
        ├── timing.py       # bench_time() timing helper
        ├── baselines.py    # Python for-loop / multiprocessing baselines
        ├── benchmarks/     # Benchmark implementations (BENCH_MAP)
        ├── run.py          # mujoco-uni-bench entry point
        ├── plotting/       # Figure generator (mujoco-uni-plot entry point)
        └── models/         # Robot model assets (package data)
            ├── unitree_go1/        # Go1 quadruped (18 DoF)
            ├── wonik_allegro/      # Allegro hand (16 DoF)
            ├── franka_emika_panda/ # Franka Panda arm (9 DoF)
            ├── humanoid/           # CMU Humanoid (56 DoF)
            └── terrain/            # Stairs height-field
```

`benchmark_results.json` (generated benchmark data) and `figures/` (generated PDF figures) are written to the current working directory and are gitignored.

## Model Sources

| Model | Source | License |
|-------|--------|---------|
| Unitree Go1 | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Apache 2.0 |
| Wonik Allegro | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Apache 2.0 |
| Franka Panda | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Apache 2.0 |
| CMU Humanoid | [dm_control](https://github.com/google-deepmind/dm_control) | Apache 2.0 |
| Stairs Terrain | [MuJoCo-LiDAR](https://github.com/discoverse-dev/MuJoCo-LiDAR) | MIT |

## Benchmark Parameters

- **Environment counts**: 32, 64, 128, 256, 512, 1024, 2048, 4096
- **C++ fast path**: warmup=5, repeat=50
- **Python baselines**: warmup=2, repeat=3 (due to significantly longer execution time)
- **Forward chunk_size**: 4 (smooths thread-pool dispatch behavior)
- **Threads**: 16 (default, configurable via `--nthread`)
