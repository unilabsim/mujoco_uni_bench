# MuJoCoUni Benchmark Suite

Benchmark suite comparing **MuJoCoUni** (`BatchEnvPool` C++ thread pool) against Python for-loop and Python multiprocessing baselines across five benchmark categories.

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

### 2. Create virtual environment and install dependencies

```bash
# mujoco-uni must be installed separately (not on PyPI)
# If you have mujoco-uni installed system-wide or in another venv,
# create the venv with access to it:
uv venv --python 3.13
source .venv/bin/activate

# Install base dependencies
uv pip install numpy matplotlib

# Install mujoco-uni (from local build or editable install)
# Option A: editable install from source
uv pip install -e /path/to/mujoco_uni/python

# Option B: if mujoco-uni is already installed, use system packages
# uv venv --system-site-packages
```

### 3. Run benchmarks

```bash
# Run all benchmarks (takes ~15-20 minutes due to slow Python baselines)
python run_benchmarks.py

# Run specific benchmarks
python run_benchmarks.py --bench 1        # Step/Forward only
python run_benchmarks.py --bench 1 4      # Step/Forward + Jacobian

# Customize parameters
python run_benchmarks.py --repeat 50 --warmup 5 --nthread 16
```

### 4. Generate figures

```bash
# Generate all figures (requires benchmark_results.json from step 3)
python plot_benchmarks.py

# Generate specific figures
python plot_benchmarks.py --fig 1         # Step/Forward figure only
python plot_benchmarks.py --fig 1 2 3 4 5 # All figures

# Custom data / output directory
python plot_benchmarks.py --data path/to/results.json --outdir ./my_figures
```

Output figures are saved to `figures/` by default.

## File Structure

```
mujoco_uni_bench/
├── pyproject.toml          # Project metadata and dependencies
├── README.md               # This file
├── run_benchmarks.py       # Benchmark runner
├── plot_benchmarks.py      # Figure generator
├── models/                 # Robot model assets
│   ├── unitree_go1/        # Go1 quadruped (18 DoF)
│   ├── wonik_allegro/      # Allegro hand (16 DoF)
│   ├── franka_emika_panda/ # Franka Panda arm (9 DoF)
│   ├── humanoid/           # CMU Humanoid (56 DoF)
│   └── terrain/            # Stairs height-field
├── benchmark_results.json  # Generated benchmark data (gitignored)
└── figures/                # Generated PDF figures (gitignored)
```

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
