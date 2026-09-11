# MuJoCoUni Benchmark Report

**Date**: 2026-09-11
**Repository**: `mujoco_uni_bench`
**Runtimes**: `mujoco_uni_runtime` 0.5.0 (`mujoco_uni.batch_env.BatchEnvPool`), `mjbatch` 0.1.0+git.6a176de (`feat/jac-site-sample-hfield`, adding the batched query interfaces `jac_site` / `sample_hfield`; see [unilabsim/mjbatch#1](https://github.com/unilabsim/mjbatch/pull/1))

[中文版](benchmark_report.md)

## Overview

This report evaluates the batched physics simulation capabilities of **MuJoCoUni**. The main subject is `BatchEnvPool` — a high-performance batched environment interface backed by a C++ thread pool. Baselines include two common Python implementations, plus **mjbatch**, a C++ batched executor also built on stock MuJoCo:

- **Python for-loop**: calls `mj_step` / `mj_forward` etc. sequentially per environment in a single process;
- **Python multiprocessing**: distributes environments across processes with `ProcessPoolExecutor`;
- **mjbatch** (`Batch`): a nanobind-bound C++ thread-pool batched executor supporting per-sim model-field expansion; reset is `mj_resetData` + pending field writes + `mj_forward`; query operations are exposed as the native batched interfaces `jac_site` / `sample_hfield`;
- **mjbatch (NumPy)**: the best achievable with stock mjbatch (no query interfaces) — after a `forward()`, results are assembled on the host with vectorized NumPy from bound derived fields (`cdof` / `subtree_com` / `site_xpos`, `geom_xpos` / `geom_xmat` / `xpos`).

Five typical RL/robotics workloads are covered: batched stepping and forward dynamics, multi-model stepping (one model instance per environment), environment reset (full and partial), site Jacobian computation, and height-field sampling. mjbatch participates in benchmarks 1, 3, 4 and 5 (throughput, reset, Jacobian, height field); benchmarks 4 and 5 additionally compare the native interfaces against the stock NumPy implementation.

## Test Environment

| Item | Configuration |
|------|---------------|
| CPU | AMD Ryzen 9 9950X3D (16 cores / 32 threads) |
| Threads | 16 (`--nthread` default) |
| OS | Linux 7.0.0-30-generic (x86_64, glibc 2.39) |
| Python | 3.12.3 |
| NumPy | 2.5.3 |
| MuJoCo | 3.11.0 |
| MuJoCoUni runtime | mujoco_uni_runtime 0.5.0 |
| mjbatch | 0.1.0+git.6a176de (feat/jac-site-sample-hfield) |

## Methodology

- **Environment counts**: 32, 64, 128, 256, 512, 1024, 2048, 4096 (each environment starts from a valid state with small random perturbations);
- **Steps per call**: `nstep = 50`;
- **Timing**: C++ fast paths use warmup=5 / repeat=50 and report the mean; Python baselines are 2–3 orders of magnitude slower, so warmup=2 / repeat=3;
- **Forward scheduling**: thread-pool `chunk_size = 4` (mjbatch has no equivalent parameter);
- **Models**: Unitree Go1 (18-DoF quadruped), Wonik Allegro Hand (16-DoF dexterous hand), Franka Panda (9-DoF arm), CMU Humanoid (56-DoF humanoid), and a stairs height-field terrain.
- **mjbatch state injection**: mjbatch's `bind("state")` exposes opaque `mjSTATE_INTEGRATION` rows (including warmstart), incompatible with the shared `mjSTATE_FULLPHYSICS` state arrays, so states are injected field by field via `bind("time"/"qpos"/"qvel"/"act")`. Because mjbatch performs change detection on bound input fields relative to the last write (and reset clobbers the sim state back to defaults), re-writing identical states would be silently skipped; the injector therefore alternates between two state copies offset by 1e-12, forcing a full copy on every call — matching the per-call state copy the other implementations always perform. Both mjbatch arms in benchmarks 4 and 5 include state injection in the timed region.
- **mjbatch query-interface methodology**: the native `jac_site` runs only `mj_kinematics` + `mj_comPos`, and `sample_hfield` runs only `mj_kinematics` — neither executes a full `mj_forward`. The stock NumPy arm must instead call `forward()` and then assemble results from derived fields (mjbatch has no kinematics-only call); that is exactly the point of the comparison. Derived fields are bound once before timing starts (mjbatch binding semantics: a field is copied out on the first call after it is bound).
- **Known methodological difference**: mjbatch's reset ends with a full `mj_forward`, while MuJoCoUni's reset uses `mj_forwardSkip`.

---

## 1. Batched Step / Forward Throughput

Step (steps/s) and forward-dynamics (forwards/s) throughput of the four robot models at 32–4096 environments (solid lines: MuJoCoUni, dashed lines: mjbatch):

![Step / Forward Throughput](figures/fig_step_forward.png)

| Model | Implementation | Step (32 envs) | Step (4096 envs) | Forward (32 envs) | Forward (4096 envs) |
|-------|----------------|---------------:|-----------------:|------------------:|--------------------:|
| Go1 | MuJoCoUni | 1.26M steps/s | 1.71M steps/s | 0.59M fwd/s | 1.81M fwd/s |
| Go1 | mjbatch | 1.30M steps/s | 1.90M steps/s | 0.78M fwd/s | 1.99M fwd/s |
| Allegro | MuJoCoUni | 2.08M steps/s | 2.87M steps/s | 0.77M fwd/s | 2.46M fwd/s |
| Allegro | mjbatch | 2.11M steps/s | 3.17M steps/s | 0.95M fwd/s | 3.12M fwd/s |
| Franka | MuJoCoUni | 0.50M steps/s | 0.68M steps/s | 0.19M fwd/s | 0.51M fwd/s |
| Franka | mjbatch | 0.48M steps/s | 0.74M steps/s | 0.35M fwd/s | 0.52M fwd/s |
| Humanoid | MuJoCoUni | 0.39M steps/s | 0.53M steps/s | 0.20M fwd/s | 0.52M fwd/s |
| Humanoid | mjbatch | 0.41M steps/s | 0.58M steps/s | 0.33M fwd/s | 0.54M fwd/s |

mjbatch vs MuJoCoUni ratios (4096 envs): Go1 step 1.11× / fwd 1.10×; Allegro step 1.10× / fwd 1.27×; Franka step 1.08× / fwd 1.04×; Humanoid step 1.09× / fwd 1.03×.

Throughput grows with the environment count and saturates around 256–1024 envs, where all 16 threads are fully occupied. Structurally simpler models (Allegro, Go1) reach about 2–3M steps/s; the higher-DoF Humanoid still exceeds 0.5M steps/s. The two C++ implementations are in the same league overall: mjbatch is consistently slightly ahead on every model (step +8–11%, forward +3–27%), most visibly on small-scale (32 envs) forwards (Go1 0.78M vs 0.59M, Franka 0.35M vs 0.19M), indicating lower fixed scheduling overhead per call.

Note: the MuJoCoUni and mjbatch datasets were collected in different sessions (same machine, same configuration), not as a strictly interleaved A/B; differences of a few percent may include machine-state drift. A conclusive comparison should rerun `mujoco-uni-bench --bench 1 --impl batch_env mjbatch` interleaved in one process.

![Step / Forward Throughput (bar)](figures/fig_step_forward_bar.png)

## 2. Multi-Model Stepping (Single Model vs Model Variants)

`BatchEnvPool` supports two construction modes: all environments share one `MjModel` (single), or each environment receives an independent model instance (model variants, suitable for domain randomization). The stepping overhead of the two is compared below:

![Multi-Model Step Throughput](figures/fig_multimodel.png)

| Model | single (4096 envs) | variants (4096 envs) | variants / single |
|-------|-------------------:|---------------------:|------------------:|
| Go1 | 1.77M steps/s | 1.71M steps/s | 96.6% |
| Allegro | 2.92M steps/s | 2.72M steps/s | 93.2% |

The throughput loss from one model instance per environment is only about 3–7%, showing that the multi-model path introduces no significant scheduling or caching overhead.

![Multi-Model Step Throughput (bar)](figures/fig_multimodel_bar.png)

## 3. Environment Reset (Full + Partial)

Reset latency for the Go1 model (milliseconds, lower is better). Left: full reset vs environment count; right: partial resets of varying fractions within 4096 environments:

![Reset Performance](figures/fig_reset.png)

**Full reset**:

| Method | 32 envs | 4096 envs | vs MuJoCoUni (4096 envs) |
|--------|--------:|----------:|-------------------------:|
| Python for-loop | 0.59 ms | 39.2 ms | 16.6× slower |
| Python multiprocessing | 296 ms | 15,100 ms | 6,400× slower |
| **MuJoCoUni** | 0.116 ms | **2.36 ms** | — |
| mjbatch | **0.044 ms** | 2.48 ms | 1.05× slower |

**Partial reset (N=4096)**:

| Method | 5% envs | 30% envs | 50% envs | 90% envs |
|--------|--------:|---------:|---------:|---------:|
| Python for-loop | 2.21 ms | — | — | 34.9 ms |
| Python multiprocessing | 955 ms | — | — | 13,600 ms |
| **MuJoCoUni** | 0.75 ms | 0.73 ms | 1.14 ms | **1.97 ms** |
| mjbatch | **0.16 ms** | 0.78 ms | 1.24 ms | 2.16 ms |

The multiprocessing baseline is dominated by process startup and serialization, needing nearly a second even to reset a handful of environments. Both C++ implementations scale roughly linearly with the reset fraction and show complementary regimes: **mjbatch is faster for small-scale / low-fraction resets** (full reset at 32 envs: 0.044 ms vs 0.116 ms; 5% partial: 0.16 ms vs 0.75 ms), thanks to lower fixed per-call overhead; **MuJoCoUni is slightly faster for large-scale resets** (full 4096: 2.36 ms vs 2.48 ms; 90% partial: 1.97 ms vs 2.16 ms, about 5–10%), consistent with its fused sparse reset kernel plus `mj_forwardSkip` (mjbatch ends with a full `mj_forward`). For the most common RL training pattern — resetting only a few terminated environments per step (5–10%) — mjbatch actually has the lower latency.

## 4. Site Jacobian Computation (Franka)

Batched position/rotation Jacobians of the Franka arm's `end_effector` site (`compute_site_jacobians`; mjbatch arms include state injection):

![Site Jacobian Computation](figures/fig_jacobian.png)

| Method | 32 envs | 4096 envs | vs MuJoCoUni (4096 envs) |
|--------|--------:|----------:|-------------------------:|
| Python for-loop | 0.35 ms | 7.9 ms | 14× slower |
| Python multiprocessing | 333 ms | 13,800 ms | 24,800× slower |
| **MuJoCoUni** | 0.12 ms | **0.56 ms** | — |
| **mjbatch (native `jac_site`)** | **0.031 ms** | 0.35 ms | **1.6× faster** |
| mjbatch (stock NumPy assembly) | 0.13 ms | 8.6 ms | 15× slower |

Both C++ batched interfaces (MuJoCoUni and the native mjbatch op) are far ahead of the Python baselines; the native `jac_site` runs only kinematics + comPos, has lower fixed overhead, and stays slightly ahead of MuJoCoUni across the whole range. Stock mjbatch with NumPy assembly, even fully vectorized, is dragged down by the full `mj_forward`, the wide derived-field copies, and host-side assembly — at 4096 envs it is slower than even the Python for-loop baseline, showing that query operations like this are only worthwhile as native batched kernels. Numerically, both mjbatch arms agree with the serial `mj_jacSite` reference bit-for-bit (max err 0.0).

![Site Jacobian Computation (bar)](figures/fig_jacobian_bar.png)

## 5. Height-Field Sampling (Stairs Terrain)

On the stairs-terrain model, 16 sample points in a 4×4 grid around the robot base (bilinear interpolation, `sample_hfield_height`; mjbatch arms include state injection):

![Height-Field Sampling](figures/fig_hfield.png)

| Method | 32 envs | 4096 envs | vs MuJoCoUni (4096 envs) |
|--------|--------:|----------:|-------------------------:|
| Python for-loop | 2.41 ms | 264 ms | 475× slower |
| Python multiprocessing | 152 ms | 446 ms | 802× slower |
| **MuJoCoUni** | 0.12 ms | **0.56 ms** | — |
| **mjbatch (native `sample_hfield`)** | **0.033 ms** | **0.27 ms** | **2.0× faster** |
| mjbatch (stock NumPy assembly) | 0.090 ms | 3.7 ms | 6.7× slower |

Pure-Python bilinear interpolation needs 2.4 ms for just 32 environments, while MuJoCoUni handles 4096 environments in 0.56 ms; the native mjbatch `sample_hfield` runs kinematics plus in-C++ interpolation and finishes the same scale in 0.27 ms. The stock NumPy arm is about 70× faster than the for-loop (vectorized interpolation amortizes the Python overhead), but the full `mj_forward` floor keeps it an order of magnitude behind the native op. Numerically, both mjbatch arms agree with the serial bilinear reference (native max err 0.0, NumPy arm ~1e-15).

![Height-Field Sampling (bar)](figures/fig_hfield_bar.png)

---

## Conclusions

1. **Throughput**: batched stepping reaches 2–3M steps/s on simple to moderately complex models (16 threads) and saturates after 256–1024 envs; mjbatch and MuJoCoUni are in the same league, with mjbatch consistently slightly ahead on all four models (step ~+10%);
2. **Multi-model**: the overhead of one model instance per environment is only 3–7%, safe to use for domain randomization;
3. **Latency-sensitive operations** (reset, Jacobian, height-field sampling): MuJoCoUni is 1–3 orders of magnitude faster than the Python for-loop and 3–4 orders faster than Python multiprocessing — the process management and data serialization overhead of multiprocessing simply cannot be amortized on fine-grained batched operations. On reset, mjbatch and MuJoCoUni occupy complementary regimes: mjbatch has lower fixed overhead for low-fraction/small-scale resets, while MuJoCoUni's fused kernel is slightly better for full resets (5–10%);
4. **Batched query interfaces**: mjbatch's new native `jac_site` / `sample_hfield` make Jacobians and height-field sampling first-class thread-pool operations (kinematics-level, no full forward needed), and both beat MuJoCoUni's counterparts by 1.6–2.0× at 4096 envs; meanwhile "stock mjbatch + NumPy assembly" is bounded by the full `mj_forward` and derived-field copies, running an order of magnitude behind the native ops — keeping query operations on the C++ side is an order-of-magnitude difference;
5. For the high-frequency reset and observation-computation paths in RL training, C++ batched interfaces eliminate the otherwise dominant Python overhead;
6. Engineering differences: mjbatch state injection requires care with the opaque `mjSTATE_INTEGRATION` row layout and the copy-on-change semantics, and derived fields must be bound before the physics call; MuJoCoUni's reset natively accepts FULLPHYSICS state arrays, a more direct interface.

## Reproduction

```bash
# Install (with the MuJoCoUni runtime; the mjbatch arms additionally need --extra mjbatch)
uv sync --extra mujoco-uni-runtime
uv sync --extra mujoco-uni-runtime --extra mjbatch

# Run all benchmarks (~12 minutes, results go to benchmark_results.json)
mujoco-uni-bench

# mjbatch throughput + reset (results go to benchmark_results_mjbatch.json)
mujoco-uni-bench --bench 1 3 --impl mjbatch --output benchmark_results_mjbatch.json

# mjbatch Jacobian + height field (native ops and stock NumPy arms)
mujoco-uni-bench --bench 4 5 --impl mjbatch --output benchmark_results_mjbatch_queries.json

# Interleaved same-process comparison of the two C++ implementations
mujoco-uni-bench --bench 1 --impl batch_env mjbatch

# Generate all figures (default output: figures/)
mujoco-uni-plot
```

Individual benchmarks can be selected with `--bench N` (1=Step/Forward, 2=multi-model, 3=reset, 4=Jacobian, 5=height field); figures with `--fig N`. See `mujoco-uni-bench --help` and `mujoco-uni-plot --help` for more options.

Data files: `benchmark_results.json` (Python baselines + MuJoCoUni, all 5 benchmarks), `benchmark_results_mjbatch.json` (mjbatch, benchmarks 1, 3, 4, 5), `benchmark_results_mjbatch_queries.json` (standalone run record of mjbatch benchmarks 4 and 5), `benchmark_results_combined.json` (everything merged; the chart data source for this report).

## Model Sources

| Model | Source | License |
|-------|--------|---------|
| Unitree Go1 | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Apache 2.0 |
| Wonik Allegro | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Apache 2.0 |
| Franka Panda | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Apache 2.0 |
| CMU Humanoid | [dm_control](https://github.com/google-deepmind/dm_control) | Apache 2.0 |
| Stairs terrain | [MuJoCo-LiDAR](https://github.com/discoverse-dev/MuJoCo-LiDAR) | MIT |
