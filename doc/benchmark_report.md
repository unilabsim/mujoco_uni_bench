# MuJoCoUni 基准测试报告

**日期**：2026-09-10
**代码仓库**：`mujoco_uni_bench`
**运行时**：`mujoco_uni_runtime` 0.5.0（`mujoco_uni.batch_env.BatchEnvPool`）

## 概述

本报告评估 **MuJoCoUni** 的批量物理仿真能力。核心被测对象是 `BatchEnvPool`——一个 C++ 线程池实现的高性能批量环境接口，与两种常见 Python 基线对比：

- **Python for-loop**：在单进程中逐环境串行调用 `mj_step` / `mj_forward` 等 API；
- **Python multiprocessing**：使用 `ProcessPoolExecutor` 将环境分发到多个进程并行执行。

测试覆盖五类典型强化学习/机器人工作负载：批量步进与前向动力学、多模型（每个环境独立模型实例）步进、环境重置（全量与部分）、站点雅可比计算、高度场采样。

## 测试环境

| 项目 | 配置 |
|------|------|
| CPU | AMD Ryzen 9 9950X3D（16 核 / 32 线程） |
| 线程数 | 16（`--nthread` 默认值） |
| OS | Linux 7.0.0-30-generic (x86_64, glibc 2.39) |
| Python | 3.12.3 |
| NumPy | 2.5.3 |
| MuJoCo | 3.11.0 |
| MuJoCoUni 运行时 | mujoco_uni_runtime 0.5.0 |

## 测试方法

- **环境数量**：32, 64, 128, 256, 512, 1024, 2048, 4096（每个环境初始状态为带微小随机扰动的合法状态）；
- **每轮步数**：`nstep = 50`；
- **计时方式**：C++ 快速路径 warmup=5 / repeat=50 取均值；Python 基线因慢 2–3 个数量级，warmup=2 / repeat=3；
- **Forward 调度**：线程池 `chunk_size = 4`；
- **测试模型**：Unitree Go1（18 DoF 四足）、Wonik Allegro Hand（16 DoF 灵巧手）、Franka Panda（9 DoF 机械臂）、CMU Humanoid（56 DoF 人形）、楼梯地形高度场。

---

## 1. 批量 Step / Forward 吞吐

四种机器人模型在 32–4096 个环境下的步进（steps/s）与前向动力学（forwards/s）吞吐：

![Step / Forward Throughput](figures/fig_step_forward.png)

| 模型 | Step（32 envs） | Step（4096 envs） | Forward（32 envs） | Forward（4096 envs） |
|------|----------------:|------------------:|-------------------:|---------------------:|
| Go1 | 1.26M steps/s | 1.71M steps/s | 0.59M fwd/s | 1.81M fwd/s |
| Allegro | 2.08M steps/s | 2.87M steps/s | 0.77M fwd/s | 2.46M fwd/s |
| Franka | 0.50M steps/s | 0.68M steps/s | 0.19M fwd/s | 0.51M fwd/s |
| Humanoid | 0.39M steps/s | 0.53M steps/s | 0.20M fwd/s | 0.52M fwd/s |

吞吐随环境数增长并在约 256–1024 envs 后趋于饱和，此时 16 个线程已全部打满。结构更简单的模型（Allegro、Go1）达到约 2–3M steps/s；自由度更高的 Humanoid 仍超过 0.5M steps/s。

![Step / Forward Throughput (bar)](figures/fig_step_forward_bar.png)

## 2. 多模型步进（单模型 vs 模型变体）

`BatchEnvPool` 支持两种构造方式：所有环境共享一个 `MjModel`（single），或为每个环境传入独立的模型实例（model variants，适用于域随机化等场景）。下面对比两者的步进开销：

![Multi-Model Step Throughput](figures/fig_multimodel.png)

| 模型 | single（4096 envs） | variants（4096 envs） | variants / single |
|------|--------------------:|----------------------:|------------------:|
| Go1 | 1.77M steps/s | 1.71M steps/s | 96.6% |
| Allegro | 2.92M steps/s | 2.72M steps/s | 93.2% |

每环境独立模型实例的吞吐损失仅约 3–7%，说明多模型路径没有引入显著的调度或缓存开销。

![Multi-Model Step Throughput (bar)](figures/fig_multimodel_bar.png)

## 3. 环境重置（全量 + 部分）

Go1 模型的重置延迟（毫秒，越低越好）。左图为全量重置随环境数的变化，右图为 4096 个环境中按不同比例部分重置：

![Reset Performance](figures/fig_reset.png)

**全量重置**：

| 方法 | 32 envs | 4096 envs | 相对 MuJoCoUni（4096 envs） |
|------|--------:|----------:|---------------------------:|
| Python for-loop | 0.59 ms | 39.2 ms | 16.6× 慢 |
| Python multiprocessing | 296 ms | 15,100 ms | 6,400× 慢 |
| **MuJoCoUni** | **0.12 ms** | **2.36 ms** | — |

**部分重置（N=4096）**：

| 方法 | 5% envs | 90% envs |
|------|--------:|---------:|
| Python for-loop | 2.21 ms | 34.9 ms |
| Python multiprocessing | 955 ms | 13,600 ms |
| **MuJoCoUni** | **0.75 ms** | **1.97 ms** |

multiprocessing 基线受进程启动与序列化开销支配，即使重置少量环境也需近 1 秒；MuJoCoUni 的部分重置延迟与重置比例近似线性，90% 重置也不到 2 ms。

## 4. 站点雅可比计算（Franka）

对 Franka 机械臂的 `end_effector` 站点批量计算位置/姿态雅可比（`compute_site_jacobians`）：

![Site Jacobian Computation](figures/fig_jacobian.png)

| 方法 | 32 envs | 4096 envs | 相对 MuJoCoUni（4096 envs） |
|------|--------:|----------:|---------------------------:|
| Python for-loop | 0.35 ms | 7.9 ms | 14× 慢 |
| Python multiprocessing | 333 ms | 13,800 ms | 24,800× 慢 |
| **MuJoCoUni** | **0.12 ms** | **0.56 ms** | — |

MuJoCoUni 的延迟在 512 envs 之后基本不再增长（约 0.6 ms 平台期），体现出良好的并行扩展性。

![Site Jacobian Computation (bar)](figures/fig_jacobian_bar.png)

## 5. 高度场采样（楼梯地形）

在楼梯地形模型上，围绕机器人基座采样 4×4 网格共 16 个点的高度（双线性插值，`sample_hfield_height`）：

![Height-Field Sampling](figures/fig_hfield.png)

| 方法 | 32 envs | 4096 envs | 相对 MuJoCoUni（4096 envs） |
|------|--------:|----------:|---------------------------:|
| Python for-loop | 2.41 ms | 264 ms | 475× 慢 |
| Python multiprocessing | 152 ms | 446 ms | 802× 慢 |
| **MuJoCoUni** | **0.12 ms** | **0.56 ms** | — |

纯 Python 实现的双线性插值即使只处理 32 个环境也需 2.4 ms，而 MuJoCoUni 处理 4096 个环境仅需 0.56 ms。

![Height-Field Sampling (bar)](figures/fig_hfield_bar.png)

---

## 结论

1. **吞吐**：批量步进在简单到中等复杂度模型上达到 2–3M steps/s（16 线程），在 256–1024 envs 后饱和；
2. **多模型**：每环境独立模型实例的额外开销仅 3–7%，可放心用于域随机化；
3. **延迟敏感操作**（重置、雅可比、高度场采样）：MuJoCoUni 相对 Python for-loop 快 1–3 个数量级，相对 Python multiprocessing 快 3–4 个数量级——multiprocessing 的进程管理与数据序列化开销在细粒度批量操作上完全无法摊销；
4. 对于 RL 训练中高频调用的 reset 与观测计算路径，C++ 批量接口能消除原本占主导的 Python 开销。

## 复现

```bash
# 安装（含 MuJoCoUni 运行时）
uv sync --extra mujoco-uni-runtime

# 运行全部基准（约 12 分钟，结果写入 benchmark_results.json）
mujoco-uni-bench

# 生成全部图（默认输出 figures/）
mujoco-uni-plot
```

单项基准可用 `--bench N` 选择（1=Step/Forward，2=多模型，3=重置，4=雅可比，5=高度场）；图可用 `--fig N` 选择。更多参数见 `mujoco-uni-bench --help` 与 `mujoco-uni-plot --help`。

## 模型来源

| 模型 | 来源 | 许可证 |
|------|------|--------|
| Unitree Go1 | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Apache 2.0 |
| Wonik Allegro | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Apache 2.0 |
| Franka Panda | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Apache 2.0 |
| CMU Humanoid | [dm_control](https://github.com/google-deepmind/dm_control) | Apache 2.0 |
| 楼梯地形 | [MuJoCo-LiDAR](https://github.com/discoverse-dev/MuJoCo-LiDAR) | MIT |
