import os

import numpy as np
import matplotlib.pyplot as plt

from .style import (
    COLOR_PINK, COLOR_BLUE,
    ROBOT_COLORS, ROBOT_HATCHES,
    METHOD_COLORS, METHOD_HATCHES, METHOD_LABELS,
    _bar_positions,
)
from .data import PLACEHOLDER


# ══════════════════════════════════════════════════════════════
# BAR CHART versions
# ══════════════════════════════════════════════════════════════

def fig1_bar(data, outdir):
    """Bar chart: step/forward throughput at selected env counts."""
    d = data.get("bench1", PLACEHOLDER["bench1"])
    num_envs = d["num_envs"]
    robots = [r for r in ["Go1", "Allegro", "Franka", "Humanoid"] if r in d["step"]]

    sel_envs = [n for n in [64, 256, 1024, 4096] if n in num_envs]
    sel_idx = [num_envs.index(n) for n in sel_envs]

    fig, (ax_step, ax_fwd) = plt.subplots(1, 2, figsize=(7.5, 4.2), sharey=True)

    bar_w, offsets = _bar_positions(len(sel_envs), len(robots))
    x = np.arange(len(sel_envs))

    for ax, key in [(ax_step, "step"), (ax_fwd, "forward")]:
        for j, robot in enumerate(robots):
            raw = d[key][robot]
            vals = [raw[i] for i in sel_idx if i < len(raw)]
            ax.bar(x[:len(vals)] + offsets[j], vals, bar_w * 0.9,
                   color=ROBOT_COLORS[robot], edgecolor="white", linewidth=0.5,
                   hatch=ROBOT_HATCHES.get(robot, ""), label=robot, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in sel_envs])
        ax.set_xlabel("Number of environments")
        ax.set_yscale("log")

    ax_step.set_ylabel("Throughput (per second)")
    ax_step.set_title("(a) Batched Step", fontsize=11, pad=0)
    ax_fwd.set_title("(b) Batched Forward", fontsize=11, pad=0)

    handles, labels = ax_step.get_legend_handles_labels()
    fig.tight_layout(w_pad=2.0)
    fig.subplots_adjust(top=0.84)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.86),
               ncol=4, frameon=False, fontsize=10)
    fig.suptitle("Step / Forward Throughput", fontsize=13, y=0.97)
    ax_step.xaxis.label.set_visible(False)
    ax_fwd.xaxis.label.set_visible(False)
    fig.text(0.5, 0.01, "Number of environments", ha='center', fontsize=13)
    path = os.path.join(outdir, "fig_step_forward_bar.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


def fig2_bar(data, outdir):
    """Bar chart: multi-model single vs multi."""
    d = data.get("bench2", PLACEHOLDER["bench2"])
    if not d:
        return
    num_envs = d["num_envs"]
    robots = [r for r in ["Go1", "Allegro"] if r in d]

    sel_envs = [n for n in [32, 64, 128, 256, 512] if n in num_envs]
    sel_idx = [num_envs.index(n) for n in sel_envs]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.2))
    # Groups: each env count; bars: robot x mode
    labels = []
    colors = []
    hatches = []
    for robot in robots:
        for mode in ["single", "multi"]:
            labels.append(f"{robot} {'single' if mode == 'single' else 'variants'}")
            colors.append(COLOR_PINK if mode == "single" else COLOR_BLUE)
            hatches.append("" if robot == "Go1" else "//")

    n_bars = len(labels)
    bar_w, offsets = _bar_positions(len(sel_envs), n_bars)
    x = np.arange(len(sel_envs))

    for j, (robot, mode) in enumerate([(r, m) for r in robots for m in ["single", "multi"]]):
        raw = d[robot][mode]
        vals = [raw[i] for i in sel_idx if i < len(raw)]
        ax.bar(x[:len(vals)] + offsets[j], vals, bar_w * 0.9,
               color=colors[j], edgecolor="white", linewidth=0.5,
               hatch=hatches[j], label=labels[j], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in sel_envs])
    ax.set_xlabel("Number of environments")
    ax.set_ylabel("Throughput (steps/s)")
    ax.set_yscale("log")

    handles, labels_leg = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.legend(handles, labels_leg, loc="lower center", bbox_to_anchor=(0.5, 0.76),
               ncol=2, frameon=False, fontsize=10)
    fig.suptitle("Multi-Model Step Throughput", fontsize=13, y=0.97)
    path = os.path.join(outdir, "fig_multimodel_bar.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


def _method_bar(data, bench_key, model_key, sel_envs_list, title, ylabel, outname, outdir):
    """Generic grouped bar chart for method comparison (3 methods x N env counts)."""
    d = data.get(bench_key, PLACEHOLDER[bench_key])
    if model_key not in d:
        return
    num_envs = d["num_envs"]
    methods = ["python-loop", "python-mp", "mujocouni-cpp"]

    # Pick sel points that actually exist in num_envs
    sel_envs = [n for n in sel_envs_list if n in num_envs]
    if not sel_envs:
        return
    sel_idx = [num_envs.index(n) for n in sel_envs]

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.2))
    bar_w, offsets = _bar_positions(len(sel_envs), len(methods))
    x = np.arange(len(sel_envs))

    for j, method in enumerate(methods):
        vals_raw = d[model_key].get(method, [])
        if not vals_raw:
            continue
        valid = [(k, i) for k, i in enumerate(sel_idx) if i < len(vals_raw)]
        if not valid:
            continue
        bar_x = [x[k] + offsets[j] for k, _ in valid]
        vals = [vals_raw[i] * 1000 for _, i in valid]
        ax.bar(bar_x, vals, bar_w * 0.9,
               color=METHOD_COLORS[method], edgecolor="white", linewidth=0.5,
               hatch=METHOD_HATCHES[method], label=METHOD_LABELS[method], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in sel_envs])
    ax.set_xlabel("Number of environments")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.82),
               ncol=3, frameon=False, fontsize=10, columnspacing=1.0, handlelength=1.5)
    fig.suptitle(title, fontsize=13, y=0.97)
    path = os.path.join(outdir, outname)
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


def fig3_bar(data, outdir):
    """Bar charts for reset benchmarks (Go1 only)."""
    d3 = data.get("bench3", PLACEHOLDER["bench3"])
    sel = [64, 256, 1024, 4096]
    methods = ["python-loop", "python-mp", "mujocouni-cpp"]

    # Full reset bar chart
    num_envs = d3["full"]["num_envs"]
    sel_envs = [n for n in sel if n in num_envs]
    sel_idx = [num_envs.index(n) for n in sel_envs]

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.2))
    bar_w, offsets = _bar_positions(len(sel_envs), len(methods))
    x = np.arange(len(sel_envs))

    for j, method in enumerate(methods):
        vals_raw = d3["full"]["Go1"].get(method, [])
        if not vals_raw:
            continue
        vals = [vals_raw[i] * 1000 for i in sel_idx if i < len(vals_raw)]
        ax.bar(x[:len(vals)] + offsets[j], vals, bar_w * 0.9,
               color=METHOD_COLORS[method], edgecolor="white", linewidth=0.5,
               hatch=METHOD_HATCHES[method], label=METHOD_LABELS[method], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in sel_envs])
    ax.set_xlabel("Number of environments")
    ax.set_ylabel("Time (ms)")
    ax.set_yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.84),
               ncol=3, frameon=False, fontsize=10, columnspacing=1.0, handlelength=1.5)
    fig.suptitle("Full Reset — Go1", fontsize=13, y=0.97)
    path = os.path.join(outdir, "fig_reset_full_bar.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")

    # Partial reset bar chart
    fracs = d3["partial"]["fractions"]
    nenv = d3["partial"]["num_envs"]

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.2))
    bar_w, offsets = _bar_positions(len(fracs), len(methods))
    x = np.arange(len(fracs))

    for j, method in enumerate(methods):
        vals = d3["partial"]["Go1"].get(method, [])
        if not vals:
            continue
        ax.bar(x + offsets[j], [v * 1000 for v in vals], bar_w * 0.9,
               color=METHOD_COLORS[method], edgecolor="white", linewidth=0.5,
               hatch=METHOD_HATCHES[method], label=METHOD_LABELS[method], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(f*100)}%" for f in fracs])
    ax.set_xlabel("Reset fraction")
    ax.set_ylabel("Time (ms)")
    ax.set_yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.84),
               ncol=3, frameon=False, fontsize=10, columnspacing=1.0, handlelength=1.5)
    fig.suptitle(f"Partial Reset — Go1 (N={nenv})", fontsize=13, y=0.97)
    path = os.path.join(outdir, "fig_reset_partial_bar.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


def fig4_bar(data, outdir):
    """Bar chart for Jacobian benchmark."""
    _method_bar(data, "bench4", "Franka", [64, 256, 1024, 4096],
                "Site Jacobian Computation — Franka", "Jacobian time (ms)",
                "fig_jacobian_bar.pdf", outdir)


def fig5_bar(data, outdir):
    """Bar chart for hfield benchmark."""
    _method_bar(data, "bench5", "Hfield", [64, 256, 1024, 4096],
                "Height-Field Sampling — Terrain", "Hfield sample time (ms)",
                "fig_hfield_bar.pdf", outdir)
