#!/usr/bin/env python3
"""
Generate benchmark figures for the MuJoCoUni technical report.

Reads benchmark data from scripts/benchmark_results.json (output of run_benchmarks.py).
Falls back to built-in placeholder data if the JSON file is not found.

Usage:
    python scripts/plot_benchmarks.py              # generate all figures
    python scripts/plot_benchmarks.py --fig 1       # generate only figure 1
    python scripts/plot_benchmarks.py --fig 1 2 3   # generate figures 1, 2, 3
    python scripts/plot_benchmarks.py --data path/to/results.json

Output directory: MuJoCoUni/Assets/Figures/
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ──────────────────────────────────────────────────────────────
# Global style
# ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(SCRIPT_DIR, "figures")
DEFAULT_JSON = os.path.join(SCRIPT_DIR, "benchmark_results.json")

# Color palette — soft pastel, distinct per robot
COLOR_PINK       = "#F0C4D8"   # soft pink (robot 1)
COLOR_BLUE       = "#7BA8D8"   # soft blue (robot 2)
COLOR_CYAN       = "#7DD4C8"   # soft cyan/teal (robot 3)
COLOR_LAVENDER   = "#C9B1E8"   # soft lavender (robot 4)

ROBOT_COLORS = {
    "Go1":       COLOR_PINK,       # pink
    "Allegro":   COLOR_BLUE,       # blue
    "Franka":    COLOR_CYAN,       # cyan
    "Humanoid":  COLOR_LAVENDER,   # lavender
}
METHOD_COLORS = {
    "python-loop":   COLOR_PINK,   # pink
    "python-mp":     COLOR_BLUE,   # blue
    "mujocouni-cpp": COLOR_CYAN,   # cyan
}
MULTIMODEL_COLORS = {
    "single": COLOR_PINK,
    "multi":  COLOR_BLUE,
}

ROBOT_MARKERS = {"Go1": "o", "Allegro": "s", "Franka": "^", "Humanoid": "D"}
METHOD_MARKERS = {"python-loop": "o", "python-mp": "s", "mujocouni-cpp": "^"}
ROBOT_LINESTYLES = {"Go1": "-", "Allegro": "--", "Franka": "-.", "Humanoid": ":"}
ROBOT_HATCHES = {"Go1": "", "Allegro": "//", "Franka": "", "Humanoid": "//"}

METHOD_LABELS = {
    "python-loop":   "Python for-loop",
    "python-mp":     "Python multiprocessing",
    "mujocouni-cpp": "MuJoCoUni",
}

# Bar chart edge + hatch for methods
METHOD_HATCHES = {
    "python-loop":   "",
    "python-mp":     "//",
    "mujocouni-cpp": "",
}


# ──────────────────────────────────────────────────────────────
# Data loader
# ──────────────────────────────────────────────────────────────
def load_data(path):
    if path and os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"Loaded benchmark data from: {path}")
        return data
    print(f"WARNING: {path} not found, using built-in placeholder data")
    return None


# ──────────────────────────────────────────────────────────────
# Built-in placeholder data (used when JSON is not available)
# ──────────────────────────────────────────────────────────────
PLACEHOLDER = {
    "bench1": {
        "num_envs": [16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        "step": {
            "Go1":      [12000, 24000, 47000, 90000, 170000, 310000, 550000, 900000],
            "Allegro":  [15000, 29000, 56000, 108000, 200000, 370000, 650000, 1050000],
            "Franka":   [10000, 19000, 37000, 70000, 130000, 240000, 420000, 700000],
            "Humanoid": [5000,  9500,  18000, 34000, 62000,  110000, 190000, 310000],
        },
        "forward": {
            "Go1":      [25000, 49000, 95000, 180000, 340000, 620000, 1100000, 1800000],
            "Allegro":  [30000, 58000, 112000, 215000, 400000, 740000, 1300000, 2100000],
            "Franka":   [20000, 39000, 75000, 142000, 265000, 490000, 860000, 1400000],
            "Humanoid": [10000, 19000, 36000, 68000, 125000, 225000, 400000, 650000],
        },
    },
    "bench2": {
        "num_envs": [32, 64, 128, 256, 512, 1024, 2048, 4096],
        "Go1":     {"single": [12000, 24000, 47000, 90000, 170000, 310000, 550000, 900000],
                    "multi":  [11500, 23000, 44000, 84000, 158000, 290000, 510000, 830000]},
        "Allegro": {"single": [15000, 29000, 56000, 108000, 200000, 370000, 650000, 1050000],
                    "multi":  [14200, 27500, 52000, 100000, 186000, 345000, 600000, 970000]},
    },
    "bench3": {
        "full": {
            "num_envs": [16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
            "Go1": {
                "python-loop":   [0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.024, 2.048, 4.096],
                "python-mp":     [0.020, 0.025, 0.035, 0.055, 0.095, 0.175, 0.33,  0.64,  1.25],
                "mujocouni-cpp": [0.0005, 0.001, 0.002, 0.003, 0.005, 0.009, 0.016, 0.030, 0.058],
            },
        },
        "partial": {
            "num_envs": 4096,
            "fractions": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
            "Go1": {
                "python-loop":   [0.20, 0.41, 1.23, 2.05, 2.87, 3.69],
                "python-mp":     [0.08, 0.15, 0.38, 0.63, 0.90, 1.16],
                "mujocouni-cpp": [0.003, 0.006, 0.017, 0.029, 0.041, 0.053],
            },
        },
    },
    "bench4": {
        "num_envs": [16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        "Franka": {
            "python-loop":   [0.015, 0.030, 0.060, 0.120, 0.240, 0.480, 0.960, 1.920],
            "python-mp":     [0.012, 0.018, 0.028, 0.048, 0.088, 0.165, 0.32,  0.62],
            "mujocouni-cpp": [0.0005, 0.0009, 0.0016, 0.0028, 0.005, 0.009, 0.017, 0.032],
        },
    },
    "bench5": {
        "num_envs": [16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        "Hfield": {
            "python-loop":   [0.018, 0.036, 0.072, 0.144, 0.288, 0.576, 1.152, 2.304],
            "python-mp":     [0.014, 0.021, 0.033, 0.057, 0.105, 0.198, 0.385, 0.75],
            "mujocouni-cpp": [0.0006, 0.001, 0.0018, 0.0032, 0.006, 0.011, 0.020, 0.038],
        },
    },
}


# ──────────────────────────────────────────────────────────────
# Helper: filter out small env counts (nenv < 32 is too noisy)
# ──────────────────────────────────────────────────────────────
MIN_ENVS = 32


def _filter_num_envs_list(num_envs, data_dict):
    """Filter a num_envs list and all corresponding arrays in data_dict."""
    if not isinstance(num_envs, list):
        return num_envs, data_dict
    idx = [i for i, n in enumerate(num_envs) if n >= MIN_ENVS]
    new_envs = [num_envs[i] for i in idx]
    new_dict = {}
    for k, v in data_dict.items():
        if k == "num_envs":
            continue
        if isinstance(v, list) and len(v) == len(num_envs):
            new_dict[k] = [v[i] for i in idx]
        elif isinstance(v, dict):
            # Nested: e.g. {"Go1": [...], "Allegro": [...]}
            inner = {}
            for kk, vv in v.items():
                if isinstance(vv, list) and len(vv) == len(num_envs):
                    inner[kk] = [vv[i] for i in idx]
                elif isinstance(vv, dict):
                    # Double nested: e.g. {"single": [...], "multi": [...]}
                    inner2 = {}
                    for kkk, vvv in vv.items():
                        if isinstance(vvv, list) and len(vvv) == len(num_envs):
                            inner2[kkk] = [vvv[i] for i in idx]
                        else:
                            inner2[kkk] = vvv
                    inner[kk] = inner2
                else:
                    inner[kk] = vv
            new_dict[k] = inner
        else:
            new_dict[k] = v
    new_dict["num_envs"] = new_envs
    return new_envs, new_dict


def _filter_small_envs(data):
    """Remove nenv < MIN_ENVS from all benchmark entries."""
    import copy
    data = copy.deepcopy(data)
    for key in list(data.keys()):
        if not key.startswith("bench"):
            continue
        bench = data[key]
        if not isinstance(bench, dict):
            continue
        # Top-level num_envs
        if "num_envs" in bench and isinstance(bench["num_envs"], list):
            _, bench = _filter_num_envs_list(bench["num_envs"], bench)
            data[key] = bench
        # Nested (e.g. bench3 has "full" and "partial" sub-dicts)
        for sub_key in list(bench.keys()):
            sub = bench[sub_key]
            if isinstance(sub, dict) and "num_envs" in sub and isinstance(sub["num_envs"], list):
                _, bench[sub_key] = _filter_num_envs_list(sub["num_envs"], sub)
    return data


# ──────────────────────────────────────────────────────────────
# Helper: setup log-scale axes
# ──────────────────────────────────────────────────────────────
def setup_log2_xaxis(ax, num_envs):
    try:
        ax.set_xscale("log", base=2)
    except (TypeError, ValueError):
        ax.set_xscale("log", basex=2)  # matplotlib < 3.3
    ax.set_xticks(num_envs)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis="x", rotation=45)


# ──────────────────────────────────────────────────────────────
# Figure 1: Step / Forward throughput
# ──────────────────────────────────────────────────────────────
def fig1_step_forward(data):
    d = data.get("bench1", PLACEHOLDER["bench1"])
    num_envs = d["num_envs"]
    robots = [r for r in ["Go1", "Allegro", "Franka", "Humanoid"] if r in d["step"]]

    fig, (ax_step, ax_fwd) = plt.subplots(1, 2, figsize=(7.5, 4.2), sharey=True)

    for robot in robots:
        kw = dict(color=ROBOT_COLORS[robot], marker=ROBOT_MARKERS[robot],
                  label=robot, markeredgecolor="white", markeredgewidth=0.5)
        step_vals = d["step"][robot]
        fwd_vals = d["forward"][robot]
        n = min(len(num_envs), len(step_vals))
        ax_step.plot(num_envs[:n], step_vals[:n], **kw)
        n = min(len(num_envs), len(fwd_vals))
        ax_fwd.plot(num_envs[:n], fwd_vals[:n], **kw)

    for ax, subtitle in [(ax_step, "(a) Batched Step"), (ax_fwd, "(b) Batched Forward")]:
        setup_log2_xaxis(ax, num_envs)
        ax.set_yscale("log")
        ax.set_xlabel("Number of environments")
        ax.set_title(subtitle, fontsize=11, pad=0)
    ax_step.set_ylabel("Throughput (per second)")

    # Title → Legend → Plot
    handles, labels = ax_step.get_legend_handles_labels()
    fig.tight_layout(w_pad=2.5)
    fig.subplots_adjust(top=0.84)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.86),
               ncol=4, frameon=False, fontsize=10)
    fig.suptitle("Step / Forward Throughput", fontsize=13, y=0.97)
    # Unify xlabel after layout is done
    ax_step.xaxis.label.set_visible(False)
    ax_fwd.xaxis.label.set_visible(False)
    fig.text(0.5, 0.01, "Number of environments", ha='center', fontsize=13)
    path = os.path.join(OUTDIR, "fig_step_forward.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ──────────────────────────────────────────────────────────────
# Figure 2: Multi-model comparison (all robots in one plot)
# ──────────────────────────────────────────────────────────────
def fig2_multimodel(data):
    d = data.get("bench2", PLACEHOLDER["bench2"])
    if not d:
        print("  SKIPPED (no data)")
        return
    num_envs = d["num_envs"]
    robots = [r for r in ["Go1", "Allegro"] if r in d]

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.2))

    for robot in robots:
        rd = d[robot]
        ls = ROBOT_LINESTYLES.get(robot, "-")
        n = min(len(num_envs), len(rd["single"]))
        ax.plot(num_envs[:n], rd["single"][:n], color=MULTIMODEL_COLORS["single"],
                marker=ROBOT_MARKERS[robot], linestyle=ls,
                label=f"{robot} \u2014 single model",
                markeredgecolor="white", markeredgewidth=0.5)
        n = min(len(num_envs), len(rd["multi"]))
        ax.plot(num_envs[:n], rd["multi"][:n], color=MULTIMODEL_COLORS["multi"],
                marker=ROBOT_MARKERS[robot], linestyle=ls,
                label=f"{robot} \u2014 model variants",
                markeredgecolor="white", markeredgewidth=0.5)

    setup_log2_xaxis(ax, num_envs)
    ax.set_yscale("log")
    ax.set_xlabel("Number of environments")
    ax.set_ylabel("Throughput (steps/s)")

    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.76),
               ncol=2, frameon=False, fontsize=10)
    fig.suptitle("Multi-Model Step Throughput", fontsize=13, y=0.97)
    path = os.path.join(OUTDIR, "fig_multimodel.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ──────────────────────────────────────────────────────────────
# Figure 3: Reset — full + partial in one row (1x2)
# ──────────────────────────────────────────────────────────────
def fig3_reset_combined(data):
    d3 = data.get("bench3", PLACEHOLDER["bench3"])
    methods = ["python-loop", "python-mp", "mujocouni-cpp"]

    fig, (ax_full, ax_partial) = plt.subplots(1, 2, figsize=(7.5, 4.2), sharey=True)

    # Left: full reset
    d_full = d3["full"]
    num_envs = d_full["num_envs"]
    for method in methods:
        vals = d_full["Go1"].get(method)
        if vals is None or all(v is None for v in vals):
            continue
        n = min(len(num_envs), len(vals))
        ax_full.plot(num_envs[:n], [v * 1000 for v in vals[:n]], color=METHOD_COLORS[method],
                     marker=METHOD_MARKERS[method], label=METHOD_LABELS[method],
                     markeredgecolor="white", markeredgewidth=0.5)
    setup_log2_xaxis(ax_full, num_envs)
    ax_full.set_yscale("log")
    ax_full.set_xlabel("Number of environments")
    ax_full.set_ylabel("Time (ms)")

    # Right: partial reset
    d_part = d3["partial"]
    fracs = d_part["fractions"]
    nenv = d_part["num_envs"]
    x_pct = [f"{int(f * 100)}%" for f in fracs]
    for method in methods:
        vals = d_part["Go1"].get(method)
        if vals is None or all(v is None for v in vals):
            continue
        ax_partial.plot(range(len(fracs)), [v * 1000 for v in vals], color=METHOD_COLORS[method],
                        marker=METHOD_MARKERS[method], label=METHOD_LABELS[method],
                        markeredgecolor="white", markeredgewidth=0.5)
    ax_full.set_title("(a) Full Reset", fontsize=11, pad=0)
    ax_partial.set_yscale("log")
    ax_partial.set_xlabel("Reset fraction")
    ax_partial.set_title(f"(b) Partial Reset (N={nenv})", fontsize=11, pad=0)
    ax_partial.set_xticks(range(len(fracs)))
    ax_partial.set_xticklabels(x_pct, rotation=45)

    handles, labels = ax_full.get_legend_handles_labels()
    fig.tight_layout(w_pad=2.5)
    fig.subplots_adjust(top=0.84)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.86),
               ncol=3, frameon=False, fontsize=10, columnspacing=1.0, handlelength=1.5)
    fig.suptitle("Reset Performance \u2014 Go1", fontsize=13, y=0.97)
    # Align xlabels after layout is done
    ax_full.xaxis.label.set_visible(False)
    ax_partial.xaxis.label.set_visible(False)
    left_center = (ax_full.get_position().x0 + ax_full.get_position().x1) / 2
    right_center = (ax_partial.get_position().x0 + ax_partial.get_position().x1) / 2
    fig.text(left_center, 0.01, "Number of environments", ha='center', fontsize=13)
    fig.text(right_center, 0.01, "Reset fraction", ha='center', fontsize=13)

    path = os.path.join(OUTDIR, "fig_reset.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ──────────────────────────────────────────────────────────────
# Figure 4: compute_site_jacobians (Franka)
# ──────────────────────────────────────────────────────────────
def fig4_jacobian(data):
    d = data.get("bench4", PLACEHOLDER["bench4"])
    num_envs = d["num_envs"]
    methods = ["python-loop", "python-mp", "mujocouni-cpp"]

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.2))

    for method in methods:
        vals = d["Franka"].get(method)
        if vals is None or all(v is None for v in vals):
            continue
        # Use min length in case data was generated with different num_envs
        n = min(len(num_envs), len(vals))
        ax.plot(num_envs[:n], [v * 1000 for v in vals[:n]], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                label=METHOD_LABELS[method], markeredgecolor="white", markeredgewidth=0.5)

    setup_log2_xaxis(ax, num_envs)
    ax.set_yscale("log")
    ax.set_xlabel("Number of environments")
    ax.set_ylabel("Time (ms)")

    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.82),
               ncol=3, frameon=False, fontsize=10, columnspacing=1.0, handlelength=1.5)
    fig.suptitle("Site Jacobian Computation \u2014 Franka", fontsize=13, y=0.97)
    path = os.path.join(OUTDIR, "fig_jacobian.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ──────────────────────────────────────────────────────────────
# Figure 5: sample_hfield_height
# ──────────────────────────────────────────────────────────────
def fig5_hfield(data):
    d = data.get("bench5", PLACEHOLDER["bench5"])
    num_envs = d["num_envs"]
    methods = ["python-loop", "python-mp", "mujocouni-cpp"]

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.2))

    for method in methods:
        vals = d["Hfield"].get(method)
        if vals is None or all(v is None for v in vals):
            continue
        n = min(len(num_envs), len(vals))
        ax.plot(num_envs[:n], [v * 1000 for v in vals[:n]], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                label=METHOD_LABELS[method], markeredgecolor="white", markeredgewidth=0.5)

    setup_log2_xaxis(ax, num_envs)
    ax.set_yscale("log")
    ax.set_xlabel("Number of environments")
    ax.set_ylabel("Time (ms)")

    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.80),
               ncol=3, frameon=False, fontsize=10, columnspacing=1.0, handlelength=1.5)
    fig.suptitle("Height-Field Sampling \u2014 Terrain", fontsize=13, y=0.95)
    path = os.path.join(OUTDIR, "fig_hfield.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ══════════════════════════════════════════════════════════════
# BAR CHART versions
# ══════════════════════════════════════════════════════════════

def _bar_positions(n_groups, n_bars, width=0.7):
    """Return x positions for grouped bar chart."""
    bar_w = width / n_bars
    offsets = np.arange(n_bars) * bar_w - width / 2 + bar_w / 2
    return bar_w, offsets


def fig1_bar(data):
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
    path = os.path.join(OUTDIR, "fig_step_forward_bar.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


def fig2_bar(data):
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
    path = os.path.join(OUTDIR, "fig_multimodel_bar.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


def _method_bar(data, bench_key, model_key, sel_envs_list, title, ylabel, outname):
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
    path = os.path.join(OUTDIR, outname)
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


def fig3_bar(data):
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
    fig.suptitle("Full Reset \u2014 Go1", fontsize=13, y=0.97)
    path = os.path.join(OUTDIR, "fig_reset_full_bar.pdf")
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
    fig.suptitle(f"Partial Reset \u2014 Go1 (N={nenv})", fontsize=13, y=0.97)
    path = os.path.join(OUTDIR, "fig_reset_partial_bar.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


def fig4_bar(data):
    """Bar chart for Jacobian benchmark."""
    _method_bar(data, "bench4", "Franka", [64, 256, 1024, 4096],
                "Site Jacobian Computation \u2014 Franka", "Jacobian time (ms)",
                "fig_jacobian_bar.pdf")


def fig5_bar(data):
    """Bar chart for hfield benchmark."""
    _method_bar(data, "bench5", "Hfield", [64, 256, 1024, 4096],
                "Height-Field Sampling \u2014 Terrain", "Hfield sample time (ms)",
                "fig_hfield_bar.pdf")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
FIGURE_MAP = {
    1: ("Step / Forward throughput",       lambda d: (fig1_step_forward(d), fig1_bar(d))),
    2: ("Multi-model comparison",          lambda d: (fig2_multimodel(d), fig2_bar(d))),
    3: ("Reset (full + partial)",          lambda d: fig3_reset_combined(d)),
    4: ("compute_site_jacobians (Franka)", lambda d: (fig4_jacobian(d), fig4_bar(d))),
    5: ("sample_hfield_height",            lambda d: (fig5_hfield(d), fig5_bar(d))),
}


def main():
    global OUTDIR
    parser = argparse.ArgumentParser(description="Generate MuJoCoUni benchmark figures")
    parser.add_argument("--fig", nargs="*", type=int, default=None,
                        help="Figure numbers to generate (default: all)")
    parser.add_argument("--data", type=str, default=DEFAULT_JSON,
                        help="Path to benchmark_results.json")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory for figures (default: MuJoCoUni/Assets/Figures)")
    args = parser.parse_args()

    if args.outdir:
        OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)

    data = load_data(args.data)
    if data is None:
        data = PLACEHOLDER

    # Global filter: remove nenv < MIN_ENVS from all benchmark data
    data = _filter_small_envs(data)

    figs = args.fig if args.fig else sorted(FIGURE_MAP.keys())
    for i in figs:
        name, func = FIGURE_MAP[i]
        print(f"[Figure {i}] {name}")
        func(data)

    print(f"\nDone. Figures saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
