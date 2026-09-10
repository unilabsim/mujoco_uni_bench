import os

import matplotlib.pyplot as plt

from .style import (
    ROBOT_COLORS, ROBOT_MARKERS, ROBOT_LINESTYLES,
    METHOD_COLORS, METHOD_MARKERS, METHOD_LABELS, MULTIMODEL_COLORS, METHODS,
    setup_log2_xaxis,
)
from .data import PLACEHOLDER


# ──────────────────────────────────────────────────────────────
# Figure 1: Step / Forward throughput
# ──────────────────────────────────────────────────────────────
def fig1_step_forward(data, outdir):
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

    # mjbatch arm: same robot colors, dashed lines
    step_mj = d.get("step_mjbatch", {})
    fwd_mj = d.get("forward_mjbatch", {})
    for robot in robots:
        if robot not in step_mj:
            continue
        kw = dict(color=ROBOT_COLORS[robot], marker=ROBOT_MARKERS[robot],
                  linestyle="--", label=f"{robot} (mjbatch)",
                  markeredgecolor="white", markeredgewidth=0.5)
        n = min(len(num_envs), len(step_mj[robot]))
        ax_step.plot(num_envs[:n], step_mj[robot][:n], **kw)
        n = min(len(num_envs), len(fwd_mj[robot]))
        ax_fwd.plot(num_envs[:n], fwd_mj[robot][:n], **kw)

    for ax, subtitle in [(ax_step, "(a) Batched Step"), (ax_fwd, "(b) Batched Forward")]:
        setup_log2_xaxis(ax, num_envs)
        ax.set_yscale("log")
        ax.set_xlabel("Number of environments")
        ax.set_title(subtitle, fontsize=11, pad=0)
    ax_step.set_ylabel("Throughput (per second)")

    # Title → Legend → Plot
    handles, labels = ax_step.get_legend_handles_labels()
    fig.tight_layout(w_pad=2.5)
    ncol = min(len(labels), 4)
    legend_rows = (len(labels) + ncol - 1) // ncol
    fig.subplots_adjust(top=0.84 - 0.06 * (legend_rows - 1))
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.86 - 0.06 * (legend_rows - 1)),
               ncol=ncol, frameon=False, fontsize=10)
    fig.suptitle("Step / Forward Throughput", fontsize=13, y=0.97)
    # Unify xlabel after layout is done
    ax_step.xaxis.label.set_visible(False)
    ax_fwd.xaxis.label.set_visible(False)
    fig.text(0.5, 0.01, "Number of environments", ha='center', fontsize=13)
    path = os.path.join(outdir, "fig_step_forward.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ──────────────────────────────────────────────────────────────
# Figure 2: Multi-model comparison (all robots in one plot)
# ──────────────────────────────────────────────────────────────
def fig2_multimodel(data, outdir):
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
                label=f"{robot} — single model",
                markeredgecolor="white", markeredgewidth=0.5)
        n = min(len(num_envs), len(rd["multi"]))
        ax.plot(num_envs[:n], rd["multi"][:n], color=MULTIMODEL_COLORS["multi"],
                marker=ROBOT_MARKERS[robot], linestyle=ls,
                label=f"{robot} — model variants",
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
    path = os.path.join(outdir, "fig_multimodel.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ──────────────────────────────────────────────────────────────
# Figure 3: Reset — full + partial in one row (1x2)
# ──────────────────────────────────────────────────────────────
def fig3_reset_combined(data, outdir):
    d3 = data.get("bench3", PLACEHOLDER["bench3"])
    methods = METHODS

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
               ncol=4, frameon=False, fontsize=10, columnspacing=1.0, handlelength=1.5)
    fig.suptitle("Reset Performance — Go1", fontsize=13, y=0.97)
    # Align xlabels after layout is done
    ax_full.xaxis.label.set_visible(False)
    ax_partial.xaxis.label.set_visible(False)
    left_center = (ax_full.get_position().x0 + ax_full.get_position().x1) / 2
    right_center = (ax_partial.get_position().x0 + ax_partial.get_position().x1) / 2
    fig.text(left_center, 0.01, "Number of environments", ha='center', fontsize=13)
    fig.text(right_center, 0.01, "Reset fraction", ha='center', fontsize=13)

    path = os.path.join(outdir, "fig_reset.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ──────────────────────────────────────────────────────────────
# Figure 4: compute_site_jacobians (Franka)
# ──────────────────────────────────────────────────────────────
def fig4_jacobian(data, outdir):
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
    fig.suptitle("Site Jacobian Computation — Franka", fontsize=13, y=0.97)
    path = os.path.join(outdir, "fig_jacobian.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ──────────────────────────────────────────────────────────────
# Figure 5: sample_hfield_height
# ──────────────────────────────────────────────────────────────
def fig5_hfield(data, outdir):
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
    fig.suptitle("Height-Field Sampling — Terrain", fontsize=13, y=0.95)
    path = os.path.join(outdir, "fig_hfield.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")
