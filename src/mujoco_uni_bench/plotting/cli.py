"""
Generate benchmark figures for the MuJoCoUni technical report.

Reads benchmark data from benchmark_results.json (output of mujoco-uni-bench).
Falls back to built-in placeholder data if the JSON file is not found.

Usage:
    mujoco-uni-plot                       # generate all figures
    mujoco-uni-plot --fig 1               # generate only figure 1
    mujoco-uni-plot --fig 1 2 3           # generate figures 1, 2, 3
    mujoco-uni-plot --data path/to/results.json

Output directory: figures/ (in the current working directory)
"""

import argparse
import os

from .data import load_data, PLACEHOLDER, _filter_small_envs
from .figures import (
    fig1_step_forward, fig2_multimodel, fig3_reset_combined,
    fig4_jacobian, fig5_hfield,
)
from .bars import fig1_bar, fig2_bar, fig3_bar, fig4_bar, fig5_bar


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
FIGURE_MAP = {
    1: ("Step / Forward throughput",       lambda d, outdir: (fig1_step_forward(d, outdir), fig1_bar(d, outdir))),
    2: ("Multi-model comparison",          lambda d, outdir: (fig2_multimodel(d, outdir), fig2_bar(d, outdir))),
    3: ("Reset (full + partial)",          lambda d, outdir: fig3_reset_combined(d, outdir)),
    4: ("compute_site_jacobians (Franka)", lambda d, outdir: (fig4_jacobian(d, outdir), fig4_bar(d, outdir))),
    5: ("sample_hfield_height",            lambda d, outdir: (fig5_hfield(d, outdir), fig5_bar(d, outdir))),
}


def main():
    parser = argparse.ArgumentParser(description="Generate MuJoCoUni benchmark figures")
    parser.add_argument("--fig", nargs="*", type=int, default=None,
                        help="Figure numbers to generate (default: all)")
    parser.add_argument("--data", type=str, default="benchmark_results.json",
                        help="Path to benchmark_results.json")
    parser.add_argument("--outdir", type=str, default="figures",
                        help="Output directory for figures (default: figures)")
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    data = load_data(args.data)
    if data is None:
        data = PLACEHOLDER

    # Global filter: remove nenv < MIN_ENVS from all benchmark data
    data = _filter_small_envs(data)

    figs = args.fig if args.fig else sorted(FIGURE_MAP.keys())
    for i in figs:
        name, func = FIGURE_MAP[i]
        print(f"[Figure {i}] {name}")
        func(data, outdir)

    print(f"\nDone. Figures saved to: {outdir}")


if __name__ == "__main__":
    main()
