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

# Color palette — soft pastel, distinct per robot
COLOR_PINK       = "#F0C4D8"   # soft pink (robot 1)
COLOR_BLUE       = "#7BA8D8"   # soft blue (robot 2)
COLOR_CYAN       = "#7DD4C8"   # soft cyan/teal (robot 3)
COLOR_LAVENDER   = "#C9B1E8"   # soft lavender (robot 4)
COLOR_AMBER      = "#F2D8A0"   # soft amber (stock-mjbatch NumPy arm)

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
    "mjbatch":       COLOR_LAVENDER,  # lavender
    "mjbatch-numpy": COLOR_AMBER,  # amber
}
MULTIMODEL_COLORS = {
    "single": COLOR_PINK,
    "multi":  COLOR_BLUE,
}

# Method arms in canonical order (mjbatch arms absent from older result
# files; figure code skips missing arms)
METHODS = ["python-loop", "python-mp", "mujocouni-cpp", "mjbatch", "mjbatch-numpy"]

ROBOT_MARKERS = {"Go1": "o", "Allegro": "s", "Franka": "^", "Humanoid": "D"}
METHOD_MARKERS = {"python-loop": "o", "python-mp": "s", "mujocouni-cpp": "^",
                  "mjbatch": "D", "mjbatch-numpy": "v"}
ROBOT_LINESTYLES = {"Go1": "-", "Allegro": "--", "Franka": "-.", "Humanoid": ":"}
ROBOT_HATCHES = {"Go1": "", "Allegro": "//", "Franka": "", "Humanoid": "//"}

METHOD_LABELS = {
    "python-loop":   "Python for-loop",
    "python-mp":     "Python multiprocessing",
    "mujocouni-cpp": "MuJoCoUni",
    "mjbatch":       "mjbatch",
    "mjbatch-numpy": "mjbatch (NumPy)",
}

# Bar chart edge + hatch for methods
METHOD_HATCHES = {
    "python-loop":   "",
    "python-mp":     "//",
    "mujocouni-cpp": "",
    "mjbatch":       "xx",
    "mjbatch-numpy": "..",
}

# Minimum env count (nenv < 32 is too noisy)
MIN_ENVS = 32


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


def _bar_positions(n_groups, n_bars, width=0.7):
    """Return x positions for grouped bar chart."""
    bar_w = width / n_bars
    offsets = np.arange(n_bars) * bar_w - width / 2 + bar_w / 2
    return bar_w, offsets
