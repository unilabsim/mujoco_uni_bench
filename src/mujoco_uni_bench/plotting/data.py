import json
import os

from .style import MIN_ENVS


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
