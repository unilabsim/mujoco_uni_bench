from .step_forward import bench_step_forward
from .multimodel import bench_multimodel
from .reset import bench_reset
from .jacobian import bench_jacobian
from .hfield import bench_hfield

BENCH_MAP = {
    1: ("Step / Forward throughput",       bench_step_forward),
    2: ("Multi-model comparison",          bench_multimodel),
    3: ("Reset (full + partial)",          bench_reset),
    4: ("compute_site_jacobians (Franka)", bench_jacobian),
    5: ("sample_hfield_height",            bench_hfield),
}
