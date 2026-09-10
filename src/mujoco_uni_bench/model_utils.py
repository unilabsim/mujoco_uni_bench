import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Attempt to import mujoco_uni_runtime (mujoco_uni.batch_env); fall back to
# mujoco-uni (mujoco.batch_env), then to standard mujoco for baselines
# ---------------------------------------------------------------------------
try:
    import mujoco
    from mujoco_uni.batch_env import BatchEnvPool
    HAS_BATCH_ENV = True
    print(f"mujoco version: {mujoco.__version__} (BatchEnvPool available)")
except ImportError:
    try:
        import mujoco
        from mujoco.batch_env import BatchEnvPool
        HAS_BATCH_ENV = True
        print(f"mujoco version: {mujoco.__version__} (BatchEnvPool available)")
    except ImportError:
        try:
            import mujoco
            BatchEnvPool = None
            HAS_BATCH_ENV = False
            print(f"mujoco version: {mujoco.__version__} (standard mujoco, no BatchEnvPool)")
        except ImportError:
            mujoco = None
            BatchEnvPool = None
            HAS_BATCH_ENV = False

# mjbatch (thirdparty/): optional batched-executor arm. Requires mujoco==3.11.0.
try:
    import mjbatch
    HAS_MJBATCH = True
    print("mjbatch available (thirdparty)")
except ImportError:
    mjbatch = None
    HAS_MJBATCH = False

# ---------------------------------------------------------------------------
# Paths (relative to this package)
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).parent / "models"

MODEL_PATHS = {
    "Go1":      str(MODELS_DIR / "unitree_go1" / "go1.xml"),
    "Allegro":  str(MODELS_DIR / "wonik_allegro" / "right_hand.xml"),
    "Franka":   str(MODELS_DIR / "franka_emika_panda" / "panda.xml"),
    "Humanoid": str(MODELS_DIR / "humanoid" / "humanoid_CMU_V2020.xml"),
    "Terrain":  str(MODELS_DIR / "terrain" / "stairs_terrain.xml"),
}


def _enable_discardvisual(root):
    compiler_tag = root.find("compiler")
    if compiler_tag is None:
        compiler_tag = ET.Element("compiler")
        root.insert(0, compiler_tag)
    compiler_tag.set("discardvisual", "true")


def _create_discardvisual_xml(model_file):
    tree = ET.parse(model_file)
    _enable_discardvisual(tree.getroot())
    fd, output_path = tempfile.mkstemp(
        suffix=".xml", dir=os.path.dirname(os.path.abspath(model_file))
    )
    os.close(fd)
    tree.write(output_path)
    return output_path


_MODEL_CACHE = {}

def load_model(name):
    path = MODEL_PATHS[name]
    if path in _MODEL_CACHE:
        return _MODEL_CACHE[path]
    tmp = _create_discardvisual_xml(path)
    model = mujoco.MjModel.from_xml_path(tmp)
    os.unlink(tmp)
    _MODEL_CACHE[path] = model
    return model


_MP_PATH_CACHE = {}

def _get_mp_path(name):
    """Cached discardvisual temp XML for multiprocessing workers."""
    if name not in _MP_PATH_CACHE:
        _MP_PATH_CACHE[name] = _create_discardvisual_xml(MODEL_PATHS[name])
    return _MP_PATH_CACHE[name]


def get_random_state(model, nbatch, rng=None):
    """Generate a random but valid initial state for the model."""
    if rng is None:
        rng = np.random.default_rng(0)
    d = mujoco.MjData(model)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    states = np.zeros((nbatch, nstate), dtype=np.float64)
    for i in range(nbatch):
        mujoco.mj_resetData(model, d)
        # Small random perturbation to qpos
        d.qpos[:] += rng.normal(0, 0.01, size=d.qpos.shape)
        mujoco.mj_forward(model, d)
        mujoco.mj_getState(model, d, states[i], mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return states


def mjbatch_state_loader(batch, model, states):
    """Return a loader that injects FULLPHYSICS `states` rows into an mjbatch Batch.

    mjbatch bind("state") rows are opaque mjSTATE_INTEGRATION vectors, wider
    than the shared mjSTATE_FULLPHYSICS state arrays (they include warmstart),
    so states are injected through the individual mjtState component fields.
    load(ids) writes only those rows (for partial resets); load() writes all.

    mjbatch copies bound input fields in only where values changed since the
    last write, so re-writing identical states after a reset (which clobbers
    the sim state back to defaults) would be silently skipped. The loader
    therefore alternates between `states` and a copy offset by 1e-12, which is
    physically identical but forces the copy-in on every call — matching the
    per-call state copy that the other benchmark arms always perform.
    """
    if not HAS_MJBATCH:
        raise RuntimeError("mjbatch is not available")
    nq, nv, na = model.nq, model.nv, model.na
    alt = states.copy()
    alt[:, 1:] += 1e-12

    def bind_views(rows):
        views = [
            (batch.bind("time"), rows[:, 0]),
            (batch.bind("qpos"), rows[:, 1:1 + nq]),
            (batch.bind("qvel"), rows[:, 1 + nq:1 + nq + nv]),
        ]
        if na:
            views.append((batch.bind("act"), rows[:, 1 + nq + nv:]))
        return views

    views_a, views_b = bind_views(states), bind_views(alt)
    state = {"alt": False}

    def load(ids=None):
        views = views_b if state["alt"] else views_a
        state["alt"] = not state["alt"]
        for view, values in views:
            if ids is None:
                view[:] = values
            else:
                view[ids] = values[ids]

    return load
