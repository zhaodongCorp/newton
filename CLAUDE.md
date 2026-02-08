# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Newton?

Newton is a GPU-accelerated physics simulation engine built on [NVIDIA Warp](https://github.com/NVIDIA/warp), targeting roboticists and simulation researchers. It extends Warp's deprecated `warp.sim` module and integrates [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) as its primary backend. The project is in active beta; the API is unstable.

## Network / Proxy

This machine is an internal devserver that can only access the external network through a proxy. For any command that requires internet access (e.g. `git push`, `pip install`, `curl`), prefix it with `with-proxy` or `pp`:

```bash
with-proxy git push
pp uv sync --extra examples
```

## Common Commands

```bash
# Setup
uv sync --extra examples          # set up environment for examples
uv sync --extra dev                # set up environment for development/testing

# Run examples
uv run -m newton.examples basic_pendulum
uv run -m newton.examples                  # list all available examples

# Run tests
uv run --extra dev -m newton.tests                              # all tests
uv run --extra dev -m newton.tests -k test_viewer_log_shapes    # specific test file
uv run --extra dev -m newton.tests -k test_basic.example_basic_shapes  # specific example test
uv run --extra dev --extra torch-cu12 -m newton.tests           # include PyTorch tests

# Lint/format (MUST run before committing)
uvx pre-commit run -a

# Inline Python
uv run python -c "..."

# Benchmarks
uvx --with virtualenv asv run --launch-method spawn main^!

# Generate API docs (after adding new public symbols)
uv run python docs/generate_api.py
```

## Architecture

### Public API / Private Implementation Boundary

`newton/_src/` contains all implementation. User code (examples, docs) must **never** import from `newton._src`. Public API modules at `newton/*.py` re-export from `_src`:

| Public module | Exposes |
|---|---|
| `newton/__init__.py` | `Model`, `ModelBuilder`, `State`, `Control`, `Contacts`, `CollisionPipeline`, `CollisionPipelineUnified`, `eval_fk`, `eval_ik`, joint/shape/geometry types |
| `newton/solvers.py` | `SolverBase`, `SolverFeatherstone`, `SolverMuJoCo`, `SolverSemiImplicit`, `SolverXPBD`, `SolverVBD`, `SolverStyle3D`, `SolverImplicitMPM` |
| `newton/geometry.py` | Broad-phase algorithms, `collide_*` functions, SDF, inertia computation |
| `newton/sensors.py` | `SensorContact`, `SensorIMU`, `SensorRaycast`, `SensorFrameTransform`, `SensorTiledCamera` |
| `newton/viewer.py` | `ViewerGL`, `ViewerUSD`, `ViewerRerun`, `ViewerViser`, `ViewerFile`, `ViewerNull` |
| `newton/ik.py` | `IKSolver`, optimizers, objectives, samplers |
| `newton/utils.py` | Mesh creation, spatial math, asset downloading, benchmarking |
| `newton/selection.py` | `ArticulationView` |
| `newton/math.py` | Smooth min/max, vector utilities |
| `newton/usd.py` | USD attribute/transform helpers |

Any new user-facing symbol added under `_src` must be re-exported in the appropriate public module.

### Core Data Model: Model / State / Control / Contacts

These four objects separate static scene definition from dynamic simulation data:

- **`Model`** - Static scene: geometry, topology (bodies, joints, articulations), materials, springs, triangles, tetrahedra, muscles, equality constraints. Supports multi-world grouping via `*_world` / `*_world_start` arrays. Created by `ModelBuilder.finalize()`.
- **`State`** - Time-varying dynamics: `particle_q/qd/f`, `body_q/qd/f`, `joint_q/qd`. Body transforms are `wp.transform` (7-DOF), velocities are `wp.spatial_vector` (6-DOF). Created via `Model.state()`.
- **`Control`** - Time-varying inputs: `joint_f`, `joint_target_pos/vel`, activations. Supports namespaced custom arrays (e.g., `control.mujoco.ctrl`). Created via `Model.control()`.
- **`Contacts`** - Per-frame collision data with separate rigid-rigid and particle-shape buffers. Created via `Model.collide(state, pipeline)`.

Typical simulation loop:
```python
state_0, state_1 = model.state(), model.state()
control = model.control()
contacts = model.collide(state_0, collision_pipeline)
solver.step(state_0, state_1, control, contacts, dt)
# swap state_0, state_1
```

### ModelBuilder Pattern

`ModelBuilder` accumulates scene data in Python lists, then `finalize()` converts to GPU-resident `wp.array`s on the `Model`. Builder methods: `add_body()`, `add_joint()`, `add_shape_*()`, `add_particle()`, etc. Multi-environment: `add_world(sub_builder)` and `replicate()`. Asset loading: `parse_mjcf()`, `parse_urdf()`, `parse_usd()` (standalone functions in `_src/utils/` that populate a builder).

### Solver Architecture

Seven solver backends inherit from `SolverBase` (`_src/solvers/solver.py`), which defines the `step(state_in, state_out, control, contacts, dt)` interface:

| Solver | Coordinates | Method | Use case |
|---|---|---|---|
| `SolverFeatherstone` | Generalized (reduced) | Explicit Euler, CRBA | Articulated rigid bodies |
| `SolverMuJoCo` | Generalized | Wraps mujoco_warp | MuJoCo-compatible simulation |
| `SolverSemiImplicit` | Maximal | Symplectic Euler | Rigid bodies, particles, cloth, soft bodies |
| `SolverXPBD` | Maximal | Position-based (XPBD) | Interactive constraint solving |
| `SolverVBD` | Particle + rigid | Vertex Block Descent | Cloth with self-collision |
| `SolverStyle3D` | Particle | Projective dynamics + PCG | Cloth simulation |
| `SolverImplicitMPM` | Particle (continuum) | MPM via warp.fem | Granular/fluid materials |

Kernel code is separated from solver logic (e.g., `kernels.py` alongside `solver_*.py`). The SemiImplicit solver's kernel modules are reused by Featherstone for particle/cloth.

### Collision Pipeline

`CollisionPipeline` and `CollisionPipelineUnified` handle broadphase + narrowphase. Three broadphase modes: `BroadPhaseAllPairs` (O(N^2)), `BroadPhaseSAP` (sweep-and-prune), `BroadPhaseExplicit` (precomputed pairs). Narrowphase uses GJK+EPA with support functions. SDF-based and hydroelastic contact models available.

### Viewer System

`ViewerBase` (`_src/viewer/viewer.py`) with six backends: `ViewerGL` (OpenGL real-time), `ViewerUSD` (offline file), `ViewerRerun`, `ViewerViser` (web-based), `ViewerFile` (record to file), `ViewerNull` (headless testing).

### Example Framework

Examples under `newton/examples/` (subfolders: `basic/`, `robot/`, `cable/`, `cloth/`, `diffsim/`, `ik/`, `mpm/`, `sensors/`, `selection/`). Each defines an `Example` class with `__init__(viewer, args)`, `step()`, `render()`, `test_final()`, and optionally `test_post_step()` and `gui(ui)`. Run via `uv run -m newton.examples <name>`. Examples support `--viewer`, `--device`, `--num-frames`, `--output-path` CLI args.

### Test Framework

Tests under `newton/tests/` use `unittest.TestCase` with `unittest_parallel` (vendored) for parallel execution. ~70+ test files organized by feature. `test_examples.py` runs examples as subprocesses via `--test` flag. Filter: `-k <pattern>`.

### Warp Usage

All performance-critical paths use `@wp.kernel` launched via `wp.launch()`. Data stored as `wp.array` with Warp dtypes (`wp.vec3`, `wp.transform`, `wp.spatial_vector`). `@wp.func` for device-side helpers. `@wp.struct` for passing complex data to kernels. CUDA graph capture via `wp.ScopedCapture` for eliminating CPU overhead.

### Internal `_src` Directory Map

| Directory | Purpose |
|---|---|
| `_src/core/` | Foundation types (`types.py`), spatial math (`spatial.py`) |
| `_src/sim/` | Model, State, Control, Contacts, ModelBuilder, collision pipelines, articulation FK/IK, joint types |
| `_src/sim/ik/` | IK solver, optimizers (L-BFGS, Levenberg-Marquardt), objectives |
| `_src/solvers/` | `solver.py` base + subdirectories for each backend |
| `_src/geometry/` | Broadphase, narrowphase (GJK/EPA/MPR), SDF, collision primitives, contact reduction, inertia |
| `_src/sensors/` | Sensor implementations + `warp_raytrace/` GPU ray-tracing renderer |
| `_src/viewer/` | Viewer base + backends, `gl/` OpenGL renderer, camera, picking |
| `_src/utils/` | MJCF/URDF/USD parsers, mesh utilities, asset downloading, benchmarking |
| `_src/usd/` | USD schema parsing, schema resolver |

## Development Guidelines

@AGENTS.md
