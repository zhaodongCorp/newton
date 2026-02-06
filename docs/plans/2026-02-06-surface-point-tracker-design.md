# Surface Point Tracker Design

## Overview

A standalone utility that samples 3D points on initial mesh surfaces of a simulated scene (rigid bodies + deformables), tracks their world-space positions each frame, and saves the resulting trajectories to a single NPZ file.

## API

```python
import newton
from newton.utils import SurfacePointTracker

model = builder.finalize()
state_0 = model.state()

# Sample points on initial mesh surfaces
tracker = SurfacePointTracker(model, state_0, num_points=10000)

# Simulation loop
for frame in range(num_frames):
    solver.step(state_0, state_1, control, contacts, dt)
    tracker.record(state_1)
    state_0, state_1 = state_1, state_0

# Save trajectories
tracker.save("trajectories.npz")
```

- `__init__(model, state, num_points)` — samples points on all visible mesh surfaces, distributed proportional to surface area.
- `record(state)` — computes and stores world-space positions for the current frame.
- `save(path)` — writes trajectories to a compressed NPZ file.

## Sampling Strategy

At construction time:

1. **Collect all triangulated surfaces.** Iterate over all shapes in the model:
   - **Rigid body meshes** — shapes with `GEO_MESH` use their triangle data directly. Primitive shapes (`GEO_BOX`, `GEO_SPHERE`, `GEO_CAPSULE`, `GEO_CYLINDER`, `GEO_CONE`) are triangulated using existing Newton utilities (`create_box_mesh`, `create_sphere_mesh`, etc.).
   - **Deformable surfaces** — cloth and soft body triangles from `model.tri_indices`, referencing `particle_q` positions.

2. **Compute per-triangle areas** across all surfaces. Allocate points per triangle proportional to `triangle_area / total_area * num_points`.

3. **Sample within each triangle** using the standard uniform-in-triangle method: generate random `u, v ~ Uniform(0,1)`, if `u + v > 1` fold to `u, v = 1-u, 1-v`, yielding barycentric coordinates `(1-u-v, u, v)`.

4. **Store per-point metadata** in flat arrays (internal, not serialized):
   - `body_index: int` — which body this point belongs to (-1 for deformable particles)
   - `tri_index: int` — triangle index (into mesh or `model.tri_indices`)
   - `bary_coords: vec3` — barycentric coordinates `(u, v, w)`
   - `is_rigid: bool` — determines the update path
   - For rigid points: `local_offset: vec3` — position in body-local space, computed from the initial state

## Per-Frame Position Update

`record(state)` runs a single `@wp.kernel` over all `num_points`:

- **Rigid body points** (`is_rigid=True`): read the body transform from `state.body_q[body_index]` and apply it to the stored `local_offset`:
  ```
  world_pos = wp.transform_point(body_transform, local_offset)
  ```

- **Deformable points** (`is_rigid=False`): look up the three vertex positions from `state.particle_q` via triangle connectivity and interpolate with barycentric coordinates:
  ```
  v0 = state.particle_q[tri_verts[0]]
  v1 = state.particle_q[tri_verts[1]]
  v2 = state.particle_q[tri_verts[2]]
  world_pos = bary.x * v0 + bary.y * v1 + bary.z * v2
  ```

Both paths execute in a single kernel launch, branching on the `is_rigid` flag. Output is written to a pre-allocated `wp.array(shape=(num_points,), dtype=wp.vec3)`, then transferred to CPU and appended to an internal list.

### Performance

For 10k points, the update kernel is lightweight relative to the physics step. The per-frame GPU-to-CPU transfer is the main cost. An alternative (pre-allocating a GPU buffer for all frames, single bulk transfer at `save()`) is possible but requires knowing `num_frames` upfront or dynamic resizing. The eager-transfer approach is simpler and sufficient for the expected scale.

## Output Format

`save(path)` writes a single compressed NPZ file:

```
positions: (num_points, num_frames, 3) float32
```

Only the trajectory positions are saved. Internal metadata (body IDs, barycentric coordinates, triangle indices) is used only during simulation for computing updates and is not serialized.

Loading:
```python
data = np.load("trajectories.npz")
trajectories = data["positions"]  # (10000, 500, 3)
```

Compression: `np.savez_compressed` is used. For 10k points x 500 frames, raw size is ~60MB, compressed typically ~10-20MB.

## File Changes

- **New:** `newton/_src/utils/surface_point_tracker.py` — `SurfacePointTracker` class, sampling logic, Warp update kernel, `save()`.
- **Edit:** `newton/utils.py` — add `SurfacePointTracker` to public re-exports.

## Dependencies

None beyond what Newton already has (NumPy via Warp, Warp itself, existing mesh utilities in `newton/_src/utils/mesh.py`).

## Scope Boundaries

- Supports rigid bodies and deformable meshes (cloth, soft bodies).
- Does not support MPM particles (no mesh surface to sample on).
- No viewer integration — purely a data utility.
- No solver coupling — works with any solver backend.
