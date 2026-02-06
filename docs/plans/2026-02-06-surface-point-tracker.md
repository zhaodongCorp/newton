# Surface Point Tracker Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use 10x-engineer:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone utility that samples 3D points on mesh surfaces, tracks their positions each simulation frame, and saves trajectories to NPZ.

**Architecture:** A single `SurfacePointTracker` class in `newton/_src/utils/surface_point_tracker.py`, re-exported via `newton/utils.py`. Sampling runs on CPU at construction time using NumPy. Per-frame position updates run as a Warp kernel on GPU. Frame data is copied to CPU eagerly and stacked at save time.

**Tech Stack:** NumPy (sampling, I/O), Warp (GPU kernels for position updates), existing Newton mesh utilities.

---

### Task 1: Scaffold the module and re-export

**Files:**
- Create: `newton/_src/utils/surface_point_tracker.py`
- Modify: `newton/utils.py`

**Step 1: Create the empty module with class stub**

Create `newton/_src/utils/surface_point_tracker.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np
import warp as wp


class SurfacePointTracker:
    """Samples 3D points on mesh surfaces and tracks their trajectories during simulation.

    Args:
        model: A finalized Newton Model.
        state: The initial simulation State (used to compute initial positions).
        num_points: Total number of points to sample, distributed proportional to surface area.
        seed: Random seed for reproducible sampling.
    """

    def __init__(self, model, state, num_points: int = 10000, seed: int = 42):
        raise NotImplementedError

    def record(self, state) -> None:
        """Record world-space positions of all tracked points for the current frame."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save recorded trajectories to a compressed NPZ file.

        The file contains a single array ``positions`` with shape ``(num_points, num_frames, 3)``.
        """
        raise NotImplementedError
```

**Step 2: Add the re-export to `newton/utils.py`**

Add the following block at the end of `newton/utils.py`, following the existing section pattern:

```python
# ==================================================================================
# surface point tracking
# ==================================================================================

from ._src.utils.surface_point_tracker import SurfacePointTracker  # noqa: E402

__all__ += [
    "SurfacePointTracker",
]
```

**Step 3: Verify the import works**

```bash
uv run python -c "from newton.utils import SurfacePointTracker; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add newton/_src/utils/surface_point_tracker.py newton/utils.py
git commit -m "$(cat <<'EOF'
Add SurfacePointTracker scaffold and public re-export

Stub class with __init__, record, save methods. Re-exported
via newton.utils for the public API.
EOF
)"
```

---

### Task 2: Implement triangle collection from model shapes

This task implements the private method that collects all triangulated surfaces from the model — both rigid body shapes (meshes and primitives) and deformable triangles.

**Files:**
- Modify: `newton/_src/utils/surface_point_tracker.py`
- Create: `newton/tests/test_surface_point_tracker.py`

**Step 1: Write the test**

Create `newton/tests/test_surface_point_tracker.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.types import GeoType


class TestTriangleCollection(unittest.TestCase):
    """Test that _collect_triangles finds surfaces from rigid shapes and deformable triangles."""

    def test_rigid_box(self):
        """A single rigid box produces 12 triangles (6 faces x 2 tris)."""
        builder = newton.ModelBuilder()
        b = builder.add_body()
        builder.add_shape_box(body=b, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize(device="cpu")
        state = model.state()

        from newton._src.utils.surface_point_tracker import SurfacePointTracker

        surfaces = SurfacePointTracker._collect_triangles(model, state)
        self.assertGreater(len(surfaces), 0)
        total_tris = sum(s["num_triangles"] for s in surfaces)
        self.assertEqual(total_tris, 12)

    def test_rigid_sphere(self):
        """A single rigid sphere produces triangles from the tessellated mesh."""
        builder = newton.ModelBuilder()
        b = builder.add_body()
        builder.add_shape_sphere(body=b, radius=1.0)
        model = builder.finalize(device="cpu")
        state = model.state()

        from newton._src.utils.surface_point_tracker import SurfacePointTracker

        surfaces = SurfacePointTracker._collect_triangles(model, state)
        self.assertGreater(len(surfaces), 0)
        total_tris = sum(s["num_triangles"] for s in surfaces)
        self.assertGreater(total_tris, 0)

    def test_deformable_cloth(self):
        """Deformable triangles from particles are collected."""
        builder = newton.ModelBuilder()
        p0 = builder.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(), 1.0)
        p1 = builder.add_particle(wp.vec3(1.0, 0.0, 0.0), wp.vec3(), 1.0)
        p2 = builder.add_particle(wp.vec3(0.0, 1.0, 0.0), wp.vec3(), 1.0)
        p3 = builder.add_particle(wp.vec3(1.0, 1.0, 0.0), wp.vec3(), 1.0)
        builder.add_triangle(p0, p1, p2)
        builder.add_triangle(p1, p3, p2)
        model = builder.finalize(device="cpu")
        state = model.state()

        from newton._src.utils.surface_point_tracker import SurfacePointTracker

        surfaces = SurfacePointTracker._collect_triangles(model, state)
        total_tris = sum(s["num_triangles"] for s in surfaces)
        self.assertEqual(total_tris, 2)

    def test_mixed_scene(self):
        """Scene with both rigid shapes and deformable triangles."""
        builder = newton.ModelBuilder()
        b = builder.add_body()
        builder.add_shape_box(body=b, hx=0.5, hy=0.5, hz=0.5)
        p0 = builder.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(), 1.0)
        p1 = builder.add_particle(wp.vec3(1.0, 0.0, 0.0), wp.vec3(), 1.0)
        p2 = builder.add_particle(wp.vec3(0.0, 1.0, 0.0), wp.vec3(), 1.0)
        builder.add_triangle(p0, p1, p2)
        model = builder.finalize(device="cpu")
        state = model.state()

        from newton._src.utils.surface_point_tracker import SurfacePointTracker

        surfaces = SurfacePointTracker._collect_triangles(model, state)
        total_tris = sum(s["num_triangles"] for s in surfaces)
        self.assertEqual(total_tris, 13)  # 12 from box + 1 deformable


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

```bash
uv run --extra dev -m newton.tests -k test_surface_point_tracker
```

Expected: FAIL (NotImplementedError or AttributeError for `_collect_triangles`)

**Step 3: Implement `_collect_triangles`**

In `newton/_src/utils/surface_point_tracker.py`, add the following imports at the top (after the existing imports):

```python
from newton._src.geometry.types import GeoType, Mesh
from newton._src.utils.mesh import (
    create_box_mesh,
    create_capsule_mesh,
    create_cone_mesh,
    create_cylinder_mesh,
    create_sphere_mesh,
)
```

Then add this static method to the `SurfacePointTracker` class:

```python
    @staticmethod
    def _collect_triangles(model, state):
        """Collect all triangulated surfaces from the model.

        Returns a list of dicts, each with:
            - vertices: np.ndarray (N, 3) — world-space vertex positions
            - indices: np.ndarray (M*3,) — flat triangle indices into vertices
            - num_triangles: int
            - body_index: int — body this surface belongs to (-1 for deformable particles)
            - is_rigid: bool
            - shape_index: int — shape index for rigid, -1 for deformable
        """
        surfaces = []

        # --- Rigid body shapes ---
        shape_count = model.shape_count
        for s_idx in range(shape_count):
            geo_type = model.shape_type.numpy()[s_idx]
            body_idx = model.shape_body.numpy()[s_idx]
            scale = model.shape_scale.numpy()[s_idx]
            shape_xform = model.shape_transform.numpy()[s_idx]

            vertices = None
            indices = None

            if geo_type == int(GeoType.BOX):
                verts_8col, indices = create_box_mesh(extents=scale)
                vertices = verts_8col[:, :3]
            elif geo_type == int(GeoType.SPHERE):
                radius = float(scale[0])
                verts_8col, indices = create_sphere_mesh(radius=radius)
                vertices = verts_8col[:, :3]
            elif geo_type == int(GeoType.CAPSULE):
                radius, half_height = float(scale[0]), float(scale[1])
                verts_8col, indices = create_capsule_mesh(radius=radius, half_height=half_height)
                vertices = verts_8col[:, :3]
            elif geo_type == int(GeoType.CYLINDER):
                radius, half_height = float(scale[0]), float(scale[1])
                verts_8col, indices = create_cylinder_mesh(radius=radius, half_height=half_height)
                vertices = verts_8col[:, :3]
            elif geo_type == int(GeoType.CONE):
                radius, half_height = float(scale[0]), float(scale[1])
                verts_8col, indices = create_cone_mesh(radius=radius, half_height=half_height)
                vertices = verts_8col[:, :3]
            elif geo_type == int(GeoType.MESH):
                mesh_src = model.shape_source[s_idx]
                if mesh_src is not None and isinstance(mesh_src, Mesh):
                    vertices = mesh_src.vertices.copy()
                    indices = mesh_src.indices.copy()
                    # Apply mesh scale
                    vertices = vertices * scale
            else:
                # Skip PLANE, SDF, HFIELD, ELLIPSOID, CONVEX_MESH, NONE
                continue

            if vertices is None or indices is None:
                continue

            num_triangles = len(indices) // 3

            # Transform vertices to world space using shape transform and body transform
            # shape_xform is a wp.transform (7 floats: px, py, pz, qx, qy, qz, qw)
            shape_tf = wp.transform(
                shape_xform[:3].tolist(),
                wp.quat(shape_xform[3], shape_xform[4], shape_xform[5], shape_xform[6]),
            )
            body_tf = wp.transform_identity()
            if body_idx >= 0:
                body_q = state.body_q.numpy()[body_idx]
                body_tf = wp.transform(
                    body_q[:3].tolist(),
                    wp.quat(body_q[3], body_q[4], body_q[5], body_q[6]),
                )
            world_tf = wp.transform_multiply(body_tf, shape_tf)

            world_verts = np.zeros_like(vertices)
            for i in range(len(vertices)):
                p = wp.transform_point(world_tf, wp.vec3(vertices[i][0], vertices[i][1], vertices[i][2]))
                world_verts[i] = [p[0], p[1], p[2]]

            surfaces.append(
                {
                    "vertices": world_verts,
                    "indices": indices.astype(np.int32),
                    "num_triangles": num_triangles,
                    "body_index": int(body_idx),
                    "is_rigid": True,
                    "shape_index": s_idx,
                }
            )

        # --- Deformable triangles (cloth / soft body) ---
        if model.tri_count > 0:
            particle_q = state.particle_q.numpy()  # (particle_count, 3)
            tri_indices = model.tri_indices.numpy()  # (tri_count * 3,)

            surfaces.append(
                {
                    "vertices": particle_q.copy(),
                    "indices": tri_indices.copy().astype(np.int32),
                    "num_triangles": model.tri_count,
                    "body_index": -1,
                    "is_rigid": False,
                    "shape_index": -1,
                }
            )

        return surfaces
```

**Step 4: Run tests to verify they pass**

```bash
uv run --extra dev -m newton.tests -k test_surface_point_tracker
```

Expected: PASS (all 4 tests in `TestTriangleCollection`)

**Step 5: Commit**

```bash
git add newton/_src/utils/surface_point_tracker.py newton/tests/test_surface_point_tracker.py
git commit -m "$(cat <<'EOF'
Implement triangle collection from model shapes

Collects triangulated surfaces from rigid shapes (box, sphere,
capsule, cylinder, cone, mesh) and deformable triangles. Transforms
rigid mesh vertices to world space.
EOF
)"
```

---

### Task 3: Implement area-proportional point sampling

**Files:**
- Modify: `newton/_src/utils/surface_point_tracker.py`
- Modify: `newton/tests/test_surface_point_tracker.py`

**Step 1: Write the test**

Add this test class to `newton/tests/test_surface_point_tracker.py`:

```python
class TestSampling(unittest.TestCase):
    """Test area-proportional point sampling on triangle surfaces."""

    def test_point_count(self):
        """Sampled point count matches requested num_points."""
        builder = newton.ModelBuilder()
        b = builder.add_body()
        builder.add_shape_box(body=b, hx=1.0, hy=1.0, hz=1.0)
        model = builder.finalize(device="cpu")
        state = model.state()

        from newton._src.utils.surface_point_tracker import SurfacePointTracker

        result = SurfacePointTracker._sample_points_on_surfaces(
            SurfacePointTracker._collect_triangles(model, state),
            num_points=500,
            seed=42,
        )
        self.assertEqual(result["bary_coords"].shape[0], 500)
        self.assertEqual(result["surface_tri_index"].shape[0], 500)

    def test_barycentric_validity(self):
        """All barycentric coordinates are non-negative and sum to 1."""
        builder = newton.ModelBuilder()
        b = builder.add_body()
        builder.add_shape_sphere(body=b, radius=1.0)
        model = builder.finalize(device="cpu")
        state = model.state()

        from newton._src.utils.surface_point_tracker import SurfacePointTracker

        result = SurfacePointTracker._sample_points_on_surfaces(
            SurfacePointTracker._collect_triangles(model, state),
            num_points=1000,
            seed=42,
        )
        bary = result["bary_coords"]
        self.assertTrue(np.all(bary >= 0.0))
        np.testing.assert_allclose(bary.sum(axis=1), 1.0, atol=1e-6)

    def test_area_proportional_distribution(self):
        """Points distribute roughly proportional to surface area across surfaces."""
        builder = newton.ModelBuilder()
        # Large box
        b1 = builder.add_body()
        builder.add_shape_box(body=b1, hx=2.0, hy=2.0, hz=2.0)
        # Small box
        b2 = builder.add_body()
        builder.add_shape_box(body=b2, hx=0.25, hy=0.25, hz=0.25)
        model = builder.finalize(device="cpu")
        state = model.state()

        from newton._src.utils.surface_point_tracker import SurfacePointTracker

        result = SurfacePointTracker._sample_points_on_surfaces(
            SurfacePointTracker._collect_triangles(model, state),
            num_points=10000,
            seed=42,
        )
        # Large box should get the vast majority of points
        large_count = np.sum(result["surface_index"] == 0)
        small_count = np.sum(result["surface_index"] == 1)
        self.assertGreater(large_count, small_count * 5)

    def test_reproducible_with_seed(self):
        """Same seed produces identical samples."""
        builder = newton.ModelBuilder()
        b = builder.add_body()
        builder.add_shape_box(body=b, hx=1.0, hy=1.0, hz=1.0)
        model = builder.finalize(device="cpu")
        state = model.state()

        from newton._src.utils.surface_point_tracker import SurfacePointTracker

        surfaces = SurfacePointTracker._collect_triangles(model, state)
        r1 = SurfacePointTracker._sample_points_on_surfaces(surfaces, num_points=100, seed=123)
        r2 = SurfacePointTracker._sample_points_on_surfaces(surfaces, num_points=100, seed=123)
        np.testing.assert_array_equal(r1["bary_coords"], r2["bary_coords"])
```

**Step 2: Run tests to verify they fail**

```bash
uv run --extra dev -m newton.tests -k TestSampling
```

Expected: FAIL (AttributeError for `_sample_points_on_surfaces`)

**Step 3: Implement `_sample_points_on_surfaces`**

Add this static method to the `SurfacePointTracker` class:

```python
    @staticmethod
    def _sample_points_on_surfaces(surfaces, num_points, seed=42):
        """Sample points on triangle surfaces proportional to area.

        Returns a dict with:
            - bary_coords: (num_points, 3) — barycentric coordinates per point
            - surface_index: (num_points,) — which surface group
            - surface_tri_index: (num_points,) — which triangle within the surface
        """
        rng = np.random.default_rng(seed)

        # Compute per-triangle areas across all surfaces
        all_areas = []
        surface_ids = []
        tri_ids = []

        for surf_idx, surf in enumerate(surfaces):
            verts = surf["vertices"]
            idxs = surf["indices"].reshape(-1, 3)
            for tri_i in range(surf["num_triangles"]):
                v0 = verts[idxs[tri_i, 0]]
                v1 = verts[idxs[tri_i, 1]]
                v2 = verts[idxs[tri_i, 2]]
                area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                all_areas.append(area)
                surface_ids.append(surf_idx)
                tri_ids.append(tri_i)

        all_areas = np.array(all_areas, dtype=np.float64)
        total_area = all_areas.sum()

        if total_area <= 0.0:
            raise ValueError("Total mesh surface area is zero; cannot sample points.")

        # Allocate points per triangle proportional to area
        probs = all_areas / total_area
        tri_assignments = rng.choice(len(all_areas), size=num_points, p=probs)

        # Sample barycentric coordinates within each assigned triangle
        u = rng.random(num_points)
        v = rng.random(num_points)
        fold = u + v > 1.0
        u[fold] = 1.0 - u[fold]
        v[fold] = 1.0 - v[fold]
        w = 1.0 - u - v

        bary_coords = np.stack([w, u, v], axis=1).astype(np.float32)
        surface_index = np.array([surface_ids[t] for t in tri_assignments], dtype=np.int32)
        surface_tri_index = np.array([tri_ids[t] for t in tri_assignments], dtype=np.int32)

        return {
            "bary_coords": bary_coords,
            "surface_index": surface_index,
            "surface_tri_index": surface_tri_index,
        }
```

**Step 4: Run tests to verify they pass**

```bash
uv run --extra dev -m newton.tests -k TestSampling
```

Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add newton/_src/utils/surface_point_tracker.py newton/tests/test_surface_point_tracker.py
git commit -m "$(cat <<'EOF'
Implement area-proportional point sampling on triangle surfaces

Distributes sample points across triangles proportional to area.
Uses uniform-in-triangle sampling with barycentric fold trick.
Reproducible via seed parameter.
EOF
)"
```

---

### Task 4: Implement `__init__` — wire sampling into per-point metadata arrays

This task connects `_collect_triangles` and `_sample_points_on_surfaces` to build the internal arrays needed by the update kernel.

**Files:**
- Modify: `newton/_src/utils/surface_point_tracker.py`
- Modify: `newton/tests/test_surface_point_tracker.py`

**Step 1: Write the test**

Add this test class to `newton/tests/test_surface_point_tracker.py`:

```python
class TestInit(unittest.TestCase):
    """Test SurfacePointTracker construction."""

    def test_init_rigid_box(self):
        """Tracker initializes with correct point count for a rigid box."""
        builder = newton.ModelBuilder()
        b = builder.add_body()
        builder.add_shape_box(body=b, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize(device="cpu")
        state = model.state()

        from newton.utils import SurfacePointTracker

        tracker = SurfacePointTracker(model, state, num_points=200)
        self.assertEqual(tracker.num_points, 200)
        self.assertEqual(len(tracker._frames), 0)

    def test_init_deformable(self):
        """Tracker initializes for deformable triangles."""
        builder = newton.ModelBuilder()
        p0 = builder.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(), 1.0)
        p1 = builder.add_particle(wp.vec3(1.0, 0.0, 0.0), wp.vec3(), 1.0)
        p2 = builder.add_particle(wp.vec3(0.0, 1.0, 0.0), wp.vec3(), 1.0)
        builder.add_triangle(p0, p1, p2)
        model = builder.finalize(device="cpu")
        state = model.state()

        from newton.utils import SurfacePointTracker

        tracker = SurfacePointTracker(model, state, num_points=100)
        self.assertEqual(tracker.num_points, 100)

    def test_init_mixed(self):
        """Tracker initializes for a mixed rigid + deformable scene."""
        builder = newton.ModelBuilder()
        b = builder.add_body()
        builder.add_shape_box(body=b, hx=0.5, hy=0.5, hz=0.5)
        p0 = builder.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(), 1.0)
        p1 = builder.add_particle(wp.vec3(1.0, 0.0, 0.0), wp.vec3(), 1.0)
        p2 = builder.add_particle(wp.vec3(0.0, 1.0, 0.0), wp.vec3(), 1.0)
        builder.add_triangle(p0, p1, p2)
        model = builder.finalize(device="cpu")
        state = model.state()

        from newton.utils import SurfacePointTracker

        tracker = SurfacePointTracker(model, state, num_points=300)
        self.assertEqual(tracker.num_points, 300)
```

**Step 2: Run tests to verify they fail**

```bash
uv run --extra dev -m newton.tests -k TestInit
```

Expected: FAIL (NotImplementedError from `__init__`)

**Step 3: Implement `__init__`**

Replace the `__init__` method in `SurfacePointTracker`:

```python
    def __init__(self, model, state, num_points: int = 10000, seed: int = 42):
        self.num_points = num_points
        self._device = model.device
        self._frames = []

        surfaces = self._collect_triangles(model, state)
        if not surfaces:
            raise ValueError("No triangulated surfaces found in model.")

        sampled = self._sample_points_on_surfaces(surfaces, num_points=num_points, seed=seed)

        # Build per-point arrays for the update kernel
        is_rigid = np.zeros(num_points, dtype=np.int32)
        body_index = np.full(num_points, -1, dtype=np.int32)
        local_offset = np.zeros((num_points, 3), dtype=np.float32)
        bary_coords = sampled["bary_coords"]  # (num_points, 3)
        # Triangle vertex indices into particle_q (for deformable) — 3 ints per point
        tri_v0 = np.zeros(num_points, dtype=np.int32)
        tri_v1 = np.zeros(num_points, dtype=np.int32)
        tri_v2 = np.zeros(num_points, dtype=np.int32)

        for i in range(num_points):
            surf_idx = sampled["surface_index"][i]
            tri_idx = sampled["surface_tri_index"][i]
            surf = surfaces[surf_idx]
            idxs = surf["indices"].reshape(-1, 3)
            verts = surf["vertices"]
            bary = bary_coords[i]

            v0_idx, v1_idx, v2_idx = idxs[tri_idx]
            v0_pos = verts[v0_idx]
            v1_pos = verts[v1_idx]
            v2_pos = verts[v2_idx]
            world_pos = bary[0] * v0_pos + bary[1] * v1_pos + bary[2] * v2_pos

            if surf["is_rigid"]:
                is_rigid[i] = 1
                body_index[i] = surf["body_index"]

                # Compute local offset: inverse transform world_pos into body space
                body_tf = wp.transform_identity()
                if body_index[i] >= 0:
                    body_q = state.body_q.numpy()[body_index[i]]
                    body_tf = wp.transform(
                        body_q[:3].tolist(),
                        wp.quat(body_q[3], body_q[4], body_q[5], body_q[6]),
                    )
                inv_tf = wp.transform_inverse(body_tf)
                local_p = wp.transform_point(inv_tf, wp.vec3(world_pos[0], world_pos[1], world_pos[2]))
                local_offset[i] = [local_p[0], local_p[1], local_p[2]]
            else:
                # Deformable: store particle indices from tri_indices
                tri_v0[i] = v0_idx
                tri_v1[i] = v1_idx
                tri_v2[i] = v2_idx

        # Upload to Warp arrays on device
        self._is_rigid = wp.array(is_rigid, dtype=wp.int32, device=self._device)
        self._body_index = wp.array(body_index, dtype=wp.int32, device=self._device)
        self._local_offset = wp.array(local_offset, dtype=wp.vec3, device=self._device)
        self._bary_coords = wp.array(bary_coords, dtype=wp.vec3, device=self._device)
        self._tri_v0 = wp.array(tri_v0, dtype=wp.int32, device=self._device)
        self._tri_v1 = wp.array(tri_v1, dtype=wp.int32, device=self._device)
        self._tri_v2 = wp.array(tri_v2, dtype=wp.int32, device=self._device)

        # Pre-allocate output buffer for a single frame
        self._frame_positions = wp.zeros(num_points, dtype=wp.vec3, device=self._device)
```

**Step 4: Run tests to verify they pass**

```bash
uv run --extra dev -m newton.tests -k TestInit
```

Expected: PASS

**Step 5: Commit**

```bash
git add newton/_src/utils/surface_point_tracker.py newton/tests/test_surface_point_tracker.py
git commit -m "$(cat <<'EOF'
Implement SurfacePointTracker.__init__

Wires triangle collection and sampling into per-point metadata
arrays (body index, local offset, barycentric coords, triangle
vertex indices). Uploads to Warp arrays on the model device.
EOF
)"
```

---

### Task 5: Implement the Warp update kernel and `record()`

**Files:**
- Modify: `newton/_src/utils/surface_point_tracker.py`
- Modify: `newton/tests/test_surface_point_tracker.py`

**Step 1: Write the test**

Add this test class to `newton/tests/test_surface_point_tracker.py`:

```python
class TestRecord(unittest.TestCase):
    """Test per-frame position recording."""

    def test_record_rigid_stationary(self):
        """Points on a stationary rigid box stay at their initial positions."""
        builder = newton.ModelBuilder()
        b = builder.add_body()
        builder.add_shape_box(body=b, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize(device="cpu")
        state = model.state()

        from newton.utils import SurfacePointTracker

        tracker = SurfacePointTracker(model, state, num_points=50, seed=42)
        tracker.record(state)
        tracker.record(state)

        self.assertEqual(len(tracker._frames), 2)
        np.testing.assert_allclose(tracker._frames[0], tracker._frames[1], atol=1e-5)

    def test_record_rigid_translated(self):
        """Points move when the body is translated."""
        builder = newton.ModelBuilder()
        b = builder.add_body()
        builder.add_shape_box(body=b, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize(device="cpu")

        state0 = model.state()
        tracker = SurfacePointTracker(model, state0, num_points=50, seed=42)
        tracker.record(state0)

        # Move the body by (1, 0, 0)
        state1 = model.state()
        body_q = state1.body_q.numpy()
        body_q[0][:3] += [1.0, 0.0, 0.0]
        state1.body_q.assign(wp.array(body_q, dtype=wp.transform, device="cpu"))
        tracker.record(state1)

        # All points should have shifted by (1, 0, 0)
        diff = tracker._frames[1] - tracker._frames[0]
        np.testing.assert_allclose(diff, np.array([1.0, 0.0, 0.0]), atol=1e-5)

    def test_record_deformable(self):
        """Points on deformable mesh update when particles move."""
        builder = newton.ModelBuilder()
        p0 = builder.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(), 1.0)
        p1 = builder.add_particle(wp.vec3(1.0, 0.0, 0.0), wp.vec3(), 1.0)
        p2 = builder.add_particle(wp.vec3(0.0, 1.0, 0.0), wp.vec3(), 1.0)
        builder.add_triangle(p0, p1, p2)
        model = builder.finalize(device="cpu")

        state0 = model.state()
        from newton.utils import SurfacePointTracker

        tracker = SurfacePointTracker(model, state0, num_points=50, seed=42)
        tracker.record(state0)

        # Move all particles up by 2.0
        state1 = model.state()
        pq = state1.particle_q.numpy()
        pq[:, 2] += 2.0
        state1.particle_q.assign(wp.array(pq, dtype=wp.vec3, device="cpu"))
        tracker.record(state1)

        # All points should have shifted by (0, 0, 2)
        diff = tracker._frames[1] - tracker._frames[0]
        np.testing.assert_allclose(diff[:, 2], 2.0, atol=1e-5)
        np.testing.assert_allclose(diff[:, :2], 0.0, atol=1e-5)
```

**Step 2: Run tests to verify they fail**

```bash
uv run --extra dev -m newton.tests -k TestRecord
```

Expected: FAIL (NotImplementedError from `record`)

**Step 3: Implement the kernel and `record()`**

Add the Warp kernel **outside** the class (at module level, after imports but before the class definition):

```python
@wp.kernel
def _update_point_positions(
    is_rigid: wp.array(dtype=wp.int32),
    body_index: wp.array(dtype=wp.int32),
    local_offset: wp.array(dtype=wp.vec3),
    bary_coords: wp.array(dtype=wp.vec3),
    tri_v0: wp.array(dtype=wp.int32),
    tri_v1: wp.array(dtype=wp.int32),
    tri_v2: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    particle_q: wp.array(dtype=wp.vec3),
    out_positions: wp.array(dtype=wp.vec3),
    has_particles: wp.int32,
    has_bodies: wp.int32,
):
    tid = wp.tid()

    if is_rigid[tid] == 1:
        # Rigid body: apply body transform to local offset
        b_idx = body_index[tid]
        if has_bodies == 1 and b_idx >= 0:
            xform = body_q[b_idx]
        else:
            xform = wp.transform_identity()
        out_positions[tid] = wp.transform_point(xform, local_offset[tid])
    else:
        # Deformable: barycentric interpolation on particle positions
        bary = bary_coords[tid]
        if has_particles == 1:
            v0 = particle_q[tri_v0[tid]]
            v1 = particle_q[tri_v1[tid]]
            v2 = particle_q[tri_v2[tid]]
            out_positions[tid] = bary[0] * v0 + bary[1] * v1 + bary[2] * v2
        else:
            out_positions[tid] = wp.vec3(0.0, 0.0, 0.0)
```

Then replace the `record` method:

```python
    def record(self, state) -> None:
        """Record world-space positions of all tracked points for the current frame."""
        has_bodies = 1 if state.body_q is not None and state.body_q.shape[0] > 0 else 0
        has_particles = 1 if state.particle_q is not None and state.particle_q.shape[0] > 0 else 0

        # Use dummy arrays if body_q or particle_q don't exist
        body_q = state.body_q if has_bodies else wp.zeros(1, dtype=wp.transform, device=self._device)
        particle_q = state.particle_q if has_particles else wp.zeros(1, dtype=wp.vec3, device=self._device)

        wp.launch(
            kernel=_update_point_positions,
            dim=self.num_points,
            inputs=[
                self._is_rigid,
                self._body_index,
                self._local_offset,
                self._bary_coords,
                self._tri_v0,
                self._tri_v1,
                self._tri_v2,
                body_q,
                particle_q,
                self._frame_positions,
                has_particles,
                has_bodies,
            ],
            device=self._device,
        )

        # Copy to CPU and store
        self._frames.append(self._frame_positions.numpy().copy())
```

**Step 4: Run tests to verify they pass**

```bash
uv run --extra dev -m newton.tests -k TestRecord
```

Expected: PASS

**Step 5: Commit**

```bash
git add newton/_src/utils/surface_point_tracker.py newton/tests/test_surface_point_tracker.py
git commit -m "$(cat <<'EOF'
Implement Warp kernel and record() for position updates

Single kernel handles both rigid (transform) and deformable
(barycentric interpolation) point updates. Copies frame data
to CPU after each kernel launch.
EOF
)"
```

---

### Task 6: Implement `save()`

**Files:**
- Modify: `newton/_src/utils/surface_point_tracker.py`
- Modify: `newton/tests/test_surface_point_tracker.py`

**Step 1: Write the test**

Add this test class to `newton/tests/test_surface_point_tracker.py`:

```python
class TestSave(unittest.TestCase):
    """Test trajectory saving to NPZ."""

    def test_save_and_load(self):
        """Saved NPZ contains positions with correct shape."""
        builder = newton.ModelBuilder()
        b = builder.add_body()
        builder.add_shape_box(body=b, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize(device="cpu")
        state = model.state()

        from newton.utils import SurfacePointTracker

        num_pts = 100
        tracker = SurfacePointTracker(model, state, num_points=num_pts)
        tracker.record(state)
        tracker.record(state)
        tracker.record(state)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name

        try:
            tracker.save(path)
            data = np.load(path)
            self.assertIn("positions", data)
            self.assertEqual(data["positions"].shape, (num_pts, 3, 3))
            self.assertEqual(data["positions"].dtype, np.float32)
        finally:
            os.unlink(path)

    def test_save_no_frames_raises(self):
        """Saving with no recorded frames raises an error."""
        builder = newton.ModelBuilder()
        b = builder.add_body()
        builder.add_shape_box(body=b, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize(device="cpu")
        state = model.state()

        from newton.utils import SurfacePointTracker

        tracker = SurfacePointTracker(model, state, num_points=50)
        with self.assertRaises(ValueError):
            tracker.save("/tmp/empty.npz")
```

**Step 2: Run tests to verify they fail**

```bash
uv run --extra dev -m newton.tests -k TestSave
```

Expected: FAIL (NotImplementedError from `save`)

**Step 3: Implement `save()`**

Replace the `save` method:

```python
    def save(self, path: str) -> None:
        """Save recorded trajectories to a compressed NPZ file.

        The file contains a single array ``positions`` with shape ``(num_points, num_frames, 3)``.
        """
        if not self._frames:
            raise ValueError("No frames recorded. Call record() at least once before save().")

        # Stack: list of (num_points, 3) -> (num_frames, num_points, 3) -> (num_points, num_frames, 3)
        stacked = np.stack(self._frames, axis=0)
        positions = np.transpose(stacked, (1, 0, 2))

        np.savez_compressed(path, positions=positions.astype(np.float32))
```

**Step 4: Run tests to verify they pass**

```bash
uv run --extra dev -m newton.tests -k TestSave
```

Expected: PASS

**Step 5: Commit**

```bash
git add newton/_src/utils/surface_point_tracker.py newton/tests/test_surface_point_tracker.py
git commit -m "$(cat <<'EOF'
Implement save() for trajectory output

Stacks recorded frames into (num_points, num_frames, 3) array
and writes compressed NPZ. Raises ValueError if no frames
recorded.
EOF
)"
```

---

### Task 7: End-to-end integration test with simulation

**Files:**
- Modify: `newton/tests/test_surface_point_tracker.py`

**Step 1: Write the test**

Add this test class to `newton/tests/test_surface_point_tracker.py`:

```python
from newton.solvers import SolverSemiImplicit


class TestEndToEnd(unittest.TestCase):
    """End-to-end test: build scene, simulate, track, save, verify."""

    def test_falling_box(self):
        """Track points on a box falling under gravity for 10 frames."""
        builder = newton.ModelBuilder()
        b = builder.add_body()
        builder.add_shape_box(body=b, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize(device="cpu")
        model.gravity = np.array([0.0, -9.81, 0.0])

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        solver = SolverSemiImplicit(model)

        from newton.utils import SurfacePointTracker

        tracker = SurfacePointTracker(model, state_0, num_points=50, seed=42)
        tracker.record(state_0)

        dt = 1.0 / 60.0
        num_frames = 10
        for _ in range(num_frames):
            state_0.clear_forces()
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, control, contacts, dt)
            tracker.record(state_1)
            state_0, state_1 = state_1, state_0

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name

        try:
            tracker.save(path)
            data = np.load(path)
            positions = data["positions"]

            self.assertEqual(positions.shape, (50, num_frames + 1, 3))

            # The box should be falling: y position at last frame < y position at first frame
            mean_y_first = positions[:, 0, 1].mean()
            mean_y_last = positions[:, -1, 1].mean()
            self.assertLess(mean_y_last, mean_y_first)
        finally:
            os.unlink(path)
```

**Step 2: Run all tracker tests**

```bash
uv run --extra dev -m newton.tests -k test_surface_point_tracker
```

Expected: PASS (all tests across all classes)

**Step 3: Run pre-commit on changed files**

```bash
uvx pre-commit run -a
```

If pre-commit modifies any files, stage them and re-run to confirm clean.

**Step 4: Commit**

```bash
git add newton/tests/test_surface_point_tracker.py
git commit -m "$(cat <<'EOF'
Add end-to-end integration test for SurfacePointTracker

Simulates a falling box with SolverSemiImplicit, records 10
frames, verifies trajectory shape and downward motion.
EOF
)"
```
