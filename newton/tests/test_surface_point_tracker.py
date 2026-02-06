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
        b1 = builder.add_body()
        builder.add_shape_box(body=b1, hx=2.0, hy=2.0, hz=2.0)
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
        from newton.utils import SurfacePointTracker

        tracker = SurfacePointTracker(model, state0, num_points=50, seed=42)
        tracker.record(state0)

        state1 = model.state()
        body_q = state1.body_q.numpy()
        body_q[0][:3] += [1.0, 0.0, 0.0]
        state1.body_q.assign(wp.array(body_q, dtype=wp.transform, device="cpu"))
        tracker.record(state1)

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

        state1 = model.state()
        pq = state1.particle_q.numpy()
        pq[:, 2] += 2.0
        state1.particle_q.assign(wp.array(pq, dtype=wp.vec3, device="cpu"))
        tracker.record(state1)

        diff = tracker._frames[1] - tracker._frames[0]
        np.testing.assert_allclose(diff[:, 2], 2.0, atol=1e-5)
        np.testing.assert_allclose(diff[:, :2], 0.0, atol=1e-5)


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


if __name__ == "__main__":
    unittest.main()
