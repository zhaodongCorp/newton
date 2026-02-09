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
from newton.solvers import SolverSemiImplicit

# =============================================================================
# Tests for _collect_triangles: verifying surface extraction from the model
# =============================================================================


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
        # A box has 6 faces, each split into 2 triangles = 12 total
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
        # Exact triangle count depends on tessellation resolution; just verify non-zero
        total_tris = sum(s["num_triangles"] for s in surfaces)
        self.assertGreater(total_tris, 0)

    def test_deformable_cloth(self):
        """Deformable triangles from particles are collected."""
        # Create a minimal 2-triangle quad from 4 particles
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
        # Add a rigid box (12 triangles)
        b = builder.add_body()
        builder.add_shape_box(body=b, hx=0.5, hy=0.5, hz=0.5)
        # Add one deformable triangle
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


# =============================================================================
# Tests for _sample_points_on_surfaces: verifying area-proportional sampling
# =============================================================================


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
        # Barycentric coords must be non-negative and sum to 1 for valid
        # points inside a triangle
        self.assertTrue(np.all(bary >= 0.0))
        np.testing.assert_allclose(bary.sum(axis=1), 1.0, atol=1e-6)

    def test_area_proportional_distribution(self):
        """Points distribute roughly proportional to surface area across surfaces."""
        builder = newton.ModelBuilder()
        # Large box: surface area = 6 * (4*4) = 96
        b1 = builder.add_body()
        builder.add_shape_box(body=b1, hx=2.0, hy=2.0, hz=2.0)
        # Small box: surface area = 6 * (0.5*0.5) = 1.5
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
        # The large box has ~64x the surface area, so it should receive
        # significantly more points than the small box
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


# =============================================================================
# Tests for SurfacePointTracker.__init__: verifying construction and setup
# =============================================================================


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
        # No frames should be recorded yet
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


# =============================================================================
# Tests for record(): verifying per-frame position tracking
# =============================================================================


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
        # Record same state twice — positions should be identical
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

        # Create a second state with the body translated by (1, 0, 0)
        state1 = model.state()
        body_q = state1.body_q.numpy()
        body_q[0][:3] += [1.0, 0.0, 0.0]
        state1.body_q.assign(wp.array(body_q, dtype=wp.transform, device="cpu"))
        tracker.record(state1)

        # All rigid points should have shifted by exactly (1, 0, 0)
        diff = tracker._frames[1] - tracker._frames[0]
        expected = np.broadcast_to(np.array([1.0, 0.0, 0.0]), diff.shape)
        np.testing.assert_allclose(diff, expected, atol=1e-5)

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

        # Move all particles uniformly along z-axis by 2.0
        state1 = model.state()
        pq = state1.particle_q.numpy()
        pq[:, 2] += 2.0
        state1.particle_q.assign(wp.array(pq, dtype=wp.vec3, device="cpu"))
        tracker.record(state1)

        # Since all particles moved by (0, 0, 2), all interpolated points
        # should also move by (0, 0, 2) regardless of barycentric coords
        diff = tracker._frames[1] - tracker._frames[0]
        np.testing.assert_allclose(diff[:, 2], 2.0, atol=1e-5)
        np.testing.assert_allclose(diff[:, :2], 0.0, atol=1e-5)


# =============================================================================
# Tests for save(): verifying trajectory serialization to NPZ
# =============================================================================


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
        # Record 3 frames
        tracker.record(state)
        tracker.record(state)
        tracker.record(state)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name

        try:
            tracker.save(path)
            positions = np.load(path)
            # Shape should be (num_points, num_frames, 3)
            self.assertEqual(positions.shape, (num_pts, 3, 3))
            self.assertEqual(positions.dtype, np.float32)
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
            tracker.save("/tmp/empty.npy")


# =============================================================================
# End-to-end integration test: full simulation loop with tracking
# =============================================================================


class TestEndToEnd(unittest.TestCase):
    """End-to-end test: build scene, simulate, track, save, verify."""

    def test_falling_box(self):
        """Track points on a box falling under gravity for 10 frames."""
        builder = newton.ModelBuilder()
        b = builder.add_body()
        builder.add_shape_box(body=b, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize(device="cpu")
        model.set_gravity((0.0, -9.81, 0.0))

        # Set up the standard Newton simulation loop with double-buffered states
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        solver = SolverSemiImplicit(model)

        from newton.utils import SurfacePointTracker

        tracker = SurfacePointTracker(model, state_0, num_points=50, seed=42)
        # Record the initial (rest) positions before any simulation steps
        tracker.record(state_0)

        dt = 1.0 / 60.0
        num_frames = 10
        for _ in range(num_frames):
            state_0.clear_forces()
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, control, contacts, dt)
            tracker.record(state_1)
            # Swap buffers: state_1 becomes input for next step
            state_0, state_1 = state_1, state_0

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name

        try:
            tracker.save(path)
            positions = np.load(path)

            # 1 initial frame + 10 simulation frames = 11 total
            self.assertEqual(positions.shape, (50, num_frames + 1, 3))

            # Verify the box is falling: mean y-position should decrease over time
            mean_y_first = positions[:, 0, 1].mean()
            mean_y_last = positions[:, -1, 1].mean()
            self.assertLess(mean_y_last, mean_y_first)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
