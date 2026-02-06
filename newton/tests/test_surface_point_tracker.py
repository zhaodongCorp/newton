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
