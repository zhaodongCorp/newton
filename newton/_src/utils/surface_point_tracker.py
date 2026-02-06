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

from newton._src.geometry.types import GeoType, Mesh
from newton._src.utils.mesh import (
    create_box_mesh,
    create_capsule_mesh,
    create_cone_mesh,
    create_cylinder_mesh,
    create_sphere_mesh,
)


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

    @staticmethod
    def _collect_triangles(model, state):
        """Collect all triangulated surfaces from the model.

        Returns a list of dicts, each with:
            - vertices: np.ndarray (N, 3) -- world-space vertex positions
            - indices: np.ndarray (M*3,) -- flat triangle indices into vertices
            - num_triangles: int
            - body_index: int -- body this surface belongs to (-1 for deformable particles)
            - is_rigid: bool
            - shape_index: int -- shape index for rigid, -1 for deformable
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
                    vertices = vertices * scale
            else:
                continue

            if vertices is None or indices is None:
                continue

            num_triangles = len(indices) // 3

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
            particle_q = state.particle_q.numpy()
            tri_indices = model.tri_indices.numpy()

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
