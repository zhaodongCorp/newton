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


# GPU kernel that computes world-space positions for all tracked surface points
# in a single launch. Handles both rigid and deformable points via branching
# on the is_rigid flag, avoiding the need for separate kernel dispatches.
@wp.kernel
def _update_point_positions(
    is_rigid: wp.array(dtype=wp.int32),  # 1 = rigid body point, 0 = deformable
    body_index: wp.array(dtype=wp.int32),  # index into body_q (-1 if ground-attached)
    local_offset: wp.array(dtype=wp.vec3),  # body-local position (rigid points only)
    bary_coords: wp.array(dtype=wp.vec3),  # barycentric coords (deformable points only)
    tri_v0: wp.array(dtype=wp.int32),  # triangle vertex 0 index into particle_q
    tri_v1: wp.array(dtype=wp.int32),  # triangle vertex 1 index into particle_q
    tri_v2: wp.array(dtype=wp.int32),  # triangle vertex 2 index into particle_q
    body_q: wp.array(dtype=wp.transform),  # current body transforms from state
    particle_q: wp.array(dtype=wp.vec3),  # current particle positions from state
    out_positions: wp.array(dtype=wp.vec3),  # output: world-space point positions
    has_particles: wp.int32,  # whether the state has particle data
    has_bodies: wp.int32,  # whether the state has body data
):
    tid = wp.tid()

    if is_rigid[tid] == 1:
        # Rigid body path: transform pre-computed body-local offset to world space
        # using the current body transform from the simulation state.
        b_idx = body_index[tid]
        if has_bodies == 1 and b_idx >= 0:
            xform = body_q[b_idx]
        else:
            # Ground-fixed body or missing body data — use identity transform
            xform = wp.transform_identity()
        out_positions[tid] = wp.transform_point(xform, local_offset[tid])
    else:
        # Deformable path: interpolate current particle positions using
        # the stored barycentric coordinates. This correctly tracks mesh
        # deformation as particles move during simulation.
        bary = bary_coords[tid]
        if has_particles == 1:
            v0 = particle_q[tri_v0[tid]]
            v1 = particle_q[tri_v1[tid]]
            v2 = particle_q[tri_v2[tid]]
            out_positions[tid] = bary[0] * v0 + bary[1] * v1 + bary[2] * v2
        else:
            # Fallback: no particle data available
            out_positions[tid] = wp.vec3(0.0, 0.0, 0.0)


class SurfacePointTracker:
    """Samples 3D points on mesh surfaces and tracks their trajectories during simulation.

    Args:
        model: A finalized Newton Model.
        state: The initial simulation State (used to compute initial positions).
        num_points: Total number of points to sample, distributed proportional to surface area.
        seed: Random seed for reproducible sampling.
    """

    def __init__(self, model, state, num_points: int = 10000, seed: int = 42):
        self.num_points = num_points
        self._device = model.device
        self._frames = []  # list of (num_points, 3) numpy arrays, one per recorded frame

        # Step 1: Extract all triangulated surfaces (rigid shapes + deformable cloth/soft)
        surfaces = self._collect_triangles(model, state)
        if not surfaces:
            raise ValueError("No triangulated surfaces found in model.")

        # Step 2: Distribute sample points across triangles proportional to surface area
        sampled = self._sample_points_on_surfaces(surfaces, num_points=num_points, seed=seed)

        # Step 3: Build per-point metadata arrays that the GPU kernel needs at runtime.
        # Each point is classified as rigid or deformable, which determines the update
        # strategy used in _update_point_positions.
        is_rigid = np.zeros(num_points, dtype=np.int32)
        body_index = np.full(num_points, -1, dtype=np.int32)
        local_offset = np.zeros((num_points, 3), dtype=np.float32)
        bary_coords = sampled["bary_coords"]  # (num_points, 3) barycentric coordinates
        # Triangle vertex indices into particle_q — only used for deformable points.
        # Stored as three separate arrays since Warp kernels can't index 2D arrays.
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

            # Compute the initial world-space position via barycentric interpolation
            v0_idx, v1_idx, v2_idx = idxs[tri_idx]
            v0_pos = verts[v0_idx]
            v1_pos = verts[v1_idx]
            v2_pos = verts[v2_idx]
            world_pos = bary[0] * v0_pos + bary[1] * v1_pos + bary[2] * v2_pos

            if surf["is_rigid"]:
                is_rigid[i] = 1
                body_index[i] = surf["body_index"]

                # For rigid points, we store the position in body-local coordinates.
                # At runtime, we transform this offset by the current body pose to
                # get the world-space position. This ensures the point moves rigidly
                # with the body regardless of how the body rotates/translates.
                body_tf = wp.transform_identity()
                if body_index[i] >= 0:
                    body_q = state.body_q.numpy()[body_index[i]]
                    body_tf = wp.transform(
                        body_q[:3].tolist(),
                        wp.quat(body_q[3], body_q[4], body_q[5], body_q[6]),
                    )
                # Invert the body transform to convert world -> body-local space
                inv_tf = wp.transform_inverse(body_tf)
                local_p = wp.transform_point(inv_tf, wp.vec3(world_pos[0], world_pos[1], world_pos[2]))
                local_offset[i] = [local_p[0], local_p[1], local_p[2]]
            else:
                # For deformable points, store the particle indices that define the
                # triangle. At runtime, the kernel reads current particle positions
                # and interpolates using the stored barycentric coords.
                tri_v0[i] = v0_idx
                tri_v1[i] = v1_idx
                tri_v2[i] = v2_idx

        # Upload all per-point metadata to GPU (or CPU device) as Warp arrays
        self._is_rigid = wp.array(is_rigid, dtype=wp.int32, device=self._device)
        self._body_index = wp.array(body_index, dtype=wp.int32, device=self._device)
        self._local_offset = wp.array(local_offset, dtype=wp.vec3, device=self._device)
        self._bary_coords = wp.array(bary_coords, dtype=wp.vec3, device=self._device)
        self._tri_v0 = wp.array(tri_v0, dtype=wp.int32, device=self._device)
        self._tri_v1 = wp.array(tri_v1, dtype=wp.int32, device=self._device)
        self._tri_v2 = wp.array(tri_v2, dtype=wp.int32, device=self._device)

        # Per-point world index, derived from body_world.
        # Used to apply viewer-style world offsets for multi-world rendering.
        body_world = model.body_world.numpy() if hasattr(model, "body_world") and model.body_world is not None else None
        point_world = np.zeros(num_points, dtype=np.int32)
        if body_world is not None:
            for i in range(num_points):
                b = body_index[i]
                if b >= 0 and b < len(body_world):
                    point_world[i] = body_world[b]
        self._point_world = point_world

        # Reusable output buffer — overwritten each frame, then copied to CPU
        self._frame_positions = wp.zeros(num_points, dtype=wp.vec3, device=self._device)

    def record(self, state) -> None:
        """Record world-space positions of all tracked points for the current frame."""
        # Determine whether body and particle data exist in this state,
        # since mixed scenes may have only one or the other.
        has_bodies = 1 if state.body_q is not None and state.body_q.shape[0] > 0 else 0
        has_particles = 1 if state.particle_q is not None and state.particle_q.shape[0] > 0 else 0

        # Provide dummy arrays when data is absent so the kernel always receives
        # valid pointers — the has_bodies/has_particles flags prevent actual reads.
        body_q = state.body_q if has_bodies else wp.zeros(1, dtype=wp.transform, device=self._device)
        particle_q = state.particle_q if has_particles else wp.zeros(1, dtype=wp.vec3, device=self._device)

        # Launch one kernel for all points — branching inside handles rigid vs deformable
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

        # Eagerly copy to CPU and store — the GPU buffer is reused next frame.
        # This trades per-frame GPU->CPU transfer cost for simplicity (no need
        # to know num_frames upfront or manage dynamic GPU buffer resizing).
        self._frames.append(self._frame_positions.numpy().copy())

    def save(self, path: str) -> None:
        """Save recorded trajectories to a compressed NPZ file.

        The file contains:
            - ``positions``: shape ``(num_points, num_frames, 3)`` -- trajectory positions.
            - ``point_world``: shape ``(num_points,)`` -- world index for each point.
        """
        if not self._frames:
            raise ValueError("No frames recorded. Call record() at least once before save().")

        # Stack per-frame arrays: list of (num_points, 3) -> (num_frames, num_points, 3)
        stacked = np.stack(self._frames, axis=0)
        # Transpose to (num_points, num_frames, 3) so each row is one point's full trajectory
        positions = np.transpose(stacked, (1, 0, 2))

        # Use compressed format to reduce file size (~3-6x smaller than uncompressed)
        np.savez_compressed(
            path,
            positions=positions.astype(np.float32),
            point_world=self._point_world,
        )

    @staticmethod
    def _sample_points_on_surfaces(surfaces, num_points, seed=42):
        """Sample points on triangle surfaces proportional to area.

        Returns a dict with:
            - bary_coords: (num_points, 3) -- barycentric coordinates per point
            - surface_index: (num_points,) -- which surface group
            - surface_tri_index: (num_points,) -- which triangle within the surface
        """
        rng = np.random.default_rng(seed)

        # Compute per-triangle areas across all surfaces to build a probability
        # distribution for sampling. Larger triangles receive more sample points.
        all_areas = []
        surface_ids = []  # maps global triangle index -> surface index
        tri_ids = []  # maps global triangle index -> local triangle index within surface

        for surf_idx, surf in enumerate(surfaces):
            verts = surf["vertices"]
            idxs = surf["indices"].reshape(-1, 3)
            for tri_i in range(surf["num_triangles"]):
                v0 = verts[idxs[tri_i, 0]]
                v1 = verts[idxs[tri_i, 1]]
                v2 = verts[idxs[tri_i, 2]]
                # Triangle area = 0.5 * |cross(edge1, edge2)|
                area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                all_areas.append(area)
                surface_ids.append(surf_idx)
                tri_ids.append(tri_i)

        all_areas = np.array(all_areas, dtype=np.float64)
        total_area = all_areas.sum()

        if total_area <= 0.0:
            raise ValueError("Total mesh surface area is zero; cannot sample points.")

        # Randomly assign each sample point to a triangle, weighted by area
        probs = all_areas / total_area
        tri_assignments = rng.choice(len(all_areas), size=num_points, p=probs)

        # Sample uniform random points within each assigned triangle using the
        # "fold" trick: generate (u, v) in [0,1]^2, then reflect points outside
        # the triangle (u+v > 1) back inside. This produces a uniform distribution
        # over the triangle with barycentric coords (w, u, v) where w = 1-u-v.
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

    @staticmethod
    def _collect_triangles(model, state):
        """Collect all triangulated surfaces from the model.

        Iterates over two categories of geometry:
        1. Rigid body shapes — primitives (box, sphere, etc.) are tessellated into
           triangle meshes using Newton's mesh utilities; actual mesh shapes use
           their stored vertices/indices directly.
        2. Deformable triangles — cloth and soft-body triangles defined by
           model.tri_indices, referencing particle positions.

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
        # Each shape in the model may be a geometric primitive or a mesh.
        # Primitives are converted to triangle meshes for uniform handling.
        shape_count = model.shape_count
        for s_idx in range(shape_count):
            geo_type = model.shape_type.numpy()[s_idx]
            body_idx = model.shape_body.numpy()[s_idx]
            scale = model.shape_scale.numpy()[s_idx]
            shape_xform = model.shape_transform.numpy()[s_idx]

            vertices = None
            indices = None

            # Tessellate each primitive type into a triangle mesh.
            # The create_*_mesh functions return (vertices_8col, indices) where
            # vertices_8col has 8 columns (pos + normal + uv); we only need xyz.
            if geo_type == int(GeoType.BOX):
                verts_8col, indices = create_box_mesh(extents=scale)
                vertices = verts_8col[:, :3]
            elif geo_type == int(GeoType.SPHERE):
                radius = float(scale[0])
                verts_8col, indices = create_sphere_mesh(radius=radius)
                vertices = verts_8col[:, :3]
            elif geo_type == int(GeoType.CAPSULE):
                radius, half_height = float(scale[0]), float(scale[1])
                # up_axis=2 to match the ray tracer which renders capsules along Z
                verts_8col, indices = create_capsule_mesh(radius=radius, half_height=half_height, up_axis=2)
                vertices = verts_8col[:, :3]
            elif geo_type == int(GeoType.CYLINDER):
                radius, half_height = float(scale[0]), float(scale[1])
                # up_axis=2 to match the ray tracer which renders cylinders along Z
                verts_8col, indices = create_cylinder_mesh(radius=radius, half_height=half_height, up_axis=2)
                vertices = verts_8col[:, :3]
            elif geo_type == int(GeoType.CONE):
                radius, half_height = float(scale[0]), float(scale[1])
                # up_axis=2 to match the ray tracer which renders cones along Z
                verts_8col, indices = create_cone_mesh(radius=radius, half_height=half_height, up_axis=2)
                vertices = verts_8col[:, :3]
            elif geo_type == int(GeoType.MESH):
                # Use the actual mesh data stored in the model
                mesh_src = model.shape_source[s_idx]
                if mesh_src is not None and isinstance(mesh_src, Mesh):
                    vertices = mesh_src.vertices.copy()
                    indices = mesh_src.indices.copy()
                    # Apply the shape scale to mesh vertices
                    vertices = vertices * scale
            else:
                # Skip unsupported types (PLANE, SDF, HFIELD, ELLIPSOID, etc.)
                continue

            if vertices is None or indices is None:
                continue

            num_triangles = len(indices) // 3

            # Compose the full world transform: body_transform * shape_transform.
            # shape_xform is stored as 7 floats: [px, py, pz, qx, qy, qz, qw]
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
            # Combined transform: first shape-local -> body-local, then body -> world
            world_tf = wp.transform_multiply(body_tf, shape_tf)

            # Transform all vertices from shape-local space to world space
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
        # These use particle positions directly and don't need shape transforms.
        # The triangle connectivity is fixed; particles move during simulation.
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
