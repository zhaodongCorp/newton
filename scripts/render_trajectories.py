#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render a Newton example with 6-camera trajectory visualization.

Usage:
    uv run python scripts/render_trajectories.py --example basic_shapes --output-dir /tmp/renders/
    uv run python scripts/render_trajectories.py --example basic_pendulum --trajectories /tmp/traj.npz --output-dir /tmp/renders/
"""

import argparse
import math
import os
import sys
import time


def compute_bounding_sphere(model, state):
    """Compute bounding sphere (center, radius) from dynamic scene content.

    Considers both rigid body shapes (body >= 0) and particle positions
    so that cloth/particle-based examples are properly framed.  Large
    static geometry (terrain meshes, ground planes) is excluded to keep
    cameras close.  Falls back to all shapes if no dynamic content exists.
    """
    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    shape_transforms = model.shape_transform.numpy()  # (num_shapes, 7) as transforms
    shape_body = model.shape_body.numpy()  # (num_shapes,)
    shape_radius = model.shape_collision_radius.numpy()  # (num_shapes,)
    body_q = state.body_q.numpy() if state.body_q is not None else None  # (num_bodies, 7)

    def _collect_shapes(include_static):
        positions = []
        radii = []
        for i in range(model.shape_count):
            r = float(shape_radius[i])
            if r > 1.0e5:
                # Skip infinite planes
                continue

            body_idx = int(shape_body[i])
            if not include_static and body_idx < 0:
                continue

            # Get shape local transform
            shape_xform = shape_transforms[i]
            shape_pos = shape_xform[:3]

            # Compose with body transform if attached
            if body_idx >= 0 and body_q is not None:
                body_xform = body_q[body_idx]
                body_pos = body_xform[:3]
                body_rot = body_xform[3:]
                rotated = wp.quat_rotate(wp.quatf(*body_rot), wp.vec3f(*shape_pos))
                world_pos = body_pos + np.array([float(rotated[0]), float(rotated[1]), float(rotated[2])])
            else:
                world_pos = shape_pos

            positions.append(world_pos)
            radii.append(r)
        return positions, radii

    # First try dynamic shapes only; fall back to all if none found
    positions, radii = _collect_shapes(include_static=False)
    if not positions:
        positions, radii = _collect_shapes(include_static=True)

    # Include particle positions for cloth/particle-based scenes
    particle_q = state.particle_q
    if particle_q is not None and particle_q.shape[0] > 0:
        pq = particle_q.numpy()  # (num_particles, 3)
        pq_min = pq.min(axis=0)
        pq_max = pq.max(axis=0)
        # Represent particle cloud as two corner points with zero radius
        positions.append(pq_min)
        positions.append(pq_max)
        radii.extend([0.0, 0.0])

    if not positions:
        return np.array([0.0, 0.0, 0.0]), 1.0

    positions = np.array(positions)
    radii = np.array(radii)

    # AABB from shape centers +/- radii
    mins = positions - radii[:, None]
    maxs = positions + radii[:, None]
    aabb_min = mins.min(axis=0)
    aabb_max = maxs.max(axis=0)

    center = (aabb_min + aabb_max) / 2.0
    radius = np.linalg.norm(aabb_max - aabb_min) / 2.0

    # Ensure minimum radius
    radius = max(radius, 0.5)
    return center, radius


def create_axis_cameras(center, radius, num_worlds=1, distance_multiplier=1.5):
    """Create 6 cameras positioned for scenes with a ground plane (Z-up).

    Camera layout:
        0-3: Four orbit cameras at 30 deg elevation, spaced 90 deg apart
             (front, right, back, left).
        4:   High overview at 60 deg elevation, azimuth 45 deg (front-right).
        5:   Top-down view (90 deg elevation).

    Args:
        center: (3,) scene center.
        radius: Bounding sphere radius.
        num_worlds: Number of simulation worlds.
        distance_multiplier: Camera distance as a multiple of the bounding
            sphere radius (default: 1.5).

    Returns a warp array of camera transforms with shape (6, num_worlds)
    suitable for SensorTiledCamera.render().
    """
    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    R = distance_multiplier * radius

    # (azimuth_deg, elevation_deg) — azimuth measured from +X axis in XY plane
    camera_angles = [
        (0.0, 30.0),  # front
        (90.0, 30.0),  # right
        (180.0, 30.0),  # back
        (270.0, 30.0),  # left
        (45.0, 60.0),  # high overview (front-right)
        (0.0, 90.0),  # top-down
    ]

    transforms = []
    for azimuth_deg, elev_deg in camera_angles:
        az = np.radians(azimuth_deg)
        el = np.radians(elev_deg)

        # Spherical to Cartesian (Z-up)
        cos_el = np.cos(el)
        d = np.array([cos_el * np.cos(az), cos_el * np.sin(az), np.sin(el)])
        cam_pos = center + R * d

        # Look-at: camera -Z axis points from cam_pos toward center
        forward = -d  # normalized direction toward center

        # Choose an up vector that isn't parallel to forward
        if abs(np.dot(forward, np.array([0.0, 0.0, 1.0]))) < 0.99:
            world_up = np.array([0.0, 0.0, 1.0])
        else:
            world_up = np.array([0.0, 1.0, 0.0])

        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # Build rotation matrix (columns = right, up, -forward in camera convention)
        # Camera convention: X=right, Y=up, Z=backward (looking along -Z)
        rot_mat = np.column_stack([right, up, -forward])  # 3x3
        quat = wp.quat_from_matrix(wp.mat33f(*rot_mat.flatten()))

        cam_xform = wp.transformf(wp.vec3f(*cam_pos), quat)
        transforms.append([cam_xform] * num_worlds)

    return wp.array(transforms, dtype=wp.transformf)


def inject_trajectory_particles(sensor, trajectory_positions, frame_idx, point_radius=0.004):
    """Inject current-frame trajectory points as renderable particle spheres.

    If the render context already contains scene particles (e.g. MPM sand),
    the trajectory dots are appended to the existing arrays so both the
    scene particles and trajectory markers are rendered together.

    Args:
        sensor: SensorTiledCamera instance.
        trajectory_positions: numpy array of shape (num_points, num_frames, 3).
        frame_idx: Current frame index into trajectory_positions.
        point_radius: Radius of each trajectory sphere (default: 0.004).
    """
    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    num_points = trajectory_positions.shape[0]

    # Trajectory dots
    traj_positions = trajectory_positions[:, frame_idx, :].astype(np.float32)
    traj_radii = np.full(num_points, point_radius, dtype=np.float32)
    traj_world = np.full(num_points, -1, dtype=np.int32)

    rc = sensor.render_context
    device = rc.device

    # If the scene already has particles (MPM sand, cloth, etc.),
    # concatenate them with trajectory dots so both are rendered.
    if rc.has_particles and rc.particles_position is not None and rc.particles_position.shape[0] > 0:
        scene_pos = rc.particles_position.numpy()
        scene_rad = rc.particles_radius.numpy()
        scene_world = rc.particles_world_index.numpy()

        all_pos = np.concatenate([scene_pos, traj_positions], axis=0)
        all_rad = np.concatenate([scene_rad, traj_radii], axis=0)
        all_world = np.concatenate([scene_world, traj_world], axis=0)
    else:
        all_pos = traj_positions
        all_rad = traj_radii
        all_world = traj_world

    rc.particles_position = wp.array(all_pos, dtype=wp.vec3f, device=device)
    rc.particles_radius = wp.array(all_rad, dtype=wp.float32, device=device)
    rc.particles_world_index = wp.array(all_world, dtype=wp.int32, device=device)


def _generate_trajectory_colors(num_points, seed=42):
    """Generate a random color (R, G, B) for each trajectory point."""
    import numpy as np  # noqa: PLC0415

    rng = np.random.RandomState(seed)
    # Use HSV with full saturation and value for vivid colors
    hues = rng.uniform(0.0, 1.0, num_points)
    colors = []
    for h in hues:
        # HSV to RGB (S=0.9, V=1.0)
        s, v = 0.9, 1.0
        c = v * s
        x = c * (1.0 - abs((h * 6.0) % 2.0 - 1.0))
        m = v - c
        if h < 1 / 6:
            r, g, b = c, x, 0.0
        elif h < 2 / 6:
            r, g, b = x, c, 0.0
        elif h < 3 / 6:
            r, g, b = 0.0, c, x
        elif h < 4 / 6:
            r, g, b = 0.0, x, c
        elif h < 5 / 6:
            r, g, b = x, 0.0, c
        else:
            r, g, b = c, 0.0, x
        colors.append((int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)))
    return colors


def _project_points_to_2d(points_3d, cam_pos, cam_quat, fov_rad, resolution):
    """Project 3D points into 2D pixel coordinates for a given camera.

    Args:
        points_3d: (N, 3) numpy array of world-space positions.
        cam_pos: (3,) camera position.
        cam_quat: (4,) camera quaternion (x, y, z, w).
        fov_rad: Field of view in radians.
        resolution: Image width/height in pixels.

    Returns:
        pixels: (N, 2) array of (x, y) pixel coordinates.
        visible: (N,) boolean mask of points in front of camera.
    """
    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    q = wp.quatf(*cam_quat)
    # Camera basis vectors
    right = np.array([float(v) for v in wp.quat_rotate(q, wp.vec3f(1.0, 0.0, 0.0))])
    up = np.array([float(v) for v in wp.quat_rotate(q, wp.vec3f(0.0, 1.0, 0.0))])
    forward = np.array([float(v) for v in wp.quat_rotate(q, wp.vec3f(0.0, 0.0, -1.0))])

    # Transform points to camera space
    rel = points_3d - cam_pos  # (N, 3)
    cam_x = rel @ right  # (N,)
    cam_y = rel @ up
    cam_z = rel @ forward  # positive = in front

    # Perspective projection
    half_size = np.tan(fov_rad / 2.0)
    visible = cam_z > 0.01  # in front of camera
    safe_z = np.where(visible, cam_z, 1.0)
    ndc_x = cam_x / (safe_z * half_size)  # [-1, 1]
    ndc_y = cam_y / (safe_z * half_size)

    px = ((ndc_x + 1.0) * 0.5 * resolution).astype(np.int32)
    py = ((1.0 - (ndc_y + 1.0) * 0.5) * resolution).astype(np.int32)  # flip Y

    pixels = np.stack([px, py], axis=-1)
    return pixels, visible


def save_camera_frames(
    color_image,
    output_dir,
    frame_idx,
    num_cameras,
    trajectory_positions,
    trail_colors,
    camera_transforms_np,
    fov_rad,
    resolution,
    prefix,
    trail_length=20,
):
    """Extract per-camera images, draw trail lines, and save as RGB JPG.

    Args:
        color_image: Warp array of shape (num_worlds, num_cameras, H, W), dtype uint32.
        output_dir: Base output directory.
        frame_idx: Frame number for filename.
        num_cameras: Number of cameras.
        trajectory_positions: (num_points, num_frames, 3) numpy array.
        trail_colors: List of (R, G, B) tuples, one per trajectory point.
        camera_transforms_np: (num_cameras, num_worlds, 7) numpy array of camera transforms.
        fov_rad: Camera FOV in radians.
        resolution: Image resolution (width = height).
        trail_length: Number of trailing frames to draw.
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    color_np = color_image.numpy()
    start_frame = max(0, frame_idx - trail_length + 1)

    for cam_idx in range(num_cameras):
        pixel_data = color_np[0, cam_idx]
        r = ((pixel_data >> 0) & 0xFF).astype(np.float32)
        g = ((pixel_data >> 8) & 0xFF).astype(np.float32)
        b = ((pixel_data >> 16) & 0xFF).astype(np.float32)

        # Detect background pixels (white from clear_data) and replace
        # with a sky gradient: blue at top -> pale at horizon -> ground at bottom.
        is_bg = (r > 253.0) & (g > 253.0) & (b > 253.0)

        # Build vertical gradient with three stops using smoothstep interpolation:
        #   top = zenith blue, middle = pale horizon, bottom = ground color
        h = pixel_data.shape[0]
        sky_zenith = np.array([70.0, 130.0, 210.0])
        sky_horizon = np.array([190.0, 210.0, 230.0])
        sky_ground = np.array([90.0, 85.0, 80.0])
        t = np.linspace(0.0, 1.0, h, dtype=np.float32)
        grad = np.empty((h, 3), dtype=np.float32)
        for i in range(h):
            if t[i] < 0.5:
                s = t[i] / 0.5
                s = s * s * (3.0 - 2.0 * s)  # smoothstep
                grad[i] = sky_zenith * (1.0 - s) + sky_horizon * s
            else:
                s = (t[i] - 0.5) / 0.5
                s = s * s * (3.0 - 2.0 * s)  # smoothstep
                grad[i] = sky_horizon * (1.0 - s) + sky_ground * s
        # Broadcast gradient to full image width
        grad_r = np.broadcast_to(grad[:, 0:1], pixel_data.shape)
        grad_g = np.broadcast_to(grad[:, 1:2], pixel_data.shape)
        grad_b = np.broadcast_to(grad[:, 2:3], pixel_data.shape)

        r = np.where(is_bg, grad_r, r)
        g = np.where(is_bg, grad_g, g)
        b = np.where(is_bg, grad_b, b)

        rgb = np.stack([r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)], axis=-1)

        img = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(img)

        cam_xform = camera_transforms_np[cam_idx, 0]
        cam_pos = cam_xform[:3]
        cam_quat = cam_xform[3:]

        # Draw trail lines for each trajectory point
        num_points = trajectory_positions.shape[0]
        for pt_idx in range(num_points):
            trail_frames = list(range(start_frame, frame_idx + 1))
            if len(trail_frames) < 2:
                continue
            trail_3d = trajectory_positions[pt_idx, trail_frames, :]
            pixels, visible = _project_points_to_2d(
                trail_3d,
                cam_pos,
                cam_quat,
                fov_rad,
                resolution,
            )
            # Draw connected line segments where both endpoints are visible
            color = trail_colors[pt_idx]
            for i in range(len(trail_frames) - 1):
                if visible[i] and visible[i + 1]:
                    x0, y0 = int(pixels[i, 0]), int(pixels[i, 1])
                    x1, y1 = int(pixels[i + 1, 0]), int(pixels[i + 1, 1])
                    # Clip to image bounds
                    if 0 <= x0 < resolution and 0 <= y0 < resolution and 0 <= x1 < resolution and 0 <= y1 < resolution:
                        draw.line([(x0, y0), (x1, y1)], fill=color, width=2)

        cam_dir = os.path.join(output_dir, f"{prefix}_cam_{cam_idx}")
        os.makedirs(cam_dir, exist_ok=True)
        filepath = os.path.join(cam_dir, f"{prefix}_frame_{frame_idx + 1:05d}.jpg")
        img.save(filepath, quality=95)



def _apply_example_shape_colors(sensor, model, example):
    """Apply custom shape colors defined by the example.

    Some examples (e.g. ik_cube_stacking) set per-shape colors via
    ``viewer.update_shape_colors()``.  Since ViewerNull discards those
    calls, we extract colors from the example's own attributes and apply
    them to the sensor's shape_colors array.

    Recognized patterns:
        - ``example.cube_colors`` (dict of shape_key -> [r,g,b])
          with ``example.shape_map`` (dict of shape_key -> shape_index)
    """
    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    colors_np = sensor.render_context.shape_colors.numpy()  # (num_shapes, 4)
    modified = False

    # Pattern: cube_colors + shape_map (ik_cube_stacking and similar examples)
    cube_colors = getattr(example, "cube_colors", None)
    shape_map = getattr(example, "shape_map", None)
    if cube_colors and shape_map:
        for key, rgb in cube_colors.items():
            if key in shape_map:
                s_idx = shape_map[key]
                if 0 <= s_idx < len(colors_np):
                    colors_np[s_idx] = [rgb[0], rgb[1], rgb[2], 1.0]
                    modified = True

    if modified:
        temp = wp.array(colors_np, dtype=wp.vec4f, device=sensor.render_context.device)
        wp.copy(sensor.render_context.shape_colors, temp)


def _compute_visibility(
    depth_image,
    trajectory_positions,
    frame_idx,
    num_cameras,
    camera_transforms_np,
    fov_rad,
    resolution,
    depth_tolerance,
):
    """Compute per-point visibility for each camera using the depth buffer.

    A point is visible from a camera if:
    1. It is in front of the camera and projects within the image frame.
    2. No scene surface occludes it — i.e. the rendered depth at the
       projected pixel is at least as far as the point's distance
       (within tolerance).

    This occlusion-based test is more robust than exact depth matching
    for particle/cloth examples where the simulation replay may not be
    perfectly deterministic: the rendered mesh can shift slightly between
    runs, but points that are not hidden behind geometry are still
    correctly marked as visible.

    Args:
        depth_image: Warp array of shape (num_worlds, num_cameras, H, W), dtype float32.
        trajectory_positions: (num_points, num_frames, 3) numpy array.
        frame_idx: Current frame index into trajectory_positions.
        num_cameras: Number of cameras.
        camera_transforms_np: (num_cameras, num_worlds, 7) numpy array.
        fov_rad: Camera FOV in radians.
        resolution: Image width/height in pixels.
        depth_tolerance: Slack added to rendered depth for the occlusion test.

    Returns:
        visibility: (num_cameras, num_points) uint8 array with 0/1 values.
    """
    import numpy as np  # noqa: PLC0415

    depth_np = depth_image.numpy()  # (num_worlds, num_cameras, H, W)
    num_points = trajectory_positions.shape[0]
    current_positions = trajectory_positions[:, frame_idx, :]  # (num_points, 3)
    visibility = np.zeros((num_cameras, num_points), dtype=np.uint8)

    for cam_idx in range(num_cameras):
        cam_xform = camera_transforms_np[cam_idx, 0]
        cam_pos = cam_xform[:3]
        cam_quat = cam_xform[3:]

        # Project points to 2D
        pixels, in_front = _project_points_to_2d(current_positions, cam_pos, cam_quat, fov_rad, resolution)

        # Actual distance from camera to each point
        dists = np.linalg.norm(current_positions - cam_pos, axis=1)  # (num_points,)

        cam_depth = depth_np[0, cam_idx]  # (H, W)

        for pt in range(num_points):
            if not in_front[pt]:
                continue
            px, py = int(pixels[pt, 0]), int(pixels[pt, 1])
            if not (0 <= px < resolution and 0 <= py < resolution):
                continue
            rendered_depth = cam_depth[py, px]
            if rendered_depth <= 0.0:
                # No surface hit at this pixel — point is in empty space,
                # nothing occludes it, so it is visible.
                visibility[cam_idx, pt] = 1
                continue
            # Point is visible if it is not behind the rendered surface
            if dists[pt] <= rendered_depth + depth_tolerance:
                visibility[cam_idx, pt] = 1

    return visibility


def _compute_world_offsets(model, state):
    """Compute viewer-style world offsets for multi-world rendering.

    Newton's physics state (body_q) stores all worlds at the same local
    positions.  ViewerGL spreads them in a 2D grid for visualization;
    SensorTiledCamera does not.  This function replicates the viewer's
    ``_auto_compute_world_offsets`` logic so we can apply the same offsets
    when rendering with the ray tracer.

    Returns an (num_worlds, 3) numpy array of per-world position offsets.
    """
    import numpy as np  # noqa: PLC0415

    from newton.utils import compute_world_offsets  # noqa: PLC0415

    num_worlds = model.num_worlds
    if num_worlds <= 1:
        return np.zeros((max(num_worlds, 1), 3), dtype=np.float32)

    # Estimate per-world extents from collision radii (simplified version
    # of ViewerBase._get_world_extents).
    shape_radii = model.shape_collision_radius.numpy()
    max_radius = 0.0
    for s in range(model.shape_count):
        r = float(shape_radii[s])
        if r < 1.0e5:  # skip infinite planes
            max_radius = max(max_radius, r)

    extent = max(max_radius * 2.0, 1.0)
    margin = 1.5
    spacing_val = float(np.ceil(extent * margin))

    # 2D grid perpendicular to up_axis (Z for Newton default)
    spacing = [spacing_val, spacing_val, spacing_val]
    spacing[model.up_axis] = 0.0

    return compute_world_offsets(num_worlds, tuple(spacing), up_axis=model.up_axis)


def _apply_world_offsets_to_body_q(state, model, world_offsets):
    """Return a new body_q array with per-world position offsets applied.

    Each body is shifted by its world's offset so that multi-world scenes
    appear spread out in a grid, matching the ViewerGL layout.
    """
    import warp as wp  # noqa: PLC0415

    if state.body_q is None:
        return None

    body_q_np = state.body_q.numpy().copy()  # (num_bodies, 7)
    body_world = model.body_world.numpy()  # (num_bodies,)

    for i in range(len(body_q_np)):
        w = int(body_world[i])
        if 0 <= w < len(world_offsets):
            body_q_np[i, :3] += world_offsets[w]

    return wp.array(body_q_np, dtype=wp.transformf, device=model.device)


def _apply_worldfixed_offsets_to_rc(rc, model, shape_world_np, world_offsets):
    """Apply world offsets to render-context transforms for body=-1 shapes.

    ``update_from_state()`` copies ``model.shape_transform`` unchanged for
    body=-1 shapes.  This post-hoc fixup shifts those transforms in the
    render context so they appear at their grid positions, without modifying
    ``model.shape_transform`` itself (which the physics / CUDA graph reads).
    """
    import warp as wp  # noqa: PLC0415

    shape_body_np = model.shape_body.numpy()
    xforms = rc.shape_transforms.numpy()  # (num_shapes, 7)
    modified = False
    for s in range(len(xforms)):
        if shape_body_np[s] >= 0:
            continue
        w = int(shape_world_np[s])
        if 0 <= w < len(world_offsets):
            xforms[s, :3] += world_offsets[w]
            modified = True

    if modified:
        temp = wp.array(xforms, dtype=wp.transformf, device=rc.device)
        wp.copy(rc.shape_transforms, temp)


def _inflate_thin_shapes(model, radius, resolution, fov_rad, camera_distance, min_pixels=1.0):
    """Compute minimum half-extent for thin shape visibility.

    Returns the minimum half-extent threshold so that every shape subtends
    at least ``min_pixels`` pixels from the farthest camera viewpoint.
    This value is applied to ``render_context.shape_sizes`` after each
    ``update_from_state()`` for the color render pass only — the depth
    render (used for visibility) keeps the original sizes so that point
    depths match accurately.
    """
    max_cam_dist = camera_distance * radius * 2.0  # conservative far bound
    pixel_size = 2.0 * max_cam_dist * math.tan(fov_rad / 2.0) / resolution
    min_half_extent = pixel_size * min_pixels / 2.0
    return min_half_extent


def _apply_shape_inflation(rc, model, min_half_extent):
    """Inflate thin dimensions in render_context.shape_sizes for the color pass.

    Must be called after update_from_state (which resets rc.shape_sizes from
    model.shape_scale) and after the depth render, but before the color render.
    """
    import warp as wp  # noqa: PLC0415

    scale = rc.shape_sizes.numpy()
    num_inflated = 0
    for s in range(len(scale)):
        inflated = False
        for d in range(3):
            if 0 < scale[s, d] < min_half_extent:
                scale[s, d] = min_half_extent
                inflated = True
        if inflated:
            num_inflated += 1

    if num_inflated > 0:
        temp = wp.array(scale, dtype=wp.vec3f, device=rc.device)
        wp.copy(rc.shape_sizes, temp)
    return num_inflated


def _apply_world_offsets_to_trajectories(trajectory_positions, point_world, world_offsets):
    """Apply per-world offsets to trajectory positions using per-point world mapping.

    Each point is shifted by its world's offset so trajectories align
    with the offset-rendered geometry.

    Args:
        trajectory_positions: (num_points, num_frames, 3) numpy array.
        point_world: (num_points,) int array mapping each point to its world index.
        world_offsets: (num_worlds, 3) numpy array of per-world position offsets.
    """
    offset_positions = trajectory_positions.copy()
    for i in range(len(point_world)):
        w = int(point_world[i])
        if 0 <= w < len(world_offsets):
            offset_positions[i] += world_offsets[w]

    return offset_positions


def _create_example(mod, viewer, args):
    """Instantiate an Example, adapting to its constructor signature.

    Newton examples have varying signatures: some take (viewer, args), others
    take (viewer, num_worlds, args), (viewer, num_worlds), (viewer,), etc.
    This helper inspects the constructor and passes matching arguments.

    The ``args`` namespace is wrapped so that example-specific attributes
    (e.g. ``use_mujoco_contacts``) return ``None`` instead of raising
    ``AttributeError`` when they weren't defined by the renderer's CLI parser.
    """
    import inspect  # noqa: PLC0415

    # Wrap args so that ``getattr(args, "foo", default)`` works correctly
    # (AttributeError is raised for missing attrs, letting the default apply),
    # while examples that guard with ``args.X if args else ...`` still work
    # because ``args`` is truthy.
    class _ForgivingArgs:
        def __init__(self, ns):
            self.__dict__["_ns"] = ns

        def __getattr__(self, name):
            return getattr(self._ns, name)

        def __setattr__(self, name, value):
            setattr(self._ns, name, value)

        def __bool__(self):
            return True

    safe_args = _ForgivingArgs(args)

    sig = inspect.signature(mod.Example.__init__)
    params = list(sig.parameters.keys())  # includes 'self'
    kwargs = {}
    unsupported = []
    for name in params[1:]:  # skip 'self'
        if name == "viewer":
            kwargs["viewer"] = viewer
        elif name == "num_worlds":
            cli_val = getattr(args, "num_worlds", None)
            if cli_val is not None:
                kwargs["num_worlds"] = cli_val
            else:
                # Use the example's own constructor default, or 1
                p = sig.parameters[name]
                if p.default is not inspect.Parameter.empty:
                    kwargs["num_worlds"] = p.default
                else:
                    kwargs["num_worlds"] = 1
        elif name == "args":
            kwargs["args"] = safe_args
        elif name == "headless":
            kwargs["headless"] = getattr(args, "headless", True)
        elif name == "test_mode":
            kwargs["test_mode"] = getattr(args, "test", False)
        elif name == "verbose":
            kwargs["verbose"] = False
        else:
            p = sig.parameters[name]
            if p.default is inspect.Parameter.empty:
                unsupported.append(name)

    if unsupported:
        print(f"ERROR: Example requires unsupported constructor arguments: {', '.join(unsupported)}")
        print("  This example has a non-standard setup that the pipeline cannot automate.")
        print("  Supported parameters: viewer, num_worlds, args, headless, test_mode, verbose")
        sys.exit(1)

    return mod.Example(**kwargs)


def discover_examples():
    """Discover available Newton examples (reused from track_surface_points.py)."""
    import newton.examples  # noqa: PLC0415

    src_dir = newton.examples.get_source_directory()
    modules = ["basic", "cable", "cloth", "contacts", "diffsim", "ik", "mpm", "multiphysics", "robot", "selection", "sensors", "softbody"]
    example_map = {}
    for module in sorted(modules):
        module_dir = os.path.join(src_dir, module)
        if not os.path.isdir(module_dir):
            continue
        for filename in sorted(os.listdir(module_dir)):
            if filename.startswith("example_") and filename.endswith(".py"):
                name = filename[8:-3]
                example_map[name] = f"newton.examples.{module}.{filename[:-3]}"
    return example_map


def pick_example(example_map):
    """Interactive example picker."""
    names = list(example_map.keys())
    print("\nAvailable Newton examples:")
    print("-" * 40)
    for i, name in enumerate(names, 1):
        print(f"  {i:3d}. {name}")
    print()
    while True:
        choice = input("Enter example number or name: ").strip()
        if not choice:
            continue
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(names):
                return names[idx]
            print(f"  Number out of range (1-{len(names)})")
            continue
        except ValueError:
            pass
        if choice in example_map:
            return choice
        matches = [n for n in names if choice in n]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"  Ambiguous, matches: {', '.join(matches)}")
        else:
            print(f"  Unknown example: {choice}")


def run_renderer(
    example_name,
    module_path,
    num_frames,
    num_points,
    resolution,
    output_dir,
    trajectories_path,
    device,
    num_worlds=None,
    camera_distance=1.5,
    traj_pct=10,
    depth_tol=1e-4,
    name_prefix=None,
    min_pixels=1.0,
    spp=1,
    point_radius=0.004,
):
    """Load example, run simulation with rendering, save JPG frames."""
    import importlib  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    import newton  # noqa: PLC0415
    import newton.viewer  # noqa: PLC0415
    from newton.sensors import SensorTiledCamera  # noqa: PLC0415
    from newton.utils import SurfacePointTracker  # noqa: PLC0415

    # Prefix for output filenames (cam dirs, visibility, etc.)
    prefix = name_prefix or example_name

    if device:
        wp.set_device(device)

    # Load example headlessly
    viewer = newton.viewer.ViewerNull(num_frames=num_frames)
    args = argparse.Namespace(
        device=device,
        viewer="null",
        headless=True,
        test=False,
        num_frames=num_frames,
        num_worlds=num_worlds,
        collision_pipeline="standard",
        broad_phase_mode="nxn",
        use_mujoco_contacts=False,
        output_path=None,
        rerun_address=None,
    )
    print(f"\nLoading example: {example_name} ({module_path})")
    mod = importlib.import_module(module_path)
    example = _create_example(mod, viewer, args)

    model = getattr(example, "model", None)
    state = getattr(example, "state_0", None) or getattr(example, "state", None)
    if state is None:
        states = getattr(example, "states", None)
        if states and len(states) > 0:
            state = states[0]
    if model is None or state is None:
        print("ERROR: Example does not expose 'model' and 'state_0'/'state'/'states'.")
        sys.exit(1)
    state_attr = "state_0" if hasattr(example, "state_0") else "state"

    print(f"  Bodies: {model.body_count}, Shapes: {model.shape_count}")

    # Load or generate trajectories
    trajectories_from_file = False
    if trajectories_path and os.path.exists(trajectories_path):
        print(f"  Loading trajectories from {trajectories_path}")
        trajectory_positions = np.load(trajectories_path)
        # Loaded trajectories are already in world-space (offsets baked in).
        trajectories_from_file = True
    else:
        print(f"  Generating trajectories on the fly ({num_points} points)...")
        tracker = SurfacePointTracker(model, state, num_points=num_points, seed=42)
        tracker.record(state)
        for _frame in range(num_frames):
            example.step()
            current_state = getattr(example, state_attr, state)
            tracker.record(current_state)
        trajectory_positions = np.stack(tracker._frames, axis=1)  # (num_points, num_frames+1, 3)
        traj_point_world = tracker._point_world
        # Reset example for the rendering pass (need fresh viewer since set_model is one-shot)
        viewer = newton.viewer.ViewerNull(num_frames=num_frames)
        example = _create_example(mod, viewer, args)
        model = example.model
        state = getattr(example, state_attr, state)

    total_frames = trajectory_positions.shape[1]
    print(f"  Trajectory: {trajectory_positions.shape[0]} points, {total_frames} frames")

    # Compute viewer-style world offsets so multi-world scenes are spread
    # out in a grid (matching ViewerGL layout). Physics body_q stores all
    # worlds at the same local positions; we apply offsets for rendering.
    world_offsets = _compute_world_offsets(model, state)
    if model.num_worlds > 1:
        print(
            f"  World offsets: {model.num_worlds} worlds, spacing ~{np.linalg.norm(world_offsets[1] - world_offsets[0]):.1f}"
        )
        # Apply offsets to trajectory positions only for on-the-fly generation.
        # Loaded trajectories are already in world-space.
        if not trajectories_from_file:
            trajectory_positions = _apply_world_offsets_to_trajectories(
                trajectory_positions,
                traj_point_world,
                world_offsets,
            )

    # For multi-world scenes, record shape world indices and body=-1 mask.
    # World offsets are applied per-frame to the render context (not to the
    # model) so that the physics / CUDA graph sees the original transforms.
    original_shape_world = None
    if model.num_worlds > 1:
        original_shape_world = model.shape_world.numpy().copy()

    # Apply offsets to a temporary state for bounding sphere computation
    offset_body_q = _apply_world_offsets_to_body_q(state, model, world_offsets)
    if offset_body_q is not None:
        # Temporarily swap body_q for bounding sphere calculation
        original_body_q = state.body_q
        state.body_q = offset_body_q

    # Compute bounding sphere and cameras.  The offset body_q ensures
    # dynamic shapes at grid positions are included.  Body=-1 shapes
    # (tables, ground) are excluded from dynamic-only mode so they
    # don't need offsets here.
    center, radius = compute_bounding_sphere(model, state)
    print(f"  Bounding sphere: center={center}, radius={radius:.3f}")

    if offset_body_q is not None:
        state.body_q = original_body_q

    # Compute minimum half-extent for inflating thin shapes in the color
    # render pass.  The depth pass (for visibility) uses original sizes so
    # that trajectory point depths match the depth buffer accurately.
    fov_rad = math.radians(60.0)
    min_half_extent = _inflate_thin_shapes(model, radius, resolution, fov_rad, camera_distance, min_pixels=min_pixels)
    print(f"  Min shape half-extent for rendering: {min_half_extent:.4f}")

    num_cameras = 6
    camera_transforms = create_axis_cameras(
        center, radius, num_worlds=model.num_worlds, distance_multiplier=camera_distance
    )

    # Particle/cloth scenes need double-sided rendering so the mesh is
    # visible from both sides.  We keep backface culling ON for the depth
    # pass (better visibility accuracy) and toggle it OFF only for the
    # color render pass below.
    has_particles = model.particle_count > 0

    # Set up sensor
    sensor = SensorTiledCamera(
        model=model,
        options=SensorTiledCamera.Options(
            default_light=False,  # We create a custom light below
            default_light_shadows=True,
            checkerboard_texture=True,
            backface_culling=True,
        ),
    )

    sensor.render_context.options.spp = spp

    # Create a directional light matching ViewerGL's sun direction.
    # ViewerGL sun_direction = (0.2, -0.3, 0.8) is the surface-to-light vector.
    # The ray tracer's light orientation is light-to-surface (negated), so we
    # negate ViewerGL's direction.
    sun_dir = np.array([-0.2, 0.3, -0.8], dtype=np.float32)
    sun_dir /= np.linalg.norm(sun_dir)
    sensor.render_context.utils.create_default_light(
        enable_shadows=True,
        direction=wp.vec3f(float(sun_dir[0]), float(sun_dir[1]), float(sun_dir[2])),
    )

    # Use a white clear color; we replace it with a sky gradient in post-processing.
    from newton._src.sensors.warp_raytrace import ClearData  # noqa: PLC0415

    clear_data = ClearData(clear_color=wp.int32(wp.uint32(0xFFFFFFFF)))

    # Move all shapes into the global world so that world 0's camera sees
    # every robot, not just world 0's.  SensorTiledCamera renders per-world;
    # shapes with world_index=-1 are placed in the global world group, which
    # is visible from every world tile.
    if model.num_worlds > 1:
        global_world = wp.array(
            np.full(model.shape_count, -1, dtype=np.int32),
            dtype=wp.int32,
            device=model.device,
        )
        # Update both model and render context — the render context copied
        # the reference during __init__, so replacing the model attribute
        # alone doesn't propagate.
        model.shape_world = global_world
        sensor.render_context.shape_world_index = global_world

    # Apply any custom shape colors defined by the example (e.g. cube colors
    # set via viewer.update_shape_colors() which ViewerNull discards).
    _apply_example_shape_colors(sensor, model, example)

    # Set triangle mesh (cloth) color to match ViewerGL's default tan (0.7, 0.5, 0.3)
    if model.tri_count > 0:
        sensor.render_context.triangle_mesh_color = (0.7, 0.5, 0.3, 1.0)

    camera_rays = sensor.compute_pinhole_camera_rays(
        resolution,
        resolution,
        [fov_rad] * num_cameras,
    )
    color_image = sensor.create_color_image_output(resolution, resolution, num_cameras)
    depth_image = sensor.create_depth_image_output(resolution, resolution, num_cameras)

    # ClearData for depth-only pass: clear depth to 0 (no-hit sentinel)
    depth_clear_data = ClearData(
        clear_color=None,
        clear_depth=wp.float32(0.0),
    )

    os.makedirs(output_dir, exist_ok=True)

    # Subsample trajectories for visualization
    num_all = trajectory_positions.shape[0]
    num_vis = max(1, int(num_all * traj_pct / 100.0))
    vis_indices = np.linspace(0, num_all - 1, num_vis, dtype=int)
    vis_positions = trajectory_positions[vis_indices]
    print(f"  Visualizing {num_vis}/{num_all} trajectories")

    # Generate per-trajectory colors and get camera transforms as numpy
    trail_colors = _generate_trajectory_colors(num_vis)
    camera_transforms_np = camera_transforms.numpy()  # (6, num_worlds, 7)

    # Render each frame
    render_frames = min(num_frames, total_frames)
    t0 = time.time()
    print(f"\n  Rendering {render_frames} frames at {resolution}x{resolution} from {num_cameras} cameras...")

    # Visibility tracking: (num_cameras, num_all_points, num_frames) uint8
    depth_tolerance = depth_tol * radius
    visibility_all = np.zeros((num_cameras, num_all, render_frames), dtype=np.uint8)

    for frame in range(render_frames):
        if frame > 0:
            example.step()
        current_state = getattr(example, state_attr, state)

        # Clear particle BVH arrays so they get rebuilt with current data.
        # For rigid-body-only scenes (no actual particles), also clear the
        # position array so has_particles returns False and update_from_state
        # doesn't try to assign a None state.particle_q.
        # For MPM/particle scenes, keep the position so update_from_state
        # propagates the current state.particle_q to the render context.
        rc = sensor.render_context
        if model.particle_count == 0:
            rc._RenderContext__particles_position = None
        rc.bvh_particles = None
        rc.bvh_particles_lowers = None
        rc.bvh_particles_uppers = None
        rc.bvh_particles_groups = None
        rc.bvh_particles_group_roots = None

        # Apply world offsets to body transforms so multi-world robots
        # appear spread out in a grid (matching ViewerGL layout).
        # Body=-1 shapes are handled separately after update_from_state.
        if model.num_worlds > 1:
            offset_bq = _apply_world_offsets_to_body_q(current_state, model, world_offsets)
            if offset_bq is not None:
                original_bq = current_state.body_q
                current_state.body_q = offset_bq

        # Update body/shape transforms from the simulation state
        sensor.update_from_state(current_state)

        # Restore original body_q so physics isn't affected
        if model.num_worlds > 1 and offset_bq is not None:
            current_state.body_q = original_bq

        # Apply world offsets to body=-1 shapes in the render context.
        # This is done per-frame (after update_from_state resets transforms
        # from the model) rather than baking into model.shape_transform,
        # which would corrupt the CUDA-graph-captured collision pipeline.
        if model.num_worlds > 1 and original_shape_world is not None:
            _apply_worldfixed_offsets_to_rc(
                sensor.render_context, model, original_shape_world, world_offsets
            )

        # Depth-only render pass (no trajectory particles) for visibility.
        # Particles are not yet injected, so the depth buffer reflects only
        # scene geometry — trajectory spheres won't occlude each other.
        sensor.render(
            None,
            camera_transforms,
            camera_rays,
            depth_image=depth_image,
            refit_bvh=True,
            clear_data=depth_clear_data,
        )

        # Compute per-point visibility against the depth buffer
        vis = _compute_visibility(
            depth_image,
            trajectory_positions,
            frame_idx=frame,
            num_cameras=num_cameras,
            camera_transforms_np=camera_transforms_np,
            fov_rad=fov_rad,
            resolution=resolution,
            depth_tolerance=depth_tolerance,
        )
        visibility_all[:, :, frame] = vis

        # Inject trajectory particles as renderable spheres.
        # Clear the particle BVH first: the depth pass built it for N scene
        # particles, but after injection the count is N + num_trajectory_dots.
        # Without clearing, refit_bvh would access out-of-bounds indices.
        rc.bvh_particles = None
        rc.bvh_particles_lowers = None
        rc.bvh_particles_uppers = None
        rc.bvh_particles_groups = None
        rc.bvh_particles_group_roots = None
        inject_trajectory_particles(sensor, vis_positions, frame_idx=frame, point_radius=point_radius)

        # Inflate thin shape dimensions for the color render so that
        # sub-pixel geometry (e.g. thin rails) is visible from all angles.
        # This must happen AFTER the depth render (which needs original
        # sizes for accurate visibility) and BEFORE the color render.
        _apply_shape_inflation(sensor.render_context, model, min_half_extent)

        # For cloth/particle scenes, disable back-face culling for the
        # color render so the mesh is visible from both sides.
        if has_particles:
            sensor.render_context.options.enable_backface_culling = False

        # Color render pass (with trajectory particles)
        sensor.render(
            None,
            camera_transforms,
            camera_rays,
            color_image=color_image,
            refit_bvh=True,
            clear_data=clear_data,
        )

        # Restore backface culling for the next frame's depth pass
        if has_particles:
            sensor.render_context.options.enable_backface_culling = True

        # Synchronize before the next iteration clears bvh_particles,
        # otherwise the render kernel may still be reading the BVH when
        # we free its GPU memory.
        wp.synchronize_device()

        # Save frames with trail lines overlaid
        save_camera_frames(
            color_image,
            output_dir,
            frame,
            num_cameras,
            vis_positions,
            trail_colors,
            camera_transforms_np,
            fov_rad,
            resolution,
            prefix,
        )

        if (frame + 1) % 10 == 0 or frame == render_frames - 1:
            elapsed = time.time() - t0
            print(f"    Frame {frame + 1}/{render_frames}  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nRendering complete in {elapsed:.1f}s")

    # Save per-camera visibility files
    for cam_idx in range(num_cameras):
        vis_path = os.path.join(output_dir, f"{prefix}_cam_{cam_idx}_visibility.npy")
        np.save(vis_path, visibility_all[cam_idx])
    total_visible = visibility_all.sum()
    total_entries = visibility_all.size
    pct_visible = 100.0 * total_visible / max(total_entries, 1)
    print(f"\nVisibility: {pct_visible:.1f}% of point-frame-camera entries visible")
    print(f"  Saved {num_cameras} files: {prefix}_cam_*_visibility.npy  shape=({num_all}, {render_frames})")
    print(f"  Depth tolerance: {depth_tolerance:.4f}")

    # Summary
    total_files = 0
    total_size = 0
    for cam_idx in range(num_cameras):
        cam_dir = os.path.join(output_dir, f"{prefix}_cam_{cam_idx}")
        if os.path.isdir(cam_dir):
            files = [f for f in os.listdir(cam_dir) if f.endswith(".jpg")]
            total_files += len(files)
            total_size += sum(os.path.getsize(os.path.join(cam_dir, f)) for f in files)
    print(f"\nOutput: {output_dir}")
    print(f"  Total files: {total_files}")
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Render a Newton example with 6-camera trajectory visualization.")
    parser.add_argument("--example", type=str, default=None, help="Example name (interactive picker if omitted)")
    parser.add_argument(
        "--trajectories",
        type=str,
        default=None,
        help="Path to NPZ from track_surface_points.py (generates on the fly if omitted)",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save rendered JPG frames")
    parser.add_argument("--num-frames", type=int, default=150, help="Number of simulation frames (default: 150)")
    parser.add_argument(
        "--num-points", type=int, default=1000, help="Number of surface points if generating on the fly (default: 1000)"
    )
    parser.add_argument("--resolution", type=int, default=512, help="Image width and height in pixels (default: 512)")
    parser.add_argument("--device", type=str, default=None, help="Warp device (e.g. cpu, cuda:0)")
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=1.5,
        help="Camera distance as a multiple of the bounding sphere radius (default: 1.5)",
    )
    parser.add_argument(
        "--traj-pct",
        type=float,
        default=10,
        help="Percentage of trajectory points to visualize (default: 10)",
    )
    parser.add_argument(
        "--depth-tol",
        type=float,
        default=1e-4,
        help="Depth tolerance as a fraction of bounding sphere radius (default: 1e-4)",
    )
    parser.add_argument(
        "--num-worlds",
        type=int,
        default=None,
        help="Number of simulation worlds (default: example's own default)",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        default=None,
        help="Prefix for output filenames (default: example name)",
    )
    parser.add_argument(
        "--min-pixels",
        type=float,
        default=1.0,
        help="Minimum shape thickness in pixels for the color render (default: 1.0)",
    )
    parser.add_argument(
        "--spp",
        type=int,
        default=1,
        help="Samples per pixel for anti-aliasing: 1, 4, 9, or 16 (default: 1)",
    )
    parser.add_argument(
        "--point-radius",
        type=float,
        default=0.004,
        help="Radius of each trajectory point sphere (default: 0.004)",
    )

    # Common example-specific args with defaults matching what examples expect
    # when not explicitly provided.  This avoids AttributeError from examples
    # that access args.X directly (guarded by ``if args``).
    parser.add_argument("--use-mujoco-contacts", action="store_true", default=False)
    parser.add_argument("--collision-pipeline", type=str, default=None)
    parser.add_argument("--broad-phase-mode", type=str, default="explicit")

    args, _ = parser.parse_known_args()

    example_map = discover_examples()

    if args.example:
        if args.example not in example_map:
            print(f"Unknown example: {args.example}")
            print("Available:", ", ".join(sorted(example_map.keys())))
            sys.exit(1)
        example_name = args.example
    else:
        example_name = pick_example(example_map)

    print(f"\n{'=' * 50}")
    print(f"  Example:       {example_name}")
    print(f"  Frames:        {args.num_frames}")
    print(f"  Resolution:    {args.resolution}x{args.resolution}")
    print(f"  Output:        {args.output_dir}")
    print(f"  Trajectories:  {args.trajectories or '(generate on the fly)'}")
    print(f"  Device:        {args.device or 'default'}")
    print(f"  Camera dist:   {args.camera_distance}x radius")
    print(f"  Traj pct:      {args.traj_pct}%")
    print(f"  Depth tol:     {args.depth_tol} * radius")
    print(f"  Num worlds:    {args.num_worlds or 'example default'}")
    print(f"{'=' * 50}")

    run_renderer(
        example_name=example_name,
        module_path=example_map[example_name],
        num_frames=args.num_frames,
        num_points=args.num_points,
        resolution=args.resolution,
        output_dir=args.output_dir,
        trajectories_path=args.trajectories,
        device=args.device,
        num_worlds=args.num_worlds,
        camera_distance=args.camera_distance,
        traj_pct=args.traj_pct,
        depth_tol=args.depth_tol,
        name_prefix=args.name_prefix,
        min_pixels=args.min_pixels,
        spp=args.spp,
        point_radius=args.point_radius,
    )


if __name__ == "__main__":
    main()
