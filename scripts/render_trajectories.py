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
    """Compute bounding sphere (center, radius) from all shapes in the scene.

    Iterates shapes, computes world-space positions using body transforms,
    and builds an AABB. The bounding sphere is centered at the AABB center
    with radius equal to half the AABB diagonal.
    """
    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    shape_transforms = model.shape_transform.numpy()  # (num_shapes, 7) as transforms
    shape_body = model.shape_body.numpy()  # (num_shapes,)
    shape_radius = model.shape_collision_radius.numpy()  # (num_shapes,)
    body_q = state.body_q.numpy() if state.body_q is not None else None  # (num_bodies, 7)

    positions = []
    radii = []
    for i in range(model.shape_count):
        r = float(shape_radius[i])
        if r > 1.0e5:
            # Skip infinite planes
            continue

        # Get shape local transform
        shape_xform = shape_transforms[i]
        shape_pos = shape_xform[:3]

        # Compose with body transform if attached
        body_idx = int(shape_body[i])
        if body_idx >= 0 and body_q is not None:
            body_xform = body_q[body_idx]
            body_pos = body_xform[:3]
            body_rot = body_xform[3:]
            # Transform shape position into world space
            rotated = wp.quat_rotate(wp.quatf(*body_rot), wp.vec3f(*shape_pos))
            world_pos = body_pos + np.array([float(rotated[0]), float(rotated[1]), float(rotated[2])])
        else:
            world_pos = shape_pos

        positions.append(world_pos)
        radii.append(r)

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


def create_axis_cameras(center, radius, num_worlds=1):
    """Create 6 axis-aligned cameras on a sphere of radius 1.5*R looking at center.

    Returns a warp array of camera transforms with shape (6, num_worlds)
    suitable for SensorTiledCamera.render().
    """
    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    R = 1.5 * radius
    # 6 axis-aligned directions: +X, -X, +Y, -Y, +Z, -Z
    directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    ]

    transforms = []
    for d in directions:
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


def inject_trajectory_particles(sensor, trajectory_positions, frame_idx, trail_length=20):
    """Inject trajectory points as renderable particles into the sensor.

    For each tracked point, adds the current-frame position and up to
    `trail_length` previous positions as small spheres. The render context's
    particle arrays are replaced each frame.

    Args:
        sensor: SensorTiledCamera instance.
        trajectory_positions: numpy array of shape (num_points, num_frames, 3).
        frame_idx: Current frame index into trajectory_positions.
        trail_length: Number of trailing frames to show (default 20).
    """
    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    num_points = trajectory_positions.shape[0]
    start_frame = max(0, frame_idx - trail_length)
    num_trail_frames = frame_idx - start_frame + 1  # includes current frame

    # Gather positions for all trail frames
    # Shape: (num_trail_frames, num_points, 3)
    trail_positions = trajectory_positions[:, start_frame : frame_idx + 1, :]
    # Reshape to (num_trail_frames * num_points, 3)
    all_positions = trail_positions.reshape(-1, 3).astype(np.float32)

    total_particles = all_positions.shape[0]

    # Radii: current frame gets larger spheres, trail gets smaller
    radii = np.full(total_particles, 0.005, dtype=np.float32)
    # Last num_points entries are the current frame -- make them bigger
    radii[-num_points:] = 0.01

    # World index: all particles belong to world 0
    world_idx = np.zeros(total_particles, dtype=np.int32)

    device = sensor.render_context.device
    sensor.render_context.particles_position = wp.array(all_positions, dtype=wp.vec3f, device=device)
    sensor.render_context.particles_radius = wp.array(radii, dtype=wp.float32, device=device)
    sensor.render_context.particles_world_index = wp.array(world_idx, dtype=wp.int32, device=device)


def save_camera_frames(color_image, output_dir, frame_idx, num_cameras=6):
    """Extract per-camera images from the rendered output and save as RGB JPG.

    The color_image has shape (num_worlds, num_cameras, height, width) with
    uint32 packed RGBA (R in low bits, A in high bits). We extract RGB
    channels and save each camera view as a separate JPG file.

    Args:
        color_image: Warp array of shape (num_worlds, num_cameras, H, W), dtype uint32.
        output_dir: Base output directory.
        frame_idx: Frame number for filename.
        num_cameras: Number of cameras (default 6).
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    # Get numpy array: (num_worlds, num_cameras, H, W)
    color_np = color_image.numpy()

    for cam_idx in range(num_cameras):
        # Extract single camera from world 0: (H, W) uint32
        pixel_data = color_np[0, cam_idx]

        # Unpack RGBA from uint32: R=bits[0:7], G=bits[8:15], B=bits[16:23]
        r = ((pixel_data >> 0) & 0xFF).astype(np.uint8)
        g = ((pixel_data >> 8) & 0xFF).astype(np.uint8)
        b = ((pixel_data >> 16) & 0xFF).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)  # (H, W, 3)

        cam_dir = os.path.join(output_dir, f"cam_{cam_idx}")
        os.makedirs(cam_dir, exist_ok=True)
        filepath = os.path.join(cam_dir, f"frame_{frame_idx + 1:05d}.jpg")
        Image.fromarray(rgb, mode="RGB").save(filepath, quality=95)
