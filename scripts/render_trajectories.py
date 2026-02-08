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


def create_axis_cameras(center, radius, num_worlds=1, distance_multiplier=1.5):
    """Create 6 axis-aligned cameras on a sphere looking at center.

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
    """Inject current-frame trajectory points as renderable particle spheres.

    Only the current-frame positions are rendered as 3D spheres. Trail lines
    are drawn as 2D overlays in save_camera_frames() instead.

    Args:
        sensor: SensorTiledCamera instance.
        trajectory_positions: numpy array of shape (num_points, num_frames, 3).
        frame_idx: Current frame index into trajectory_positions.
        trail_length: Unused, kept for API compatibility.
    """
    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    num_points = trajectory_positions.shape[0]

    # Only inject current-frame positions as spheres
    current_positions = trajectory_positions[:, frame_idx, :].astype(np.float32)
    radii = np.full(num_points, 0.008, dtype=np.float32)
    # Use global world (-1) so particles are visible from any world tile
    world_idx = np.full(num_points, -1, dtype=np.int32)

    device = sensor.render_context.device
    sensor.render_context.particles_position = wp.array(current_positions, dtype=wp.vec3f, device=device)
    sensor.render_context.particles_radius = wp.array(radii, dtype=wp.float32, device=device)
    sensor.render_context.particles_world_index = wp.array(world_idx, dtype=wp.int32, device=device)


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
        # with a sky gradient: light blue at top -> white at bottom.
        is_bg = (r > 253.0) & (g > 253.0) & (b > 253.0)

        # Build vertical gradient: row 0 = top (light blue), last row = bottom (white)
        h = pixel_data.shape[0]
        sky_top = np.array([135.0, 206.0, 250.0])  # light sky blue
        sky_bot = np.array([255.0, 255.0, 255.0])  # white
        t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]  # (H, 1)
        grad = sky_top * (1.0 - t) + sky_bot * t  # (H, 3)
        # Broadcast gradient to full image width
        grad_r = np.broadcast_to(grad[:, 0:1], pixel_data.shape)
        grad_g = np.broadcast_to(grad[:, 1:2], pixel_data.shape)
        grad_b = np.broadcast_to(grad[:, 2:3], pixel_data.shape)

        r = np.where(is_bg, grad_r, r)
        g = np.where(is_bg, grad_g, g)
        b = np.where(is_bg, grad_b, b)

        # Brighten non-background pixels to compensate for the ray tracer's
        # lower ambient intensity (0.5) compared to ViewerGL (~1.0).
        brightness = 1.5
        not_bg = ~is_bg
        r = np.where(not_bg, np.minimum(r * brightness, 255.0), r)
        g = np.where(not_bg, np.minimum(g * brightness, 255.0), g)
        b = np.where(not_bg, np.minimum(b * brightness, 255.0), b)

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

        cam_dir = os.path.join(output_dir, f"cam_{cam_idx}")
        os.makedirs(cam_dir, exist_ok=True)
        filepath = os.path.join(cam_dir, f"frame_{frame_idx + 1:05d}.jpg")
        img.save(filepath, quality=95)


def _assign_viewer_colors(sensor, model):
    """Assign shape colors matching ViewerGL's Paul Tol Bright palette.

    Replicates the color scheme from ViewerGL._shape_color_map() and the
    dark gray + checkerboard appearance for ground planes.
    """
    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    import newton  # noqa: PLC0415

    # Paul Tol Bright 9-color palette (same as ViewerGL._shape_color_map)
    palette = [
        [68, 119, 170],  # blue
        [102, 204, 238],  # cyan
        [34, 136, 51],  # green
        [204, 187, 68],  # yellow
        [238, 102, 119],  # red
        [170, 51, 119],  # magenta
        [187, 187, 187],  # grey
        [238, 153, 51],  # orange
        [0, 153, 136],  # teal
    ]

    num_shapes = model.shape_count
    geo_types = model.shape_type.numpy()
    colors = np.ones((num_shapes, 4), dtype=np.float32)

    for s in range(num_shapes):
        if int(geo_types[s]) == int(newton.GeoType.PLANE):
            # Ground plane: dark gray (matches ViewerGL)
            colors[s] = [0.125, 0.125, 0.15, 1.0]
        else:
            c = palette[s % len(palette)]
            colors[s] = [c[0] / 255.0, c[1] / 255.0, c[2] / 255.0, 1.0]

    sensor.render_context.shape_colors = wp.array(
        colors,
        dtype=wp.vec4f,
        device=sensor.render_context.device,
    )


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
    """
    import inspect  # noqa: PLC0415

    sig = inspect.signature(mod.Example.__init__)
    params = list(sig.parameters.keys())  # includes 'self'
    kwargs = {}
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
            kwargs["args"] = args
        elif name == "headless":
            kwargs["headless"] = getattr(args, "headless", True)
        elif name == "test_mode":
            kwargs["test_mode"] = getattr(args, "test", False)
        elif name == "verbose":
            kwargs["verbose"] = False
    return mod.Example(**kwargs)


def discover_examples():
    """Discover available Newton examples (reused from track_surface_points.py)."""
    import newton.examples  # noqa: PLC0415

    src_dir = newton.examples.get_source_directory()
    modules = ["basic", "cable", "cloth", "contacts", "diffsim", "ik", "mpm", "robot", "selection", "sensors"]
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
):
    """Load example, run simulation with rendering, save JPG frames."""
    import importlib  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    import newton  # noqa: PLC0415
    import newton.viewer  # noqa: PLC0415
    from newton.sensors import SensorTiledCamera  # noqa: PLC0415
    from newton.utils import SurfacePointTracker  # noqa: PLC0415

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
        output_path=None,
        rerun_address=None,
    )
    print(f"\nLoading example: {example_name} ({module_path})")
    mod = importlib.import_module(module_path)
    example = _create_example(mod, viewer, args)

    model = getattr(example, "model", None)
    state = getattr(example, "state_0", None)
    if model is None or state is None:
        print("ERROR: Example does not expose 'model' and 'state_0'.")
        sys.exit(1)

    print(f"  Bodies: {model.body_count}, Shapes: {model.shape_count}")

    # Load or generate trajectories
    trajectories_pre_offset = False  # Whether trajectory positions already include world offsets
    if trajectories_path and os.path.exists(trajectories_path):
        print(f"  Loading trajectories from {trajectories_path}")
        traj_data = np.load(trajectories_path)
        trajectory_positions = traj_data["positions"]
        # Trajectories saved by track_surface_points.py already have world
        # offsets baked in (point_world is included for reference).
        if "point_world" in traj_data:
            trajectories_pre_offset = True
            traj_point_world = traj_data["point_world"]
        else:
            traj_point_world = None
    else:
        print(f"  Generating trajectories on the fly ({num_points} points)...")
        tracker = SurfacePointTracker(model, state, num_points=num_points, seed=42)
        tracker.record(state)
        for _frame in range(num_frames):
            example.step()
            current_state = getattr(example, "state_0", state)
            tracker.record(current_state)
        trajectory_positions = np.stack(tracker._frames, axis=1)  # (num_points, num_frames+1, 3)
        traj_point_world = tracker._point_world
        # Reset example for the rendering pass (need fresh viewer since set_model is one-shot)
        viewer = newton.viewer.ViewerNull(num_frames=num_frames)
        example = _create_example(mod, viewer, args)
        model = example.model
        state = getattr(example, "state_0", state)

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
        # Apply offsets to trajectory positions only if not already baked in.
        # Pre-computed trajectories from track_surface_points.py have offsets
        # applied at save time; on-the-fly trajectories need offsets here.
        if not trajectories_pre_offset and traj_point_world is not None:
            trajectory_positions = _apply_world_offsets_to_trajectories(
                trajectory_positions,
                traj_point_world,
                world_offsets,
            )

    # Apply offsets to a temporary state for bounding sphere computation
    offset_body_q = _apply_world_offsets_to_body_q(state, model, world_offsets)
    if offset_body_q is not None:
        # Temporarily swap body_q for bounding sphere calculation
        original_body_q = state.body_q
        state.body_q = offset_body_q

    # Compute bounding sphere and cameras
    center, radius = compute_bounding_sphere(model, state)
    print(f"  Bounding sphere: center={center}, radius={radius:.3f}")

    if offset_body_q is not None:
        state.body_q = original_body_q

    num_cameras = 6
    camera_transforms = create_axis_cameras(
        center, radius, num_worlds=model.num_worlds, distance_multiplier=camera_distance
    )

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

    # Assign shape colors to match ViewerGL appearance (Paul Tol Bright palette)
    _assign_viewer_colors(sensor, model)
    fov_rad = math.radians(60.0)
    camera_rays = sensor.compute_pinhole_camera_rays(
        resolution,
        resolution,
        [fov_rad] * num_cameras,
    )
    color_image = sensor.create_color_image_output(resolution, resolution, num_cameras)

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
    render_frames = min(num_frames, total_frames - 1)
    t0 = time.time()
    print(f"\n  Rendering {render_frames} frames at {resolution}x{resolution} from {num_cameras} cameras...")

    for frame in range(render_frames):
        if frame > 0:
            example.step()
        current_state = getattr(example, "state_0", state)

        # Clear injected particles before update_from_state so that
        # has_particles returns False (avoids crash when state.particle_q
        # is None for rigid-body-only scenes). Also clear BVH bounds arrays
        # since particle count changes each frame as the trail grows.
        rc = sensor.render_context
        rc._RenderContext__particles_position = None
        rc.bvh_particles = None
        rc.bvh_particles_lowers = None
        rc.bvh_particles_uppers = None
        rc.bvh_particles_groups = None
        rc.bvh_particles_group_roots = None

        # Apply world offsets to body transforms so multi-world robots
        # appear spread out in a grid (matching ViewerGL layout).
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

        # Inject trajectory particles as renderable spheres
        inject_trajectory_particles(sensor, vis_positions, frame_idx=frame + 1)

        # Render (state=None since we already updated above)
        sensor.render(
            None,
            camera_transforms,
            camera_rays,
            color_image=color_image,
            refit_bvh=True,
            clear_data=clear_data,
        )

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
        )

        if (frame + 1) % 10 == 0 or frame == render_frames - 1:
            elapsed = time.time() - t0
            print(f"    Frame {frame + 1}/{render_frames}  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nRendering complete in {elapsed:.1f}s")

    # Summary
    total_files = 0
    total_size = 0
    for cam_idx in range(num_cameras):
        cam_dir = os.path.join(output_dir, f"cam_{cam_idx}")
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
    parser.add_argument("--num-frames", type=int, default=60, help="Number of simulation frames (default: 60)")
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
        "--num-worlds",
        type=int,
        default=None,
        help="Number of simulation worlds (default: example's own default)",
    )
    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()
