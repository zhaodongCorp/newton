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
        [68, 119, 170],   # blue
        [102, 204, 238],  # cyan
        [34, 136, 51],    # green
        [204, 187, 68],   # yellow
        [238, 102, 119],  # red
        [170, 51, 119],   # magenta
        [187, 187, 187],  # grey
        [238, 153, 51],   # orange
        [0, 153, 136],    # teal
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
        colors, dtype=wp.vec4f, device=sensor.render_context.device,
    )


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
            kwargs["num_worlds"] = getattr(args, "num_worlds", 1)
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


def run_renderer(example_name, module_path, num_frames, num_points, resolution, output_dir, trajectories_path, device):
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
    if trajectories_path and os.path.exists(trajectories_path):
        print(f"  Loading trajectories from {trajectories_path}")
        traj_data = np.load(trajectories_path)
        trajectory_positions = traj_data["positions"]
    else:
        print(f"  Generating trajectories on the fly ({num_points} points)...")
        tracker = SurfacePointTracker(model, state, num_points=num_points, seed=42)
        tracker.record(state)
        for frame in range(num_frames):
            example.step()
            current_state = getattr(example, "state_0", state)
            tracker.record(current_state)
        trajectory_positions = np.stack(tracker._frames, axis=1)  # (num_points, num_frames+1, 3)
        # Reset example for the rendering pass (need fresh viewer since set_model is one-shot)
        viewer = newton.viewer.ViewerNull(num_frames=num_frames)
        example = _create_example(mod, viewer, args)
        model = example.model
        state = getattr(example, "state_0", state)

    total_frames = trajectory_positions.shape[1]
    print(f"  Trajectory: {trajectory_positions.shape[0]} points, {total_frames} frames")

    # Compute bounding sphere and cameras
    center, radius = compute_bounding_sphere(model, state)
    print(f"  Bounding sphere: center={center}, radius={radius:.3f}")

    num_cameras = 6
    camera_transforms = create_axis_cameras(center, radius, num_worlds=model.num_worlds)

    # Set up sensor
    sensor = SensorTiledCamera(
        model=model,
        options=SensorTiledCamera.Options(
            default_light=True,
            default_light_shadows=True,
            checkerboard_texture=True,
            backface_culling=True,
        ),
    )

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
        # is None for rigid-body-only scenes). Use the private attribute
        # since the setter doesn't accept None when particles already exist.
        rc = sensor.render_context
        rc._RenderContext__particles_position = None
        rc.bvh_particles = None

        # Update body/shape transforms from the simulation state
        sensor.update_from_state(current_state)

        # Inject trajectory particles as renderable spheres
        inject_trajectory_particles(sensor, trajectory_positions, frame_idx=frame + 1)

        # Render (state=None since we already updated above)
        sensor.render(
            None,
            camera_transforms,
            camera_rays,
            color_image=color_image,
            refit_bvh=True,
        )

        # Save frames
        save_camera_frames(color_image, output_dir, frame, num_cameras)

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
    )


if __name__ == "__main__":
    main()
