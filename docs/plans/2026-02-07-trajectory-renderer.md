# Trajectory Renderer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use 10x-engineer:executing-plans to implement this plan task-by-task.

**Goal:** Create `scripts/render_trajectories.py` that re-runs a Newton example with 6 axis-aligned cameras and renders the scene + 3D trajectory particles to RGB JPG frames.

**Architecture:** The script reuses the example discovery/loading pattern from `scripts/track_surface_points.py`. It computes a bounding sphere from shape positions, places 6 cameras on its surface, injects trajectory particles into `SensorTiledCamera`'s `RenderContext`, and saves per-camera JPG frames each simulation step.

**Tech Stack:** Newton `SensorTiledCamera` (GPU raytrace), `warp`, `numpy`, `PIL` (Pillow for JPG)

---

### Task 1: Bounding sphere computation

**Files:**
- Create: `scripts/render_trajectories.py`

**Step 1: Write the bounding sphere function and a smoke test**

Write the initial script file with the bounding sphere computation and a `__main__` block that tests it on a simple box model.

```python
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
            world_pos = body_pos + wp.quat_rotate(wp.quatf(*body_rot), wp.vec3f(*shape_pos)).numpy()
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
```

**Step 2: Run smoke test**

```bash
uv run python -c "
import warp as wp
import newton
from scripts.render_trajectories import compute_bounding_sphere

builder = newton.ModelBuilder()
b = builder.add_body(xform=wp.transform(p=wp.vec3(1.0, 2.0, 3.0), q=wp.quat_identity()))
builder.add_shape_box(body=b, hx=0.5, hy=0.5, hz=0.5)
model = builder.finalize(device='cpu')
state = model.state()
center, radius = compute_bounding_sphere(model, state)
print(f'Center: {center}, Radius: {radius}')
assert abs(center[0] - 1.0) < 0.1
assert radius > 0
print('PASS')
"
```

Expected: PASS with center near (1, 2, 3).

**Step 3: Commit**

```bash
git add scripts/render_trajectories.py && git commit -m "Add bounding sphere computation for trajectory renderer"
```

---

### Task 2: Camera placement

**Files:**
- Modify: `scripts/render_trajectories.py`

**Step 1: Add 6-camera creation function**

Add after `compute_bounding_sphere`:

```python
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
```

**Step 2: Run smoke test**

```bash
uv run python -c "
import numpy as np
import warp as wp
from scripts.render_trajectories import create_axis_cameras

center = np.array([0.0, 0.0, 0.0])
radius = 2.0
cams = create_axis_cameras(center, radius, num_worlds=1)
print(f'Camera transforms shape: {cams.shape}')
assert cams.shape == (6, 1), f'Expected (6, 1), got {cams.shape}'
# Check +X camera is at (3, 0, 0)
cam0 = cams.numpy()[0, 0]
print(f'Camera 0 position: {cam0[:3]}')
assert abs(cam0[0] - 3.0) < 0.01
print('PASS')
"
```

Expected: PASS with shape (6, 1).

**Step 3: Commit**

```bash
git add scripts/render_trajectories.py && git commit -m "Add 6-camera axis-aligned placement"
```

---

### Task 3: Trajectory particle injection

**Files:**
- Modify: `scripts/render_trajectories.py`

**Step 1: Add particle injection function**

This function builds the trajectory particle arrays (current dots + 20-frame trailing) and assigns them to the sensor's render context.

Add after `create_axis_cameras`:

```python
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
```

**Step 2: Smoke test**

```bash
uv run python -c "
import numpy as np
# Just verify the array shapes are correct
num_points = 100
num_frames = 30
traj = np.random.randn(num_points, num_frames, 3).astype(np.float32)

# Simulate what inject_trajectory_particles does
frame_idx = 25
trail_length = 20
start_frame = max(0, frame_idx - trail_length)
num_trail_frames = frame_idx - start_frame + 1
trail = traj[:, start_frame:frame_idx + 1, :]
all_pos = trail.reshape(-1, 3)
print(f'Trail frames: {num_trail_frames}, Total particles: {all_pos.shape[0]}')
assert all_pos.shape[0] == num_trail_frames * num_points
print('PASS')
"
```

Expected: PASS with 21 trail frames, 2100 particles.

**Step 3: Commit**

```bash
git add scripts/render_trajectories.py && git commit -m "Add trajectory particle injection for rendering"
```

---

### Task 4: Frame saving (uint32 to RGB JPG)

**Files:**
- Modify: `scripts/render_trajectories.py`

**Step 1: Add frame extraction and saving function**

Add after `inject_trajectory_particles`:

```python
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
```

**Step 2: Smoke test**

```bash
uv run python -c "
import numpy as np
import os, tempfile

# Simulate packed uint32 RGBA: red pixel
r, g, b, a = 255, 0, 0, 255
packed = np.uint32(r | (g << 8) | (b << 16) | (a << 24))
print(f'Packed: {packed:#010x}')
# Unpack
r_out = (packed >> 0) & 0xFF
g_out = (packed >> 8) & 0xFF
b_out = (packed >> 16) & 0xFF
assert r_out == 255 and g_out == 0 and b_out == 0
print('Unpack PASS')

# Test JPG save
from PIL import Image
img = np.zeros((64, 64, 3), dtype=np.uint8)
img[:, :, 0] = 255  # red
tmpdir = tempfile.mkdtemp()
path = os.path.join(tmpdir, 'test.jpg')
Image.fromarray(img, mode='RGB').save(path, quality=95)
assert os.path.exists(path)
print(f'Saved to {path}, size: {os.path.getsize(path)} bytes')
print('PASS')
"
```

Expected: PASS.

**Step 3: Commit**

```bash
git add scripts/render_trajectories.py && git commit -m "Add per-camera JPG frame saving"
```

---

### Task 5: Main rendering loop and CLI

**Files:**
- Modify: `scripts/render_trajectories.py`

**Step 1: Add the main rendering function and CLI**

Add after `save_camera_frames`:

```python
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


def run_renderer(example_name, module_path, num_frames, num_points, resolution,
                 output_dir, trajectories_path, device):
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
        device=device, viewer="null", headless=True, test=False,
        num_frames=num_frames, collision_pipeline="standard",
        broad_phase_mode="nxn", output_path=None, rerun_address=None,
    )
    print(f"\nLoading example: {example_name} ({module_path})")
    mod = importlib.import_module(module_path)
    example = mod.Example(viewer, args)

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
        # Reset example for the rendering pass
        mod = importlib.import_module(module_path)
        example = mod.Example(viewer, args)
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
            colors_per_shape=True,
            backface_culling=True,
        ),
    )
    camera_rays = sensor.compute_pinhole_camera_rays(
        resolution, resolution, math.radians(60.0),
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

        # Inject trajectory particles
        inject_trajectory_particles(sensor, trajectory_positions, frame_idx=frame + 1)

        # Render
        sensor.render(
            current_state, camera_transforms, camera_rays,
            color_image=color_image, refit_bvh=True,
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
    parser = argparse.ArgumentParser(
        description="Render a Newton example with 6-camera trajectory visualization."
    )
    parser.add_argument("--example", type=str, default=None,
                        help="Example name (interactive picker if omitted)")
    parser.add_argument("--trajectories", type=str, default=None,
                        help="Path to NPZ from track_surface_points.py (generates on the fly if omitted)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save rendered JPG frames")
    parser.add_argument("--num-frames", type=int, default=60,
                        help="Number of simulation frames (default: 60)")
    parser.add_argument("--num-points", type=int, default=1000,
                        help="Number of surface points if generating on the fly (default: 1000)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Image width and height in pixels (default: 512)")
    parser.add_argument("--device", type=str, default=None,
                        help="Warp device (e.g. cpu, cuda:0)")
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
```

**Step 2: Run end-to-end test**

```bash
uv run python scripts/render_trajectories.py \
    --example basic_shapes \
    --output-dir /tmp/test_renders/ \
    --num-frames 10 \
    --num-points 200 \
    --resolution 512 \
    --device cuda:0
```

Expected: 6 subdirectories `cam_0/` through `cam_5/`, each with 10 JPG files.

**Step 3: Verify output**

```bash
ls /tmp/test_renders/cam_0/ | head -5
uv run python -c "
from PIL import Image
img = Image.open('/tmp/test_renders/cam_0/frame_00001.jpg')
print(f'Size: {img.size}, Mode: {img.mode}')
assert img.size == (512, 512)
assert img.mode == 'RGB'
print('PASS')
"
```

**Step 4: Commit**

```bash
git add scripts/render_trajectories.py && git commit -m "Add trajectory renderer with 6-camera JPG output"
```

---

### Task 6: Debug and iterate

This is a debugging task. The rendering pipeline has several moving parts that may need adjustment:

**Potential issues to check:**

1. **Camera orientation**: The look-at quaternion math may produce inverted or rotated images. Run the renderer and visually inspect the JPGs. Fix the rotation matrix construction in `create_axis_cameras` if needed.

2. **Particle visibility**: The trajectory particles may be too small or too large to see. Adjust `0.01` / `0.005` radii in `inject_trajectory_particles`. Scale them relative to the bounding sphere radius.

3. **Scene not visible**: If the camera FOV or distance doesn't capture the scene, adjust the `1.5` multiplier in `create_axis_cameras` or the `60.0` degree FOV.

4. **Model without particles**: If the model's `particle_q` is None, `SensorTiledCamera` may not enable particle rendering. Check that `render_context.options.enable_particles` is True after injecting particles.

**Debug commands:**

```bash
# Quick render with fewer frames for fast iteration
uv run python scripts/render_trajectories.py \
    --example basic_pendulum \
    --output-dir /tmp/debug_renders/ \
    --num-frames 3 \
    --num-points 100 \
    --resolution 256

# View a frame
uv run python -c "
from PIL import Image
Image.open('/tmp/debug_renders/cam_0/frame_00001.jpg').show()
"
```

**Step: Fix any issues found, then commit**

```bash
git add scripts/render_trajectories.py && git commit -m "Fix rendering issues in trajectory renderer"
```

---

### Task 7: Final verification

**Step 1: Run full end-to-end with a real example**

```bash
# First generate trajectories
uv run python scripts/track_surface_points.py \
    --example basic_shapes \
    --num-frames 30 \
    --num-points 500 \
    --output /tmp/basic_shapes_traj.npz

# Then render with pre-computed trajectories
uv run python scripts/render_trajectories.py \
    --example basic_shapes \
    --trajectories /tmp/basic_shapes_traj.npz \
    --output-dir /tmp/basic_shapes_renders/ \
    --num-frames 30 \
    --resolution 512
```

**Step 2: Verify all outputs**

```bash
uv run python -c "
import os
from PIL import Image

output_dir = '/tmp/basic_shapes_renders'
for cam in range(6):
    cam_dir = os.path.join(output_dir, f'cam_{cam}')
    files = sorted(os.listdir(cam_dir))
    print(f'cam_{cam}: {len(files)} files')
    assert len(files) == 30, f'Expected 30, got {len(files)}'
    # Check first frame
    img = Image.open(os.path.join(cam_dir, files[0]))
    assert img.size == (512, 512)
    assert img.mode == 'RGB'
print('All checks PASS')
"
```

**Step 3: Commit the design doc**

```bash
git add docs/plans/2026-02-07-trajectory-renderer-design.md && git commit -m "Add trajectory renderer design document"
```
