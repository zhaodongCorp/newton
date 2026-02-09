# Trajectory Renderer Design

**Date:** 2026-02-07
**Status:** Approved

## Overview

A standalone script `scripts/render_trajectories.py` that renders a Newton simulation scene with overlaid 3D point trajectories from 6 axis-aligned cameras, producing RGB JPG frames.

This is step 2 of the surface point tracking pipeline. Step 1 (`scripts/track_surface_points.py`) generates and saves 3D point trajectories to NPZ. This step re-runs the simulation, places 6 cameras around the scene, and renders each frame with the scene geometry plus trajectory particles.

## Pipeline

1. Re-run the simulation headlessly (same pattern as `track_surface_points.py`)
2. Compute a bounding sphere from the initial scene geometry
3. Place 6 axis-aligned cameras on a sphere of radius 1.5x the bounding sphere radius
4. Each frame: render the scene via `SensorTiledCamera`, with trajectory points injected as 3D particles (current-frame dots + 20-frame trailing points)
5. Save RGB JPG frames to `output_dir/cam_N/frame_NNNNN.jpg`

## Camera Setup

### Bounding Sphere

Iterate all shapes in the model at the initial state. For each shape, get its world-space position from `body_q` + `shape_transform`. Use `shape_collision_radius` as the shape's extent. Compute the axis-aligned bounding box, then derive the bounding sphere: center = AABB center, radius = half-diagonal of the AABB.

### Camera Placement

6 cameras at distance `R = 1.5 * bounding_sphere_radius` from the center, along the 6 axis directions. Each camera's orientation makes it look toward the center.

| Camera | Position (relative to center) | Look direction |
|--------|-------------------------------|----------------|
| cam_0  | `(+R, 0, 0)`                | `-X`           |
| cam_1  | `(-R, 0, 0)`                | `+X`           |
| cam_2  | `(0, +R, 0)`                | `-Y`           |
| cam_3  | `(0, -R, 0)`                | `+Y`           |
| cam_4  | `(0, 0, +R)`                | `-Z`           |
| cam_5  | `(0, 0, -R)`                | `+Z`           |

- FOV: 60 degrees
- Resolution: 512x512

## Trajectory Rendering

Trajectories are rendered as 3D particles in the scene using the raytrace renderer's native particle support.

### Per-frame particle injection

At each frame `t`:

- **Current points** (frame `t`): All tracked points as small spheres (bright color, e.g. red).
- **Trailing points** (frames `t-1` through `t-19`): Same tracked points at previous positions as smaller/faded spheres showing motion trails.

Total particles per frame: `num_points * min(t+1, 21)`. For 1000 tracked points, up to 21,000 particles.

### Implementation approach

Use the `SensorTiledCamera`'s underlying `RenderContext` to inject trajectory particles for visualization. If direct injection isn't straightforward, fall back to adding non-simulated particles (zero mass, no forces) to the model and updating their positions each frame before rendering.

## Script Interface

```
uv run python scripts/render_trajectories.py \
    --example basic_shapes \
    --trajectories /tmp/basic_shapes_trajectories.npz \
    --output-dir /tmp/basic_shapes_renders/ \
    --num-frames 60 \
    --num-points 1000 \
    --resolution 512 \
    --device cuda:0
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--example` | Newton example name | required |
| `--trajectories` | Path to NPZ from step 1. If omitted, generates trajectories on the fly | None |
| `--output-dir` | Where to save JPG frames | required |
| `--num-frames` | Number of simulation frames | 60 |
| `--num-points` | Points to track if generating on the fly | 1000 |
| `--resolution` | Image width and height in pixels | 512 |
| `--device` | Warp device | auto |

### Output structure

```
output_dir/
  cam_0/          # +X camera
    frame_00001.jpg
    frame_00002.jpg
    ...
  cam_1/          # -X camera
  cam_2/          # +Y camera
  cam_3/          # -Y camera
  cam_4/          # +Z camera
  cam_5/          # -Z camera
```

Each JPG is an RGB image at 512x512.

## Key Components

1. **`compute_bounding_sphere(model, state)`** -- Scene center and radius from shape transforms and collision radii.
2. **`create_cameras(center, radius)`** -- 6 `wp.transformf` camera transforms on the sphere surface.
3. **`render_frame(sensor, state, cameras, rays, trajectory_points, color_image)`** -- Updates particles, renders, extracts per-camera images.
4. **`save_frame(images, output_dir, frame_idx)`** -- Saves 6 RGB JPG files.
5. **Main loop** -- Discovers example, runs simulation, loads/generates trajectories, renders, saves.

## Dependencies

- `PIL` (Pillow) -- JPG saving with RGB conversion (already available)
- `numpy` -- already available
- `warp` -- already available
- `newton.sensors.SensorTiledCamera` -- existing Newton API
- No new dependencies

## Scope Boundaries

- No video encoding (user assembles with `ffmpeg`)
- No interactive viewer -- purely offline batch rendering
- Trajectory data from step 1 NPZ or generated on the fly
- Single-world scenes only
