#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run any Newton example headlessly and save surface point trajectories.

Usage:
    uv run python scripts/track_surface_points.py
    uv run python scripts/track_surface_points.py --example basic_shapes
    uv run python scripts/track_surface_points.py --example basic_shapes --num-frames 120 --num-points 2000
"""

import argparse
import os
import sys
import time


def discover_examples():
    """Discover available Newton examples, same logic as newton.examples.main()."""
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
                name = filename[8:-3]  # strip "example_" and ".py"
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
        # Try as number
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(names):
                return names[idx]
            print(f"  Number out of range (1-{len(names)})")
            continue
        except ValueError:
            pass
        # Try as name
        if choice in example_map:
            return choice
        # Try partial match
        matches = [n for n in names if choice in n]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"  Ambiguous, matches: {', '.join(matches)}")
        else:
            print(f"  Unknown example: {choice}")


def run_tracker(example_name, module_path, num_frames, num_points, output_path, device):
    """Load an example, run it headlessly, track surface points, and save."""
    import importlib  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    import newton  # noqa: PLC0415
    import newton.viewer  # noqa: PLC0415
    from newton.utils import SurfacePointTracker  # noqa: PLC0415

    if device:
        wp.set_device(device)

    # Create a headless viewer
    viewer = newton.viewer.ViewerNull(num_frames=num_frames)

    # Build a minimal args namespace matching what examples expect
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

    # Import and instantiate the example
    print(f"\nLoading example: {example_name} ({module_path})")
    mod = importlib.import_module(module_path)
    example = mod.Example(viewer, args)

    # Find model and initial state
    model = getattr(example, "model", None)
    state = getattr(example, "state_0", None)
    if model is None:
        print("ERROR: Example does not expose 'model' attribute.")
        sys.exit(1)
    if state is None:
        print("ERROR: Example does not expose 'state_0' attribute.")
        sys.exit(1)

    print(f"  Bodies: {model.body_count}, Shapes: {model.shape_count}, "
          f"Particles: {model.particle_count}, Triangles: {model.tri_count}")

    # Create tracker
    try:
        tracker = SurfacePointTracker(model, state, num_points=num_points, seed=42)
    except ValueError as e:
        print(f"ERROR: Cannot create tracker: {e}")
        sys.exit(1)

    print(f"  Tracking {tracker.num_points} surface points over {num_frames} frames...")
    tracker.record(state)

    # Run simulation
    t0 = time.time()
    for frame in range(num_frames):
        example.step()
        # After step, state_0 has the latest state (examples swap internally)
        current_state = getattr(example, "state_0", state)
        tracker.record(current_state)

        if (frame + 1) % 20 == 0 or frame == num_frames - 1:
            elapsed = time.time() - t0
            print(f"  Frame {frame + 1}/{num_frames}  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nSimulation complete in {elapsed:.1f}s")

    # Save
    tracker.save(output_path)

    # Report
    data = np.load(output_path)
    positions = data["positions"]
    print(f"\nSaved: {output_path}")
    print(f"  Shape: {positions.shape}  (num_points, num_frames, xyz)")
    print(f"  Dtype: {positions.dtype}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")

    # Quick sanity check: did anything move?
    displacement = np.linalg.norm(positions[:, -1, :] - positions[:, 0, :], axis=1)
    print(f"  Mean displacement (first -> last frame): {displacement.mean():.4f}")
    print(f"  Max displacement:  {displacement.max():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a Newton example and save surface point trajectories."
    )
    parser.add_argument("--example", type=str, default=None, help="Example name (interactive picker if omitted)")
    parser.add_argument("--num-frames", type=int, default=60, help="Number of simulation frames (default: 60)")
    parser.add_argument("--num-points", type=int, default=1000, help="Number of surface points to track (default: 1000)")
    parser.add_argument("--output", type=str, default=None, help="Output NPZ path (default: /tmp/<example>_trajectories.npz)")
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

    output_path = args.output or f"/tmp/{example_name}_trajectories.npz"

    print(f"\n{'=' * 50}")
    print(f"  Example:    {example_name}")
    print(f"  Frames:     {args.num_frames}")
    print(f"  Points:     {args.num_points}")
    print(f"  Output:     {output_path}")
    print(f"  Device:     {args.device or 'default'}")
    print(f"{'=' * 50}")

    run_tracker(
        example_name=example_name,
        module_path=example_map[example_name],
        num_frames=args.num_frames,
        num_points=args.num_points,
        output_path=output_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
