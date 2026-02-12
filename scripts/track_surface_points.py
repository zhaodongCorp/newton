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
    """Discover available Newton examples, same logic as newton.examples.main().

    Scans the example subdirectories for files matching the ``example_*.py``
    naming convention and builds a mapping from short names (e.g. "basic_shapes")
    to fully qualified module paths (e.g. "newton.examples.basic.example_basic_shapes").
    """
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
                name = filename[8:-3]  # strip "example_" prefix and ".py" suffix
                example_map[name] = f"newton.examples.{module}.{filename[:-3]}"
    return example_map


def pick_example(example_map):
    """Interactive example picker.

    Presents a numbered list of available examples and accepts either the
    number or name as input. Supports partial name matching for convenience.
    """
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


def _create_example(mod, viewer, args):
    """Instantiate an Example, adapting to its constructor signature.

    Newton examples have varying signatures: some take (viewer, args), others
    take (viewer, num_worlds, args), (viewer, num_worlds), (viewer,), etc.
    This helper inspects the constructor and passes matching arguments.

    The ``args`` namespace is wrapped so that example-specific attributes
    (e.g. ``use_mujoco_contacts``) return ``None`` instead of raising
    ``AttributeError`` when they weren't defined by the tracker's CLI parser.
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
    supported = {"self", "viewer", "num_worlds", "args", "headless", "test_mode", "verbose"}
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


def run_tracker(example_name, module_path, num_frames, num_points, output_path, device, num_worlds=None):
    """Load an example, run it headlessly, track surface points, and save.

    This function:
    1. Dynamically imports the specified Newton example module
    2. Instantiates it with a headless (ViewerNull) viewer
    3. Creates a SurfacePointTracker to sample points on the scene's meshes
    4. Runs the simulation loop, recording point positions each frame
    5. Saves the trajectory data to a compressed NPZ file
    """
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

    # Build a minimal args namespace matching what Example constructors expect.
    # These fields mirror the CLI args from newton.examples.__main__.
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

    # Import and instantiate the example.
    # Examples have varying constructor signatures, so we inspect and adapt.
    print(f"\nLoading example: {example_name} ({module_path})")
    mod = importlib.import_module(module_path)
    example = _create_example(mod, viewer, args)

    # Access the model and initial state from the example.
    # Examples use varying attribute names: state_0, state, states (list), etc.
    model = getattr(example, "model", None)
    state = getattr(example, "state_0", None) or getattr(example, "state", None)
    if state is None:
        states = getattr(example, "states", None)
        if states and len(states) > 0:
            state = states[0]
    if model is None:
        print("ERROR: Example does not expose 'model' attribute.")
        sys.exit(1)
    if state is None:
        print("ERROR: Example does not expose 'state_0', 'state', or 'states' attribute.")
        sys.exit(1)
    state_attr = "state_0" if hasattr(example, "state_0") else "state"

    print(
        f"  Bodies: {model.body_count}, Shapes: {model.shape_count}, "
        f"Particles: {model.particle_count}, Triangles: {model.tri_count}"
    )

    # Create tracker
    try:
        tracker = SurfacePointTracker(model, state, num_points=num_points, seed=42)
    except ValueError as e:
        print(f"ERROR: Cannot create tracker: {e}")
        sys.exit(1)

    print(f"  Tracking {tracker.num_points} surface points over {num_frames} frames...")
    tracker.record(state)

    # Run simulation loop, recording surface point positions after each step.
    # Examples internally swap state buffers, so we re-read the state each frame
    # to get the latest simulation state.
    t0 = time.time()
    for frame in range(num_frames):
        example.step()
        # After step, read the latest state (examples may swap internally)
        current_state = getattr(example, state_attr, state)
        tracker.record(current_state)

        # Progress reporting every 20 frames
        if (frame + 1) % 20 == 0 or frame == num_frames - 1:
            elapsed = time.time() - t0
            print(f"  Frame {frame + 1}/{num_frames}  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nSimulation complete in {elapsed:.1f}s")

    # Apply viewer-style world offsets so that multi-world trajectory
    # positions are spread out in a grid matching ViewerGL layout.
    # Physics body_q stores all worlds at overlapping local positions;
    # without offsets, all robots' trajectories would overlap.
    if model.num_worlds > 1:
        from newton.utils import compute_world_offsets  # noqa: PLC0415

        # Compute grid spacing (same logic as ViewerGL._auto_compute_world_offsets)
        shape_radii = model.shape_collision_radius.numpy()
        max_radius = 0.0
        for s in range(model.shape_count):
            r = float(shape_radii[s])
            if r < 1.0e5:
                max_radius = max(max_radius, r)
        extent = max(max_radius * 2.0, 1.0)
        spacing_val = float(np.ceil(extent * 1.5))
        spacing = [spacing_val, spacing_val, spacing_val]
        spacing[model.up_axis] = 0.0
        world_offsets = compute_world_offsets(model.num_worlds, tuple(spacing), up_axis=model.up_axis)

        # Apply per-point world offset using the tracker's point_world mapping
        point_world = tracker._point_world
        stacked = np.stack(tracker._frames, axis=0)  # (num_frames, num_points, 3)
        for i in range(len(point_world)):
            w = int(point_world[i])
            if 0 <= w < len(world_offsets):
                stacked[:, i, :] += world_offsets[w]
        tracker._frames = [stacked[f] for f in range(stacked.shape[0])]
        print(f"  Applied world offsets for {model.num_worlds} worlds (spacing ~{spacing_val:.1f})")

    # Save trajectories and report statistics
    tracker.save(output_path)

    # Load back and display summary info for quick verification
    positions = np.load(output_path)
    print(f"\nSaved: {output_path}")
    print(f"  Shape: {positions.shape}  (num_points, num_frames, xyz)")
    print(f"  Dtype: {positions.dtype}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")

    # Quick sanity check: report how much the tracked points moved overall.
    # A displacement of 0 would indicate the scene is static or tracking failed.
    displacement = np.linalg.norm(positions[:, -1, :] - positions[:, 0, :], axis=1)
    print(f"  Mean displacement (first -> last frame): {displacement.mean():.4f}")
    print(f"  Max displacement:  {displacement.max():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run a Newton example and save surface point trajectories.")
    parser.add_argument("--example", type=str, default=None, help="Example name (interactive picker if omitted)")
    parser.add_argument("--num-frames", type=int, default=150, help="Number of simulation frames (default: 150)")
    parser.add_argument(
        "--num-points", type=int, default=1000, help="Number of surface points to track (default: 1000)"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output NPY path (default: /tmp/<example>_trajectories.npy)"
    )
    parser.add_argument("--device", type=str, default=None, help="Warp device (e.g. cpu, cuda:0)")
    parser.add_argument(
        "--num-worlds",
        type=int,
        default=None,
        help="Number of simulation worlds (default: example's own default)",
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

    output_path = args.output or f"/tmp/{example_name}_trajectories.npy"

    print(f"\n{'=' * 50}")
    print(f"  Example:    {example_name}")
    print(f"  Frames:     {args.num_frames}")
    print(f"  Points:     {args.num_points}")
    print(f"  Output:     {output_path}")
    print(f"  Device:     {args.device or 'default'}")
    print(f"  Num worlds: {args.num_worlds or 'example default'}")
    print(f"{'=' * 50}")

    run_tracker(
        example_name=example_name,
        module_path=example_map[example_name],
        num_frames=args.num_frames,
        num_points=args.num_points,
        output_path=output_path,
        device=args.device,
        num_worlds=args.num_worlds,
    )


if __name__ == "__main__":
    main()
