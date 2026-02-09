#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end pipeline: track surface points, render trajectories, generate videos.

Runs all three steps sequentially for a chosen Newton example:
  1. Track surface points  ->  trajectories .npy
  2. Render 6-camera frames + visibility  ->  JPG frames + visibility .npy
  3. Convert frames to MP4 videos

Usage:
    uv run python scripts/run_pipeline.py --example basic_urdf --device cuda:0
    uv run python scripts/run_pipeline.py --device cuda:0   # interactive picker
"""

import argparse
import os
import subprocess
import sys
import time


def run_step(description, cmd):
    """Run a subprocess command, printing its output in real time."""
    print(f"\n{'=' * 60}")
    print(f"  STEP: {description}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\nERROR: Step failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n  Completed in {elapsed:.1f}s")
    return result


def discover_examples():
    """Discover available Newton examples."""
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


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: track -> render -> video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--example", type=str, default=None, help="Example name (interactive picker if omitted)")
    parser.add_argument("--output-dir", type=str, default=None, help="Base output directory (default: /tmp/<example>)")
    parser.add_argument("--device", type=str, default=None, help="Warp device (e.g. cpu, cuda:0)")

    # Tracking params
    parser.add_argument("--num-frames", type=int, default=60, help="Number of simulation frames")
    parser.add_argument("--num-points", type=int, default=1000, help="Number of surface points to track")
    parser.add_argument("--num-worlds", type=int, default=None, help="Number of simulation worlds")

    # Rendering params
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution (width=height)")
    parser.add_argument("--camera-distance", type=float, default=1.5, help="Camera distance multiplier")
    parser.add_argument("--traj-pct", type=float, default=10, help="Percentage of trajectories to visualize")
    parser.add_argument("--depth-tol", type=float, default=0.01, help="Depth tolerance (fraction of radius)")

    # Video params
    parser.add_argument("--fps", type=int, default=24, help="Video frames per second")

    args = parser.parse_args()

    # Pick example
    example_map = discover_examples()
    if args.example:
        if args.example not in example_map:
            print(f"Unknown example: {args.example}")
            print("Available:", ", ".join(sorted(example_map.keys())))
            sys.exit(1)
        example_name = args.example
    else:
        example_name = pick_example(example_map)

    # Set up paths
    output_dir = args.output_dir or f"/tmp/{example_name}"
    trajectories_path = os.path.join(output_dir, f"{example_name}_trajectories.npy")
    renders_dir = os.path.join(output_dir, "renders")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"  Pipeline: {example_name}")
    print(f"  Output:   {output_dir}")
    print(f"{'#' * 60}")

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    t_total = time.time()

    # Step 1: Track surface points
    track_cmd = [
        sys.executable,
        os.path.join(scripts_dir, "track_surface_points.py"),
        "--example",
        example_name,
        "--output",
        trajectories_path,
        "--num-frames",
        str(args.num_frames),
        "--num-points",
        str(args.num_points),
    ]
    if args.device:
        track_cmd += ["--device", args.device]
    if args.num_worlds is not None:
        track_cmd += ["--num-worlds", str(args.num_worlds)]

    run_step("Track surface points", track_cmd)

    # Step 2: Render trajectories
    render_cmd = [
        sys.executable,
        os.path.join(scripts_dir, "render_trajectories.py"),
        "--example",
        example_name,
        "--trajectories",
        trajectories_path,
        "--output-dir",
        renders_dir,
        "--num-frames",
        str(args.num_frames),
        "--resolution",
        str(args.resolution),
        "--camera-distance",
        str(args.camera_distance),
        "--traj-pct",
        str(args.traj_pct),
        "--depth-tol",
        str(args.depth_tol),
    ]
    if args.device:
        render_cmd += ["--device", args.device]
    if args.num_worlds is not None:
        render_cmd += ["--num-worlds", str(args.num_worlds)]

    run_step("Render 6-camera frames + visibility", render_cmd)

    # Step 3: Generate videos
    video_cmd = [
        sys.executable,
        os.path.join(scripts_dir, "frames_to_video.py"),
        "--input-dir",
        renders_dir,
        "--name",
        example_name,
        "--fps",
        str(args.fps),
    ]

    run_step("Generate MP4 videos", video_cmd)

    elapsed_total = time.time() - t_total

    # Final summary
    videos_dir = os.path.join(renders_dir, "videos")
    print(f"\n{'#' * 60}")
    print(f"  Pipeline complete in {elapsed_total:.1f}s")
    print(f"{'#' * 60}")
    print(f"\n  Trajectories: {trajectories_path}")
    print(f"  Frames:       {renders_dir}/{example_name}_cam_*/")
    print(f"  Visibility:   {renders_dir}/{example_name}_cam_*_visibility.npy")
    print(f"  Videos:       {videos_dir}/")
    if os.path.isdir(videos_dir):
        for f in sorted(os.listdir(videos_dir)):
            if f.endswith(".mp4"):
                fpath = os.path.join(videos_dir, f)
                size_mb = os.path.getsize(fpath) / 1024 / 1024
                print(f"    {f}  ({size_mb:.1f} MB)")
    print()


if __name__ == "__main__":
    main()
