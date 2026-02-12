#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compose 6-camera videos into 2x3 grids and concatenate all examples.

For each simulation example, the 6 per-camera MP4s are arranged into a
2-row x 3-column grid.  A title overlay shows the example name.  All
per-example grids are then concatenated into a single long video.

Requires: ffmpeg on PATH.

Usage:
    uv run python scripts/compose_videos.py --input-dir /path/to/videos
    uv run python scripts/compose_videos.py --input-dir /path/to/videos --output final.mp4
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile


def find_examples(input_dir):
    """Group MP4 files by example name.

    Expects filenames like ``<example>_cam_0.mp4`` … ``<example>_cam_5.mp4``.

    Returns:
        Sorted list of (example_name, [cam0.mp4, …, cam5.mp4]) tuples.
    """
    pattern = re.compile(r"^(.+)_cam_(\d+)\.mp4$")
    groups = {}
    for fname in os.listdir(input_dir):
        m = pattern.match(fname)
        if not m:
            continue
        name, cam_idx = m.group(1), int(m.group(2))
        groups.setdefault(name, {})[cam_idx] = os.path.join(input_dir, fname)

    results = []
    for name in sorted(groups):
        cams = groups[name]
        if len(cams) != 6:
            print(f"  WARNING: {name} has {len(cams)} cameras (expected 6), skipping")
            continue
        cam_files = [cams[i] for i in range(6)]
        results.append((name, cam_files))
    return results


def compose_grid(cam_files, output_path, title, crf):
    """Create a 2x3 grid video from 6 camera videos with a title overlay.

    Layout:
        cam_0 | cam_1 | cam_2
        cam_3 | cam_4 | cam_5
    """
    # Build ffmpeg filter graph.
    # Scale each input to the same size, pad to even dimensions, then xstack.
    inputs = []
    for f in cam_files:
        inputs.extend(["-i", f])

    # Use xstack filter for the 2x3 grid, then overlay title text.
    filter_parts = []
    # Scale each stream to match the first input's size and ensure even dims
    for i in range(6):
        filter_parts.append(
            f"[{i}:v]scale=iw:ih:force_original_aspect_ratio=decrease,"
            f"pad=ceil(iw/2)*2:ceil(ih/2)*2[v{i}]"
        )
    # xstack with 2 rows x 3 columns
    stack_inputs = "".join(f"[v{i}]" for i in range(6))
    filter_parts.append(f"{stack_inputs}xstack=inputs=6:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0[grid]")

    # Title overlay — white text with dark background bar at the top
    filter_parts.append(
        "[grid]drawtext="
        f"text='{title}':"
        "fontsize=36:fontcolor=white:"
        "x=(w-text_w)/2:y=20:"
        "box=1:boxcolor=black@0.6:boxborderw=8"
        "[out]"
    )

    filter_graph = ";\n".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_graph,
        "-map", "[out]",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"  ERROR composing {title}:")
        # Show last few lines of stderr (most relevant)
        lines = result.stderr.strip().splitlines()
        for line in lines[-10:]:
            print(f"    {line}")
        return False
    return True


def concatenate_videos(video_paths, output_path, crf):
    """Concatenate a list of videos into a single video using ffmpeg."""
    # Use ffmpeg concat demuxer via a temp file list
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for path in video_paths:
            # ffmpeg concat format requires escaping single quotes
            escaped = path.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
        list_path = f.name

    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c:v", "libx264",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print("  ERROR concatenating videos:")
            lines = result.stderr.strip().splitlines()
            for line in lines[-10:]:
                print(f"    {line}")
            return False
        return True
    finally:
        os.remove(list_path)


def main():
    parser = argparse.ArgumentParser(
        description="Compose 6-camera videos into 2x3 grids and concatenate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing <example>_cam_0..5.mp4 files",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for the final concatenated video (default: <input-dir>/composed.mp4)",
    )
    parser.add_argument(
        "--crf", type=int, default=18,
        help="H.264 CRF quality (0-51, lower = higher quality)",
    )
    parser.add_argument(
        "--keep-grids", action="store_true",
        help="Keep intermediate per-example grid videos",
    )
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found on PATH")
        sys.exit(1)

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"ERROR: {input_dir} is not a directory")
        sys.exit(1)

    examples = find_examples(input_dir)
    if not examples:
        print(f"ERROR: No valid 6-camera example sets found in {input_dir}")
        print("  Expected files like: <example>_cam_0.mp4 ... <example>_cam_5.mp4")
        sys.exit(1)

    output_path = args.output or os.path.join(input_dir, "composed.mp4")
    grids_dir = os.path.join(input_dir, "grids")
    os.makedirs(grids_dir, exist_ok=True)

    print(f"Input:    {input_dir}")
    print(f"Output:   {output_path}")
    print(f"Examples: {len(examples)}")
    print(f"CRF:      {args.crf}")
    print()

    # Step 1: Compose each example's 6 cameras into a 2x3 grid
    grid_paths = []
    for name, cam_files in examples:
        grid_path = os.path.join(grids_dir, f"{name}_grid.mp4")
        print(f"  Composing {name} ...")
        if compose_grid(cam_files, grid_path, title=name, crf=args.crf):
            size_mb = os.path.getsize(grid_path) / 1024 / 1024
            print(f"    -> {grid_path} ({size_mb:.1f} MB)")
            grid_paths.append(grid_path)
        else:
            print(f"    FAILED, skipping {name}")

    if not grid_paths:
        print("\nERROR: No grid videos were created")
        sys.exit(1)

    # Step 2: Concatenate all grids into a single video
    print(f"\n  Concatenating {len(grid_paths)} examples ...")
    if concatenate_videos(grid_paths, output_path, crf=args.crf):
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"    -> {output_path} ({size_mb:.1f} MB)")
    else:
        print("  FAILED to concatenate")
        sys.exit(1)

    # Cleanup intermediate grids unless --keep-grids
    if not args.keep_grids:
        for p in grid_paths:
            os.remove(p)
        try:
            os.rmdir(grids_dir)
        except OSError:
            pass  # directory not empty or doesn't exist

    print("\nDone.")


if __name__ == "__main__":
    main()
