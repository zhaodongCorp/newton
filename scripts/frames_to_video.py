#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Convert rendered camera frame folders into MP4 videos.

Usage:
    uv run python scripts/frames_to_video.py --input-dir /tmp/renders/ --name basic_urdf
    uv run python scripts/frames_to_video.py --input-dir /tmp/renders/ --name basic_urdf --fps 30
"""

import argparse
import glob
import os
import sys


def make_video(frame_dir, output_path, fps, crf=18):
    """Create an MP4 video from a directory of JPG frames.

    Args:
        frame_dir: Directory containing *_frame_00001.jpg or frame_00001.jpg files.
        output_path: Output .mp4 file path.
        fps: Frames per second.
        crf: Constant Rate Factor for H.264 encoding (0-51, lower = higher
            quality).  Default 18 is visually lossless.
    """
    from PIL import Image  # noqa: PLC0415

    # Match both prefixed (*_frame_*.jpg) and unprefixed (frame_*.jpg) patterns
    frames = sorted(glob.glob(os.path.join(frame_dir, "*_frame_*.jpg")))
    if not frames:
        frames = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    if not frames:
        print(f"  Skipping {frame_dir} (no *frame_*.jpg files)")
        return False

    # Read first frame to get dimensions
    with Image.open(frames[0]) as img:
        width, height = img.size

    try:
        import cv2  # noqa: PLC0415

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for f in frames:
            frame = cv2.imread(f)
            writer.write(frame)
        writer.release()

        # Re-encode with ffmpeg for proper H.264 + CRF quality control.
        # OpenCV's mp4v codec doesn't support CRF, so we transcode the
        # output file.
        import shutil  # noqa: PLC0415
        import subprocess  # noqa: PLC0415

        if shutil.which("ffmpeg"):
            tmp_path = output_path + ".tmp.mp4"
            os.rename(output_path, tmp_path)
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                tmp_path,
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            os.remove(tmp_path)
            if result.returncode != 0:
                print(f"  ffmpeg re-encode failed: {result.stderr.strip()}")
                return False
    except ImportError:
        # Fallback: use ffmpeg subprocess.
        # Derive the input pattern from the actual first filename so that
        # both prefixed (e.g. basic_urdf_frame_00001.jpg) and unprefixed
        # (frame_00001.jpg) naming conventions work.
        import re  # noqa: PLC0415
        import subprocess  # noqa: PLC0415

        first_name = os.path.basename(frames[0])
        # Replace the 5-digit number with ffmpeg's %05d placeholder
        ffmpeg_name = re.sub(r"\d{5}", "%05d", first_name, count=1)
        pattern = os.path.join(frame_dir, ffmpeg_name)
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"  ffmpeg failed: {result.stderr.strip()}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Convert rendered camera frame folders into MP4 videos.")
    parser.add_argument("--input-dir", type=str, required=True, help="Root output directory containing cam_* folders")
    parser.add_argument("--name", type=str, required=True, help="Base name for video files (e.g. basic_urdf)")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second (default: 24)")
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 CRF quality (0-51, lower = higher quality, default: 18 visually lossless)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"ERROR: {input_dir} is not a directory")
        sys.exit(1)

    # Find camera folders: match both prefixed (e.g. basic_urdf_cam_0) and
    # unprefixed (cam_0) naming conventions.
    cam_dirs = sorted(glob.glob(os.path.join(input_dir, f"{args.name}_cam_*")))
    cam_dirs = [d for d in cam_dirs if os.path.isdir(d)]
    if not cam_dirs:
        # Fallback to unprefixed pattern
        cam_dirs = sorted(glob.glob(os.path.join(input_dir, "cam_*")))
        cam_dirs = [d for d in cam_dirs if os.path.isdir(d)]
    if not cam_dirs:
        print(f"ERROR: No camera folders found in {input_dir}")
        sys.exit(1)

    # Create videos subfolder
    videos_dir = os.path.join(input_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    print(f"Input:  {input_dir}")
    print(f"Output: {videos_dir}")
    print(f"FPS:    {args.fps}")
    print(f"CRF:    {args.crf}")
    print(f"Cameras: {len(cam_dirs)}")
    print()

    created = 0
    for cam_dir in cam_dirs:
        cam_name = os.path.basename(cam_dir)  # e.g. "basic_urdf_cam_0" or "cam_0"
        # Avoid double-prefixing: if the folder name already starts with the
        # video name prefix, use it directly; otherwise prepend the name.
        if cam_name.startswith(f"{args.name}_"):
            video_name = f"{cam_name}.mp4"
        else:
            video_name = f"{args.name}_{cam_name}.mp4"
        video_path = os.path.join(videos_dir, video_name)

        num_frames = len(glob.glob(os.path.join(cam_dir, "*frame_*.jpg")))
        print(f"  {cam_name}: {num_frames} frames -> {video_name}")

        if make_video(cam_dir, video_path, args.fps, crf=args.crf):
            size_mb = os.path.getsize(video_path) / 1024 / 1024
            print(f"    -> {size_mb:.1f} MB")
            created += 1

    print(f"\nCreated {created}/{len(cam_dirs)} videos in {videos_dir}")


if __name__ == "__main__":
    main()
