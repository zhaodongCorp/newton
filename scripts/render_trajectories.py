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
