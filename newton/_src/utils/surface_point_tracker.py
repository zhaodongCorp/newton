# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np
import warp as wp


class SurfacePointTracker:
    """Samples 3D points on mesh surfaces and tracks their trajectories during simulation.

    Args:
        model: A finalized Newton Model.
        state: The initial simulation State (used to compute initial positions).
        num_points: Total number of points to sample, distributed proportional to surface area.
        seed: Random seed for reproducible sampling.
    """

    def __init__(self, model, state, num_points: int = 10000, seed: int = 42):
        raise NotImplementedError

    def record(self, state) -> None:
        """Record world-space positions of all tracked points for the current frame."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save recorded trajectories to a compressed NPZ file.

        The file contains a single array ``positions`` with shape ``(num_points, num_frames, 3)``.
        """
        raise NotImplementedError
