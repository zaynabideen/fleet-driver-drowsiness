"""Mouth Aspect Ratio (MAR) on the inner lip contour.

    MAR = mean(|upper_i - lower_i| for 3 vertical pairs) / |left_corner - right_corner|

The inner lips are used (not the outer) because lip thickness and smiling
change the outer contour much more than the actual opening. Closed mouth ~0,
speech usually < 0.4, a full yawn typically > 0.6.
"""

from __future__ import annotations

import numpy as np

from drowsiness.features.landmarks import MOUTH_CORNERS, MOUTH_VERTICAL_PAIRS


def mouth_aspect_ratio(landmarks_px: np.ndarray) -> float:
    pts = np.asarray(landmarks_px, dtype=float)[:, :2]
    width = np.linalg.norm(pts[MOUTH_CORNERS[0]] - pts[MOUTH_CORNERS[1]])
    if width < 1e-6:
        return 0.0
    vertical = np.mean([np.linalg.norm(pts[a] - pts[b]) for a, b in MOUTH_VERTICAL_PAIRS])
    return float(vertical / width)
