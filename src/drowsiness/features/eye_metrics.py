"""Eye Aspect Ratio (EAR).

    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

Soukupova & Cech (2016). Two vertical lid distances normalised by eye width,
so it is invariant to face scale and in-plane rotation. Open eyes are
typically ~0.25-0.35 and fall towards ~0.05-0.15 when closed. The absolute
value varies between people, which is why the analysis layer compares it to a
per-driver calibrated baseline instead of a fixed number.

EAR is a 2D measure: it is only meaningful when the eye is seen roughly
face-on. The analysis layer marks eyes UNOBSERVABLE at large yaw/pitch.
"""

from __future__ import annotations

import numpy as np

from drowsiness.features.landmarks import LEFT_EYE_EAR, RIGHT_EYE_EAR


def eye_aspect_ratio(points: np.ndarray) -> float:
    """EAR for six (x, y) points ordered p1..p6. Returns 0.0 for a degenerate eye."""
    p = np.asarray(points, dtype=float)[:, :2]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    if horizontal < 1e-6:
        return 0.0
    return float(vertical / (2.0 * horizontal))


def eye_aspect_ratios(landmarks_px: np.ndarray) -> tuple[float, float]:
    """(left, right) EAR from pixel-space landmarks."""
    left = eye_aspect_ratio(landmarks_px[list(LEFT_EYE_EAR)])
    right = eye_aspect_ratio(landmarks_px[list(RIGHT_EYE_EAR)])
    return left, right
