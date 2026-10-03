"""Head pose from MediaPipe's 3D landmarks.

Instead of solvePnP against a generic 3D face model (which needs camera
intrinsics and is sensitive to the chosen model points), we build the face's
own coordinate frame directly from the 3D landmarks:

    X_face = right-eye-outer -> left-eye-outer      (across the face)
    Y_face = forehead -> chin, orthogonalised to X  (down the face)
    Z_face = X_face x Y_face                        (out of the back of the head)

Expressed in camera coordinates (x right, y down, z away from the camera,
MediaPipe's convention) these columns form the rotation matrix R. For an
upright face looking at the camera R = I. Euler angles (R = Rz Ry Rx):

    pitch = atan2(R[2,1], R[2,2])   positive = chin down
    yaw   = asin(-R[2,0])           sign consistent, only magnitude used downstream
    roll  = atan2(R[1,0], R[0,0])

Absolute angles depend on where the camera is mounted, so the analysis
layer works with angles *relative to the driver's calibrated neutral pose*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from drowsiness.features.landmarks import CHIN, FOREHEAD, LEFT_EYE_OUTER, RIGHT_EYE_OUTER


@dataclass(frozen=True, slots=True)
class HeadPose:
    pitch_deg: float
    yaw_deg: float
    roll_deg: float


def rotation_from_landmarks(landmarks_px: np.ndarray) -> np.ndarray | None:
    p = np.asarray(landmarks_px, dtype=float)
    across = p[LEFT_EYE_OUTER] - p[RIGHT_EYE_OUTER]
    down = p[CHIN] - p[FOREHEAD]
    nx = np.linalg.norm(across)
    if nx < 1e-6:
        return None
    x_axis = across / nx
    down = down - np.dot(down, x_axis) * x_axis
    ny = np.linalg.norm(down)
    if ny < 1e-6:
        return None
    y_axis = down / ny
    z_axis = np.cross(x_axis, y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def euler_from_rotation(r: np.ndarray) -> HeadPose:
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return HeadPose(pitch, yaw, roll)


def estimate_head_pose(landmarks_px: np.ndarray) -> HeadPose | None:
    r = rotation_from_landmarks(landmarks_px)
    return None if r is None else euler_from_rotation(r)
