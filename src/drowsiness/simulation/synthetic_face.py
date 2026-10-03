"""Synthetic face landmarks and images with *known* geometry.

Used to unit-test the feature layer (EAR, MAR, head pose, quality) against
ground truth without a camera or MediaPipe. This is a geometric test
fixture, not a model of real faces; it says nothing about real-world accuracy.
"""

from __future__ import annotations

import math

import numpy as np

from drowsiness.features import landmarks as L
from drowsiness.schemas.detection import LandmarkResult


def _rotation(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    p, y, r = (math.radians(a) for a in (pitch_deg, yaw_deg, roll_deg))
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _place_eye(pts: np.ndarray, ear_idx: tuple[int, ...], contour: tuple[int, ...], cx: float, ear: float) -> None:
    width = 0.36
    opening = ear * width  # EAR = vertical / width for this layout
    cy = -0.3
    corner_a, upper_a, upper_b, corner_b, lower_b, lower_a = ear_idx  # p1, p2, p3, p4, p5, p6
    pts[corner_a] = (cx - width / 2, cy, 0.0)
    pts[corner_b] = (cx + width / 2, cy, 0.0)
    x_a, x_b = cx - width / 2, cx + width / 2
    for up, low, frac in ((upper_a, lower_a, 1 / 3), (upper_b, lower_b, 2 / 3)):
        x = x_a + (x_b - x_a) * frac
        pts[up] = (x, cy - opening / 2, 0.0)
        pts[low] = (x, cy + opening / 2, 0.0)
    used = set(ear_idx)
    rest = [i for i in contour if i not in used]
    for k, i in enumerate(rest):
        t = 2 * math.pi * k / len(rest)
        pts[i] = (cx + width / 2 * math.cos(t), cy + opening / 2 * math.sin(t), 0.0)


def synthetic_landmarks(
    ear: float = 0.30,
    mar: float = 0.02,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    roll_deg: float = 0.0,
    image_size: tuple[int, int] = (640, 480),
    face_height_px: float = 220.0,
    ear_left: float | None = None,
    ear_right: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """(478, 3) normalised landmarks with exact EAR/MAR and a known head rotation."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((L.NUM_LANDMARKS, 3))
    # Filler points spread over a face-shaped ellipse so bounding boxes are realistic.
    t = rng.uniform(0, 2 * math.pi, L.NUM_LANDMARKS)
    rad = np.sqrt(rng.uniform(0, 1, L.NUM_LANDMARKS))
    pts[:, 0] = 0.8 * rad * np.cos(t)
    pts[:, 1] = 1.0 * rad * np.sin(t)
    pts[:, 2] = 0.0

    _place_eye(pts, L.RIGHT_EYE_EAR, L.RIGHT_EYE_CONTOUR, -0.4, ear if ear_right is None else ear_right)
    _place_eye(pts, L.LEFT_EYE_EAR, L.LEFT_EYE_CONTOUR, 0.4, ear if ear_left is None else ear_left)
    # Eye outer corners (33, 263) are placed by _place_eye and also anchor head pose.

    mouth_w = 0.5
    pts[L.MOUTH_CORNERS[0]] = (-mouth_w / 2, 0.5, 0.0)
    pts[L.MOUTH_CORNERS[1]] = (mouth_w / 2, 0.5, 0.0)
    for (up, low), x in zip(L.MOUTH_VERTICAL_PAIRS, (-0.1, 0.0, 0.1), strict=True):
        pts[up] = (x, 0.5 - mar * mouth_w / 2, 0.0)
        pts[low] = (x, 0.5 + mar * mouth_w / 2, 0.0)

    pts[L.FOREHEAD] = (0.0, -1.0, 0.1)
    pts[L.CHIN] = (0.0, 1.0, 0.1)
    pts[L.NOSE_TIP] = (0.0, 0.1, -0.3)

    rotated = pts @ _rotation(pitch_deg, yaw_deg, roll_deg).T
    w, h = image_size
    scale = face_height_px / 2.0
    px = rotated * scale + np.array([w / 2, h / 2, 0.0])
    return px / np.array([w, h, w])


def synthetic_result(
    blendshapes: dict[str, float] | None = None, image_size: tuple[int, int] = (640, 480), **kwargs: float
) -> LandmarkResult:
    lms = synthetic_landmarks(image_size=image_size, **kwargs)  # type: ignore[arg-type]
    return LandmarkResult(lms, image_size[0], image_size[1], dict(blendshapes or {}))


def synthetic_face_image(
    result: LandmarkResult, brightness: int = 140, sunglasses: bool = False, seed: int = 0
) -> np.ndarray:
    """A textured grey image with a lit 'face' region where the landmarks are."""
    import cv2

    rng = np.random.default_rng(seed)
    h, w = result.image_height, result.image_width
    img = rng.integers(40, 90, (h, w), dtype=np.uint8)
    px = result.pixels()[:, :2].astype(np.int32)
    hull = cv2.convexHull(px)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    texture = np.clip(rng.normal(brightness, 25, (h, w)), 0, 255).astype(np.uint8)
    img[mask > 0] = texture[mask > 0]
    if sunglasses:
        for contour in (L.LEFT_EYE_CONTOUR, L.RIGHT_EYE_CONTOUR):
            pts = px[list(contour)]
            x0, y0 = pts.min(axis=0) - 25
            x1, y1 = pts.max(axis=0) + 25
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), 10, -1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
