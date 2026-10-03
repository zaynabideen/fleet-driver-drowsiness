"""Signal quality: can this frame's measurements be trusted?

Two levels:
  * CameraHealthMonitor - is the camera producing usable images at all
    (covered lens, black frames, frozen feed)?
  * assess_face_quality - given a detected face, are the eyes/mouth/head
    measurable (size, lighting, blur, sunglasses/occlusion, asymmetry)?

A low-quality measurement is reported as *unobservable*, never as a value
that happens to look like "eyes closed" or "eyes open".
"""

from __future__ import annotations

import cv2
import numpy as np

from drowsiness.config.settings import CameraSettings, QualitySettings
from drowsiness.features.landmarks import (
    LEFT_EYE_CONTOUR,
    LEFT_EYE_OUTER,
    RIGHT_EYE_CONTOUR,
    RIGHT_EYE_OUTER,
)
from drowsiness.schemas.detection import FrameQuality


class CameraHealthMonitor:
    """Detects frames that carry no usable information about the driver."""

    def __init__(self, settings: CameraSettings) -> None:
        self._s = settings
        self._last_signature: bytes | None = None
        self._identical_count = 0

    def check(self, image_bgr: np.ndarray | None) -> str | None:
        """Return an issue code, or None if the frame is usable."""
        if image_bgr is None or image_bgr.size == 0:
            return "no_frame"
        grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        small = cv2.resize(grey, (32, 24), interpolation=cv2.INTER_AREA)
        signature = small.tobytes()
        if signature == self._last_signature:
            self._identical_count += 1
        else:
            self._identical_count = 0
        self._last_signature = signature
        if self._identical_count >= self._s.frozen_frame_count:
            return "frozen_feed"
        if float(grey.mean()) < self._s.dark_frame_mean:
            return "dark_or_covered"
        if float(grey.std()) < self._s.flat_frame_std:
            return "no_image_detail"
        return None


def _bbox(points: np.ndarray, w: int, h: int, pad: float = 0.0) -> tuple[int, int, int, int] | None:
    x0, y0 = points[:, 0].min(), points[:, 1].min()
    x1, y1 = points[:, 0].max(), points[:, 1].max()
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0 = int(max(0, x0 - px)), int(max(0, y0 - py))
    x1, y1 = int(min(w, x1 + px)), int(min(h, y1 + py))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def assess_face_quality(
    grey: np.ndarray,
    landmarks_px: np.ndarray,
    ear_left: float,
    ear_right: float,
    settings: QualitySettings,
) -> FrameQuality:
    """Quality of a detected face. ``grey`` is the full greyscale frame."""
    h, w = grey.shape[:2]
    issues: list[str] = []
    score = 1.0
    eyes_ok = mouth_ok = True

    inter_ocular = float(np.linalg.norm(landmarks_px[LEFT_EYE_OUTER, :2] - landmarks_px[RIGHT_EYE_OUTER, :2]))
    if inter_ocular / w < settings.min_face_width_ratio:
        issues.append("face_too_small")
        score *= 0.5
        eyes_ok = False

    face_box = _bbox(landmarks_px[:, :2], w, h)
    if face_box is None:
        return FrameQuality(0.0, False, False, False, ("face_out_of_frame",))
    x0, y0, x1, y1 = face_box
    face = grey[y0:y1, x0:x1]
    face_brightness = float(face.mean())
    if face_brightness < settings.min_face_brightness:
        issues.append("low_light")
        score *= 0.6
    elif face_brightness > settings.max_face_brightness:
        issues.append("overexposed")
        score *= 0.6
    if float(cv2.Laplacian(face, cv2.CV_64F).var()) < settings.min_sharpness:
        issues.append("blurred")
        score *= 0.7

    eye_pts = np.vstack([landmarks_px[list(LEFT_EYE_CONTOUR), :2], landmarks_px[list(RIGHT_EYE_CONTOUR), :2]])
    eye_box = _bbox(eye_pts, w, h, pad=0.25)
    if eye_box is not None and face_brightness > 1.0:
        ex0, ey0, ex1, ey1 = eye_box
        eye_ratio = float(grey[ey0:ey1, ex0:ex1].mean()) / face_brightness
        if eye_ratio < settings.eye_region_dark_ratio:
            issues.append("eyes_occluded_or_sunglasses")
            score *= 0.8
            eyes_ok = False

    if abs(ear_left - ear_right) > settings.max_eye_asymmetry:
        issues.append("eye_asymmetry")
        score *= 0.8

    if score < 0.3:
        eyes_ok = mouth_ok = False
    return FrameQuality(
        score=round(score, 4),
        eyes_observable=eyes_ok,
        mouth_observable=mouth_ok,
        head_observable=True,
        issues=tuple(issues),
    )


NO_FACE_QUALITY = FrameQuality(0.0, False, False, False, ("no_face_detected",))
CAMERA_FAILED_QUALITY = FrameQuality(0.0, False, False, False, ("camera_unavailable",))
