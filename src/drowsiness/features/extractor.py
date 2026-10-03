"""Turns a frame + landmark result into a FrameFeatures observation."""

from __future__ import annotations

import cv2

from drowsiness.config.settings import Settings
from drowsiness.features.eye_metrics import eye_aspect_ratios
from drowsiness.features.head_pose import estimate_head_pose
from drowsiness.features.mouth_metrics import mouth_aspect_ratio
from drowsiness.features.signal_quality import (
    CAMERA_FAILED_QUALITY,
    NO_FACE_QUALITY,
    assess_face_quality,
)
from drowsiness.schemas.detection import Frame, FrameFeatures, LandmarkResult


class FeatureExtractor:
    def __init__(self, settings: Settings) -> None:
        self._s = settings

    def camera_failure(self, timestamp_s: float, issue: str) -> FrameFeatures:
        return FrameFeatures(
            timestamp_s=timestamp_s,
            camera_ok=False,
            face_present=False,
            quality=CAMERA_FAILED_QUALITY,
            camera_issue=issue,
        )

    def extract(self, frame: Frame, result: LandmarkResult | None) -> FrameFeatures:
        if result is None:
            return FrameFeatures(frame.timestamp_s, camera_ok=True, face_present=False, quality=NO_FACE_QUALITY)

        px = result.pixels()
        ear_l, ear_r = eye_aspect_ratios(px)
        grey = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        quality = assess_face_quality(grey, px, ear_l, ear_r, self._s.quality)
        pose = estimate_head_pose(px)
        bs = result.blendshapes
        blink = None
        if "eyeBlinkLeft" in bs and "eyeBlinkRight" in bs:
            blink = (bs["eyeBlinkLeft"] + bs["eyeBlinkRight"]) / 2.0
        return FrameFeatures(
            timestamp_s=frame.timestamp_s,
            camera_ok=True,
            face_present=True,
            quality=quality,
            ear_left=ear_l,
            ear_right=ear_r,
            blink_score=blink,
            mar=mouth_aspect_ratio(px),
            jaw_open=bs.get("jawOpen"),
            pitch_deg=pose.pitch_deg if pose else None,
            yaw_deg=pose.yaw_deg if pose else None,
            roll_deg=pose.roll_deg if pose else None,
        )
