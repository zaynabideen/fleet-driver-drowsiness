"""Data produced by the detection and feature layers.

These are *observations*. Nothing here says whether the driver is drowsy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from drowsiness.schemas.states import EyeState


@dataclass(frozen=True, slots=True)
class Frame:
    """One image from a source, with a monotonic timestamp in seconds."""

    image: np.ndarray  # BGR, HxWx3
    timestamp_s: float
    index: int


@dataclass(frozen=True, slots=True)
class LandmarkResult:
    """Raw output of a face landmark detector for one face.

    ``landmarks`` is an (N, 3) array in MediaPipe convention: x, y normalised
    to [0, 1] by image width/height, z roughly in units of image width.
    """

    landmarks: np.ndarray
    image_width: int
    image_height: int
    blendshapes: dict[str, float] = field(default_factory=dict)

    def pixels(self) -> np.ndarray:
        """Landmarks as (N, 3) in pixel-like units (x*W, y*H, z*W)."""
        scale = np.array([self.image_width, self.image_height, self.image_width], dtype=float)
        return self.landmarks[:, :3] * scale


@dataclass(frozen=True, slots=True)
class FrameQuality:
    """How trustworthy this frame's measurements are, and why."""

    score: float  # 0..1
    eyes_observable: bool
    mouth_observable: bool
    head_observable: bool
    issues: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FrameFeatures:
    """Per-frame numeric features. This is the stream that can be recorded and replayed.

    Head angles are in degrees, absolute (not yet relative to the driver's
    calibrated neutral). Positive pitch = chin down.
    """

    timestamp_s: float
    camera_ok: bool
    face_present: bool
    quality: FrameQuality
    ear_left: float | None = None
    ear_right: float | None = None
    blink_score: float | None = None  # mean eyeBlink blendshape, if available
    mar: float | None = None
    jaw_open: float | None = None
    pitch_deg: float | None = None
    yaw_deg: float | None = None
    roll_deg: float | None = None
    camera_issue: str | None = None

    @property
    def ear(self) -> float | None:
        if self.ear_left is None or self.ear_right is None:
            return None
        return (self.ear_left + self.ear_right) / 2.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["quality"]["issues"] = list(self.quality.issues)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameFeatures:
        q = dict(data["quality"])
        q["issues"] = tuple(q.get("issues", ()))
        return cls(**{**data, "quality": FrameQuality(**q)})


@dataclass(frozen=True, slots=True)
class EyeObservation:
    state: EyeState
    confidence: float
    ear: float | None
    threshold: float | None
    reason: str
