"""Face landmark detection.

Why MediaPipe Face Landmarker (Tasks API):
  * 478 3D landmarks incl. a dense eye contour -> stable EAR/MAR and a 3D head pose
    without camera intrinsics.
  * Blendshape scores (eyeBlink*, jawOpen) give an *independent* second opinion
    on eye closure and mouth opening.
  * Runs in real time on a laptop CPU; Apache-2.0.
Rejected: Haar cascades (no landmarks), dlib-68 (sparse eye points, HOG face
detector fails at moderate yaw). The legacy ``mp.solutions.face_mesh`` API is
removed from current MediaPipe releases, so the Tasks API is used.

The pipeline depends only on the ``LandmarkDetector`` protocol, so tests and
replays never need MediaPipe or a camera.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from drowsiness.schemas.detection import LandmarkResult

DEFAULT_MODEL_PATH = Path("data/models/face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


class LandmarkDetector(Protocol):
    def detect(self, image_bgr: np.ndarray, timestamp_s: float) -> LandmarkResult | None:
        """Return landmarks for the most prominent face, or None if no face is found."""

    def close(self) -> None: ...


class ModelNotFoundError(FileNotFoundError):
    pass


class MediaPipeFaceLandmarker:
    """MediaPipe Tasks Face Landmarker in VIDEO mode (uses temporal tracking between frames)."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ModelNotFoundError(
                f"Face landmarker model not found at {model_path}. "
                "Run: python scripts/download_model.py"
            )
        import mediapipe as mp  # imported lazily so the rest of the system works without it
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, image_bgr: np.ndarray, timestamp_s: float) -> LandmarkResult | None:
        # VIDEO mode requires strictly increasing integer millisecond timestamps.
        ts_ms = max(int(round(timestamp_s * 1000)), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        if not result.face_landmarks:
            return None
        lms = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=float)
        blend: dict[str, float] = {}
        if result.face_blendshapes:
            blend = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
        h, w = image_bgr.shape[:2]
        return LandmarkResult(landmarks=lms, image_width=w, image_height=h, blendshapes=blend)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> MediaPipeFaceLandmarker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
