"""Smoke tests against the real MediaPipe model. Skipped when the model is not downloaded."""

from pathlib import Path

import numpy as np
import pytest

MODEL = Path(__file__).resolve().parents[2] / "data" / "models" / "face_landmarker.task"
SAMPLE = Path(__file__).resolve().parents[2] / "data" / "samples" / "face.jpg"

pytestmark = [
    pytest.mark.requires_model,
    pytest.mark.skipif(not MODEL.is_file(), reason="run scripts/download_model.py first"),
]


def test_blank_image_has_no_face():
    from drowsiness.detection.landmark_detector import MediaPipeFaceLandmarker

    with MediaPipeFaceLandmarker(MODEL) as det:
        assert det.detect(np.full((480, 640, 3), 128, np.uint8), 0.0) is None


@pytest.mark.skipif(not SAMPLE.is_file(), reason="put a frontal face photo at data/samples/face.jpg")
def test_real_face_produces_plausible_features():
    import cv2

    from drowsiness.config.settings import Settings
    from drowsiness.detection.landmark_detector import MediaPipeFaceLandmarker
    from drowsiness.features.extractor import FeatureExtractor
    from drowsiness.schemas.detection import Frame

    img = cv2.imread(str(SAMPLE))
    with MediaPipeFaceLandmarker(MODEL) as det:
        res = det.detect(img, 0.0)
    assert res is not None and res.landmarks.shape == (478, 3)
    f = FeatureExtractor(Settings()).extract(Frame(img, 0.0, 0), res)
    assert 0.1 < f.ear < 0.5
    assert abs(f.yaw_deg) < 30
