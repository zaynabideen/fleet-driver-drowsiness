import numpy as np
import pytest

from drowsiness.config.settings import Settings
from drowsiness.features.extractor import FeatureExtractor
from drowsiness.features.eye_metrics import eye_aspect_ratio, eye_aspect_ratios
from drowsiness.features.head_pose import estimate_head_pose
from drowsiness.features.mouth_metrics import mouth_aspect_ratio
from drowsiness.features.signal_quality import CameraHealthMonitor
from drowsiness.schemas.detection import Frame, FrameFeatures
from drowsiness.simulation.synthetic_face import synthetic_face_image, synthetic_landmarks, synthetic_result

SIZE = (640, 480)


def _px(lms: np.ndarray) -> np.ndarray:
    return lms * np.array([SIZE[0], SIZE[1], SIZE[0]])


# ---------- EAR ----------

def test_ear_formula_on_known_points():
    # width 4, two vertical gaps of 1 -> EAR = (1 + 1) / (2 * 4) = 0.25
    pts = np.array([[0, 0], [1, -0.5], [3, -0.5], [4, 0], [3, 0.5], [1, 0.5]])
    assert eye_aspect_ratio(pts) == pytest.approx(0.25)


def test_ear_degenerate_eye_is_zero_not_crash():
    assert eye_aspect_ratio(np.zeros((6, 2))) == 0.0


@pytest.mark.parametrize("ear", [0.05, 0.15, 0.30, 0.40])
def test_ear_recovered_from_landmarks(ear):
    left, right = eye_aspect_ratios(_px(synthetic_landmarks(ear=ear)))
    assert left == pytest.approx(ear, abs=1e-6)
    assert right == pytest.approx(ear, abs=1e-6)


def test_ear_is_invariant_to_roll_and_scale():
    base = eye_aspect_ratios(_px(synthetic_landmarks(ear=0.3)))
    rolled = eye_aspect_ratios(_px(synthetic_landmarks(ear=0.3, roll_deg=25, face_height_px=120)))
    assert rolled == pytest.approx(base, abs=1e-6)


def test_ear_drops_when_yawed_far_which_is_why_eyes_become_unobservable():
    frontal = np.mean(eye_aspect_ratios(_px(synthetic_landmarks(ear=0.3))))
    turned = np.mean(eye_aspect_ratios(_px(synthetic_landmarks(ear=0.3, yaw_deg=60))))
    assert turned > frontal * 1.5  # foreshortened width inflates EAR: EAR is invalid here


# ---------- MAR ----------

@pytest.mark.parametrize("mar", [0.0, 0.2, 0.7])
def test_mar_recovered(mar):
    assert mouth_aspect_ratio(_px(synthetic_landmarks(mar=mar))) == pytest.approx(mar, abs=1e-6)


# ---------- Head pose ----------

@pytest.mark.parametrize(
    "pitch,yaw,roll", [(0, 0, 0), (20, 0, 0), (-15, 0, 0), (0, 30, 0), (0, -25, 0), (0, 0, 15), (18, 12, -8)]
)
def test_head_pose_recovers_known_rotation(pitch, yaw, roll):
    pose = estimate_head_pose(_px(synthetic_landmarks(pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll)))
    assert pose is not None
    assert pose.pitch_deg == pytest.approx(pitch, abs=0.5)
    assert pose.yaw_deg == pytest.approx(yaw, abs=0.5)
    assert pose.roll_deg == pytest.approx(roll, abs=0.5)


def test_chin_down_is_positive_pitch():
    pose = estimate_head_pose(_px(synthetic_landmarks(pitch_deg=25)))
    assert pose.pitch_deg > 20


def test_head_pose_degenerate_returns_none():
    assert estimate_head_pose(np.zeros((478, 3))) is None


# ---------- Camera health ----------

def test_camera_black_frame_is_flagged():
    mon = CameraHealthMonitor(Settings().camera)
    assert mon.check(np.zeros((480, 640, 3), np.uint8)) == "dark_or_covered"


def test_camera_missing_frame_is_flagged():
    assert CameraHealthMonitor(Settings().camera).check(None) == "no_frame"


def test_camera_frozen_feed_is_flagged():
    mon = CameraHealthMonitor(Settings().camera)
    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    results = [mon.check(img) for _ in range(Settings().camera.frozen_frame_count + 1)]
    assert results[0] is None
    assert results[-1] == "frozen_feed"


def test_camera_normal_noise_is_ok():
    mon = CameraHealthMonitor(Settings().camera)
    rng = np.random.default_rng(1)
    for _ in range(5):
        assert mon.check(rng.integers(30, 200, (480, 640, 3), dtype=np.uint8)) is None


# ---------- Extractor + face quality ----------

def _extract(result, image):
    return FeatureExtractor(Settings()).extract(Frame(image, 1.0, 0), result)


def test_extractor_good_face():
    res = synthetic_result(ear=0.3, mar=0.1, blendshapes={"eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.2, "jawOpen": 0.05})
    f = _extract(res, synthetic_face_image(res))
    assert f.face_present and f.camera_ok
    assert f.ear == pytest.approx(0.3, abs=1e-6)
    assert f.blink_score == pytest.approx(0.15)
    assert f.quality.eyes_observable
    assert f.quality.score > 0.7, f.quality.issues


def test_extractor_no_face_is_not_a_measurement():
    f = FeatureExtractor(Settings()).extract(Frame(np.zeros((10, 10, 3), np.uint8), 2.0, 0), None)
    assert not f.face_present
    assert f.ear is None and f.quality.score == 0.0
    assert "no_face_detected" in f.quality.issues


def test_sunglasses_make_eyes_unobservable():
    res = synthetic_result(ear=0.3)
    f = _extract(res, synthetic_face_image(res, sunglasses=True))
    assert not f.quality.eyes_observable
    assert "eyes_occluded_or_sunglasses" in f.quality.issues


def test_low_light_reduces_quality():
    res = synthetic_result(ear=0.3)
    f = _extract(res, synthetic_face_image(res, brightness=20))
    assert "low_light" in f.quality.issues
    assert f.quality.score < 0.7


def test_small_face_makes_eyes_unobservable():
    res = synthetic_result(ear=0.3, face_height_px=40)
    f = _extract(res, synthetic_face_image(res))
    assert "face_too_small" in f.quality.issues
    assert not f.quality.eyes_observable


def test_one_eye_occluded_flags_asymmetry():
    res = synthetic_result(ear_left=0.30, ear_right=0.08)
    f = _extract(res, synthetic_face_image(res))
    assert "eye_asymmetry" in f.quality.issues


def test_features_roundtrip_through_dict():
    res = synthetic_result(ear=0.3)
    f = _extract(res, synthetic_face_image(res))
    assert FrameFeatures.from_dict(f.to_dict()) == f
