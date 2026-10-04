"""Full pipeline from images, with a fake landmark detector standing in for MediaPipe."""

import json

import numpy as np

from drowsiness.config.settings import load_settings
from drowsiness.pipeline.driver_monitor import DriverMonitor
from drowsiness.presentation.overlay import render
from drowsiness.schemas.detection import Frame
from drowsiness.schemas.states import DriverState
from drowsiness.simulation.synthetic_face import synthetic_face_image, synthetic_result
from drowsiness.sources.frame_sources import read_features


class ScriptedDetector:
    """Returns synthetic landmarks whose EAR/pitch follow a script of (until_s, ear, pitch, face)."""

    def __init__(self, script):
        self.script = script
        self.closed = False

    def detect(self, image, t):
        for until, ear, pitch, face in self.script:
            if t < until:
                return synthetic_result(ear=ear, pitch_deg=pitch, blendshapes={
                    "eyeBlinkLeft": 0.9 if ear < 0.15 else 0.05, "eyeBlinkRight": 0.9 if ear < 0.15 else 0.05,
                    "jawOpen": 0.02}) if face else None
        return None

    def close(self):
        self.closed = True


def run(script, seconds, settings=None, log_dir=None, fps=15.0, camera_fail_after=None):
    det = ScriptedDetector(script)
    mon = DriverMonitor(settings or load_settings(), detector=det, alert_sinks=[], log_dir=log_dir)
    # Two alternating images: a real sensor never delivers byte-identical frames (that is a frozen feed).
    imgs = [synthetic_face_image(synthetic_result(ear=0.3), seed=k) for k in (0, 1)]
    outs = []
    for i in range(int(seconds * fps)):
        t = i / fps
        frame = None if (camera_fail_after is not None and t >= camera_fail_after) else Frame(imgs[i % 2], t, i)
        outs.append(mon.process_frame(frame, t))
    mon.close()
    assert det.closed
    return outs


def test_image_pipeline_reaches_critical_on_closure_with_head_drop():
    outs = run([(25, 0.30, 0, True), (31, 0.30, 0, True), (40, 0.07, 25, True)], 35)
    assert outs[-1].decision.state == DriverState.CRITICAL_SLEEP_RISK
    assert outs[-1].features.quality.score > 0.7  # real quality checks ran on real pixels


def test_image_pipeline_face_lost_is_unknown():
    outs = run([(28, 0.30, 0, True), (99, 0.3, 0, False)], 31)
    assert outs[-1].decision.state == DriverState.UNKNOWN


def test_missing_frames_become_camera_unavailable():
    outs = run([(99, 0.30, 0, True)], 30, camera_fail_after=27)
    assert outs[-1].decision.state == DriverState.CAMERA_UNAVAILABLE
    assert outs[-1].features.camera_issue == "no_frame"


def test_logs_written_and_no_pixels_stored_by_default(tmp_path):
    run([(25, 0.30, 0, True), (31, 0.30, 0, True), (40, 0.07, 0, True)], 35, log_dir=tmp_path)
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(e["event_type"] == "alert" for e in events)
    assert any(e.get("driver_state") == "CRITICAL_SLEEP_RISK" for e in events)
    assert (tmp_path / "decisions.log").exists()
    assert not (tmp_path / "features.jsonl").exists()
    assert not list(tmp_path.rglob("*.jpg")) and not list(tmp_path.rglob("*.mp4"))


def test_feature_recording_replays_to_identical_decisions(tmp_path):
    settings = load_settings(privacy={"record_features": True})
    outs = run([(25, 0.30, 0, True), (30, 0.30, 0, True), (34, 0.07, 20, True), (99, 0.3, 0, False)], 38,
               settings=settings, log_dir=tmp_path)
    replay = DriverMonitor(settings, alert_sinks=[])
    replayed = [replay.process_features(f).decision.state for f in read_features(tmp_path / "features.jsonl")]
    assert replayed == [o.decision.state for o in outs]


def test_overlay_renders_every_state():
    outs = run([(25, 0.30, 0, True), (31, 0.30, 0, True), (35, 0.07, 25, True)], 35)
    outs += run([(28, 0.30, 0, True), (99, 0.3, 0, False)], 31)
    seen = set()
    img = np.zeros((480, 640, 3), np.uint8)
    for o in outs:
        if o.decision.state not in seen:
            seen.add(o.decision.state)
            canvas = render(img, o, [], 30.0, 0.5)
            assert canvas.shape == (480, 640 + 360, 3)
    assert {DriverState.MONITORING, DriverState.CRITICAL_SLEEP_RISK, DriverState.UNKNOWN} <= seen


def test_state_change_during_calibration_is_logged_without_crashing(tmp_path):
    """Regression: on a real webcam the face often appears ~1 s after start (camera warm-up).
    The UNKNOWN -> MONITORING change happens before calibration ends, when the neutral pose
    was still a NumPy value, and writing that event to JSON crashed the app."""
    from drowsiness.simulation.streams import StreamBuilder

    mon = DriverMonitor(load_settings(), alert_sinks=[], log_dir=tmp_path)
    for f in StreamBuilder().no_face(1.5).alert(3, pitch=5.0).segment(1.2, pitch=5.0, pitch_to=30.0).frames:
        mon.process_features(f)
    mon.close()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    changes = [e for e in events if e["event_type"] == "state_change"]
    assert [e["driver_state"] for e in changes[:2]] == ["UNKNOWN", "MONITORING"]
    head = changes[1]["signals"]
    assert isinstance(head["head_nod"], bool) and isinstance(head["head_pitch_deg"], float)


def test_event_writer_never_crashes_on_numpy_values(tmp_path):
    from drowsiness.events.event_log import JsonlWriter

    w = JsonlWriter(tmp_path / "x.jsonl")
    w.write({"a": np.bool_(True), "b": np.float64(1.5), "c": np.int64(3), "d": np.zeros(2)})
    w.close()
    assert json.loads((tmp_path / "x.jsonl").read_text()) == {"a": True, "b": 1.5, "c": 3, "d": [0.0, 0.0]}
