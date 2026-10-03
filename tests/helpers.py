from __future__ import annotations

from dataclasses import replace

from drowsiness.analysis.calibration import DriverBaseline
from drowsiness.config.settings import Settings
from drowsiness.pipeline.driver_monitor import DriverMonitor, MonitorOutput
from drowsiness.schemas.analysis import BehaviourSnapshot
from drowsiness.schemas.detection import FrameFeatures
from drowsiness.schemas.states import EyeState, ObservationStatus


def run_stream(frames: list[FrameFeatures], settings: Settings | None = None,
               calibrated: bool = False) -> list[MonitorOutput]:
    """Replay a feature stream through the full decision pipeline (no logging to disk)."""
    s = settings or Settings()
    baseline = DriverBaseline.fixed(s.calibration, open_ear=0.30) if calibrated else None
    monitor = DriverMonitor(s, baseline=baseline, alert_sinks=[])
    return [monitor.process_features(f) for f in frames]


def states(outputs: list[MonitorOutput]) -> list[str]:
    return [o.decision.state.value for o in outputs]


def snapshot(**overrides) -> BehaviourSnapshot:
    """A calm, fully observed, calibrated driver; override fields to build test cases."""
    base = BehaviourSnapshot(
        timestamp_s=100.0,
        status=ObservationStatus.OBSERVED,
        observation_confidence=0.95,
        calibrated=True,
        eye_state=EyeState.OPEN,
        eye_confidence=0.95,
        current_closure_s=0.0,
        perclos=0.03,
        perclos_coverage=0.98,
        blink_rate_per_min=15.0,
        slow_blinks=0,
        long_closures=0,
        mean_blink_duration_s=0.15,
        yawning_now=False,
        current_mouth_open_s=0.0,
        yawns=0,
        rel_pitch_deg=0.0,
        rel_yaw_deg=0.0,
        roll_deg=0.0,
        head_down_s=0.0,
        eyes_off_road_s=0.0,
        nods=0,
        head_dropping=False,
        looking_away=False,
        face_lost_s=0.0,
        lost_while_impaired=False,
        camera_unavailable_s=0.0,
        camera_issue=None,
    )
    return replace(base, **overrides)
