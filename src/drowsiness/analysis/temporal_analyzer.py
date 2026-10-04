r"""Temporal analysis: frame-level observations -> behaviour over time.

    FrameFeatures --> calibration --> eye state --> blink / PERCLOS
                                  \-> yawn
                                  \-> head movement
                                  \-> observation gaps (face lost, camera down)
                  --> BehaviourSnapshot (the feature vector for the risk engine)

This layer interprets behaviour but makes no safety decision.
"""

from __future__ import annotations

from drowsiness.analysis.blink_analyzer import BlinkAnalyzer
from drowsiness.analysis.calibration import DriverBaseline
from drowsiness.analysis.eye_state import EyeStateClassifier
from drowsiness.analysis.head_movement import HeadMovementAnalyzer
from drowsiness.analysis.yawn_analyzer import YawnAnalyzer
from drowsiness.config.settings import Settings
from drowsiness.schemas.analysis import BehaviourSnapshot
from drowsiness.schemas.detection import EyeObservation, FrameFeatures
from drowsiness.schemas.states import EyeState, ObservationStatus

DEGRADED_CONFIDENCE_FACTOR = 0.5


class TemporalAnalyzer:
    def __init__(self, settings: Settings, baseline: DriverBaseline | None = None) -> None:
        self._s = settings
        self.baseline = baseline or DriverBaseline(settings.calibration)
        self._eyes = EyeStateClassifier(settings.eyes, settings.quality)
        self._blinks = BlinkAnalyzer(settings.eyes, settings.blink)
        self._yawns = YawnAnalyzer(settings.mouth)
        self._head = HeadMovementAnalyzer(settings.head)
        self._face_lost_since: float | None = None
        self._camera_down_since: float | None = None
        self._last_impaired_s: float | None = None
        self._unobserved_since: float | None = None
        self._lost_while_impaired = False
        self.last_eye: EyeObservation | None = None

    def update(self, f: FrameFeatures) -> BehaviourSnapshot:
        t = f.timestamp_s
        self.baseline.update(f)
        eye = self._eyes.classify(f, self.baseline)
        self.last_eye = eye
        blink = self._blinks.update(t, eye.state)
        yawn = self._yawns.update(f)

        rel_pitch = rel_yaw = None
        if f.face_present and f.pitch_deg is not None and f.yaw_deg is not None:
            n_pitch, n_yaw, _ = self.baseline.neutral_pose
            rel_pitch, rel_yaw = float(f.pitch_deg - n_pitch), float(f.yaw_deg - n_yaw)
        head = self._head.update(t, rel_pitch, rel_yaw, eye.state)

        # --- observation gaps -------------------------------------------------
        self._camera_down_since = None if f.camera_ok else (self._camera_down_since if self._camera_down_since is not None else t)
        face_lost = f.camera_ok and not f.face_present
        self._face_lost_since = (self._face_lost_since if self._face_lost_since is not None else t) if face_lost else None

        impaired_now = (
            (eye.state == EyeState.CLOSED and blink.current_closure_s >= self._s.eyes.slow_blink_s)
            or head.head_dropping
            or head.head_down_s > 0
        )
        if impaired_now:
            self._last_impaired_s = t
        self._track_unobserved_after_impairment(t, eye.state)

        if not f.camera_ok:
            status = ObservationStatus.CAMERA_UNAVAILABLE
        elif not f.face_present:
            status = ObservationStatus.FACE_LOST
        elif eye.state == EyeState.UNOBSERVABLE:
            status = ObservationStatus.DEGRADED
        else:
            status = ObservationStatus.OBSERVED

        if status == ObservationStatus.OBSERVED:
            obs_conf = eye.confidence
        elif status == ObservationStatus.DEGRADED:
            obs_conf = f.quality.score * DEGRADED_CONFIDENCE_FACTOR
        else:
            obs_conf = 0.0

        return BehaviourSnapshot(
            timestamp_s=t,
            status=status,
            observation_confidence=round(obs_conf, 4),
            calibrated=self.baseline.calibrated,
            eye_state=eye.state,
            eye_confidence=eye.confidence,
            current_closure_s=blink.current_closure_s,
            perclos=blink.perclos,
            perclos_coverage=blink.perclos_coverage,
            blink_rate_per_min=blink.blink_rate_per_min,
            slow_blinks=blink.slow_blinks,
            long_closures=blink.long_closures,
            mean_blink_duration_s=blink.mean_blink_duration_s,
            yawning_now=yawn.yawning_now,
            current_mouth_open_s=yawn.current_open_s,
            yawns=yawn.yawns,
            rel_pitch_deg=rel_pitch,
            rel_yaw_deg=rel_yaw,
            roll_deg=f.roll_deg,
            head_down_s=head.head_down_s,
            eyes_off_road_s=head.eyes_off_road_s,
            nods=head.nods,
            head_dropping=head.head_dropping,
            looking_away=head.looking_away,
            face_lost_s=0.0 if self._face_lost_since is None else t - self._face_lost_since,
            lost_while_impaired=self._lost_while_impaired,
            camera_unavailable_s=0.0 if self._camera_down_since is None else t - self._camera_down_since,
            camera_issue=f.camera_issue,
        )

    def _track_unobserved_after_impairment(self, t: float, eye_state: EyeState) -> None:
        """A driver who slumps can disappear from view: that is not a neutral gap."""
        if eye_state != EyeState.UNOBSERVABLE:
            self._unobserved_since = None
            self._lost_while_impaired = False
            return
        if self._unobserved_since is None:
            self._unobserved_since = t
            self._lost_while_impaired = (
                self._last_impaired_s is not None
                and t - self._last_impaired_s <= self._s.head.impaired_lookback_s
            )
