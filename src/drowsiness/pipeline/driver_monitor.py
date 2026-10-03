"""The end-to-end pipeline, with each layer kept separate.

    CAMERA -> camera health -> LANDMARKS -> FEATURES     (observation)
           -> TEMPORAL ANALYSIS                          (interpretation)
           -> RISK ENGINE -> STATE MACHINE               (decision)
           -> ALERT MANAGER -> EVENT LOG                 (action / record)

``process_frame`` runs the whole chain on an image. ``process_features``
starts from recorded features, so the decision layers can be replayed
deterministically without video (privacy-friendly testing and evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from drowsiness.analysis.calibration import DriverBaseline
from drowsiness.analysis.temporal_analyzer import TemporalAnalyzer
from drowsiness.config.settings import Settings
from drowsiness.detection.landmark_detector import LandmarkDetector
from drowsiness.events.event_log import EventRecorder, EvidenceSnapshotter, FeatureRecorder
from drowsiness.features.extractor import FeatureExtractor
from drowsiness.features.signal_quality import CameraHealthMonitor
from drowsiness.safety.alert_manager import ACTIONS, AlertManager, AlertSink, LoggingAlertSink
from drowsiness.safety.risk_engine import RiskModel, RuleBasedRiskEngine
from drowsiness.safety.state_machine import SafetyStateMachine
from drowsiness.schemas.analysis import BehaviourSnapshot
from drowsiness.schemas.detection import Frame, FrameFeatures, LandmarkResult
from drowsiness.schemas.events import Alert, RiskAssessment, StateDecision


@dataclass(frozen=True, slots=True)
class MonitorOutput:
    features: FrameFeatures
    snapshot: BehaviourSnapshot
    risk: RiskAssessment
    decision: StateDecision
    alert: Alert | None
    landmarks: LandmarkResult | None = None


class DriverMonitor:
    def __init__(
        self,
        settings: Settings,
        detector: LandmarkDetector | None = None,
        risk_model: RiskModel | None = None,
        baseline: DriverBaseline | None = None,
        alert_sinks: list[AlertSink] | None = None,
        log_dir: str | Path | None = None,
        snapshot_dir: str | Path = "data/snapshots",
    ) -> None:
        self.settings = settings
        self._detector = detector
        self._camera = CameraHealthMonitor(settings.camera)
        self._extractor = FeatureExtractor(settings)
        self.analyzer = TemporalAnalyzer(settings, baseline)
        self.risk_model: RiskModel = risk_model or RuleBasedRiskEngine(settings.risk)
        self.state_machine = SafetyStateMachine(settings)
        self.alerts = AlertManager(settings.alerts, alert_sinks if alert_sinks is not None else [LoggingAlertSink()])

        self._events: EventRecorder | None = None
        self._features_out: FeatureRecorder | None = None
        self._snapshots: EvidenceSnapshotter | None = None
        if log_dir is not None:
            self._events = EventRecorder(log_dir)
            if settings.privacy.record_features:
                self._features_out = FeatureRecorder(Path(log_dir) / "features.jsonl")
        if settings.privacy.save_snapshots:
            self._snapshots = EvidenceSnapshotter(snapshot_dir, settings.privacy.snapshot_max_width)

    # ------------------------------------------------------------------ entry points
    def process_frame(self, frame: Frame | None, timestamp_s: float) -> MonitorOutput:
        """Run the full chain. Pass ``frame=None`` when the source failed to deliver a frame."""
        issue = self._camera.check(None if frame is None else frame.image)
        landmarks = None
        if issue is not None:
            features = self._extractor.camera_failure(timestamp_s, issue)
        else:
            assert frame is not None
            if self._detector is None:
                raise RuntimeError("DriverMonitor has no detector; use process_features for replays")
            landmarks = self._detector.detect(frame.image, frame.timestamp_s)
            features = self._extractor.extract(frame, landmarks)
        out = self.process_features(features, landmarks=landmarks)
        if out.alert is not None and self._snapshots is not None and frame is not None and out.alert.severity >= 2:
            self._snapshots.save(frame.image, out.alert)
        return out

    def process_features(self, features: FrameFeatures, landmarks: LandmarkResult | None = None) -> MonitorOutput:
        if self._features_out is not None:
            self._features_out.record(features)
        snapshot = self.analyzer.update(features)
        risk = self.risk_model.assess(snapshot)
        decision = self.state_machine.step(snapshot, risk)
        alert = self.alerts.process(decision)
        if self._events is not None:
            if decision.changed:
                self._events.record_decision(decision, snapshot, ACTIONS.get(decision.state))
            if alert is not None:
                self._events.record_alert(alert, snapshot)
        return MonitorOutput(features, snapshot, risk, decision, alert, landmarks)

    def close(self) -> None:
        for closable in (self._events, self._features_out, self._detector):
            if closable is not None:
                closable.close()

