"""Structured event logging and optional evidence.

Three separate outputs, so privacy choices are explicit:

  logs/events.jsonl      state changes and alerts (metadata only, always on)
  logs/decisions.log     human-readable "what / why / how confident / how long"
  logs/features.jsonl    per-frame numeric features, no pixels (opt-in, for replay/eval)
  data/snapshots/*.jpg   downscaled frame on HIGH/CRITICAL alerts (opt-in)

Raw video is never written by this module.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

import numpy as np

from drowsiness.schemas.analysis import BehaviourSnapshot
from drowsiness.schemas.detection import FrameFeatures
from drowsiness.schemas.events import Alert, StateDecision


class JsonFormatter(logging.Formatter):
    """One JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if isinstance(getattr(record, "event", None), dict):
            payload["event"] = record.event  # type: ignore[attr-defined]
        return json.dumps(payload)


def configure_logging(directory: str | Path = "logs", console_level: str = "INFO") -> None:
    """Console: readable lines. File: JSON lines."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("drowsiness")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    file_handler = logging.FileHandler(Path(directory) / "system.jsonl", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(console)
    root.addHandler(file_handler)


def _wall(ts_s: float, origin_wall: datetime | None) -> str:
    if origin_wall is None:
        return datetime.now(UTC).isoformat(timespec="milliseconds")
    return datetime.fromtimestamp(origin_wall.timestamp() + ts_s, UTC).isoformat(timespec="milliseconds")


def signals_from(s: BehaviourSnapshot | None) -> dict[str, Any]:
    if s is None:
        return {}
    return {
        "eye_state": s.eye_state.value,
        "eyes_closed": s.eye_state.value == "CLOSED",
        "eye_closure_s": round(s.current_closure_s, 3),
        "perclos": None if s.perclos is None else round(s.perclos, 3),
        "blink_rate_per_min": None if s.blink_rate_per_min is None else round(s.blink_rate_per_min, 1),
        "slow_blinks": s.slow_blinks,
        "long_closures": s.long_closures,
        "mean_blink_duration_s": None if s.mean_blink_duration_s is None else round(s.mean_blink_duration_s, 3),
        "yawn_detected": s.yawning_now,
        "yawns": s.yawns,
        "head_pitch_deg": None if s.rel_pitch_deg is None else round(s.rel_pitch_deg, 1),
        "head_yaw_deg": None if s.rel_yaw_deg is None else round(s.rel_yaw_deg, 1),
        "head_nod": s.head_dropping,
        "nods": s.nods,
        "head_down_s": round(s.head_down_s, 2),
        "eyes_off_road_s": round(s.eyes_off_road_s, 2),
        "observation": s.status.value,
        "face_lost_s": round(s.face_lost_s, 2),
    }


def decision_event(d: StateDecision, snapshot: BehaviourSnapshot | None, action: str | None,
                   origin_wall: datetime | None = None) -> dict[str, Any]:
    risk = d.risk
    return {
        "event_id": uuid.uuid4().hex,
        "event_type": "state_change",
        "timestamp": _wall(d.timestamp_s, origin_wall),
        "stream_time_s": round(d.timestamp_s, 3),
        "driver_state": d.state.value,
        "previous_state": d.previous_state.value,
        "confidence": round(d.confidence, 3),
        "risk_score": None if risk is None else risk.risk_score,
        "risk_level": None if risk is None else risk.level.name,
        "rule": None if risk is None else risk.rule,
        "evidence": [] if risk is None else [e.to_dict() for e in risk.active_evidence],
        "reason": d.reason,
        "signals": signals_from(snapshot),
        "action": action,
    }


def alert_event(a: Alert, snapshot: BehaviourSnapshot | None, origin_wall: datetime | None = None) -> dict[str, Any]:
    return {
        "event_id": a.alert_id,
        "event_type": "alert",
        "timestamp": _wall(a.timestamp_s, origin_wall),
        "stream_time_s": round(a.timestamp_s, 3),
        "alert_type": a.alert_type.value,
        "severity": a.severity,
        "confidence": round(a.confidence, 3),
        "reason": a.reason,
        "evidence": list(a.evidence),
        "cooldown_s": a.cooldown_s,
        "signals": signals_from(snapshot),
        "action": a.action,
    }


def format_decision(d: StateDecision, origin_wall: datetime | None = None) -> str:
    """Observability line: WHAT happened, WHY, HOW CONFIDENT, HOW LONG."""
    lines = [
        _wall(d.timestamp_s, origin_wall),
        f"{d.previous_state.value} -> {d.state.value}",
        f"Confidence: {d.confidence:.2f}",
    ]
    if d.risk is not None:
        lines.append(f"Risk: {d.risk.level.name} ({d.risk.risk_score:.2f}) via rule '{d.risk.rule}'")
        if d.risk.active_evidence:
            lines.append("Evidence:")
            lines.extend(f"  * {e.description}" for e in d.risk.active_evidence)
    lines.append(f"Reason: {d.reason}")
    return "\n".join(lines)


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] = open(self.path, "a", encoding="utf-8")

    def write(self, obj: dict[str, Any]) -> None:
        self._fh.write(json.dumps(obj) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class EventRecorder:
    """Writes state changes and alerts to events.jsonl, and readable decisions to decisions.log."""

    def __init__(self, directory: str | Path, origin_wall: datetime | None = None) -> None:
        directory = Path(directory)
        self.events = JsonlWriter(directory / "events.jsonl")
        self._decisions = open(directory / "decisions.log", "a", encoding="utf-8")
        self.origin_wall = origin_wall or datetime.now(UTC)

    def record_decision(self, d: StateDecision, snapshot: BehaviourSnapshot | None, action: str | None) -> None:
        self.events.write(decision_event(d, snapshot, action, self.origin_wall))
        self._decisions.write(format_decision(d, self.origin_wall) + "\n\n")
        self._decisions.flush()

    def record_alert(self, a: Alert, snapshot: BehaviourSnapshot | None) -> None:
        self.events.write(alert_event(a, snapshot, self.origin_wall))

    def close(self) -> None:
        self.events.close()
        self._decisions.close()


class FeatureRecorder:
    """Per-frame numeric features (no pixels) for deterministic replay and evaluation."""

    def __init__(self, path: str | Path) -> None:
        self._w = JsonlWriter(path)

    def record(self, f: FrameFeatures) -> None:
        self._w.write(f.to_dict())

    def close(self) -> None:
        self._w.close()


class EvidenceSnapshotter:
    """Opt-in: save a downscaled still when a serious alert fires. Never video."""

    def __init__(self, directory: str | Path, max_width: int = 320) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_width = max_width

    def save(self, image_bgr: np.ndarray, alert: Alert) -> Path:
        import cv2

        h, w = image_bgr.shape[:2]
        if w > self._max_width:
            image_bgr = cv2.resize(image_bgr, (self._max_width, int(h * self._max_width / w)))
        path = self._dir / f"{alert.alert_id}_{alert.alert_type.value}.jpg"
        cv2.imwrite(str(path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return path
