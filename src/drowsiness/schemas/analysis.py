"""Output of the temporal analysis layer: behaviour over time, not single frames.

``BehaviourSnapshot`` is the feature vector handed to the risk engine. It is
the seam where a learned model (XGBoost, LSTM, ...) can replace or sit beside
the rule-based engine without touching detection or the state machine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from drowsiness.schemas.states import EyeState, ObservationStatus


@dataclass(frozen=True, slots=True)
class BehaviourSnapshot:
    timestamp_s: float
    status: ObservationStatus
    observation_confidence: float  # 0..1, from frame quality and calibration
    calibrated: bool

    # Eyes
    eye_state: EyeState
    eye_confidence: float
    current_closure_s: float  # ongoing continuous closure, 0 if open
    perclos: float | None  # None when coverage is too low to trust
    perclos_coverage: float
    blink_rate_per_min: float | None
    slow_blinks: int
    long_closures: int
    mean_blink_duration_s: float | None

    # Mouth
    yawning_now: bool
    current_mouth_open_s: float
    yawns: int

    # Head (relative to calibrated neutral; positive pitch = chin down)
    rel_pitch_deg: float | None
    rel_yaw_deg: float | None
    roll_deg: float | None
    head_down_s: float  # sustained head-down while eyes NOT confirmed open
    eyes_off_road_s: float  # head down with eyes open: distraction context only
    nods: int
    head_dropping: bool  # recent nod and head still down
    looking_away: bool

    # Observation gaps
    face_lost_s: float
    lost_while_impaired: bool  # face vanished right after a nod / head-drop / eye closure
    camera_unavailable_s: float
    camera_issue: str | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["eye_state"] = self.eye_state.value
        return data
