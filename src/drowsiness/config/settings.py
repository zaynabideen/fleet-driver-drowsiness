"""Central configuration.

Every threshold the system uses lives here, with the reasoning for its default.
Nothing in the detection, analysis or safety layers should contain an inline
magic number that changes behaviour.

Defaults are *starting points* taken from the drowsiness literature and from
common practice with landmark-based driver monitoring. They are not tuned
values. Tune them on a development split and confirm them on a separate
validation split (see docs/EVALUATION.md); never tune on the test split.

Override any value with a YAML file, e.g. ``configs/default.yaml``:

    eyes:
      closed_ratio: 0.6
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CameraSettings(_Section):
    timeout_s: float = Field(
        2.0,
        gt=0,
        description="No usable frame for this long -> CAMERA_UNAVAILABLE. "
        "Long enough to ride out a dropped frame or a USB hiccup.",
    )
    dark_frame_mean: float = Field(
        12.0,
        ge=0,
        le=255,
        description="Whole-frame mean grey level below which the lens is treated as covered/black.",
    )
    flat_frame_std: float = Field(
        4.0,
        ge=0,
        description="Whole-frame grey std below which the image carries no information "
        "(lens covered, uniform glare).",
    )
    frozen_frame_count: int = Field(
        45,
        ge=2,
        description="Identical consecutive frames before the feed is considered frozen "
        "(~1.5 s at 30 FPS).",
    )


class QualitySettings(_Section):
    min_face_width_ratio: float = Field(
        0.06,
        gt=0,
        description="Inter-ocular distance / image width. Below this the face is too small "
        "for reliable eye geometry.",
    )
    min_face_brightness: float = Field(35.0, description="Mean grey level of the face region; darker is unreliable.")
    max_face_brightness: float = Field(235.0, description="Mean grey level of the face region; brighter is washed out.")
    min_sharpness: float = Field(
        8.0,
        ge=0,
        description="Variance of the Laplacian over the face region. Below this the face is motion-blurred.",
    )
    max_yaw_for_eyes_deg: float = Field(
        35.0,
        description="EAR is a 2D ratio; beyond this head yaw the eye contour is foreshortened "
        "and EAR no longer measures openness.",
    )
    max_pitch_for_eyes_deg: float = Field(
        35.0,
        description="Beyond this pitch (relative to calibrated neutral) the eyelids are viewed too "
        "obliquely for EAR to be meaningful.",
    )
    eye_region_dark_ratio: float = Field(
        0.45,
        description="Eye-region brightness / face brightness below this suggests sunglasses or "
        "occlusion; eyes are then reported UNOBSERVABLE instead of CLOSED.",
    )
    max_eye_asymmetry: float = Field(
        0.12,
        description="|EAR_left - EAR_right| above this (with a frontal head) suggests one eye is "
        "occluded or mis-tracked; confidence is reduced.",
    )


class CalibrationSettings(_Section):
    duration_s: float = Field(
        20.0,
        gt=0,
        description="Seconds of good-quality frames used to learn this driver's open-eye EAR and neutral head pose.",
    )
    min_samples: int = Field(150, ge=10, description="Minimum good frames before a baseline is accepted.")
    ear_percentile: float = Field(
        85.0,
        description="Open-eye baseline = this percentile of EAR during calibration. A high percentile "
        "ignores blinks and is robust if the driver is already slightly tired.",
    )
    default_open_ear: float = Field(0.28, description="Population-typical open-eye EAR used before calibration completes.")
    min_open_ear: float = Field(0.18, description="Lower clamp on the learned baseline (guards against calibrating on a tired driver).")
    max_open_ear: float = Field(0.42, description="Upper clamp on the learned baseline.")


class EyeSettings(_Section):
    closed_ratio: float = Field(
        0.65,
        gt=0,
        lt=1,
        description="Eye CLOSED when EAR < baseline * closed_ratio. With a typical 0.30 baseline "
        "this is ~0.20, the commonly used absolute EAR threshold.",
    )
    reopen_ratio: float = Field(
        0.75,
        gt=0,
        lt=1,
        description="Eye returns to OPEN only when EAR > baseline * reopen_ratio. The gap to "
        "closed_ratio is per-frame hysteresis that stops chatter around the threshold.",
    )
    blendshape_closed: float = Field(
        0.55,
        description="MediaPipe eyeBlink blendshape score treated as 'closed'. Used as an independent "
        "second opinion: agreement raises confidence, disagreement lowers it.",
    )
    min_blink_s: float = Field(
        0.05,
        ge=0,
        description="Closures shorter than this are tracker noise, not blinks (real blinks last >= ~75 ms) "
        "and are discarded.",
    )
    normal_blink_max_s: float = Field(
        0.4,
        description="Typical spontaneous blinks last ~0.1-0.4 s. Closures up to this length are normal blinks.",
    )
    slow_blink_s: float = Field(0.5, description="Closures at least this long count as slow blinks (an early drowsiness sign).")
    long_closure_s: float = Field(1.0, description="Closures at least this long count as prolonged eye closures.")
    unobserved_gap_tolerance_s: float = Field(
        0.3,
        description="A closure survives an unobservable gap up to this long; a longer gap ends the "
        "episode as 'interrupted' rather than guessing it continued.",
    )


class BlinkSettings(_Section):
    window_s: float = Field(60.0, gt=0, description="Rolling window for blink rate, slow-blink counts and PERCLOS.")
    perclos_min_coverage: float = Field(
        0.6,
        description="PERCLOS is only trusted when eyes were observable for at least this fraction of the window.",
    )
    excessive_rate_per_min: float = Field(
        30.0,
        description="Resting blink rate is roughly 10-20/min; sustained rates above this are flagged.",
    )


class MouthSettings(_Section):
    yawn_mar: float = Field(
        0.5,
        description="Inner-lip MAR above this = wide mouth opening. Speech rarely exceeds ~0.4.",
    )
    yawn_blendshape: float = Field(0.6, description="jawOpen blendshape score that independently indicates a wide opening.")
    min_yawn_s: float = Field(
        2.0,
        description="A wide opening must last this long to be a yawn. Talking and laughing produce "
        "short openings; yawns typically last several seconds.",
    )
    merge_gap_s: float = Field(0.4, description="Openings separated by less than this are merged into one episode.")
    window_s: float = Field(300.0, gt=0, description="Rolling window for yawn frequency (5 minutes).")


class HeadSettings(_Section):
    down_pitch_deg: float = Field(
        20.0,
        description="Pitch (relative to calibrated neutral, positive = chin down) beyond which the head is 'down'.",
    )
    down_min_s: float = Field(
        2.0,
        description="Head must stay down this long before it counts as sustained head-down.",
    )
    nod_drop_deg: float = Field(
        12.0,
        description="A nod is a pitch drop of at least this many degrees...",
    )
    nod_max_fall_s: float = Field(
        0.7,
        description="...occurring within this time. Deliberate glances at the dashboard are usually slower and shallower.",
    )
    nod_reset_deg: float = Field(
        6.0,
        description="Pitch must return within this many degrees of neutral before another nod can be counted.",
    )
    away_yaw_deg: float = Field(
        40.0,
        description="Yaw beyond this = looking away (mirror check, passenger). Reported as distraction context, "
        "never as drowsiness evidence.",
    )
    window_s: float = Field(60.0, gt=0, description="Rolling window for nod counts.")
    impaired_lookback_s: float = Field(
        3.0,
        description="If the face disappears within this many seconds of a nod/head-drop/eye closure, "
        "the loss is treated as possible slumping, not a neutral sensor gap.",
    )


class ObservationSettings(_Section):
    face_loss_grace_s: float = Field(
        1.0,
        description="Face missing for less than this: hold the last state with decaying confidence "
        "(detector flicker). Longer: UNKNOWN.",
    )
    confidence_decay_per_s: float = Field(0.5, description="Confidence lost per second while holding state through a face dropout.")


class RiskSettings(_Section):
    """Indicator thresholds. Each list gives the values for severity 1, 2, 3.

    Use a very large number to disable a severity level for an indicator.
    """

    ongoing_closure_s: tuple[float, float, float] = Field(
        (1.0, 2.0, 3.0),
        description="Current continuous eye closure. 2 s with eyes shut at 70 mph is ~60 m driven blind; "
        "3 s is treated as a probable microsleep.",
    )
    perclos: tuple[float, float, float] = Field(
        (0.15, 0.25, 0.40),
        description="Fraction of observed time eyes were closed in the window. ~0.15 is a commonly cited "
        "onset of drowsiness in the PERCLOS literature.",
    )
    long_closures: tuple[float, float, float] = Field(
        (2, 3, 1e9), description="Number of closures >= long_closure_s in the blink window."
    )
    slow_blinks: tuple[float, float, float] = Field(
        (4, 1e9, 1e9), description="Number of slow blinks in the blink window (capped at severity 1)."
    )
    nods: tuple[float, float, float] = Field(
        (2, 3, 1e9),
        description="Nods in the head window. One nod alone is ignored: a single quick glance down looks the same.",
    )
    head_down_s: tuple[float, float, float] = Field(
        (2.0, 4.0, 1e9),
        description="Sustained head-down time *while eyes are not confirmed open*. Head down with eyes open "
        "is a dashboard glance (distraction), not drowsiness.",
    )
    yawns: tuple[float, float, float] = Field(
        (2, 1e9, 1e9), description="Yawns in the yawn window. Yawning alone never exceeds severity 1."
    )
    blink_rate: tuple[float, float, float] = Field(
        (30.0, 1e9, 1e9), description="Blinks per minute. Excess blinking alone never exceeds severity 1."
    )
    critical_combo_closure_s: float = Field(
        2.0,
        description="Eye closure of at least this long *combined with* the head down/dropping is CRITICAL.",
    )
    min_categories_for_high: int = Field(
        3,
        description="Three independent signal categories (eyes, head, mouth/blink) all at severity >= 1 "
        "also make HIGH risk.",
    )


class StateMachineSettings(_Section):
    escalate_confirm_s: dict[str, float] = Field(
        default_factory=lambda: {
            "DROWSINESS_WARNING": 0.5,
            "HIGH_DROWSINESS_RISK": 0.3,
            "CRITICAL_SLEEP_RISK": 0.0,
        },
        description="Target level must persist this long before escalating. Critical is immediate because "
        "its rule already contains a multi-second duration.",
    )
    deescalate_hold_s: dict[str, float] = Field(
        default_factory=lambda: {
            "CRITICAL_SLEEP_RISK": 5.0,
            "HIGH_DROWSINESS_RISK": 8.0,
            "DROWSINESS_WARNING": 10.0,
        },
        description="Lower risk must persist this long before stepping down ONE level from the given state. "
        "Fast to escalate, slow to recover.",
    )
    recovery_confirm_s: float = Field(
        2.0,
        description="After UNKNOWN / CAMERA_UNAVAILABLE / MONITORING, low risk must be observed this long "
        "before reporting ALERT.",
    )


class AlertSettings(_Section):
    cooldown_s: dict[str, float] = Field(
        default_factory=lambda: {
            "DROWSINESS_WARNING": 30.0,
            "HIGH_DROWSINESS_RISK": 15.0,
            "CRITICAL_SLEEP_RISK": 5.0,
            "UNKNOWN": 60.0,
            "CAMERA_UNAVAILABLE": 60.0,
        },
        description="Minimum seconds between repeat alerts of the same type. An escalation to a more "
        "severe type always fires immediately.",
    )


class PrivacySettings(_Section):
    save_snapshots: bool = Field(False, description="Save a downscaled JPEG when a HIGH/CRITICAL alert fires. Off by default.")
    snapshot_max_width: int = Field(320, description="Snapshots are downscaled to at most this width.")
    record_video: bool = Field(False, description="Record raw video. Off by default; the system never needs it.")
    record_features: bool = Field(
        False,
        description="Record the per-frame numeric feature stream (no pixels) for replay and evaluation.",
    )


class LoggingSettings(_Section):
    directory: str = Field("logs", description="Where event and decision logs are written.")
    console_level: str = Field("INFO")


class Settings(_Section):
    camera: CameraSettings = CameraSettings()
    quality: QualitySettings = QualitySettings()
    calibration: CalibrationSettings = CalibrationSettings()
    eyes: EyeSettings = EyeSettings()
    blink: BlinkSettings = BlinkSettings()
    mouth: MouthSettings = MouthSettings()
    head: HeadSettings = HeadSettings()
    observation: ObservationSettings = ObservationSettings()
    risk: RiskSettings = RiskSettings()
    state_machine: StateMachineSettings = StateMachineSettings()
    alerts: AlertSettings = AlertSettings()
    privacy: PrivacySettings = PrivacySettings()
    logging: LoggingSettings = LoggingSettings()

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.eyes.reopen_ratio <= self.eyes.closed_ratio:
            raise ValueError("eyes.reopen_ratio must be greater than eyes.closed_ratio (hysteresis)")
        if not self.eyes.normal_blink_max_s < self.eyes.slow_blink_s <= self.eyes.long_closure_s:
            raise ValueError("expected normal_blink_max_s < slow_blink_s <= long_closure_s")
        for name, levels in self.risk.model_dump().items():
            if isinstance(levels, (list, tuple)) and list(levels) != sorted(levels):
                raise ValueError(f"risk.{name} severities must be non-decreasing")
        return self


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(path: str | Path | None = None, **overrides: Any) -> Settings:
    """Load defaults, then a YAML file (if given), then keyword overrides."""
    data: dict[str, Any] = Settings().model_dump()
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            data = _deep_merge(data, yaml.safe_load(fh) or {})
    if overrides:
        data = _deep_merge(data, overrides)
    return Settings.model_validate(data)
