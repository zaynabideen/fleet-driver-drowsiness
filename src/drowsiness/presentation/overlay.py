"""On-screen driver-monitoring panel (OpenCV).

Presentation only: it reads a MonitorOutput and draws it. It never makes a
decision. The video is left unobstructed; all information goes in a side
panel, and a single restrained banner appears only for elevated or
unobservable states.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from drowsiness.features.landmarks import (
    LEFT_EYE_CONTOUR,
    MOUTH_CORNERS,
    MOUTH_VERTICAL_PAIRS,
    RIGHT_EYE_CONTOUR,
)
from drowsiness.pipeline.driver_monitor import MonitorOutput
from drowsiness.schemas.events import Alert
from drowsiness.schemas.states import DriverState, EyeState, ObservationStatus

PANEL_W = 360
VIDEO_H = 480
FONT = cv2.FONT_HERSHEY_SIMPLEX
BOLD = cv2.FONT_HERSHEY_DUPLEX

# BGR. Muted palette; only the state colour carries meaning, and the label always says it in words.
BG = (34, 30, 28)
RULE = (70, 64, 60)
TEXT = (232, 228, 224)
MUTED = (160, 152, 146)
STATE_COLOUR = {
    DriverState.ALERT: (110, 180, 90),
    DriverState.MONITORING: (190, 160, 110),
    DriverState.DROWSINESS_WARNING: (60, 190, 240),
    DriverState.HIGH_DROWSINESS_RISK: (40, 130, 245),
    DriverState.CRITICAL_SLEEP_RISK: (60, 60, 225),
    DriverState.UNKNOWN: (150, 150, 150),
    DriverState.CAMERA_UNAVAILABLE: (170, 130, 150),
}
BANNER_TEXT = {
    DriverState.DROWSINESS_WARNING: "DROWSINESS WARNING",
    DriverState.HIGH_DROWSINESS_RISK: "HIGH DROWSINESS RISK",
    DriverState.CRITICAL_SLEEP_RISK: "CRITICAL SLEEP RISK",
    DriverState.UNKNOWN: "DRIVER NOT VISIBLE - STATE UNKNOWN",
    DriverState.CAMERA_UNAVAILABLE: "CAMERA UNAVAILABLE",
}
RISK_WORD = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
BANNER_REASON = {
    "prolonged_eye_closure": "EYES CLOSED",
    "high_perclos": "EYES CLOSING OFTEN",
    "repeated_long_closures": "REPEATED EYE CLOSURES",
    "head_nodding": "HEAD NODDING",
    "downward_head_pose": "HEAD DOWN",
    "frequent_yawning": "YAWNING",
    "repeated_slow_blinks": "SLOW BLINKS",
}


def _text(img: np.ndarray, s: str, org: tuple[int, int], scale: float = 0.5, colour=TEXT, font=FONT,
          thick: int = 1) -> None:
    cv2.putText(img, s, org, font, scale, colour, thick, cv2.LINE_AA)


def _row(img: np.ndarray, y: int, label: str, value: str, colour=TEXT) -> int:
    _text(img, label, (18, y), 0.47, MUTED)
    _text(img, value, (150, y), 0.5, colour)
    return y + 22


def _section(img: np.ndarray, y: int, title: str) -> int:
    cv2.line(img, (18, y - 8), (PANEL_W - 18, y - 8), RULE, 1)
    _text(img, title, (18, y + 8), 0.4, MUTED)
    return y + 30


def draw_landmarks(video: np.ndarray, out: MonitorOutput) -> None:
    lm = out.landmarks
    if lm is None:
        return
    px = lm.pixels()[:, :2]
    sx, sy = video.shape[1] / lm.image_width, video.shape[0] / lm.image_height
    eye_colour = (90, 200, 255) if out.snapshot.eye_state == EyeState.CLOSED else (200, 200, 120)
    for idx in (*LEFT_EYE_CONTOUR, *RIGHT_EYE_CONTOUR):
        cv2.circle(video, (int(px[idx, 0] * sx), int(px[idx, 1] * sy)), 1, eye_colour, -1, cv2.LINE_AA)
    for idx in (*MOUTH_CORNERS, *[i for pair in MOUTH_VERTICAL_PAIRS for i in pair]):
        cv2.circle(video, (int(px[idx, 0] * sx), int(px[idx, 1] * sy)), 1, (180, 180, 180), -1, cv2.LINE_AA)


def render(
    image_bgr: np.ndarray,
    out: MonitorOutput,
    recent_alerts: Sequence[Alert] = (),
    fps: float | None = None,
    calibration_progress: float = 1.0,
    show_landmarks: bool = True,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    video = cv2.resize(image_bgr, (int(w * VIDEO_H / h), VIDEO_H))
    if show_landmarks:
        draw_landmarks(video, out)

    d, s, r = out.decision, out.snapshot, out.risk
    colour = STATE_COLOUR[d.state]

    if d.state in BANNER_TEXT:
        overlay = video.copy()
        cv2.rectangle(overlay, (0, 0), (video.shape[1], 46), colour, -1)
        cv2.addWeighted(overlay, 0.85, video, 0.15, 0, video)
        banner = BANNER_TEXT[d.state]
        if d.state.is_elevated and r.observable:
            top = r.active_evidence[0].code if r.active_evidence else ""
            reason = BANNER_REASON.get(top)
            if s.yawning_now and d.state == DriverState.DROWSINESS_WARNING:
                reason = "YAWNING"
            if reason:
                banner = f"{banner}  -  {reason}"
        _text(video, banner, (16, 31), 0.75, (255, 255, 255), BOLD, 1)

    panel = np.full((VIDEO_H, PANEL_W, 3), BG, dtype=np.uint8)
    _text(panel, "DRIVER MONITORING", (18, 30), 0.6, TEXT, BOLD)
    if fps is not None:
        _text(panel, f"{fps:4.1f} fps", (PANEL_W - 82, 30), 0.42, MUTED)

    cv2.rectangle(panel, (18, 44), (PANEL_W - 18, 84), colour, -1)
    _text(panel, d.state.value.replace("_", " "), (30, 71), 0.62, (255, 255, 255), BOLD)
    if d.recovery_progress is not None:
        _text(panel, f"recovering {d.recovery_progress:.0%}", (PANEL_W - 140, 70), 0.42, (255, 255, 255))
        cv2.rectangle(panel, (18, 81), (18 + int((PANEL_W - 36) * d.recovery_progress), 84), (255, 255, 255), -1)

    y = 110
    risk_word = RISK_WORD[r.level.value] if r.observable else "N/A"
    y = _row(panel, y, "Drowsiness risk", f"{risk_word}  ({r.risk_score:.2f})")
    bar_w = PANEL_W - 168
    cv2.rectangle(panel, (150, y - 14), (150 + bar_w, y - 8), RULE, -1)
    cv2.rectangle(panel, (150, y - 14), (150 + int(bar_w * r.risk_score), y - 8), colour, -1)
    y += 6
    y = _row(panel, y, "Confidence", f"{d.confidence:.0%}    in state {d.duration_s:.1f}s")

    y = _section(panel, y + 2, "SIGNALS")
    eye = s.eye_state.value
    if s.eye_state == EyeState.CLOSED:
        eye += f"  {s.current_closure_s:.1f}s"
    y = _row(panel, y, "Eyes", eye)
    rate = "--" if s.blink_rate_per_min is None else f"{s.blink_rate_per_min:.0f}/min"
    perclos = "--" if s.perclos is None else f"{s.perclos:.0%}"
    y = _row(panel, y, "Blink rate", f"{rate}   PERCLOS {perclos}")
    y = _row(panel, y, "Slow / long", f"{s.slow_blinks} / {s.long_closures} (60s)")
    if s.yawning_now:
        yawn = f"YES {s.current_mouth_open_s:.1f}s"
    elif s.current_mouth_open_s > 0:
        yawn = f"opening {s.current_mouth_open_s:.1f}s"
    else:
        yawn = "NO"
    y = _row(panel, y, "Yawn", f"{yawn}   ({s.yawns} in 5 min)")
    if s.rel_pitch_deg is None:
        head = "--"
    else:
        tag = "DOWN" if s.head_down_s > 0 or s.eyes_off_road_s > 0 else ("AWAY" if s.looking_away else "NORMAL")
        head = f"{tag}  p{s.rel_pitch_deg:+.0f} y{s.rel_yaw_deg:+.0f}"
    y = _row(panel, y, "Head pose", head)
    y = _row(panel, y, "Nods", f"{s.nods} (60s)")

    y = _section(panel, y + 2, "SENSOR")
    camera = {
        ObservationStatus.OBSERVED: "GOOD",
        ObservationStatus.DEGRADED: "DEGRADED",
        ObservationStatus.FACE_LOST: "NO FACE",
        ObservationStatus.CAMERA_UNAVAILABLE: "UNAVAILABLE",
    }[s.status]
    issues = ", ".join(i.replace("_", " ") for i in out.features.quality.issues[:2])
    y = _row(panel, y, "Camera", camera + (f" ({issues})" if issues and s.status != ObservationStatus.OBSERVED else ""))
    cal = "done" if calibration_progress >= 1.0 else f"{calibration_progress:.0%}"
    y = _row(panel, y, "Calibration", cal)

    y = _section(panel, y + 2, "RECENT ALERTS")
    alerts = list(recent_alerts)[-3:][::-1]
    if not alerts:
        _text(panel, "none", (18, y), 0.45, MUTED)
    for a in alerts:
        if y > VIDEO_H - 6:
            break
        _text(panel, f"{a.timestamp_s:7.1f}s  {a.alert_type.value.replace('_', ' ')}", (18, y), 0.44,
              STATE_COLOUR[a.alert_type])
        y += 19

    return np.hstack([video, panel])
