"""Guided demo: on-screen prompts that walk the person in front of the camera through
each behaviour, so a complete demo video can be recorded in one take.

Presentation only. The prompts never influence detection or decisions; the system
reacts to what the camera actually sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DemoStep:
    prompt: str
    seconds: float
    expect: str  # what viewers should see on the panel


STEPS: tuple[DemoStep, ...] = (
    DemoStep("Blink normally", 5, "Stays ALERT"),
    DemoStep("Look down at your desk - eyes OPEN", 4, "A glance is not drowsiness"),
    DemoStep("Sit up, look at the screen", 3, ""),
    DemoStep("Yawn once, slowly", 7, "Yawn detected - still not critical"),
    DemoStep("Close your eyes for 2 seconds", 5, "Prolonged closure -> WARNING"),
    DemoStep("Open your eyes, sit normally", 4, ""),
    DemoStep("Close your eyes AND drop your head", 6, "Closure + head drop -> CRITICAL"),
    DemoStep("Sit up, eyes open", 4, ""),
    DemoStep("Cover the camera with your hand", 5, "No image -> CAMERA UNAVAILABLE, never 'safe'"),
    DemoStep("Uncover the camera", 3, ""),
)
CALIBRATION_PROMPT = "Calibrating: look at the screen normally"
MAX_CALIBRATION_S = 45.0  # don't wait forever (e.g. sunglasses); continue the script anyway
DONE_HOLD_S = 2.5


class GuidedDemo:
    """Tracks which prompt to show, from a monotonic clock."""

    def __init__(self, steps: tuple[DemoStep, ...] = STEPS) -> None:
        self._steps = steps
        self._start: float | None = None
        self._script_start: float | None = None

    def update(self, now_s: float, calibrated: bool) -> tuple[str, str, float | None, bool]:
        """Return (prompt, expectation, seconds_left_in_step, finished)."""
        if self._start is None:
            self._start = now_s
        if self._script_start is None:
            if calibrated or now_s - self._start >= MAX_CALIBRATION_S:
                self._script_start = now_s
            else:
                return CALIBRATION_PROMPT, "Learning your eyes and head position", None, False
        t = now_s - self._script_start
        for step in self._steps:
            if t < step.seconds:
                return step.prompt, step.expect, step.seconds - t, False
            t -= step.seconds
        return "Demo complete", "", None, t >= DONE_HOLD_S


def draw_prompt(canvas: np.ndarray, video_width: int, prompt: str, expect: str,
                seconds_left: float | None, step_label: str = "") -> None:
    """Caption bar along the bottom of the video area (not over the side panel)."""
    h = canvas.shape[0]
    bar_h = 64
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (video_width - 1, h - 1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
    title = prompt if seconds_left is None else f"{prompt}  ({seconds_left:0.0f}s)"
    cv2.putText(canvas, title, (16, h - bar_h + 26), cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    sub = expect if not step_label else f"{step_label}   {expect}"
    if sub:
        cv2.putText(canvas, sub, (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 200, 210), 1, cv2.LINE_AA)
