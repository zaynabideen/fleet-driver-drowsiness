"""Yawn detection.

A frame shows a *wide opening* when inner-lip MAR >= yawn_mar or the jawOpen
blendshape >= yawn_blendshape. A yawn is a wide opening that persists for
at least min_yawn_s (openings separated by < merge_gap_s are merged, so a
brief lip movement mid-yawn doesn't split it).

Why duration matters: talking, laughing and singing produce frequent but
short openings; yawns are long. A single yawn is weak evidence and the risk
engine caps yawning at severity 1 on its own.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from drowsiness.config.settings import MouthSettings
from drowsiness.schemas.detection import FrameFeatures


@dataclass(frozen=True, slots=True)
class YawnMetrics:
    yawning_now: bool
    current_open_s: float
    yawns: int


class YawnAnalyzer:
    def __init__(self, settings: MouthSettings) -> None:
        self._s = settings
        self._start: float | None = None
        self._last_open: float | None = None
        self._yawn_ends: deque[float] = deque()

    def is_wide_open(self, f: FrameFeatures) -> bool | None:
        """True/False, or None when the mouth cannot be observed."""
        if not f.face_present or not f.quality.mouth_observable or f.mar is None:
            return None
        if f.mar >= self._s.yawn_mar:
            return True
        return f.jaw_open is not None and f.jaw_open >= self._s.yawn_blendshape

    def update(self, f: FrameFeatures) -> YawnMetrics:
        t = f.timestamp_s
        wide = self.is_wide_open(f)
        if wide:
            if self._start is None:
                self._start = t
            self._last_open = t
        elif self._start is not None and self._last_open is not None and t - self._last_open > self._s.merge_gap_s:
            if self._last_open - self._start >= self._s.min_yawn_s:
                self._yawn_ends.append(self._last_open)
            self._start = self._last_open = None

        while self._yawn_ends and self._yawn_ends[0] < t - self._s.window_s:
            self._yawn_ends.popleft()

        current = 0.0 if self._start is None or self._last_open is None else self._last_open - self._start
        yawning_now = current >= self._s.min_yawn_s
        return YawnMetrics(yawning_now, current, len(self._yawn_ends) + (1 if yawning_now else 0))
