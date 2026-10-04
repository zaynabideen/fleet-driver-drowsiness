"""Per-driver baseline.

EAR at rest differs a lot between people (eye shape, ethnicity, glasses,
camera angle), and absolute head angles depend on where the camera is
mounted. A fixed EAR threshold therefore either misses some drivers or
constantly flags others. The baseline learns, from the first seconds of
good-quality observation:

  * open-eye EAR   - a high percentile, so blinks don't drag it down and a
                     driver who is already tired doesn't calibrate "closed" as normal
  * neutral pitch / yaw / roll - the median pose = "looking at the road"

Until calibration completes, population defaults are used and the
analysis reports ``calibrated=False`` (which lowers confidence and keeps the
state at MONITORING instead of ALERT).
"""

from __future__ import annotations

import numpy as np

from drowsiness.config.settings import CalibrationSettings
from drowsiness.schemas.detection import FrameFeatures

_GOOD_QUALITY = 0.6


def _median_pose(poses: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Median pose as plain Python floats (NumPy scalars must not leak into logs/JSON)."""
    p, y, r = (float(v) for v in np.median(np.array(poses), axis=0))
    return p, y, r


class DriverBaseline:
    def __init__(self, settings: CalibrationSettings) -> None:
        self._s = settings
        self._ears: list[float] = []
        self._poses: list[tuple[float, float, float]] = []
        self._start_s: float | None = None
        self._last_s = 0.0
        self._calibrated = False
        self._open_ear = settings.default_open_ear
        self._neutral = (0.0, 0.0, 0.0)

    @classmethod
    def fixed(
        cls,
        settings: CalibrationSettings,
        open_ear: float,
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0,
    ) -> DriverBaseline:
        """A pre-computed baseline (e.g. stored per driver ID, or for tests)."""
        b = cls(settings)
        b._open_ear = float(np.clip(open_ear, settings.min_open_ear, settings.max_open_ear))
        b._neutral = (pitch, yaw, roll)
        b._calibrated = True
        return b

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def open_ear(self) -> float:
        return self._open_ear

    @property
    def neutral_pose(self) -> tuple[float, float, float]:
        return self._neutral

    @property
    def progress(self) -> float:
        if self._calibrated:
            return 1.0
        if self._start_s is None:
            return 0.0
        by_samples = len(self._ears) / self._s.min_samples
        by_time = (self._last_s - self._start_s) / self._s.duration_s
        return min(0.99, by_samples, by_time)

    def update(self, f: FrameFeatures) -> None:
        if self._calibrated or not f.face_present or f.quality.score < _GOOD_QUALITY:
            return
        if f.pitch_deg is not None and f.yaw_deg is not None and f.roll_deg is not None:
            self._poses.append((f.pitch_deg, f.yaw_deg, f.roll_deg))
            # Provisional neutral pose so relative angles are sensible during calibration.
            self._neutral = _median_pose(self._poses)
        if not f.quality.eyes_observable or f.ear is None:
            return
        if self._start_s is None:
            self._start_s = f.timestamp_s
        self._ears.append(f.ear)
        self._last_s = f.timestamp_s
        elapsed = f.timestamp_s - self._start_s
        if elapsed >= self._s.duration_s and len(self._ears) >= self._s.min_samples:
            self._finalise()

    def _finalise(self) -> None:
        ear = float(np.percentile(self._ears, self._s.ear_percentile))
        self._open_ear = float(np.clip(ear, self._s.min_open_ear, self._s.max_open_ear))
        if self._poses:
            self._neutral = _median_pose(self._poses)
        self._calibrated = True
        self._ears.clear()
        self._poses.clear()
