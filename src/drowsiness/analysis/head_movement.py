"""Head movement: sustained head-down, nods, looking away.

All angles are relative to the driver's calibrated neutral; positive pitch = chin down.

Nod: pitch rises by >= nod_drop_deg within nod_max_fall_s (a fast, involuntary
drop). Another nod is only counted after the head returns within nod_reset_deg
of neutral, so one long drop is one nod, not thirty.

Head-down is split by what the eyes are doing:
  * eyes NOT confirmed open (closed / unobservable)  -> head_down_s   (drowsiness evidence)
  * eyes OPEN                                       -> eyes_off_road_s (dashboard/phone glance:
                                                       distraction context, NOT drowsiness)
This is the main defence against "every look at the dashboard is drowsiness".
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from drowsiness.config.settings import HeadSettings
from drowsiness.schemas.states import EyeState


@dataclass(frozen=True, slots=True)
class HeadMetrics:
    head_down_s: float
    eyes_off_road_s: float
    nods: int
    head_dropping: bool
    looking_away: bool
    last_nod_s: float | None


class HeadMovementAnalyzer:
    def __init__(self, settings: HeadSettings) -> None:
        self._s = settings
        self._history: deque[tuple[float, float]] = deque()
        self._armed = True
        self._nods: deque[float] = deque()
        self._impaired_down_since: float | None = None
        self._off_road_since: float | None = None

    def update(self, t: float, rel_pitch: float | None, rel_yaw: float | None, eyes: EyeState) -> HeadMetrics:
        while self._nods and self._nods[0] < t - self._s.window_s:
            self._nods.popleft()
        if rel_pitch is None:
            # No head observation: timers stop (we do not invent continuity).
            self._impaired_down_since = self._off_road_since = None
            self._history.clear()
            return HeadMetrics(0.0, 0.0, len(self._nods), False, False, self._last_nod())

        self._detect_nod(t, rel_pitch)

        down = rel_pitch >= self._s.down_pitch_deg
        if down and eyes != EyeState.OPEN:
            self._impaired_down_since = self._impaired_down_since if self._impaired_down_since is not None else t
            self._off_road_since = None
        elif down:
            self._off_road_since = self._off_road_since if self._off_road_since is not None else t
            self._impaired_down_since = None
        else:
            self._impaired_down_since = self._off_road_since = None

        last_nod = self._last_nod()
        dropping = down and last_nod is not None and t - last_nod <= self._s.impaired_lookback_s
        away = rel_yaw is not None and abs(rel_yaw) >= self._s.away_yaw_deg
        return HeadMetrics(
            head_down_s=0.0 if self._impaired_down_since is None else float(t - self._impaired_down_since),
            eyes_off_road_s=0.0 if self._off_road_since is None else float(t - self._off_road_since),
            nods=len(self._nods),
            head_dropping=bool(dropping),
            looking_away=bool(away),
            last_nod_s=last_nod,
        )

    def _detect_nod(self, t: float, pitch: float) -> None:
        self._history.append((t, pitch))
        while self._history and self._history[0][0] < t - self._s.nod_max_fall_s:
            self._history.popleft()
        if not self._armed and pitch <= self._s.nod_reset_deg:
            self._armed = True
        if self._armed:
            lowest = min(p for _, p in self._history)
            if pitch - lowest >= self._s.nod_drop_deg:
                self._nods.append(t)
                self._armed = False

    def _last_nod(self) -> float | None:
        return self._nods[-1] if self._nods else None
