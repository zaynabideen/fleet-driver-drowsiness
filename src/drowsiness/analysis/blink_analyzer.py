"""Eye-closure episodes, blink metrics and PERCLOS.

A *closure episode* runs from the first CLOSED frame to the first OPEN frame.
Durations are measured with timestamps, never frame counts, so a drop from 30
to 12 FPS does not change what "1 second of closure" means.

Episode classes (configurable):
    duration <  min_blink_s                 discarded (single-frame tracker noise)
    duration <= normal_blink_max_s          normal blink
    slow_blink_s <= duration < long_closure_s   slow blink
    duration >= long_closure_s               prolonged closure

If the eyes become UNOBSERVABLE mid-closure, the episode survives a short gap
(unobserved_gap_tolerance_s). After that it is closed as *interrupted*, with
its duration as a lower bound (time until the last frame actually seen
closed). We don't pretend to know what happened while we couldn't see.

PERCLOS here is EAR-based: the fraction of *observed* time the eye was
classified CLOSED in the window. Unobservable time is excluded from both
numerator and denominator, and PERCLOS is withheld (None) when the eyes were
observable for too little of the window.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from drowsiness.config.settings import BlinkSettings, EyeSettings
from drowsiness.schemas.states import EyeState

MAX_SAMPLE_DT_S = 0.2  # a gap longer than this between frames is not credited as observed time


@dataclass(frozen=True, slots=True)
class ClosureEpisode:
    start_s: float
    end_s: float
    interrupted: bool

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class BlinkMetrics:
    current_closure_s: float
    perclos: float | None
    perclos_coverage: float
    blink_rate_per_min: float | None
    slow_blinks: int
    long_closures: int
    mean_blink_duration_s: float | None


class BlinkAnalyzer:
    def __init__(self, eyes: EyeSettings, blink: BlinkSettings) -> None:
        self._e = eyes
        self._b = blink
        self._episode_start: float | None = None
        self._last_closed_s: float | None = None
        self._episodes: deque[ClosureEpisode] = deque()
        self._samples: deque[tuple[float, float, bool]] = deque()  # (t, dt, closed) for observed frames
        self._prev_t: float | None = None
        self._first_t: float | None = None

    @property
    def episodes(self) -> tuple[ClosureEpisode, ...]:
        return tuple(self._episodes)

    def update(self, t: float, state: EyeState, suppress: bool = False) -> BlinkMetrics:
        """``suppress``: eyes closed as part of a yawn - not counted as a closure, nor in PERCLOS."""
        if self._first_t is None:
            self._first_t = t
        dt = 0.0 if self._prev_t is None else min(max(t - self._prev_t, 0.0), MAX_SAMPLE_DT_S)
        self._prev_t = t

        if suppress:
            self._episode_start = None
            self._last_closed_s = None
        elif state == EyeState.CLOSED:
            if self._episode_start is None:
                self._episode_start = t
            self._last_closed_s = t
            self._samples.append((t, dt, True))
        elif state == EyeState.OPEN:
            if self._episode_start is not None:
                self._end_episode(t, interrupted=False)
            self._samples.append((t, dt, False))
        else:  # UNOBSERVABLE
            if self._episode_start is not None and self._last_closed_s is not None:
                if t - self._last_closed_s > self._e.unobserved_gap_tolerance_s:
                    self._end_episode(self._last_closed_s, interrupted=True)

        self._prune(t)
        return self._metrics(t)

    def _end_episode(self, end_s: float, interrupted: bool) -> None:
        assert self._episode_start is not None
        if end_s - self._episode_start >= self._e.min_blink_s:  # shorter = single-frame tracker noise
            self._episodes.append(ClosureEpisode(self._episode_start, end_s, interrupted))
        self._episode_start = None
        self._last_closed_s = None

    def _prune(self, t: float) -> None:
        horizon = t - self._b.window_s
        while self._episodes and self._episodes[0].end_s < horizon:
            self._episodes.popleft()
        while self._samples and self._samples[0][0] < horizon:
            self._samples.popleft()

    def current_closure_s(self, t: float) -> float:
        return 0.0 if self._episode_start is None else t - self._episode_start

    def _metrics(self, t: float) -> BlinkMetrics:
        observed = sum(dt for _, dt, _ in self._samples)
        closed = sum(dt for _, dt, c in self._samples if c)
        assert self._first_t is not None
        span = min(self._b.window_s, t - self._first_t)
        coverage = observed / span if span > 0 else 0.0
        # PERCLOS needs most of a full window: a few seconds of data is not a percentage of anything.
        full_enough = span >= self._b.window_s * self._b.perclos_min_coverage
        perclos = closed / observed if (full_enough and coverage >= self._b.perclos_min_coverage and observed > 0) else None

        durations = [e.duration_s for e in self._episodes]
        slow = sum(1 for d in durations if self._e.slow_blink_s <= d < self._e.long_closure_s)
        long_ = sum(1 for d in durations if d >= self._e.long_closure_s)
        rate = None
        if observed >= 15.0:  # need a meaningful observation span for a per-minute rate
            rate = len(durations) / (observed / 60.0)
        mean_dur = sum(durations) / len(durations) if durations else None
        return BlinkMetrics(
            current_closure_s=self.current_closure_s(t),
            perclos=perclos,
            perclos_coverage=min(1.0, coverage),
            blink_rate_per_min=rate,
            slow_blinks=slow,
            long_closures=long_,
            mean_blink_duration_s=mean_dur,
        )
