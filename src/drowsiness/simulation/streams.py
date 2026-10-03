"""Synthetic FrameFeatures streams for scenario tests.

Each scenario is written as timed segments ("30 s alert driving with normal
blinks, then a 1.5 s eye closure, ..."). The streams go through the *real*
temporal analysis, risk engine and state machine. They verify the decision
logic behaves as designed; they are NOT evidence of real-world accuracy,
which needs labelled video (see docs/EVALUATION.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from drowsiness.schemas.detection import FrameFeatures, FrameQuality

OPEN_EAR = 0.30
CLOSED_EAR = 0.08
GOOD = FrameQuality(0.95, True, True, True, ())
SUNGLASSES = FrameQuality(0.76, False, True, True, ("eyes_occluded_or_sunglasses",))
NO_FACE = FrameQuality(0.0, False, False, False, ("no_face_detected",))
CAMERA_DOWN = FrameQuality(0.0, False, False, False, ("camera_unavailable",))


@dataclass
class StreamBuilder:
    fps: float = 30.0
    t: float = 0.0
    frames: list[FrameFeatures] = field(default_factory=list)

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    def segment(
        self,
        seconds: float,
        ear: float = OPEN_EAR,
        mar: float = 0.05,
        pitch: float = 0.0,
        yaw: float = 0.0,
        face: bool = True,
        camera_ok: bool = True,
        quality: FrameQuality = GOOD,
        blink_every_s: float | None = None,
        blink_s: float = 0.15,
        pitch_to: float | None = None,
    ) -> StreamBuilder:
        """Append ``seconds`` of frames. ``pitch_to`` ramps pitch linearly across the segment."""
        n = max(1, round(seconds * self.fps))
        for i in range(n):
            local = i * self.dt
            e = ear
            if blink_every_s and (local % blink_every_s) > blink_every_s - blink_s:
                e = CLOSED_EAR
            p = pitch if pitch_to is None else pitch + (pitch_to - pitch) * i / max(1, n - 1)
            self.frames.append(self._frame(e, mar, p, yaw, face, camera_ok, quality))
            self.t += self.dt
        return self

    def alert(self, seconds: float, blink_every_s: float = 4.0, **kw: float) -> StreamBuilder:
        """Normal driving: eyes open, a ~150 ms blink every few seconds (~15/min)."""
        return self.segment(seconds, blink_every_s=blink_every_s, **kw)  # type: ignore[arg-type]

    def eyes_closed(self, seconds: float, **kw: float) -> StreamBuilder:
        return self.segment(seconds, ear=CLOSED_EAR, **kw)  # type: ignore[arg-type]

    def no_face(self, seconds: float) -> StreamBuilder:
        return self.segment(seconds, face=False, quality=NO_FACE)

    def camera_down(self, seconds: float) -> StreamBuilder:
        return self.segment(seconds, face=False, camera_ok=False, quality=CAMERA_DOWN)

    def _frame(self, ear: float, mar: float, pitch: float, yaw: float, face: bool, camera_ok: bool,
               quality: FrameQuality) -> FrameFeatures:
        if not camera_ok:
            return FrameFeatures(self.t, False, False, quality, camera_issue="no_frame")
        if not face:
            return FrameFeatures(self.t, True, False, quality)
        blink = 0.85 if ear < 0.15 else 0.05
        return FrameFeatures(
            timestamp_s=self.t,
            camera_ok=True,
            face_present=True,
            quality=quality,
            ear_left=ear,
            ear_right=ear,
            blink_score=blink,
            mar=mar,
            jaw_open=min(1.0, mar * 1.2),
            pitch_deg=pitch,
            yaw_deg=yaw,
            roll_deg=0.0,
        )
