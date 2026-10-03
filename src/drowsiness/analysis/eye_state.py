"""Per-frame eye state: OPEN / CLOSED / UNOBSERVABLE.

Rules, in order:
 1. No face, poor quality, sunglasses/occlusion  -> UNOBSERVABLE (never "open")
 2. Head yaw/pitch outside the range where EAR is geometrically valid -> UNOBSERVABLE
 3. EAR vs the driver's baseline with two thresholds (hysteresis):
        OPEN -> CLOSED when EAR < baseline * closed_ratio
        CLOSED -> OPEN when EAR > baseline * reopen_ratio
 4. Confidence = frame quality x agreement with the eyeBlink blendshape
    (an independent estimate from the same model). Agreement keeps
    confidence; disagreement lowers it, but EAR still decides the state
    because it is the interpretable, documented measure.
"""

from __future__ import annotations

from drowsiness.analysis.calibration import DriverBaseline
from drowsiness.config.settings import EyeSettings, QualitySettings
from drowsiness.schemas.detection import EyeObservation, FrameFeatures
from drowsiness.schemas.states import EyeState

AGREE_FACTOR = 1.0
DISAGREE_FACTOR = 0.6
NO_BLENDSHAPE_FACTOR = 0.85
UNCALIBRATED_FACTOR = 0.8


class EyeStateClassifier:
    def __init__(self, eyes: EyeSettings, quality: QualitySettings) -> None:
        self._e = eyes
        self._q = quality
        self._last = EyeState.OPEN

    def classify(self, f: FrameFeatures, baseline: DriverBaseline) -> EyeObservation:
        unobservable = self._unobservable_reason(f, baseline)
        if unobservable is not None:
            return EyeObservation(EyeState.UNOBSERVABLE, 0.0, f.ear, None, unobservable)

        ear = f.ear
        assert ear is not None
        close_thr = baseline.open_ear * self._e.closed_ratio
        open_thr = baseline.open_ear * self._e.reopen_ratio
        if self._last == EyeState.CLOSED:
            closed = ear < open_thr
            threshold = open_thr
        else:
            closed = ear < close_thr
            threshold = close_thr
        state = EyeState.CLOSED if closed else EyeState.OPEN
        self._last = state

        conf = f.quality.score
        if f.blink_score is None:
            conf *= NO_BLENDSHAPE_FACTOR
            reason = "EAR only"
        elif (f.blink_score >= self._e.blendshape_closed) == closed:
            conf *= AGREE_FACTOR
            reason = "EAR and blink blendshape agree"
        else:
            conf *= DISAGREE_FACTOR
            reason = "EAR and blink blendshape disagree"
        if not baseline.calibrated:
            conf *= UNCALIBRATED_FACTOR
            reason += "; uncalibrated baseline"
        return EyeObservation(state, round(conf, 4), ear, threshold, reason)

    def _unobservable_reason(self, f: FrameFeatures, baseline: DriverBaseline) -> str | None:
        if not f.camera_ok:
            return "camera unavailable"
        if not f.face_present:
            return "no face"
        if not f.quality.eyes_observable:
            return "eyes not measurable: " + ", ".join(f.quality.issues)
        if f.ear is None:
            return "no eye landmarks"
        n_pitch, n_yaw, _ = baseline.neutral_pose
        if f.yaw_deg is not None and abs(f.yaw_deg - n_yaw) > self._q.max_yaw_for_eyes_deg:
            return "head turned beyond EAR validity"
        if f.pitch_deg is not None and abs(f.pitch_deg - n_pitch) > self._q.max_pitch_for_eyes_deg:
            return "head pitched beyond EAR validity"
        return None
