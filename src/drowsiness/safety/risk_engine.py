"""Multi-signal, explainable risk engine.

Step 1 - indicators. Each behavioural measure becomes an indicator with a
severity 0-3 from configurable thresholds (settings.risk):

    category  indicator                  source
    eyes      ongoing eye closure (s)    BlinkAnalyzer
    eyes      PERCLOS                    BlinkAnalyzer (only when coverage is sufficient)
    eyes      prolonged closures (count) BlinkAnalyzer
    blink     slow blinks (count)        capped at 1
    blink     blink rate (/min)          capped at 1
    head      nods (count)               HeadMovementAnalyzer
    head      head down, eyes not open   HeadMovementAnalyzer
    mouth     yawns (count)              capped at 1

Step 2 - level rules (first match wins), each named in the output:

    CRITICAL  ongoing closure at severity 3 (microsleep-length)
              | PERCLOS at severity 3
              | ongoing closure >= critical_combo_closure_s AND head down/dropping
    HIGH      any eye indicator at severity >= 2
              | eyes >= 1 AND head >= 1             (independent corroboration)
              | nods at severity >= 2               (repeated involuntary head drops; also
                                                     the main signal when eyes are hidden)
              | >= min_categories_for_high categories at >= 1
    MODERATE  any indicator at severity >= 1
    LOW       otherwise

Design choices:
  * History (window counts, PERCLOS) escalates, but does not hold an alarm on once the
    driver has been *observed* alert for history_relevance_s; a new sign re-activates it.
  * Weak, ambiguous signals (yawns, blink rate, slow blinks) are capped so on
    their own they can never exceed MODERATE.
  * HIGH needs either strong eye evidence or two independent categories.
  * risk_score lies in the band of the level (LOW 0-.25, MODERATE .25-.5,
    HIGH .5-.75, CRITICAL .75-1), positioned by how far the strongest
    indicator is past its threshold. Score and level never contradict.
  * confidence is separate from risk: it is how trustworthy the observations
    are (frame quality, blendshape agreement, calibration) times how many
    independent categories corroborate an elevated level.

``RiskModel`` is the seam for a learned model: anything that maps a
BehaviourSnapshot to a RiskAssessment can replace or sit beside this class.
"""

from __future__ import annotations

from typing import Protocol

from drowsiness.config.settings import RiskSettings
from drowsiness.schemas.analysis import BehaviourSnapshot
from drowsiness.schemas.events import Evidence, RiskAssessment
from drowsiness.schemas.states import ObservationStatus, RiskLevel, SignalCategory

BAND_WIDTH = 0.25
# Window-based evidence: describes the recent past, not the present.
HISTORY_CODES = frozenset({
    "high_perclos", "repeated_long_closures", "repeated_slow_blinks",
    "excessive_blinking", "head_nodding", "frequent_yawning",
})
DISABLED = 1e8


class RiskModel(Protocol):
    def assess(self, snapshot: BehaviourSnapshot) -> RiskAssessment: ...


def severity(value: float, thresholds: tuple[float, float, float]) -> int:
    return sum(1 for thr in thresholds if thr < DISABLED and value >= thr)


def graded(value: float, thresholds: tuple[float, float, float]) -> float:
    """Severity plus fractional progress towards the next threshold.

    Ranges 0..4: beyond the third threshold, progress continues at the same
    step size so the score keeps rising inside the CRITICAL band.
    """
    sev = severity(value, thresholds)
    if sev >= 3:
        step = thresholds[2] - thresholds[1]
        return 3.0 + (min(0.999, (value - thresholds[2]) / step) if step > 0 else 0.0)
    nxt = thresholds[sev]
    if nxt >= DISABLED:
        return float(sev)
    prev = thresholds[sev - 1] if sev > 0 else 0.0
    span = nxt - prev
    return sev + (max(0.0, value - prev) / span if span > 0 else 0.0)


class RuleBasedRiskEngine:
    def __init__(self, settings: RiskSettings) -> None:
        self._r = settings

    # ------------------------------------------------------------------ indicators
    def indicators(self, s: BehaviourSnapshot) -> list[tuple[Evidence, float]]:
        r = self._r
        out: list[tuple[Evidence, float]] = []

        def add(code: str, cat: SignalCategory, value: float, thr: tuple[float, float, float], desc: str) -> None:
            out.append((Evidence(code, cat, severity(value, thr), value, desc), graded(value, thr)))

        eyes_measured = s.status == ObservationStatus.OBSERVED or s.current_closure_s > 0
        if eyes_measured:
            add("prolonged_eye_closure", SignalCategory.EYES, s.current_closure_s, r.ongoing_closure_s,
                f"eye closure {s.current_closure_s:.1f}s (ongoing)")
        if s.perclos is not None:
            add("high_perclos", SignalCategory.EYES, s.perclos, r.perclos,
                f"PERCLOS {s.perclos:.2f} (eye coverage {s.perclos_coverage:.0%})")
        add("repeated_long_closures", SignalCategory.EYES, s.long_closures, r.long_closures,
            f"{s.long_closures} prolonged eye closures in window")
        add("repeated_slow_blinks", SignalCategory.BLINK, s.slow_blinks, r.slow_blinks,
            f"{s.slow_blinks} slow blinks in window")
        if s.blink_rate_per_min is not None:
            add("excessive_blinking", SignalCategory.BLINK, s.blink_rate_per_min, r.blink_rate,
                f"blink rate {s.blink_rate_per_min:.0f}/min")
        add("head_nodding", SignalCategory.HEAD, s.nods, r.nods, f"{s.nods} head nods in window")
        pitch = f", pitch {s.rel_pitch_deg:+.0f}°" if s.rel_pitch_deg is not None else ""
        add("downward_head_pose", SignalCategory.HEAD, s.head_down_s, r.head_down_s,
            f"head down {s.head_down_s:.1f}s with eyes not open{pitch}")
        add("frequent_yawning", SignalCategory.MOUTH, s.yawns, r.yawns, f"{s.yawns} yawns in window")
        return out

    # ------------------------------------------------------------------ decision
    def assess(self, s: BehaviourSnapshot) -> RiskAssessment:
        if s.status in (ObservationStatus.CAMERA_UNAVAILABLE, ObservationStatus.FACE_LOST):
            return self._unobservable(s)

        scored = self._retire_stale_history(self.indicators(s), s)
        evidence = tuple(e for e, _ in scored)
        max_sev = {cat: 0 for cat in SignalCategory}
        for e in evidence:
            max_sev[e.category] = max(max_sev[e.category], e.severity)
        active_categories = sum(1 for v in max_sev.values() if v >= 1)
        by_code = {e.code: e for e in evidence}
        closure = by_code.get("prolonged_eye_closure")
        perclos = by_code.get("high_perclos")
        head_down = s.head_dropping or s.head_down_s > 0

        level, rule = RiskLevel.LOW, "no_indicator_above_threshold"
        if closure and closure.severity >= 3:
            level, rule = RiskLevel.CRITICAL, "microsleep_length_eye_closure"
        elif perclos and perclos.severity >= 3:
            level, rule = RiskLevel.CRITICAL, "extreme_perclos"
        elif s.current_closure_s >= self._r.critical_combo_closure_s and head_down:
            level, rule = RiskLevel.CRITICAL, "eye_closure_with_head_drop"
        elif max_sev[SignalCategory.EYES] >= 2:
            level, rule = RiskLevel.HIGH, "strong_eye_evidence"
        elif max_sev[SignalCategory.EYES] >= 1 and max_sev[SignalCategory.HEAD] >= 1:
            level, rule = RiskLevel.HIGH, "eyes_corroborated_by_head"
        elif by_code["head_nodding"].severity >= 2:
            level, rule = RiskLevel.HIGH, "repeated_nodding"
        elif active_categories >= self._r.min_categories_for_high:
            level, rule = RiskLevel.HIGH, "multiple_independent_categories"
        elif active_categories >= 1:
            level, rule = RiskLevel.MODERATE, "single_category_indicator"

        top = max((g for _, g in scored), default=0.0)
        within = min(0.999, max(0.0, top - level.value)) if level != RiskLevel.LOW else min(0.999, top)
        score = round(level.value * BAND_WIDTH + within * BAND_WIDTH, 4)

        confidence = s.observation_confidence
        if level >= RiskLevel.MODERATE:
            confidence *= 0.7 + 0.1 * min(3, active_categories)
        return RiskAssessment(
            timestamp_s=s.timestamp_s,
            level=level,
            risk_score=score,
            confidence=round(min(1.0, confidence), 4),
            evidence=tuple(sorted(evidence, key=lambda e: -e.severity)),
            rule=rule,
            observable=True,
        )

    def _retire_stale_history(self, scored: list[tuple[Evidence, float]],
                              s: BehaviourSnapshot) -> list[tuple[Evidence, float]]:
        """Once the driver has been observed alert for history_relevance_s, history evidence is
        still reported but no longer sets the level. A new impairment sign makes it count again."""
        if s.observed_alert_s < self._r.history_relevance_s:
            return scored
        out = []
        for e, g in scored:
            if e.code in HISTORY_CODES and e.severity > 0:
                e = Evidence(e.code, e.category, 0, e.value,
                             f"{e.description} (inactive: driver alert {s.observed_alert_s:.0f}s)")
                g = 0.0
            out.append((e, g))
        return out

    def _unobservable(self, s: BehaviourSnapshot) -> RiskAssessment:
        if s.status == ObservationStatus.CAMERA_UNAVAILABLE:
            ev = Evidence("camera_unavailable", SignalCategory.EYES, 0, s.camera_unavailable_s,
                          f"camera unavailable {s.camera_unavailable_s:.1f}s ({s.camera_issue or 'no frames'})")
        else:
            ev = Evidence("face_not_visible", SignalCategory.EYES, 0, s.face_lost_s,
                          f"face not visible {s.face_lost_s:.1f}s"
                          + (" right after impairment signs" if s.lost_while_impaired else ""))
        return RiskAssessment(s.timestamp_s, RiskLevel.LOW, 0.0, 0.0, (ev,), "not_observable", observable=False)
