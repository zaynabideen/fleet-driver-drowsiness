"""Safety state machine.

Turns a stream of noisy risk assessments into a stable, explainable state.

Observation states (checked first):
  * camera failing for >= camera.timeout_s           -> CAMERA_UNAVAILABLE
  * camera failing for less                          -> hold last state, confidence decays
  * face lost for < face_loss_grace_s                -> hold last state, confidence decays
  * face/eyes lost right after impairment signs while
    an elevated state is active                      -> HOLD the elevated state (possible slump);
                                                        never fall back to a "neutral" state
  * face lost for longer                             -> UNKNOWN  (never ALERT)

Drowsiness ladder:  ALERT -> DROWSINESS_WARNING -> HIGH_DROWSINESS_RISK -> CRITICAL_SLEEP_RISK
  * Escalation: the target level must persist for escalate_confirm_s[target]
    (temporal confirmation). Levels can be skipped - a microsleep goes
    straight to CRITICAL.
  * De-escalation: lower risk must persist for deescalate_hold_s[current],
    then the state steps down ONE level and the timer restarts (hysteresis +
    cooldown). Fast to escalate, slow to recover.
  * ALERT is only reported when the driver is actually observed, calibrated,
    and low risk has been seen for recovery_confirm_s. Otherwise MONITORING.
"""

from __future__ import annotations

from drowsiness.config.settings import Settings
from drowsiness.schemas.analysis import BehaviourSnapshot
from drowsiness.schemas.events import RiskAssessment, StateDecision
from drowsiness.schemas.states import ELEVATED_LADDER, DriverState, ObservationStatus, RiskLevel


class SafetyStateMachine:
    def __init__(self, settings: Settings, initial: DriverState = DriverState.MONITORING) -> None:
        self._s = settings
        self.state = initial
        self._since = 0.0
        self._started = False
        self._pending_since: float | None = None
        self._lower_since: float | None = None
        self._low_since: float | None = None
        self._confidence = 0.0
        self._hold_from: float | None = None

    # ------------------------------------------------------------------ public
    def step(self, snap: BehaviourSnapshot, risk: RiskAssessment) -> StateDecision:
        t = snap.timestamp_s
        if not self._started:
            self._since, self._started = t, True

        if snap.status == ObservationStatus.CAMERA_UNAVAILABLE:
            if snap.camera_unavailable_s >= self._s.camera.timeout_s:
                return self._go(t, DriverState.CAMERA_UNAVAILABLE, 0.0, risk,
                                f"camera unavailable for {snap.camera_unavailable_s:.1f}s ({snap.camera_issue})")
            return self._hold(t, risk, f"camera gap {snap.camera_unavailable_s:.1f}s (< timeout)")

        if snap.status == ObservationStatus.FACE_LOST:
            if self.state.is_elevated and snap.lost_while_impaired:
                return self._hold(t, risk, "face lost right after impairment signs - possible slump, holding state",
                                  decay=False)
            if snap.face_lost_s < self._s.observation.face_loss_grace_s:
                return self._hold(t, risk, f"face not visible {snap.face_lost_s:.1f}s (< grace)")
            return self._go(t, DriverState.UNKNOWN, 0.0, risk,
                            f"face not visible for {snap.face_lost_s:.1f}s - driver cannot be assessed")

        self._hold_from = None
        return self._ladder(t, snap, risk)

    # ------------------------------------------------------------------ ladder
    def _ladder(self, t: float, snap: BehaviourSnapshot, risk: RiskAssessment) -> StateDecision:
        sm = self._s.state_machine
        target = risk.level.to_state()
        current_rank = self.state.risk_rank
        why = "; ".join(e.description for e in risk.active_evidence) or "no drowsiness indicators"

        # Eyes unobservable right after impairment while elevated: same logic as a lost face.
        if (snap.status == ObservationStatus.DEGRADED and snap.lost_while_impaired
                and self.state.is_elevated and target.risk_rank < current_rank):
            return self._hold(t, risk, "eyes unobservable right after impairment signs - holding state", decay=False)

        if target.risk_rank > current_rank:
            self._lower_since = None
            self._low_since = None
            if self._pending_since is None:
                self._pending_since = t
            confirm = sm.escalate_confirm_s.get(target.value, 0.0)
            if t - self._pending_since >= confirm:
                return self._go(t, target, risk.confidence, risk, f"{risk.rule}: {why}")
            return self._stay(t, risk, f"confirming {target.value} ({t - self._pending_since:.1f}/{confirm:.1f}s)")

        self._pending_since = None

        if current_rank > 0:
            if target.risk_rank == current_rank:
                self._lower_since = None
                return self._stay(t, risk, f"{risk.rule}: {why}")
            if self._lower_since is None:
                self._lower_since = t
            hold = sm.deescalate_hold_s.get(self.state.value, 0.0)
            if t - self._lower_since >= hold:
                self._lower_since = t  # each further step needs its own hold period
                step_down = ELEVATED_LADDER[current_rank - 1]
                if step_down == DriverState.ALERT:
                    step_down = DriverState.MONITORING
                    self._low_since = t
                return self._go(t, step_down, risk.confidence, risk,
                                f"risk below {self.state.value} for {hold:.0f}s - stepping down")
            return self._stay(t, risk, f"recovering: lower risk for {t - self._lower_since:.1f}/{hold:.0f}s",
                              recovery=min(1.0, (t - self._lower_since) / hold) if hold > 0 else 1.0)

        # Not elevated and target is LOW.
        assert risk.level == RiskLevel.LOW
        observed = snap.status == ObservationStatus.OBSERVED and snap.calibrated
        if not observed:
            self._low_since = None
            reason = "driver not fully observable (" + snap.status.value.lower() + ")" if snap.calibrated else (
                "calibrating driver baseline")
            return self._go(t, DriverState.MONITORING, risk.confidence, risk, reason)
        if self.state == DriverState.ALERT:
            return self._stay(t, risk, "no drowsiness indicators")
        if self._low_since is None:
            self._low_since = t
        if t - self._low_since >= sm.recovery_confirm_s:
            return self._go(t, DriverState.ALERT, risk.confidence, risk,
                            f"driver observed with low risk for {sm.recovery_confirm_s:.0f}s")
        return self._go(t, DriverState.MONITORING, risk.confidence, risk, "confirming alert state")

    # ------------------------------------------------------------------ helpers
    def _go(self, t: float, new: DriverState, conf: float, risk: RiskAssessment, reason: str) -> StateDecision:
        prev = self.state
        changed = new != prev
        if changed:
            self.state, self._since = new, t
            self._pending_since = None
            if not new.is_elevated:
                self._lower_since = None
            if new in (DriverState.UNKNOWN, DriverState.CAMERA_UNAVAILABLE):
                self._low_since = None  # ALERT must be re-earned after losing sight of the driver
        self._confidence = conf
        return StateDecision(t, self.state, prev, changed, self._since, conf, risk, reason)

    def _stay(self, t: float, risk: RiskAssessment, reason: str, recovery: float | None = None) -> StateDecision:
        self._confidence = risk.confidence
        return StateDecision(t, self.state, self.state, False, self._since, risk.confidence, risk, reason, recovery)

    def _hold(self, t: float, risk: RiskAssessment, reason: str, decay: bool = True) -> StateDecision:
        if self._hold_from is None:
            self._hold_from = t
        conf = self._confidence
        if decay:
            conf = max(0.0, conf * (1.0 - self._s.observation.confidence_decay_per_s * (t - self._hold_from)))
        return StateDecision(t, self.state, self.state, False, self._since, round(conf, 4), risk, reason)
