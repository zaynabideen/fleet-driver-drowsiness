import json
from dataclasses import replace

import pytest
from helpers import snapshot

from drowsiness.config.settings import Settings
from drowsiness.events.event_log import EventRecorder, format_decision
from drowsiness.safety.alert_manager import AlertManager
from drowsiness.safety.risk_engine import RuleBasedRiskEngine, graded, severity
from drowsiness.safety.state_machine import SafetyStateMachine
from drowsiness.schemas.events import RiskAssessment, StateDecision
from drowsiness.schemas.states import DriverState, EyeState, ObservationStatus, RiskLevel

S = Settings()
ENGINE = RuleBasedRiskEngine(S.risk)


# ---------------------------------------------------------------- severity helpers

def test_severity_counts_thresholds():
    assert severity(0.5, (1, 2, 3)) == 0
    assert severity(2.0, (1, 2, 3)) == 2
    assert severity(9, (1, 2, 3)) == 3
    assert severity(9, (2, 1e9, 1e9)) == 1  # disabled levels never reached


def test_graded_progress():
    assert graded(1.5, (1, 2, 3)) == pytest.approx(1.5)
    assert graded(0.5, (1, 2, 3)) == pytest.approx(0.5)
    assert graded(5, (2, 1e9, 1e9)) == 1.0


# ---------------------------------------------------------------- risk engine

def assess(**kw) -> RiskAssessment:
    return ENGINE.assess(snapshot(**kw))


def test_normal_driver_low_risk():
    r = assess()
    assert r.level == RiskLevel.LOW
    assert r.risk_score < 0.25
    assert r.active_evidence == ()


@pytest.mark.parametrize(
    "closure,level",
    [(0.3, RiskLevel.LOW), (1.2, RiskLevel.MODERATE), (2.2, RiskLevel.HIGH), (3.1, RiskLevel.CRITICAL)],
)
def test_ongoing_eye_closure_levels(closure, level):
    r = assess(current_closure_s=closure, eye_state=EyeState.CLOSED)
    assert r.level == level


def test_closure_plus_head_drop_is_critical_sooner():
    r = assess(current_closure_s=2.1, eye_state=EyeState.CLOSED, head_dropping=True, rel_pitch_deg=25)
    assert r.level == RiskLevel.CRITICAL
    assert r.rule == "eye_closure_with_head_drop"


def test_yawning_alone_never_exceeds_moderate():
    r = assess(yawns=10, yawning_now=True)
    assert r.level == RiskLevel.MODERATE


def test_blinking_alone_never_exceeds_moderate():
    assert assess(blink_rate_per_min=60, slow_blinks=9).level == RiskLevel.MODERATE


def test_single_nod_is_ignored():
    assert assess(nods=1).level == RiskLevel.LOW


def test_eyes_corroborated_by_head_is_high():
    r = assess(long_closures=2, nods=2)
    assert r.level == RiskLevel.HIGH
    assert r.rule == "eyes_corroborated_by_head"


def test_three_weak_categories_is_high():
    r = assess(slow_blinks=5, yawns=3, nods=2)
    assert r.level == RiskLevel.HIGH
    assert r.rule == "multiple_independent_categories"


def test_perclos_levels():
    assert assess(perclos=0.18).level == RiskLevel.MODERATE
    assert assess(perclos=0.30).level == RiskLevel.HIGH
    assert assess(perclos=0.45).level == RiskLevel.CRITICAL


def test_score_always_inside_level_band():
    cases = [{}, {"perclos": 0.18}, {"perclos": 0.3}, {"perclos": 0.5}, {"current_closure_s": 1.9},
             {"current_closure_s": 5.0}, {"yawns": 3}]
    for kw in cases:
        r = assess(**kw)
        assert r.level.value * 0.25 <= r.risk_score < (r.level.value + 1) * 0.25 + 1e-9, (kw, r)


def test_every_elevated_decision_has_evidence_and_rule():
    r = assess(current_closure_s=2.5, eye_state=EyeState.CLOSED)
    assert r.active_evidence
    assert "eye closure 2.5s" in r.active_evidence[0].description
    assert r.rule == "strong_eye_evidence"


def test_corroboration_raises_confidence():
    single = assess(current_closure_s=1.2)
    multi = assess(current_closure_s=1.2, nods=2, yawns=2)
    assert multi.confidence > single.confidence


def test_history_evidence_retires_once_driver_observed_alert():
    fresh = assess(long_closures=3, nods=3, yawns=2, observed_alert_s=2.0)
    assert fresh.level == RiskLevel.HIGH
    stale = assess(long_closures=3, nods=3, yawns=2, observed_alert_s=S.risk.history_relevance_s + 1)
    assert stale.level == RiskLevel.LOW
    assert all(e.severity == 0 for e in stale.evidence)
    assert any("inactive: driver alert" in e.description for e in stale.evidence)


def test_acute_evidence_never_retired():
    r = assess(current_closure_s=2.5, eye_state=EyeState.CLOSED, observed_alert_s=60.0)
    assert r.level == RiskLevel.HIGH


def test_unobservable_is_not_low_risk_with_confidence():
    r = assess(status=ObservationStatus.FACE_LOST, face_lost_s=3.0)
    assert not r.observable
    assert r.confidence == 0.0
    assert r.evidence[0].code == "face_not_visible"


def test_degraded_status_still_uses_head_and_mouth():
    r = assess(status=ObservationStatus.DEGRADED, eye_state=EyeState.UNOBSERVABLE, perclos=None,
               nods=3, head_down_s=4.5)
    assert r.level == RiskLevel.HIGH
    assert r.rule == "repeated_nodding"
    # sustained head-down alone (eyes hidden) is only a warning: could be reading the dashboard
    assert assess(status=ObservationStatus.DEGRADED, eye_state=EyeState.UNOBSERVABLE, perclos=None,
                  head_down_s=4.5).level == RiskLevel.MODERATE


# ---------------------------------------------------------------- state machine

def drive(sm, seq):
    """seq: [(seconds, snapshot_kwargs)] at 30 FPS -> list of decisions."""
    out, t = [], getattr(sm, "_t", 0.0)
    for secs, kw in seq:
        for _ in range(max(1, round(secs * 30))):
            snap = snapshot(timestamp_s=t, **kw)
            out.append(sm.step(snap, ENGINE.assess(snap)))
            t += 1 / 30
    sm._t = t
    return out


def test_starts_monitoring_then_alert_after_confirmation():
    sm = SafetyStateMachine(S)
    d = drive(sm, [(0.1, {})])
    assert d[0].state == DriverState.MONITORING
    d = drive(sm, [(2.5, {})])
    assert d[-1].state == DriverState.ALERT


def test_uncalibrated_driver_stays_monitoring():
    sm = SafetyStateMachine(S)
    d = drive(sm, [(5, {"calibrated": False})])
    assert {x.state for x in d} == {DriverState.MONITORING}


def test_single_noisy_frame_does_not_escalate():
    sm = SafetyStateMachine(S)
    drive(sm, [(3, {})])
    d = drive(sm, [(1 / 30, {"current_closure_s": 2.5}), (1, {})])
    assert all(x.state == DriverState.ALERT for x in d)


def test_escalation_requires_temporal_confirmation():
    sm = SafetyStateMachine(S)
    drive(sm, [(3, {})])
    d = drive(sm, [(0.4, {"perclos": 0.18})])
    assert d[-1].state == DriverState.ALERT  # 0.4 s < 0.5 s confirmation
    d = drive(sm, [(0.2, {"perclos": 0.18})])
    assert d[-1].state == DriverState.DROWSINESS_WARNING


def test_critical_is_immediate_and_can_skip_levels():
    sm = SafetyStateMachine(S)
    drive(sm, [(3, {})])
    d = drive(sm, [(1 / 30, {"current_closure_s": 3.2, "eye_state": EyeState.CLOSED})])
    assert d[-1].state == DriverState.CRITICAL_SLEEP_RISK
    assert d[-1].previous_state == DriverState.ALERT


def test_deescalation_needs_sustained_lower_risk_then_drops_to_supported_level():
    hold = S.state_machine.deescalate_hold_s
    sm = SafetyStateMachine(S)
    drive(sm, [(3, {}), (0.1, {"current_closure_s": 3.2})])
    assert sm.state == DriverState.CRITICAL_SLEEP_RISK
    d = drive(sm, [(hold["CRITICAL_SLEEP_RISK"] - 0.5, {})])
    assert d[-1].state == DriverState.CRITICAL_SLEEP_RISK   # confirmation not yet met
    assert 0 < d[-1].recovery_progress < 1                  # progress shown while recovering
    d = drive(sm, [(1.0, {})])
    assert d[-1].state == DriverState.MONITORING            # evidence supports LOW: no rung-by-rung crawl
    d = drive(sm, [(S.state_machine.recovery_confirm_s + 0.2, {})])
    assert d[-1].state == DriverState.ALERT


def test_partial_recovery_stops_at_the_level_still_supported():
    sm = SafetyStateMachine(S)
    drive(sm, [(3, {}), (0.1, {"current_closure_s": 3.2})])
    d = drive(sm, [(S.state_machine.deescalate_hold_s["CRITICAL_SLEEP_RISK"] + 0.5, {"perclos": 0.18})])
    assert d[-1].state == DriverState.DROWSINESS_WARNING


def test_hysteresis_no_flapping_with_oscillating_risk():
    sm = SafetyStateMachine(S)
    drive(sm, [(3, {}), (1, {"perclos": 0.18})])
    assert sm.state == DriverState.DROWSINESS_WARNING
    d = drive(sm, [(0.5, {}), (0.5, {"perclos": 0.18})] * 10)
    assert {x.state for x in d} == {DriverState.DROWSINESS_WARNING}
    assert sum(x.changed for x in d) == 0


def test_face_loss_within_grace_holds_state_with_decaying_confidence():
    sm = SafetyStateMachine(S)
    drive(sm, [(3, {})])
    d = drive(sm, [(0.8, {"status": ObservationStatus.FACE_LOST, "face_lost_s": 0.5})])
    assert d[-1].state == DriverState.ALERT
    assert d[-1].confidence < d[0].confidence


def test_face_loss_beyond_grace_is_unknown_not_alert():
    sm = SafetyStateMachine(S)
    drive(sm, [(3, {})])
    d = drive(sm, [(0.1, {"status": ObservationStatus.FACE_LOST, "face_lost_s": 1.5})])
    assert d[-1].state == DriverState.UNKNOWN
    assert d[-1].confidence == 0.0


def test_face_lost_after_impairment_holds_elevated_state():
    sm = SafetyStateMachine(S)
    drive(sm, [(3, {}), (0.5, {"current_closure_s": 2.5})])
    assert sm.state == DriverState.HIGH_DROWSINESS_RISK
    d = drive(sm, [(20, {"status": ObservationStatus.FACE_LOST, "face_lost_s": 5.0, "lost_while_impaired": True})])
    assert d[-1].state == DriverState.HIGH_DROWSINESS_RISK


def test_camera_failure_after_timeout():
    sm = SafetyStateMachine(S)
    drive(sm, [(3, {})])
    d = drive(sm, [(0.1, {"status": ObservationStatus.CAMERA_UNAVAILABLE, "camera_unavailable_s": 1.0})])
    assert d[-1].state == DriverState.ALERT
    d = drive(sm, [(0.1, {"status": ObservationStatus.CAMERA_UNAVAILABLE, "camera_unavailable_s": 2.5})])
    assert d[-1].state == DriverState.CAMERA_UNAVAILABLE


def test_recovery_from_unknown_requires_confirmation():
    sm = SafetyStateMachine(S)
    drive(sm, [(3, {}), (0.1, {"status": ObservationStatus.FACE_LOST, "face_lost_s": 2.0})])
    assert sm.state == DriverState.UNKNOWN
    d = drive(sm, [(0.5, {})])
    assert d[-1].state == DriverState.MONITORING
    d = drive(sm, [(2.0, {})])
    assert d[-1].state == DriverState.ALERT


def test_sunglasses_driver_is_monitoring_not_alert():
    sm = SafetyStateMachine(S)
    d = drive(sm, [(5, {"status": ObservationStatus.DEGRADED, "eye_state": EyeState.UNOBSERVABLE})])
    assert d[-1].state == DriverState.MONITORING


# ---------------------------------------------------------------- alerts

def decision(t, state, prev=DriverState.ALERT, changed=True):
    return StateDecision(t, state, prev, changed, t, 0.9, None, "test")


class Collect:
    def __init__(self):
        self.alerts = []

    def emit(self, alert, decision):
        self.alerts.append(alert)


def test_alert_fires_on_entering_warning_state():
    sink = Collect()
    am = AlertManager(S.alerts, [sink])
    a = am.process(decision(0, DriverState.DROWSINESS_WARNING))
    assert a is not None and a.severity == 1 and a.action == "VISUAL_WARNING"
    assert sink.alerts == [a]


def test_no_alert_for_alert_or_monitoring():
    am = AlertManager(S.alerts)
    assert am.process(decision(0, DriverState.ALERT)) is None
    assert am.process(decision(0, DriverState.MONITORING)) is None


def test_alert_cooldown_prevents_spam():
    am = AlertManager(S.alerts)
    fired = [am.process(decision(t * 0.1, DriverState.DROWSINESS_WARNING, changed=False)) for t in range(200)]
    assert sum(a is not None for a in fired) == 1  # 20 s < 30 s cooldown


def test_reminder_after_cooldown():
    am = AlertManager(S.alerts)
    assert am.process(decision(0, DriverState.DROWSINESS_WARNING))
    assert am.process(decision(10, DriverState.DROWSINESS_WARNING, changed=False)) is None
    assert am.process(decision(31, DriverState.DROWSINESS_WARNING, changed=False)) is not None


def test_escalation_not_delayed_by_lower_alert_cooldown():
    am = AlertManager(S.alerts)
    assert am.process(decision(0, DriverState.DROWSINESS_WARNING))
    assert am.process(decision(1, DriverState.HIGH_DROWSINESS_RISK))
    a = am.process(decision(2, DriverState.CRITICAL_SLEEP_RISK))
    assert a is not None and a.severity == 3


def test_flicker_between_states_does_not_spam():
    am = AlertManager(S.alerts)
    fired = []
    for i in range(20):
        st = DriverState.DROWSINESS_WARNING if i % 2 else DriverState.ALERT
        fired.append(am.process(decision(i * 0.5, st)))
    assert sum(a is not None for a in fired) == 1


def test_unknown_raises_system_alert():
    a = AlertManager(S.alerts).process(decision(0, DriverState.UNKNOWN))
    assert a.severity == 0 and a.action == "CHECK_CAMERA_VIEW"


# ---------------------------------------------------------------- event log

def test_event_log_is_structured_json(tmp_path):
    snap = snapshot(current_closure_s=2.5, eye_state=EyeState.CLOSED)
    risk = ENGINE.assess(snap)
    d = StateDecision(1.0, DriverState.HIGH_DROWSINESS_RISK, DriverState.ALERT, True, 1.0, risk.confidence, risk,
                      "strong_eye_evidence")
    rec = EventRecorder(tmp_path)
    rec.record_decision(d, snap, "ALERT_DRIVER")
    rec.close()
    event = json.loads((tmp_path / "events.jsonl").read_text().splitlines()[0])
    for key in ("event_id", "timestamp", "driver_state", "confidence", "risk_score", "evidence", "signals", "action"):
        assert key in event
    assert event["driver_state"] == "HIGH_DROWSINESS_RISK"
    assert event["signals"]["eyes_closed"] is True
    assert event["evidence"][0]["code"] == "prolonged_eye_closure"
    log = (tmp_path / "decisions.log").read_text()
    assert "Evidence:" in log and "eye closure 2.5s" in log


def test_format_decision_explains_what_why_how_confident():
    snap = snapshot(current_closure_s=2.5)
    risk = ENGINE.assess(snap)
    text = format_decision(StateDecision(1.0, DriverState.HIGH_DROWSINESS_RISK, DriverState.ALERT, True, 1.0,
                                         0.91, risk, "r"))
    assert "ALERT -> HIGH_DROWSINESS_RISK" in text and "Confidence: 0.91" in text and "Reason:" in text


def test_snapshot_helper_replace_works():
    assert replace(snapshot(), nods=2).nods == 2
