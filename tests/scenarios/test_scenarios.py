"""End-to-end behavioural scenarios through the real pipeline (synthetic feature streams).

Each stream starts with 25 s of normal driving so the driver baseline
calibrates exactly as it would on the road. These tests prove the DECISION
LOGIC does what the spec says. They do not measure real-world accuracy.
"""

import pytest
from helpers import run_stream, states

from drowsiness.schemas.states import DriverState
from drowsiness.simulation.streams import SUNGLASSES, StreamBuilder

CAL = 25.0  # seconds of normal driving for calibration
RANK = {s.value: s.risk_rank for s in DriverState}


def calibrated(fps: float = 30.0) -> StreamBuilder:
    return StreamBuilder(fps=fps).alert(CAL)


def max_state(outs, after_s=CAL):
    elevated = [o.decision.state for o in outs if o.decision.timestamp_s >= after_s]
    return max(elevated, key=lambda s: s.risk_rank)


# ------------------------------------------------------------- the 8 required scenarios

def test_scenario_1_normal_blinking_is_alert():
    outs = run_stream(calibrated().alert(60).frames)
    assert outs[-1].decision.state == DriverState.ALERT
    assert max_state(outs).risk_rank == 0
    assert all(o.alert is None for o in outs)


def test_scenario_2_long_eye_closure_is_warning():
    outs = run_stream(calibrated().alert(5).eyes_closed(1.6).alert(5).frames)
    assert max_state(outs) == DriverState.DROWSINESS_WARNING
    warn = next(o for o in outs if o.decision.state == DriverState.DROWSINESS_WARNING)
    assert "prolonged_eye_closure" in [e.code for e in warn.risk.active_evidence]
    assert warn.alert is not None and warn.alert.severity == 1


def test_scenario_3_repeated_long_closures_and_nodding_is_high():
    b = calibrated().alert(5)
    for _ in range(3):
        b.eyes_closed(0.3, pitch=0, pitch_to=22).eyes_closed(1.0, pitch=22).alert(0.5, pitch=22, pitch_to=0).alert(4)
    outs = run_stream(b.frames)
    assert max_state(outs) == DriverState.HIGH_DROWSINESS_RISK
    high = next(o for o in outs if o.decision.state == DriverState.HIGH_DROWSINESS_RISK)
    codes = {e.code for e in high.risk.active_evidence}
    assert {"prolonged_eye_closure", "head_nodding"} <= codes or "repeated_long_closures" in codes


def test_scenario_4_persistent_closure_and_head_drop_is_critical():
    outs = run_stream(calibrated().alert(5).eyes_closed(0.3, pitch=0, pitch_to=25).eyes_closed(3.0, pitch=25).frames)
    crit = [o for o in outs if o.decision.state == DriverState.CRITICAL_SLEEP_RISK]
    assert crit, states(outs)[-5:]
    first = crit[0]
    assert first.risk.rule == "eye_closure_with_head_drop"
    assert first.snapshot.current_closure_s == pytest.approx(2.0, abs=0.1)  # before the 3 s microsleep rule
    assert first.alert is not None and first.alert.action == "ALERT_DRIVER_URGENT_AND_NOTIFY_FLEET"


def test_scenario_5_face_disappears_is_unknown_not_alert():
    outs = run_stream(calibrated().alert(5).no_face(5).frames)
    tail = [o.decision.state for o in outs if o.decision.timestamp_s > CAL + 5 + 1.1]
    assert set(tail) == {DriverState.UNKNOWN}
    assert DriverState.ALERT not in tail


def test_scenario_6_camera_fails_is_camera_unavailable():
    outs = run_stream(calibrated().alert(5).camera_down(4).frames)
    assert outs[-1].decision.state == DriverState.CAMERA_UNAVAILABLE
    assert any(o.alert and o.alert.action == "REPORT_CAMERA_FAULT" for o in outs)


def test_scenario_7_brief_look_down_is_not_sleep():
    # Quick glance down: head drops fast (one 'nod'), eyelids lower so EAR reads 'closed' for ~0.9 s.
    b = calibrated().alert(5).segment(0.3, ear=0.17, pitch=0, pitch_to=25).segment(0.6, ear=0.17, pitch=25)
    b.alert(0.3, pitch=25, pitch_to=0).alert(10)
    outs = run_stream(b.frames)
    assert max_state(outs).risk_rank == 0
    assert outs[-1].decision.state == DriverState.ALERT


def test_scenario_7b_dashboard_check_with_eyes_open_is_not_drowsiness():
    outs = run_stream(calibrated().alert(5).alert(3, pitch=25).alert(5).frames)
    assert max_state(outs).risk_rank == 0
    assert max(o.snapshot.eyes_off_road_s for o in outs) > 2.5  # reported as context, not risk


def test_scenario_8_single_yawn_is_not_critical():
    outs = run_stream(calibrated().alert(5).segment(4.0, mar=0.8).alert(10).frames)
    assert max_state(outs).risk_rank <= 1
    assert any(o.snapshot.yawns == 1 for o in outs)


# ------------------------------------------------------------- additional robustness scenarios

def test_calibration_period_reports_monitoring_not_alert():
    outs = run_stream(StreamBuilder().alert(15).frames)
    assert {o.decision.state for o in outs} == {DriverState.MONITORING}


def test_microsleep_without_head_movement_is_critical():
    outs = run_stream(calibrated().alert(5).eyes_closed(3.5).frames)
    assert outs[-1].decision.state == DriverState.CRITICAL_SLEEP_RISK
    assert outs[-1].risk.rule == "microsleep_length_eye_closure"


def test_gradual_drowsiness_via_perclos():
    b = calibrated().alert(40)
    for _ in range(25):  # 0.7 s closures every 2.5 s: slow blinks, PERCLOS ~0.28
        b.eyes_closed(0.7).alert(1.8, blink_every_s=0)
    outs = run_stream(b.frames)
    assert max_state(outs).risk_rank >= 2
    assert any(e.code == "high_perclos" and e.severity >= 1 for o in outs for e in o.risk.active_evidence)


def test_talking_and_laughing_stay_alert():
    b = calibrated().alert(5)
    for _ in range(30):
        b.segment(0.3, mar=0.35).segment(0.2, mar=0.05)
    for _ in range(4):
        b.segment(0.8, mar=0.6).segment(0.5)
    outs = run_stream(b.frames)
    assert max_state(outs).risk_rank == 0


def test_sunglasses_driver_is_monitoring_not_alert():
    outs = run_stream(calibrated().alert(5).segment(20, quality=SUNGLASSES).frames)
    assert outs[-1].decision.state == DriverState.MONITORING
    assert max_state(outs).risk_rank == 0


def test_slump_out_of_view_holds_elevated_state():
    # Eyes close, head drops, then the face leaves the frame: not a neutral sensor gap.
    b = calibrated().alert(5).eyes_closed(0.3, pitch=0, pitch_to=25).eyes_closed(2.2, pitch=25).no_face(10)
    outs = run_stream(b.frames)
    assert outs[-1].decision.state == DriverState.CRITICAL_SLEEP_RISK
    assert "possible slump" in outs[-1].decision.reason


def test_driver_turning_head_to_check_mirror_is_not_drowsiness():
    outs = run_stream(calibrated().alert(5).alert(1.5, yaw=55).alert(5).frames)
    assert max_state(outs).risk_rank == 0


def test_noisy_eye_signal_does_not_trigger():
    import random

    rng = random.Random(0)
    b = calibrated()
    for _ in range(600):  # 20 s of single-frame dropouts in EAR (tracker noise)
        b.segment(1 / 30, ear=0.08 if rng.random() < 0.08 else 0.30)
    outs = run_stream(b.frames)
    assert max_state(outs).risk_rank == 0


@pytest.mark.parametrize("fps", [30.0, 15.0, 10.0])
def test_critical_detection_is_frame_rate_independent(fps):
    outs = run_stream(calibrated(fps).alert(5).eyes_closed(0.3, pitch=0, pitch_to=25).eyes_closed(2.5, pitch=25).frames)
    first = next(o for o in outs if o.decision.state == DriverState.CRITICAL_SLEEP_RISK)
    assert first.snapshot.current_closure_s == pytest.approx(2.0, abs=1.5 / fps)


def test_recovery_after_critical_goes_through_monitoring_and_is_prompt():
    b = calibrated().alert(5).eyes_closed(3.2)
    t0 = b.t
    outs = run_stream(b.alert(40).frames)
    seq = []
    for st in states(outs):
        if not seq or seq[-1] != st:
            seq.append(st)
    assert "CRITICAL_SLEEP_RISK" in seq
    assert seq[-2:] == ["MONITORING", "ALERT"]       # never jumps straight to ALERT
    took = _seconds_to_alert(outs, t0)
    assert took is not None and took <= 12, took


# ------------------------------------------------------------- recovery (regression: alarm stuck after driver recovered)

def _seconds_to_alert(outs, normal_from):
    for o in outs:
        if o.decision.timestamp_s >= normal_from and o.decision.state == DriverState.ALERT:
            return o.decision.timestamp_s - normal_from
    return None


def test_returns_to_alert_promptly_after_repeated_episodes():
    b = calibrated().alert(3)
    for _ in range(3):
        b.eyes_closed(0.3, pitch=0, pitch_to=25).eyes_closed(2.2, pitch=25).alert(3)
    t0 = b.t
    outs = run_stream(b.alert(60).frames)
    assert max_state(outs) == DriverState.CRITICAL_SLEEP_RISK
    took = _seconds_to_alert(outs, t0)
    assert took is not None and took < 15, took  # was ~69 s, then ~27 s on a real run


def test_two_yawns_do_not_hold_warning_for_minutes():
    b = calibrated().alert(3).segment(4, mar=0.8).alert(5).segment(4, mar=0.8).alert(3).eyes_closed(2.2).alert(1)
    t0 = b.t
    outs = run_stream(b.alert(60).frames)
    took = _seconds_to_alert(outs, t0)
    assert took is not None and took < 15, took  # was ~5 minutes before the fix


def test_repeated_microsleep_pattern_still_escalates_despite_alert_gaps():
    b = calibrated().alert(3)
    for _ in range(3):  # 1.3 s closures separated by ~18 s of normal driving
        b.eyes_closed(1.3).alert(18)
    outs = run_stream(b.frames)
    assert max_state(outs).risk_rank >= 2


def test_unobservable_time_does_not_count_as_recovery():
    b = calibrated().alert(3)
    for _ in range(3):
        b.eyes_closed(0.3, pitch=0, pitch_to=22).eyes_closed(1.0, pitch=22).alert(0.5, pitch=22, pitch_to=0).alert(2)
    b.segment(20, quality=SUNGLASSES)  # eyes can't be seen: history must stay active
    outs = run_stream(b.frames)
    assert outs[-1].snapshot.observed_alert_s == 0.0
    assert outs[-1].decision.state.risk_rank >= 1



# ------------------------------------------------------------- yawns (regression from a real webcam run)

def test_moderate_real_world_yawn_is_detected():
    outs = run_stream(calibrated().alert(3).segment(3.5, mar=0.45).alert(3).frames)
    assert any(o.snapshot.yawning_now for o in outs)
    assert outs[-1].snapshot.yawns == 1


def test_eyes_closing_during_a_yawn_is_not_counted_as_drowsy_eye_closure():
    # Real run: eyes shut for ~2.3 s during a yawn pushed the state to HIGH.
    b = calibrated().alert(3)
    b.segment(0.4, mar=0.6).segment(3.0, mar=0.7, ear=0.08).segment(0.4, mar=0.6).alert(5)
    outs = run_stream(b.frames)
    assert max_state(outs).risk_rank <= 1
    assert all(o.snapshot.long_closures == 0 for o in outs)


def test_mouth_hanging_open_does_not_hide_a_microsleep():
    # Asleep with the mouth open: the yawn exemption expires after max_yawn_s.
    b = calibrated().alert(3).segment(12, mar=0.7, ear=0.08, pitch=0, pitch_to=-15)
    outs = run_stream(b.frames)
    assert max_state(outs) == DriverState.CRITICAL_SLEEP_RISK



def test_briefly_hiding_the_face_does_not_revive_cleared_history():
    # Real run: after recovering, covering the face then uncovering jumped back to HIGH on old history.
    b = calibrated().alert(3)
    for _ in range(3):
        b.eyes_closed(1.4).alert(1)
    b.alert(12).no_face(3)
    t0 = b.t
    outs = run_stream(b.alert(8).frames)
    after = [o.decision.state for o in outs if o.decision.timestamp_s >= t0]
    assert all(st.risk_rank == 0 for st in after), set(after)
    assert after[-1] == DriverState.ALERT



def test_yawn_held_3_seconds_raises_a_warning_and_clears_after():
    b = calibrated().alert(3).segment(3.6, mar=0.6)
    t_yawn_end = b.t
    outs = run_stream(b.alert(15).frames)
    warn = [o for o in outs if o.decision.state == DriverState.DROWSINESS_WARNING]
    assert warn, "a 3+ s yawn must raise a warning"
    assert warn[0].alert is not None and warn[0].alert.severity == 1
    assert "frequent_yawning" in {e.code for e in warn[0].risk.active_evidence}
    assert max_state(outs) == DriverState.DROWSINESS_WARNING  # never more than a warning
    assert _seconds_to_alert(outs, t_yawn_end) < 12
    assert outs[-1].decision.state == DriverState.ALERT


def test_mouth_open_2_seconds_does_not_alert():
    outs = run_stream(calibrated().alert(3).segment(2.0, mar=0.6).alert(5).frames)
    assert max_state(outs).risk_rank == 0



def test_yawn_warning_appears_3_seconds_after_mouth_opens():
    b = calibrated().alert(3)
    t_open = b.t
    outs = run_stream(b.segment(4.0, mar=0.6).alert(3).frames)
    first = next(o for o in outs if o.decision.state == DriverState.DROWSINESS_WARNING)
    assert first.decision.timestamp_s - t_open == pytest.approx(3.0, abs=0.1)
