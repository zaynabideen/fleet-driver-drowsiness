from dataclasses import replace

import pytest

from drowsiness.analysis.blink_analyzer import BlinkAnalyzer
from drowsiness.analysis.calibration import DriverBaseline
from drowsiness.analysis.eye_state import EyeStateClassifier
from drowsiness.analysis.head_movement import HeadMovementAnalyzer
from drowsiness.analysis.temporal_analyzer import TemporalAnalyzer
from drowsiness.analysis.yawn_analyzer import YawnAnalyzer
from drowsiness.config.settings import Settings
from drowsiness.schemas.states import EyeState, ObservationStatus
from drowsiness.simulation.streams import GOOD, SUNGLASSES, StreamBuilder

S = Settings()


def fixed_baseline(**kw):
    return DriverBaseline.fixed(S.calibration, open_ear=0.30, **kw)


def feed_blinks(states_with_durations, fps=30.0):
    """[(EyeState, seconds), ...] -> BlinkAnalyzer, last metrics."""
    ba = BlinkAnalyzer(S.eyes, S.blink)
    t, m = 0.0, None
    for state, secs in states_with_durations:
        for _ in range(max(1, round(secs * fps))):
            m = ba.update(t, state)
            t += 1 / fps
    return ba, m


# ---------------------------------------------------------------- calibration

def test_calibration_learns_open_ear_despite_blinks_and_camera_mount_angle():
    frames = StreamBuilder().alert(25, ear=0.26, pitch=12.0, yaw=-5.0).frames
    b = DriverBaseline(S.calibration)
    for f in frames:
        b.update(f)
    assert b.calibrated
    assert b.open_ear == pytest.approx(0.26, abs=0.005)  # blinks ignored by the high percentile
    assert b.neutral_pose[0] == pytest.approx(12.0) and b.neutral_pose[1] == pytest.approx(-5.0)


def test_calibration_not_complete_before_duration():
    b = DriverBaseline(S.calibration)
    for f in StreamBuilder().alert(5).frames:
        b.update(f)
    assert not b.calibrated
    assert b.open_ear == S.calibration.default_open_ear
    assert 0 < b.progress < 1


def test_calibration_clamps_implausible_baseline():
    b = DriverBaseline(S.calibration)
    for f in StreamBuilder().segment(25, ear=0.12).frames:  # e.g. calibrating on a very tired driver
        b.update(f)
    assert b.open_ear == S.calibration.min_open_ear


def test_calibration_ignores_poor_quality_frames():
    b = DriverBaseline(S.calibration)
    for f in StreamBuilder().segment(25, quality=replace(GOOD, score=0.3)).frames:
        b.update(f)
    assert not b.calibrated


# ---------------------------------------------------------------- eye state

def _frame(**kw):
    return StreamBuilder().segment(0.01, **kw).frames[0]


def test_eye_closed_and_open_against_baseline():
    c = EyeStateClassifier(S.eyes, S.quality)
    assert c.classify(_frame(ear=0.30), fixed_baseline()).state == EyeState.OPEN
    assert c.classify(_frame(ear=0.10), fixed_baseline()).state == EyeState.CLOSED


def test_eye_state_hysteresis_prevents_chatter():
    c = EyeStateClassifier(S.eyes, S.quality)
    b = fixed_baseline()  # close < 0.195, reopen > 0.225
    assert c.classify(_frame(ear=0.21), b).state == EyeState.OPEN      # in the band, was open
    assert c.classify(_frame(ear=0.15), b).state == EyeState.CLOSED
    assert c.classify(_frame(ear=0.21), b).state == EyeState.CLOSED    # in the band, was closed
    assert c.classify(_frame(ear=0.24), b).state == EyeState.OPEN


def test_threshold_is_relative_to_driver():
    c = EyeStateClassifier(S.eyes, S.quality)
    narrow_eyes = DriverBaseline.fixed(S.calibration, open_ear=0.20)
    # 0.17 would be "closed" for an average driver but is normal for this one
    assert c.classify(_frame(ear=0.17), narrow_eyes).state == EyeState.OPEN
    assert EyeStateClassifier(S.eyes, S.quality).classify(_frame(ear=0.17), fixed_baseline()).state == EyeState.CLOSED


@pytest.mark.parametrize(
    "kw,reason",
    [
        ({"quality": SUNGLASSES}, "eyes not measurable"),
        ({"face": False}, "no face"),
        ({"yaw": 50.0}, "head turned"),
        ({"pitch": 45.0}, "head pitched"),
        ({"camera_ok": False}, "camera"),
    ],
)
def test_eyes_unobservable_never_reported_open(kw, reason):
    obs = EyeStateClassifier(S.eyes, S.quality).classify(_frame(**kw), fixed_baseline())
    assert obs.state == EyeState.UNOBSERVABLE
    assert reason in obs.reason
    assert obs.confidence == 0.0


def test_blendshape_disagreement_lowers_confidence():
    c = EyeStateClassifier(S.eyes, S.quality)
    agree = c.classify(_frame(ear=0.30), fixed_baseline())
    f = replace(_frame(ear=0.30), blink_score=0.9)
    disagree = EyeStateClassifier(S.eyes, S.quality).classify(f, fixed_baseline())
    assert disagree.state == EyeState.OPEN
    assert disagree.confidence < agree.confidence


def test_uncalibrated_lowers_confidence():
    f = _frame(ear=0.30)
    cal = EyeStateClassifier(S.eyes, S.quality).classify(f, fixed_baseline())
    uncal = EyeStateClassifier(S.eyes, S.quality).classify(f, DriverBaseline(S.calibration))
    assert uncal.confidence < cal.confidence


# ---------------------------------------------------------------- blinks / PERCLOS

def test_blink_duration_measured():
    ba, _ = feed_blinks([(EyeState.OPEN, 1), (EyeState.CLOSED, 0.2), (EyeState.OPEN, 1)])
    assert ba.episodes[0].duration_s == pytest.approx(0.2, abs=0.04)


def test_blink_duration_is_frame_rate_independent():
    durations = []
    for fps in (30.0, 10.0):
        ba, _ = feed_blinks([(EyeState.OPEN, 1), (EyeState.CLOSED, 1.2), (EyeState.OPEN, 1)], fps=fps)
        durations.append(ba.episodes[0].duration_s)
    assert durations[0] == pytest.approx(1.2, abs=0.05)
    assert durations[1] == pytest.approx(1.2, abs=0.11)


def test_slow_blinks_and_long_closures_counted():
    seq = [(EyeState.OPEN, 2)]
    for d in (0.15, 0.6, 0.7, 1.3):
        seq += [(EyeState.CLOSED, d), (EyeState.OPEN, 2)]
    _, m = feed_blinks(seq)
    assert m.slow_blinks == 2
    assert m.long_closures == 1


def test_ongoing_closure_reported():
    _, m = feed_blinks([(EyeState.OPEN, 1), (EyeState.CLOSED, 1.5)])
    assert m.current_closure_s == pytest.approx(1.5, abs=0.05)


def test_blink_frequency_normal_driving():
    ba = BlinkAnalyzer(S.eyes, S.blink)
    m = None
    for f in StreamBuilder().alert(60, blink_every_s=4.0).frames:
        m = ba.update(f.timestamp_s, EyeState.CLOSED if f.ear < 0.15 else EyeState.OPEN)
    assert m.blink_rate_per_min == pytest.approx(15, abs=1.5)
    assert m.perclos == pytest.approx(0.15 / 4, abs=0.01)
    assert m.slow_blinks == 0 and m.long_closures == 0


def test_perclos_withheld_when_eyes_rarely_observable():
    _, m = feed_blinks([(EyeState.OPEN, 10), (EyeState.UNOBSERVABLE, 55)])
    assert m.perclos is None
    assert m.perclos_coverage < S.blink.perclos_min_coverage


def test_short_unobservable_gap_does_not_split_closure():
    ba, m = feed_blinks([(EyeState.OPEN, 1), (EyeState.CLOSED, 0.6), (EyeState.UNOBSERVABLE, 0.1),
                         (EyeState.CLOSED, 0.6), (EyeState.OPEN, 0.5)])
    assert len(ba.episodes) == 1
    assert ba.episodes[0].duration_s == pytest.approx(1.3, abs=0.05)


def test_long_unobservable_gap_interrupts_with_lower_bound():
    ba, m = feed_blinks([(EyeState.OPEN, 1), (EyeState.CLOSED, 0.8), (EyeState.UNOBSERVABLE, 2.0)])
    ep = ba.episodes[0]
    assert ep.interrupted
    assert ep.duration_s == pytest.approx(0.8, abs=0.05)  # only what was actually seen
    assert m.current_closure_s == 0.0


# ---------------------------------------------------------------- yawns

def _yawns(builder):
    ya = YawnAnalyzer(S.mouth)
    m = None
    for f in builder.frames:
        m = ya.update(f)
    return m


def test_long_wide_opening_is_a_yawn():
    m = _yawns(StreamBuilder().segment(1).segment(3.5, mar=0.75).segment(1))
    assert m.yawns == 1 and not m.yawning_now


def test_yawning_now_flag_during_yawn():
    m = _yawns(StreamBuilder().segment(1).segment(3.5, mar=0.75))
    assert m.yawning_now and m.yawns == 1


def test_talking_is_not_yawning():
    b = StreamBuilder().segment(1)
    for _ in range(20):
        b.segment(0.3, mar=0.35).segment(0.2, mar=0.05)
    assert _yawns(b).yawns == 0


def test_laughing_short_wide_openings_not_yawns():
    b = StreamBuilder().segment(1)
    for _ in range(5):
        b.segment(0.8, mar=0.6).segment(0.6, mar=0.1)
    assert _yawns(b).yawns == 0


def test_brief_closure_inside_yawn_is_merged():
    m = _yawns(StreamBuilder().segment(1).segment(1.8, mar=0.75).segment(0.2).segment(1.8, mar=0.75).segment(1))
    assert m.yawns == 1


# ---------------------------------------------------------------- head movement

def _head(seq, eyes=EyeState.CLOSED, fps=30.0):
    """seq: [(pitch_from, pitch_to, seconds)]"""
    ha = HeadMovementAnalyzer(S.head)
    t, m = 0.0, None
    for p0, p1, secs in seq:
        n = max(1, round(secs * fps))
        for i in range(n):
            m = ha.update(t, p0 + (p1 - p0) * i / max(1, n - 1), 0.0, eyes)
            t += 1 / fps
    return m


def test_fast_drop_is_a_nod_counted_once():
    m = _head([(0, 0, 1), (0, 25, 0.3), (25, 25, 2)])
    assert m.nods == 1
    assert m.head_dropping


def test_slow_look_down_is_not_a_nod():
    m = _head([(0, 0, 1), (0, 25, 3.0), (25, 25, 1)], eyes=EyeState.OPEN)
    assert m.nods == 0


def test_repeated_nods_need_reset_between():
    m = _head([(0, 0, 1)] + [(0, 20, 0.3), (20, 0, 0.5), (0, 0, 1.0)] * 3)
    assert m.nods == 3


def test_head_down_with_eyes_open_is_distraction_not_drowsiness():
    m = _head([(0, 25, 0.2), (25, 25, 3)], eyes=EyeState.OPEN)
    assert m.head_down_s == 0.0
    assert m.eyes_off_road_s == pytest.approx(3.0, abs=0.1)


def test_head_down_with_eyes_closed_counts():
    m = _head([(25, 25, 3)], eyes=EyeState.CLOSED)
    assert m.head_down_s == pytest.approx(3.0, abs=0.1)


def test_looking_away_flag():
    ha = HeadMovementAnalyzer(S.head)
    assert ha.update(0.0, 0.0, 50.0, EyeState.UNOBSERVABLE).looking_away


# ---------------------------------------------------------------- temporal analyzer

def _analyze(frames, baseline=None):
    ta = TemporalAnalyzer(S, baseline or fixed_baseline())
    return [ta.update(f) for f in frames]


def test_face_lost_status_and_duration():
    snaps = _analyze(StreamBuilder().alert(2).no_face(3).frames)
    assert snaps[-1].status == ObservationStatus.FACE_LOST
    assert snaps[-1].face_lost_s == pytest.approx(3.0, abs=0.05)
    assert snaps[-1].eye_state == EyeState.UNOBSERVABLE


def test_face_lost_after_eye_closure_flags_possible_slump():
    snaps = _analyze(StreamBuilder().alert(2).eyes_closed(1.5, pitch=25).no_face(2).frames)
    assert snaps[-1].lost_while_impaired


def test_face_lost_while_alert_is_not_flagged():
    snaps = _analyze(StreamBuilder().alert(10).no_face(2).frames)
    assert not snaps[-1].lost_while_impaired


def test_camera_down_status():
    snaps = _analyze(StreamBuilder().alert(1).camera_down(1).frames)
    assert snaps[-1].status == ObservationStatus.CAMERA_UNAVAILABLE
    assert snaps[-1].camera_unavailable_s == pytest.approx(1.0, abs=0.05)


def test_sunglasses_degrade_but_head_still_tracked():
    snaps = _analyze(StreamBuilder().segment(2, quality=SUNGLASSES, pitch=5).frames)
    assert snaps[-1].status == ObservationStatus.DEGRADED
    assert snaps[-1].rel_pitch_deg == pytest.approx(5.0)
    assert snaps[-1].observation_confidence < 0.5


def test_head_coming_back_up_is_not_a_nod():
    """Regression from a real webcam run: lifting the head from looking UP back to neutral
    (pitch -20 -> 0) was counted as 9 nods in half a second."""
    m = _head([(0, 0, 1), (0, -20, 1.0), (-20, -20, 1), (-20, 1, 0.4), (1, 1, 2)], eyes=EyeState.OPEN)
    assert m.nods == 0


def test_one_drop_is_one_nod_even_when_it_hovers_near_reset():
    m = _head([(0, 0, 1), (0, 22, 0.3), (22, 4, 0.3), (4, 5, 1.0)], eyes=EyeState.CLOSED)
    assert m.nods == 1



def test_mouth_open_2s_is_not_a_yawn():
    m = _yawns(StreamBuilder().segment(1).segment(2.0, mar=0.75).segment(1))
    assert m.yawns == 0
