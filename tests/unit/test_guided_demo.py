import numpy as np

from drowsiness.presentation.guided_demo import CALIBRATION_PROMPT, MAX_CALIBRATION_S, STEPS, GuidedDemo, draw_prompt


def test_waits_for_calibration_then_runs_script_and_finishes():
    g = GuidedDemo()
    assert g.update(0.0, calibrated=False)[0] == CALIBRATION_PROMPT
    assert g.update(10.0, calibrated=False)[0] == CALIBRATION_PROMPT
    prompt, _, left, done = g.update(20.0, calibrated=True)
    assert prompt == STEPS[0].prompt and left == STEPS[0].seconds and not done
    total = sum(s.seconds for s in STEPS)
    assert g.update(20.0 + STEPS[0].seconds + 0.1, True)[0] == STEPS[1].prompt
    assert not g.update(20.0 + total + 0.5, True)[3]
    assert g.update(20.0 + total + 5.0, True)[3]


def test_script_starts_even_if_calibration_never_finishes():
    g = GuidedDemo()
    g.update(0.0, calibrated=False)
    assert g.update(MAX_CALIBRATION_S + 0.1, calibrated=False)[0] == STEPS[0].prompt


def test_prompt_draws_only_over_video_area():
    canvas = np.zeros((480, 1000, 3), np.uint8)
    draw_prompt(canvas, 640, "Close your eyes", "WARNING expected", 3.2)
    assert canvas[-30:, :640].any()
    assert not canvas[:, 640:].any()
