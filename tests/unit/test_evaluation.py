import json

import pytest

from drowsiness.evaluation.dataset import assign_splits, check_no_subject_leakage, split_for_subject
from drowsiness.evaluation.metrics import Interval, Prediction, event_metrics, frame_metrics

LABELS = [Interval(0, 100, "alert"), Interval(100, 120, "drowsy"), Interval(120, 125, "ignore"),
          Interval(125, 200, "alert")]


def timeline(spans):
    """spans: [(start, end, state)] at 10 Hz"""
    out = []
    for start, end, state in spans:
        t = start
        while t < end - 1e-9:
            out.append(Prediction(round(t, 2), state))
            t += 0.1
    return out


def test_perfect_detector():
    preds = timeline([(0, 100, "ALERT"), (100, 120, "HIGH_DROWSINESS_RISK"), (120, 200, "ALERT")])
    m = frame_metrics(preds, LABELS).summary()
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    assert m["false_positive_rate"] == 0.0 and m["coverage"] == 1.0


def test_ignore_intervals_are_not_scored():
    preds = timeline([(120, 125, "CRITICAL_SLEEP_RISK")])
    assert frame_metrics(preds, LABELS).total == 0


def test_warning_is_not_positive_by_default():
    preds = timeline([(100, 120, "DROWSINESS_WARNING")])
    m = frame_metrics(preds, LABELS)
    assert m.fn == 200 and m.tp == 0


def test_abstentions_lower_conservative_recall_not_hide_misses():
    preds = timeline([(100, 110, "HIGH_DROWSINESS_RISK"), (110, 120, "UNKNOWN")])
    s = frame_metrics(preds, LABELS).summary()
    assert s["recall"] == 1.0                      # on covered frames
    assert s["recall_conservative"] == pytest.approx(0.5)
    assert s["coverage"] == pytest.approx(0.5)


def test_event_recall_latency_and_false_alarms():
    preds = timeline([(0, 50, "ALERT"), (50, 53, "HIGH_DROWSINESS_RISK"), (53, 103.5, "ALERT"),
                      (103.5, 120, "CRITICAL_SLEEP_RISK"), (120, 200, "ALERT")])
    ev = event_metrics(preds, LABELS).summary()
    assert ev["drowsy_episodes"] == 1 and ev["event_recall"] == 1.0
    assert ev["latency_median_s"] == pytest.approx(3.5, abs=0.11)
    assert ev["false_alarms"] == 1
    assert ev["false_alarms_per_hour"] == pytest.approx(1 / (175 / 3600))


def test_missed_episode():
    ev = event_metrics(timeline([(0, 200, "ALERT")]), LABELS).summary()
    assert ev["event_recall"] == 0.0 and ev["latency_median_s"] is None


def test_subject_split_is_deterministic_and_leak_free():
    manifest = {"videos": [{"video": f"v{i}_{k}.mp4", "subject": f"S{i:02d}", "labels": "x.json"}
                           for i in range(40) for k in range(3)]}
    a, b = assign_splits(manifest), assign_splits(manifest)
    assert a == b
    for split in a.values():
        for item in split:
            same = [x for x in manifest["videos"] if x["subject"] == item["subject"]]
            assert all(x in split for x in same)  # all of a subject's videos in one split
    assert all(len(v) > 0 for v in a.values())
    assert split_for_subject("S01") == split_for_subject("S01")


def test_leakage_check_raises():
    with pytest.raises(ValueError):
        check_no_subject_leakage({"development": [{"subject": "A"}], "test": [{"subject": "A"}]})


def test_evaluate_script_refuses_test_split_without_flag(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("evaluate", Path(__file__).parents[2] / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "m.json").write_text(json.dumps({"videos": []}))
    assert mod.main(["--manifest", str(tmp_path / "m.json"), "--split", "test", "--predictions", str(tmp_path)]) == 2
