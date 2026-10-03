"""Evaluation metrics against labelled driver video.

Inputs
  * predictions: per-frame timeline rows {"t": seconds, "state": "..."} (written by --timeline)
  * labels:      intervals [{"start_s", "end_s", "label"}] with label "drowsy", "alert" or "ignore"

Two complementary views:

FRAME LEVEL - each prediction frame is compared with the label at its time.
  Positive prediction = state in ``positive_states`` (default HIGH + CRITICAL).
  UNKNOWN / CAMERA_UNAVAILABLE are *abstentions*: they are reported as
  coverage, and recall is given twice - on covered frames, and conservatively
  with abstentions counted as misses (a system that hides in UNKNOWN must not
  look good).

EVENT LEVEL - what matters operationally.
  * event recall: fraction of labelled drowsy episodes with a positive
    prediction between (start - tolerance) and (end + tolerance)
  * detection latency: first positive prediction minus episode start
  * false alarms per hour: predicted positive episodes that overlap no
    labelled drowsy episode (with tolerance), per hour of labelled-alert driving

Nothing here produces a number unless real labelled data is supplied.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

ABSTAIN_STATES = frozenset({"UNKNOWN", "CAMERA_UNAVAILABLE"})
DEFAULT_POSITIVE = frozenset({"HIGH_DROWSINESS_RISK", "CRITICAL_SLEEP_RISK"})


@dataclass(frozen=True)
class Interval:
    start_s: float
    end_s: float
    label: str  # "drowsy" | "alert" | "ignore"


@dataclass(frozen=True)
class Prediction:
    t: float
    state: str


@dataclass
class FrameMetrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    abstain_pos: int = 0
    abstain_neg: int = 0

    def _div(self, a: float, b: float) -> float | None:
        return a / b if b else None

    @property
    def covered(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def total(self) -> int:
        return self.covered + self.abstain_pos + self.abstain_neg

    def summary(self) -> dict[str, float | int | None]:
        precision = self._div(self.tp, self.tp + self.fp)
        recall = self._div(self.tp, self.tp + self.fn)
        f1 = None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        return {
            "frames_total": self.total,
            "coverage": self._div(self.covered, self.total),
            "accuracy": self._div(self.tp + self.tn, self.covered),
            "precision": precision,
            "recall": recall,
            "recall_conservative": self._div(self.tp, self.tp + self.fn + self.abstain_pos),
            "f1": f1,
            "false_positive_rate": self._div(self.fp, self.fp + self.tn),
            "false_negative_rate": self._div(self.fn, self.fn + self.tp),
            **{k: v for k, v in asdict(self).items()},
        }


@dataclass
class EventMetrics:
    episodes: int
    detected: int
    latencies_s: list[float]
    false_alarms: int
    alert_hours: float

    def summary(self) -> dict[str, float | int | None]:
        lat = sorted(self.latencies_s)
        return {
            "drowsy_episodes": self.episodes,
            "detected_episodes": self.detected,
            "event_recall": self.detected / self.episodes if self.episodes else None,
            "latency_median_s": statistics.median(lat) if lat else None,
            "latency_p90_s": lat[min(len(lat) - 1, int(round(0.9 * (len(lat) - 1))))] if lat else None,
            "false_alarms": self.false_alarms,
            "labelled_alert_hours": round(self.alert_hours, 4),
            "false_alarms_per_hour": self.false_alarms / self.alert_hours if self.alert_hours > 0 else None,
        }


def label_at(t: float, intervals: Sequence[Interval], default: str = "ignore") -> str:
    for iv in intervals:
        if iv.start_s <= t < iv.end_s:
            return iv.label
    return default


def frame_metrics(preds: Iterable[Prediction], intervals: Sequence[Interval],
                  positive_states: frozenset[str] = DEFAULT_POSITIVE, unlabelled: str = "ignore",
                  acc: FrameMetrics | None = None) -> FrameMetrics:
    m = acc or FrameMetrics()
    for p in preds:
        truth = label_at(p.t, intervals, unlabelled)
        if truth == "ignore":
            continue
        positive_truth = truth == "drowsy"
        if p.state in ABSTAIN_STATES:
            if positive_truth:
                m.abstain_pos += 1
            else:
                m.abstain_neg += 1
            continue
        positive_pred = p.state in positive_states
        if positive_pred and positive_truth:
            m.tp += 1
        elif positive_pred:
            m.fp += 1
        elif positive_truth:
            m.fn += 1
        else:
            m.tn += 1
    return m


def predicted_episodes(preds: Sequence[Prediction], positive_states: frozenset[str]) -> list[tuple[float, float]]:
    episodes: list[tuple[float, float]] = []
    start = prev = None
    for p in preds:
        if p.state in positive_states:
            start = p.t if start is None else start
            prev = p.t
        elif start is not None:
            episodes.append((start, prev if prev is not None else start))
            start = prev = None
    if start is not None:
        episodes.append((start, prev if prev is not None else start))
    return episodes


def event_metrics(preds: Sequence[Prediction], intervals: Sequence[Interval],
                  positive_states: frozenset[str] = DEFAULT_POSITIVE, tolerance_s: float = 2.0) -> EventMetrics:
    preds = sorted(preds, key=lambda p: p.t)
    drowsy = [iv for iv in intervals if iv.label == "drowsy"]
    positives = [p.t for p in preds if p.state in positive_states]
    detected, latencies = 0, []
    for iv in drowsy:
        hits = [t for t in positives if iv.start_s - tolerance_s <= t <= iv.end_s + tolerance_s]
        if hits:
            detected += 1
            latencies.append(max(0.0, hits[0] - iv.start_s))
    false_alarms = 0
    for start, end in predicted_episodes(preds, positive_states):
        overlaps = any(start <= iv.end_s + tolerance_s and end >= iv.start_s - tolerance_s for iv in drowsy)
        if not overlaps and label_at(start, intervals) != "ignore":
            false_alarms += 1
    alert_hours = sum(iv.end_s - iv.start_s for iv in intervals if iv.label == "alert") / 3600.0
    return EventMetrics(len(drowsy), detected, latencies, false_alarms, alert_hours)
