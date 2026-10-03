"""Dataset manifests, label files and subject-wise splits.

Splits are made by SUBJECT (driver), never by frame or clip: frames from the
same person are highly correlated, and a driver seen during threshold tuning
must not appear in the test set. Assignment is a deterministic hash of the
subject ID, so a split is reproducible and stable when new subjects are added.

Manifest (JSON):
    {"videos": [{"video": "path.mp4", "subject": "S01", "labels": "labels/S01_a.json"}, ...]}

Label file (JSON):
    {"intervals": [{"start_s": 0, "end_s": 300, "label": "alert"},
                   {"start_s": 300, "end_s": 360, "label": "drowsy"},
                   {"start_s": 360, "end_s": 370, "label": "ignore"}]}
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from drowsiness.evaluation.metrics import Interval, Prediction

SPLITS = ("development", "validation", "test")


def split_for_subject(subject: str, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2), salt: str = "v1") -> str:
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    h = int(hashlib.sha256(f"{salt}:{subject}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0
    for name, frac in zip(SPLITS, fractions, strict=True):
        cumulative += frac
        if h <= cumulative:
            return name
    return SPLITS[-1]


def assign_splits(manifest: dict[str, Any], **kw: Any) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for item in manifest["videos"]:
        out[split_for_subject(str(item["subject"]), **kw)].append(item)
    check_no_subject_leakage(out)
    return out


def check_no_subject_leakage(splits: dict[str, list[dict[str, Any]]]) -> None:
    seen: dict[str, str] = {}
    for name, items in splits.items():
        for item in items:
            subj = str(item["subject"])
            if seen.setdefault(subj, name) != name:
                raise ValueError(f"subject {subj} appears in both {seen[subj]} and {name}")


def load_intervals(path: str | Path) -> list[Interval]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Interval(float(i["start_s"]), float(i["end_s"]), str(i["label"])) for i in data["intervals"]]


def load_timeline(path: str | Path) -> list[Prediction]:
    preds = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                preds.append(Prediction(float(row["t"]), str(row["state"])))
    return preds
