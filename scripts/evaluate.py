"""Evaluate prediction timelines against labelled intervals for one split.

    # 1. make predictions (one timeline per video, named <video stem>.jsonl)
    python scripts/process_dataset.py --manifest data/manifest.json --split validation --out runs/val
    # 2. score them
    python scripts/evaluate.py --manifest data/manifest.json --split validation --predictions runs/val

The test split needs --i-am-not-tuning: look at it once, after thresholds are frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drowsiness.evaluation.dataset import assign_splits, load_intervals, load_timeline  # noqa: E402
from drowsiness.evaluation.metrics import DEFAULT_POSITIVE, EventMetrics, event_metrics, frame_metrics  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", choices=["development", "validation", "test"], required=True)
    p.add_argument("--predictions", required=True, help="directory of <video stem>.jsonl timelines")
    p.add_argument("--positive", default=",".join(sorted(DEFAULT_POSITIVE)),
                   help="comma-separated states counted as a positive (drowsy) prediction")
    p.add_argument("--tolerance", type=float, default=2.0, help="event matching tolerance in seconds")
    p.add_argument("--out", default=None, help="write the report as JSON here")
    p.add_argument("--i-am-not-tuning", action="store_true", help="required for the test split")
    args = p.parse_args(argv)

    if args.split == "test" and not args.i_am_not_tuning:
        print("Refusing to score the test split without --i-am-not-tuning. "
              "Tune on development, choose on validation, report test once.")
        return 2

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = assign_splits(manifest)[args.split]
    positive = frozenset(s.strip() for s in args.positive.split(",") if s.strip())

    frames = None
    events = EventMetrics(0, 0, [], 0, 0.0)
    missing = []
    for item in items:
        timeline = Path(args.predictions) / f"{Path(item['video']).stem}.jsonl"
        if not timeline.is_file():
            missing.append(str(timeline))
            continue
        preds = load_timeline(timeline)
        intervals = load_intervals(manifest_path.parent / item["labels"])
        frames = frame_metrics(preds, intervals, positive, acc=frames)
        ev = event_metrics(preds, intervals, positive, args.tolerance)
        events = EventMetrics(events.episodes + ev.episodes, events.detected + ev.detected,
                              events.latencies_s + ev.latencies_s, events.false_alarms + ev.false_alarms,
                              events.alert_hours + ev.alert_hours)

    if frames is None:
        print("No timelines found for this split.", *missing[:5], sep="\n  ")
        return 1
    report = {
        "split": args.split,
        "videos_scored": len(items) - len(missing),
        "videos_missing": len(missing),
        "subjects": sorted({str(i["subject"]) for i in items}),
        "positive_states": sorted(positive),
        "frame_level": frames.summary(),
        "event_level": events.summary(),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
