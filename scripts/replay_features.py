"""Replay a recorded feature stream (no video) through the decision layers.

Lets you change thresholds in a YAML config and re-run the analysis, risk
engine and state machine in seconds, without re-running computer vision.

    python scripts/replay_features.py logs/features.jsonl --config my.yaml --timeline runs/replay.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drowsiness.config.settings import load_settings  # noqa: E402
from drowsiness.pipeline.driver_monitor import DriverMonitor  # noqa: E402
from drowsiness.sources.frame_sources import read_features  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("features")
    p.add_argument("--config", default=None)
    p.add_argument("--timeline", default=None)
    args = p.parse_args()

    monitor = DriverMonitor(load_settings(args.config), alert_sinks=[])
    counts: Counter[str] = Counter()
    rows = []
    for f in read_features(args.features):
        out = monitor.process_features(f)
        counts[out.decision.state.value] += 1
        rows.append({"t": round(f.timestamp_s, 3), "state": out.decision.state.value,
                     "risk_score": out.risk.risk_score, "confidence": round(out.decision.confidence, 3)})
        if out.decision.changed:
            print(f"{f.timestamp_s:8.2f}s  {out.decision.previous_state.value:>22} -> {out.decision.state.value:<22}"
                  f"  {out.decision.reason}")
    if args.timeline:
        Path(args.timeline).parent.mkdir(parents=True, exist_ok=True)
        Path(args.timeline).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    total = sum(counts.values()) or 1
    print("\nTime in state:", ", ".join(f"{k} {v / total:.1%}" for k, v in counts.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
