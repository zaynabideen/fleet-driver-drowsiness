"""Run the monitor headless over every video in one split, writing a timeline per video.

    python scripts/process_dataset.py --manifest data/manifest.json --split validation --out runs/val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drowsiness.cli import demo_main  # noqa: E402
from drowsiness.evaluation.dataset import assign_splits  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", choices=["development", "validation", "test"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--record-features", action="store_true", help="also save features for fast re-tuning replays")
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    items = assign_splits(json.loads(manifest_path.read_text(encoding="utf-8")))[args.split]
    out = Path(args.out)
    for item in items:
        video = manifest_path.parent / item["video"]
        stem = Path(item["video"]).stem
        argv = ["--source", str(video), "--headless", "--timeline", str(out / f"{stem}.jsonl"),
                "--log-dir", str(out / "logs" / stem)]
        if args.config:
            argv += ["--config", args.config]
        if args.record_features:
            argv.append("--record-features")
        print(f"[{args.split}] {video}")
        demo_main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
