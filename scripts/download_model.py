"""Download the MediaPipe Face Landmarker model (Apache-2.0, ~3.6 MB).

    python scripts/download_model.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drowsiness.detection.landmark_detector import MODEL_URL  # noqa: E402

DEST = ROOT / "data" / "models" / "face_landmarker.task"


def main() -> int:
    if DEST.is_file() and DEST.stat().st_size > 1_000_000:
        print(f"Model already present: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_URL}")
    tmp = DEST.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.replace(DEST)
    print(f"Saved to {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
