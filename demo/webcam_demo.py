"""Live driver-monitoring demo.

    python scripts/download_model.py        # once
    python demo/webcam_demo.py              # webcam 0
    python demo/webcam_demo.py --source my_drive.mp4
    python demo/webcam_demo.py --help

Look at the screen normally for the first ~20 s while the system calibrates
to your eyes and head position. Press q or Esc to quit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drowsiness.cli import demo_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(demo_main())
