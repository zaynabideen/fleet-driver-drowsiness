"""Render the monitoring panel for a few synthetic states (for the README).

The 'face' is a synthetic geometric fixture, not a real driver, and the
states come from synthetic feature streams through the real pipeline.

    python scripts/render_ui_preview.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drowsiness.config.settings import Settings  # noqa: E402
from drowsiness.pipeline.driver_monitor import DriverMonitor  # noqa: E402
from drowsiness.presentation.overlay import render  # noqa: E402
from drowsiness.schemas.states import DriverState  # noqa: E402
from drowsiness.simulation.streams import StreamBuilder  # noqa: E402
from drowsiness.simulation.synthetic_face import synthetic_face_image, synthetic_result  # noqa: E402

OUT = ROOT / "docs" / "images"


def capture(builder: StreamBuilder, want: DriverState, ear: float, pitch: float, name: str,
            face: bool = True) -> None:
    monitor = DriverMonitor(Settings(), alert_sinks=[])
    hit = None
    for f in builder.frames:
        out = monitor.process_features(f)
        if out.decision.state == want:
            hit = out
    assert hit is not None, f"{name}: never reached {want}"
    res = synthetic_result(ear=ear, pitch_deg=pitch)
    image = synthetic_face_image(res, brightness=150)
    if not face:  # driver out of view: background only, no landmarks
        image = synthetic_face_image(synthetic_result(ear=ear, face_height_px=1), brightness=60)
    image = cv2.GaussianBlur(image, (0, 0), 3)  # smooth the noise texture: smaller PNGs for the README
    hit = replace(hit, landmarks=res if face else None)
    img = render(image, hit, monitor.alerts.recent_alerts, 29.7, 1.0)
    cv2.imwrite(str(OUT / f"{name}.png"), img)
    print("wrote", OUT / f"{name}.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = lambda: StreamBuilder().alert(25)  # noqa: E731
    capture(base().alert(5), DriverState.ALERT, 0.30, 0, "ui_alert")
    capture(base().alert(5).eyes_closed(1.7), DriverState.DROWSINESS_WARNING, 0.08, 0, "ui_warning")
    capture(base().alert(5).eyes_closed(0.3, pitch=0, pitch_to=25).eyes_closed(2.4, pitch=25),
            DriverState.CRITICAL_SLEEP_RISK, 0.08, 25, "ui_critical")
    capture(base().alert(5).no_face(3), DriverState.UNKNOWN, 0.30, 0, "ui_unknown", face=False)


if __name__ == "__main__":
    main()
