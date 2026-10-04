"""Command-line entry points.

    python demo/webcam_demo.py                          # live webcam with the monitoring panel
    python demo/webcam_demo.py --source driver.mp4      # a recorded driver video
    python demo/webcam_demo.py --source driver.mp4 --headless --timeline out/timeline.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from drowsiness.config.settings import Settings, load_settings
from drowsiness.events.event_log import configure_logging

log = logging.getLogger("drowsiness.cli")

RECORD_FPS = 20


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fleet driver drowsiness monitor")
    p.add_argument("--source", default="0", help="webcam index (e.g. 0) or path to a video file")
    p.add_argument("--config", default=None, help="YAML config overriding defaults")
    p.add_argument("--model", default="data/models/face_landmarker.task")
    p.add_argument("--log-dir", default=None, help="event log directory (default from config)")
    p.add_argument("--headless", action="store_true", help="no window; process as fast as possible")
    p.add_argument("--timeline", default=None, help="write a per-frame state timeline (JSONL) for evaluation")
    p.add_argument("--record-features", action="store_true", help="record numeric features (no pixels) for replay")
    p.add_argument("--save-snapshots", action="store_true", help="save a small still on HIGH/CRITICAL alerts")
    p.add_argument("--save-annotated", default=None,
                   help="OPT-IN: write the annotated video to this path (contains the driver's face)")
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the webcam image")
    p.add_argument("--no-landmarks", action="store_true", help="hide eye/mouth landmark dots")
    p.add_argument("--guided-demo", action="store_true",
                   help="show on-screen prompts for each behaviour, record to demo.mp4 (unless "
                        "--save-annotated is given) and stop automatically when the script ends")
    args = p.parse_args(argv)
    if args.guided_demo and not args.save_annotated:
        args.save_annotated = "demo.mp4"
    return args


def _settings(args: argparse.Namespace) -> Settings:
    overrides: dict = {}
    if args.record_features or args.save_snapshots:
        overrides["privacy"] = {
            "record_features": args.record_features,
            "save_snapshots": args.save_snapshots,
        }
    return load_settings(args.config, **overrides)


def demo_main(argv: list[str] | None = None) -> int:
    import cv2

    from drowsiness.detection.landmark_detector import MediaPipeFaceLandmarker, ModelNotFoundError
    from drowsiness.pipeline.driver_monitor import DriverMonitor
    from drowsiness.presentation.guided_demo import GuidedDemo, draw_prompt
    from drowsiness.presentation.overlay import PANEL_W, render
    from drowsiness.sources.frame_sources import VideoFileSource, WebcamSource

    args = _parse(argv)
    settings = _settings(args)
    log_dir = args.log_dir or settings.logging.directory
    configure_logging(log_dir, settings.logging.console_level)

    try:
        detector = MediaPipeFaceLandmarker(args.model)
    except ModelNotFoundError as exc:
        print(exc)
        return 2

    source = WebcamSource(int(args.source), mirror=not args.no_mirror) if args.source.isdigit() \
        else VideoFileSource(args.source)
    if not source.opened:
        print(f"Could not open camera {args.source}")
        return 2

    monitor = DriverMonitor(settings, detector=detector, log_dir=log_dir)
    timeline = None
    if args.timeline:
        Path(args.timeline).parent.mkdir(parents=True, exist_ok=True)
        timeline = open(args.timeline, "w", encoding="utf-8")
    writer = None
    guide = GuidedDemo() if args.guided_demo else None
    fps_est, last = None, time.perf_counter()
    log.info("Monitoring started (source=%s). Calibrating for ~%.0fs: look at the road normally.",
             args.source, settings.calibration.duration_s)
    try:
        while not source.finished:
            frame = source.read()
            if frame is None and source.finished:
                break
            out = monitor.process_frame(frame, source.now() if frame is None else frame.timestamp_s)
            if timeline:
                timeline.write(json.dumps({
                    "t": round(out.decision.timestamp_s, 3),
                    "state": out.decision.state.value,
                    "risk_score": out.risk.risk_score,
                    "confidence": round(out.decision.confidence, 3),
                }) + "\n")
            if args.headless or frame is None:
                if frame is None and not args.headless:
                    cv2.waitKey(30)
                continue

            now = time.perf_counter()
            inst = 1.0 / max(now - last, 1e-6)
            last = now
            fps_est = inst if fps_est is None else 0.9 * fps_est + 0.1 * inst
            canvas = render(frame.image, out, monitor.alerts.recent_alerts, fps_est,
                            monitor.analyzer.baseline.progress, show_landmarks=not args.no_landmarks)
            finished = False
            if guide is not None:
                prompt, expect, left, finished = guide.update(time.perf_counter(),
                                                              monitor.analyzer.baseline.calibrated)
                draw_prompt(canvas, canvas.shape[1] - PANEL_W, prompt, expect, left)
            if args.save_annotated:
                if writer is None:
                    writer = cv2.VideoWriter(args.save_annotated, cv2.VideoWriter_fourcc(*"mp4v"), RECORD_FPS,
                                             (canvas.shape[1], canvas.shape[0]))
                    rec_start, written = time.perf_counter(), 0
                # Pace by wall-clock time so the recording plays back in real time even if
                # processing runs faster or slower than RECORD_FPS (a 3 s eye closure stays 3 s).
                target = int((time.perf_counter() - rec_start) * RECORD_FPS) + 1
                while written < target:
                    writer.write(canvas)
                    written += 1
            cv2.imshow("Driver Monitoring", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or finished:
                break
    finally:
        source.close()
        monitor.close()
        if timeline:
            timeline.close()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
    log.info("Stopped. Events written to %s/events.jsonl", log_dir)
    if args.save_annotated:
        log.info("Video saved to %s", Path(args.save_annotated).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(demo_main())
