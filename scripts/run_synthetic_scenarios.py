"""Run the behavioural scenarios and print what the system decided, and how fast.

SYNTHETIC: these are generated feature streams through the real decision
layers. They show the logic behaves as designed and measure decision
latency *from the onset of a simulated behaviour*. They are not
real-world accuracy and must not be reported as such.

    python scripts/run_synthetic_scenarios.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drowsiness.config.settings import Settings  # noqa: E402
from drowsiness.pipeline.driver_monitor import DriverMonitor  # noqa: E402
from drowsiness.schemas.states import DriverState  # noqa: E402
from drowsiness.simulation.streams import SUNGLASSES, StreamBuilder  # noqa: E402

CAL = 25.0
ONSET = CAL + 5.0


def scenarios() -> list[tuple[str, str, StreamBuilder]]:
    def base() -> StreamBuilder:
        return StreamBuilder().alert(CAL).alert(5)

    nod = base()
    for _ in range(3):
        nod.eyes_closed(0.3, pitch=0, pitch_to=22).eyes_closed(1.0, pitch=22).alert(0.5, pitch=22, pitch_to=0).alert(4)
    glance = base().segment(0.3, ear=0.17, pitch=0, pitch_to=25).segment(0.6, ear=0.17, pitch=25).alert(10)
    return [
        ("1 Normal blinking", "ALERT", base().alert(60)),
        ("2 Long eye closure (1.6 s)", "DROWSINESS_WARNING", base().eyes_closed(1.6).alert(5)),
        ("3 Repeated long closures + nods", "HIGH_DROWSINESS_RISK", nod),
        ("4 Closure + head drop", "CRITICAL_SLEEP_RISK",
         base().eyes_closed(0.3, pitch=0, pitch_to=25).eyes_closed(3.0, pitch=25)),
        ("5 Face disappears", "UNKNOWN", base().no_face(5)),
        ("6 Camera fails", "CAMERA_UNAVAILABLE", base().camera_down(4)),
        ("7 Brief look down", "ALERT", glance),
        ("8 Single yawn", "ALERT", base().segment(4.0, mar=0.8).alert(10)),
        ("  Microsleep, head still (3.5 s)", "CRITICAL_SLEEP_RISK", base().eyes_closed(3.5)),
        ("  Sunglasses", "MONITORING", base().segment(20, quality=SUNGLASSES)),
        ("  Talking", "ALERT", base().segment(15, mar=0.3)),
    ]


def worst(states: list[DriverState]) -> DriverState:
    elevated = [s for s in states if s.is_elevated]
    if elevated:
        return max(elevated, key=lambda s: s.risk_rank)
    return states[-1]


def main() -> None:
    print("SYNTHETIC scenario results (decision logic only - not real-world accuracy)\n")
    print(f"{'Scenario':36} {'Expected':22} {'Got':22} {'Onset->state':>12}  OK")
    print("-" * 100)
    per_frame: list[float] = []
    for name, expected, builder in scenarios():
        monitor = DriverMonitor(Settings(), alert_sinks=[])
        got, first = [], None
        for f in builder.frames:
            t0 = time.perf_counter()
            out = monitor.process_features(f)
            per_frame.append(time.perf_counter() - t0)
            if f.timestamp_s >= ONSET:
                got.append(out.decision.state)
                if first is None and out.decision.state.value == expected and out.decision.changed:
                    first = f.timestamp_s - ONSET
        final = worst(got)
        lat = "-" if first is None else f"{first:.2f}s"
        ok = "yes" if final.value == expected else "NO"
        print(f"{name:36} {expected:22} {final.value:22} {lat:>12}  {ok}")
    per_frame.sort()
    print(f"\nDecision layers (analysis + risk + state machine + alerts): "
          f"median {per_frame[len(per_frame) // 2] * 1e3:.3f} ms/frame, "
          f"p99 {per_frame[int(len(per_frame) * 0.99)] * 1e3:.3f} ms/frame on this machine.")


if __name__ == "__main__":
    main()
