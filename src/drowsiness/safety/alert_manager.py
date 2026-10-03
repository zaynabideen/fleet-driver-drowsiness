"""Alert manager: decides WHEN to tell someone, and fans alerts out to sinks.

Rules:
  * Alertable states: the three drowsiness states (severity 1-3) and the two
    sensor states UNKNOWN / CAMERA_UNAVAILABLE (severity 0, "system").
  * Cooldowns are per alert TYPE. Entering an alertable state fires unless
    that same type fired within its cooldown, so a state that flickers
    WARNING/ALERT/WARNING does not spam the driver.
  * Escalation is never delayed by a less severe alert: CRITICAL has its own
    (short) cooldown, independent of any WARNING/HIGH alert just raised.
  * While a state persists, a reminder fires each time its cooldown elapses.

Sinks implement ``AlertSink``. Console/log and event-file sinks ship here;
the on-screen alert is drawn by the presentation layer from
``recent_alerts``. An audio or buzzer sink only needs an ``emit`` method.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Protocol

from drowsiness.config.settings import AlertSettings
from drowsiness.schemas.events import Alert, StateDecision
from drowsiness.schemas.states import DriverState

log = logging.getLogger("drowsiness.alerts")

SEVERITY = {
    DriverState.DROWSINESS_WARNING: 1,
    DriverState.HIGH_DROWSINESS_RISK: 2,
    DriverState.CRITICAL_SLEEP_RISK: 3,
    DriverState.UNKNOWN: 0,
    DriverState.CAMERA_UNAVAILABLE: 0,
}
ACTIONS = {
    DriverState.DROWSINESS_WARNING: "VISUAL_WARNING",
    DriverState.HIGH_DROWSINESS_RISK: "ALERT_DRIVER",
    DriverState.CRITICAL_SLEEP_RISK: "ALERT_DRIVER_URGENT_AND_NOTIFY_FLEET",
    DriverState.UNKNOWN: "CHECK_CAMERA_VIEW",
    DriverState.CAMERA_UNAVAILABLE: "REPORT_CAMERA_FAULT",
}


class AlertSink(Protocol):
    def emit(self, alert: Alert, decision: StateDecision) -> None: ...


class LoggingAlertSink:
    """Human-readable alert lines on the console / log file."""

    def emit(self, alert: Alert, decision: StateDecision) -> None:
        level = logging.WARNING if alert.severity >= 1 else logging.INFO
        log.log(level, "ALERT %s (severity %d, confidence %.2f) action=%s | %s",
                alert.alert_type.value, alert.severity, alert.confidence, alert.action, alert.reason)


class AlertManager:
    def __init__(self, settings: AlertSettings, sinks: list[AlertSink] | None = None, history: int = 10) -> None:
        self._s = settings
        self._sinks = list(sinks or [])
        self._last_fired: dict[DriverState, float] = {}
        self.recent_alerts: deque[Alert] = deque(maxlen=history)

    def add_sink(self, sink: AlertSink) -> None:
        self._sinks.append(sink)

    def process(self, d: StateDecision) -> Alert | None:
        if d.state not in SEVERITY:
            return None
        sev = SEVERITY[d.state]
        cooldown = self._s.cooldown_s.get(d.state.value, 30.0)
        last = self._last_fired.get(d.state)
        if last is not None and d.timestamp_s - last < cooldown:
            return None

        evidence = tuple(e.description for e in d.risk.active_evidence) if d.risk else ()
        alert = Alert(
            alert_type=d.state,
            severity=sev,
            timestamp_s=d.timestamp_s,
            reason=d.reason,
            confidence=d.confidence,
            evidence=evidence,
            cooldown_s=cooldown,
            action=ACTIONS[d.state],
        )
        self._last_fired[d.state] = d.timestamp_s
        self.recent_alerts.append(alert)
        for sink in self._sinks:
            sink.emit(alert, d)
        return alert
