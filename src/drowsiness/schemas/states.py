"""Enumerations shared by every layer."""

from __future__ import annotations

from enum import Enum


class DriverState(str, Enum):
    """Final safety state reported by the state machine.

    ``UNKNOWN`` and ``CAMERA_UNAVAILABLE`` are deliberately separate from
    ``ALERT``: not being able to see the driver is never reported as safe.
    """

    ALERT = "ALERT"
    MONITORING = "MONITORING"  # observing, but not confident enough to claim ALERT
    DROWSINESS_WARNING = "DROWSINESS_WARNING"
    HIGH_DROWSINESS_RISK = "HIGH_DROWSINESS_RISK"
    CRITICAL_SLEEP_RISK = "CRITICAL_SLEEP_RISK"
    UNKNOWN = "UNKNOWN"
    CAMERA_UNAVAILABLE = "CAMERA_UNAVAILABLE"

    @property
    def risk_rank(self) -> int:
        """Ordering of the drowsiness states; non-drowsiness states rank 0."""
        return _RANK.get(self, 0)

    @property
    def is_elevated(self) -> bool:
        return self.risk_rank >= 1


_RANK = {
    DriverState.DROWSINESS_WARNING: 1,
    DriverState.HIGH_DROWSINESS_RISK: 2,
    DriverState.CRITICAL_SLEEP_RISK: 3,
}

ELEVATED_LADDER: tuple[DriverState, ...] = (
    DriverState.ALERT,
    DriverState.DROWSINESS_WARNING,
    DriverState.HIGH_DROWSINESS_RISK,
    DriverState.CRITICAL_SLEEP_RISK,
)


class RiskLevel(int, Enum):
    LOW = 0
    MODERATE = 1
    HIGH = 2
    CRITICAL = 3

    def to_state(self) -> DriverState:
        return ELEVATED_LADDER[self.value]


class EyeState(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    UNOBSERVABLE = "UNOBSERVABLE"  # cannot be measured: never assumed open


class ObservationStatus(str, Enum):
    OBSERVED = "OBSERVED"  # face and eyes measurable
    DEGRADED = "DEGRADED"  # face visible but eyes not measurable (sunglasses, head turned...)
    FACE_LOST = "FACE_LOST"
    CAMERA_UNAVAILABLE = "CAMERA_UNAVAILABLE"


class SignalCategory(str, Enum):
    """Independent evidence families. Corroboration means different categories agree."""

    EYES = "eyes"
    HEAD = "head"
    MOUTH = "mouth"
    BLINK = "blink"
