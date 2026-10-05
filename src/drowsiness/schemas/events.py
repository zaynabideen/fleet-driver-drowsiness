"""Decisions, alerts and logged events.

Every decision carries: state, confidence, evidence, timestamp, duration and
a human-readable reason.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from drowsiness.schemas.states import DriverState, RiskLevel, SignalCategory


@dataclass(frozen=True, slots=True)
class Evidence:
    """One indicator that contributed to a risk decision."""

    code: str  # machine-readable, e.g. "prolonged_eye_closure"
    category: SignalCategory
    severity: int  # 0..3
    value: float
    description: str  # human-readable, e.g. "eye closure 2.1 s (ongoing)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "category": self.category.value,
            "severity": self.severity,
            "value": round(self.value, 4),
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    timestamp_s: float
    level: RiskLevel
    risk_score: float  # 0..1, consistent with level bands
    confidence: float  # how sure we are of the observations (NOT how risky)
    evidence: tuple[Evidence, ...]
    rule: str  # which rule produced the level
    observable: bool  # False -> risk could not be assessed

    @property
    def active_evidence(self) -> tuple[Evidence, ...]:
        return tuple(e for e in self.evidence if e.severity > 0)


@dataclass(frozen=True, slots=True)
class StateDecision:
    timestamp_s: float
    state: DriverState
    previous_state: DriverState
    changed: bool
    state_since_s: float
    confidence: float
    risk: RiskAssessment | None
    reason: str
    recovery_progress: float | None = None  # 0..1 while counting down to the next lower level

    @property
    def duration_s(self) -> float:
        return self.timestamp_s - self.state_since_s


@dataclass(frozen=True, slots=True)
class Alert:
    alert_type: DriverState
    severity: int  # 0 system, 1 warning, 2 high, 3 critical
    timestamp_s: float
    reason: str
    confidence: float
    evidence: tuple[str, ...]
    cooldown_s: float
    action: str
    alert_id: str = field(default_factory=lambda: uuid.uuid4().hex)


def iso_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")
