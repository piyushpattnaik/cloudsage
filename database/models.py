"""
CloudSage — Data Models
Typed schema definitions for Cosmos DB documents.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import uuid


@dataclass
class IncidentRecord:
    """Primary incident document stored in Cosmos DB."""
    service: str
    severity: str                         # P1 | P2 | P3 | P4
    description: str
    action: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # BUG 2 FIX (models side): timestamp is Optional[str] so callers can pass the real
    # incident start from the event payload. __post_init__ fills it with utcnow() when
    # no timestamp is provided, preserving backward-compat with existing call sites.
    timestamp: Optional[str] = None
    status: str = "open"                  # open | resolved | awaiting_approval
    agent: Optional[str] = None
    incident_confirmed: bool = True
    root_cause: Optional[str] = None
    affected_components: List[str] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)
    automation_result: Optional[Dict] = None
    policy_approved: bool = False
    resolved_at: Optional[str] = None
    mttr_minutes: Optional[float] = None
    reliability_impact_score: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Fill timestamp with utcnow() only when caller didn't provide one.
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    def resolve(self, resolution_note: str = "") -> None:
        """Mark incident as resolved and compute MTTR.

        FIXED: datetime.now(timezone.utc) returns tz-naive; if self.timestamp was stored with
        timezone info (e.g. from datetime.now(timezone.utc).isoformat()), subtracting
        them raised TypeError. Now both datetimes are normalised to UTC-naive before
        subtraction so MTTR is always computed correctly.
        """
        from datetime import timezone as _tz

        self.status = "resolved"
        self.resolved_at = datetime.now(timezone.utc).isoformat()

        def _to_naive_utc(ts_str: str) -> datetime:
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is not None:
                dt = dt.astimezone(_tz.utc).replace(tzinfo=None)
            return dt

        try:
            start = _to_naive_utc(self.timestamp)
            end = _to_naive_utc(self.resolved_at)
            self.mttr_minutes = round((end - start).total_seconds() / 60, 2)
        except (ValueError, TypeError):
            self.mttr_minutes = None

        if resolution_note:
            self.tags["resolution_note"] = resolution_note


@dataclass
class AIDecisionRecord:
    """Records every AI decision for audit and feedback loop."""
    service: str
    agent: str
    decision: Dict[str, Any]
    outcome: str = "pending"             # pending | success | failure
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    execution_time_ms: float = 0.0
    model_used: str = "gpt-4"
    confidence: Optional[float] = None
    feedback_score: Optional[int] = None  # -1 | 0 | 1 (thumbs down / neutral / up)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReliabilitySnapshot:
    """Periodic reliability scorecard snapshot."""
    service: str
    period: str                           # e.g., "2024-01" for monthly
    reliability_score: float              # 0-100
    mttr_minutes: float
    incident_count: int
    p1_count: int
    p2_count: int
    deployment_stability_index: float     # 0-1
    cost_efficiency_index: float          # 0-1
    security_risk_score: float            # 0-100
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    computed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)
