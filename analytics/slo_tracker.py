"""
CloudSage — SLO Error Budget Tracker
=====================================================
Drives ALL automation decisions from error budget burn rate — the same
abstraction used by Google SRE and Netflix.

Why this matters:
  Static severity (P1/P2/P3) is an approximation of business impact.
  Error budget burn rate IS business impact — it's how fast you're
  consuming your customers' tolerance for downtime.

Model:
  - Each service has an SLO target (e.g., 99.9% availability = 43.8 min/month budget)
  - Burn rate = how fast you're consuming that budget right now
  - 1x burn rate = will exhaust budget exactly at end of month
  - 14x burn rate = will exhaust budget in 2 days (Google's "fast burn" alert)

Policy integration:
  - Error budget > 50% remaining → normal automation policy
  - Error budget < 50% → tighten approval requirements
  - Error budget < 20% → manual approval required for all actions
  - Error budget exhausted → immediate escalation, freeze deployments
"""

import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("CloudSage.SLOTracker")


# ---------------------------------------------------------------------------
# SLO Definitions
# ---------------------------------------------------------------------------

@dataclass
class SLODefinition:
    service: str
    slo_target_pct: float          # e.g., 99.9
    window_days: int = 30          # rolling window
    # Burn rate thresholds (multiples of 1x)
    fast_burn_threshold: float = 14.4   # exhausts budget in 2 days
    slow_burn_threshold: float = 6.0    # exhausts budget in 5 days


@dataclass
class ErrorBudgetStatus:
    service: str
    slo_target_pct: float
    window_days: int

    # Budget
    total_budget_minutes: float
    consumed_budget_minutes: float
    remaining_budget_minutes: float
    remaining_budget_pct: float

    # Burn rate
    current_burn_rate: float          # multiple of 1x sustainable rate
    burn_rate_classification: str     # "normal" | "slow_burn" | "fast_burn" | "exhausted"

    # Policy recommendation
    policy_tier: str                  # "standard" | "conservative" | "restricted" | "freeze"
    requires_human_approval: bool
    deployment_freeze_recommended: bool
    time_to_budget_exhaustion_hours: Optional[float]

    # Alert
    alert_message: str


# ---------------------------------------------------------------------------
# SLO Tracker
# ---------------------------------------------------------------------------

class SLOErrorBudgetTracker:
    """
    Tracks error budget consumption and burn rate per service.
    Integrates with PolicyEngine to tighten automation guardrails
    when budget is burning fast.
    """

    def __init__(self, slo_definitions: dict = None):
        """
        Args:
            slo_definitions: dict of service_name -> SLODefinition config dict
                             Loaded from config.json["slo_definitions"]
        """
        self._definitions: dict = {}
        self._incident_log: dict = {}  # service -> list of (timestamp, duration_minutes)

        if slo_definitions:
            for service, cfg in slo_definitions.items():
                self._definitions[service] = SLODefinition(
                    service=service,
                    slo_target_pct=cfg.get("slo_target_pct") or cfg.get("target_pct", 99.9),
                    window_days=cfg.get("window_days", 30),
                    fast_burn_threshold=cfg.get("fast_burn_threshold") or cfg.get("fast_burn_multiplier", 14.4),
                    slow_burn_threshold=cfg.get("slow_burn_threshold") or cfg.get("slow_burn_multiplier", 6.0),
                )

    def _get_definition(self, service: str) -> SLODefinition:
        return self._definitions.get(service, SLODefinition(service=service, slo_target_pct=99.9))

    def record_downtime(self, service: str, duration_minutes: float, timestamp: str = None):
        """Record an outage/degradation against the error budget."""
        if service not in self._incident_log:
            self._incident_log[service] = []
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        self._incident_log[service].append((ts, duration_minutes))
        logger.info(f"Recorded downtime: service={service} duration={duration_minutes}m")

    def compute_status(
        self,
        service: str,
        recent_incidents: list = None,
        current_error_rate_pct: float = 0.0,
    ) -> ErrorBudgetStatus:
        """
        Compute current error budget status.

        Args:
            service: Service name
            recent_incidents: list of resolved incident dicts from Cosmos
                              (with mttr_minutes and timestamp fields)
            current_error_rate_pct: Current error rate (0-100) for burn rate calc
        """
        defn = self._get_definition(service)
        window_minutes = defn.window_days * 24 * 60

        # Total error budget = (1 - SLO) × window
        error_budget_fraction = 1 - (defn.slo_target_pct / 100)
        total_budget_minutes = window_minutes * error_budget_fraction

        # Consumed budget from resolved incidents
        consumed = 0.0
        if recent_incidents:
            now = datetime.now(timezone.utc)
            window_start = now.timestamp() - (defn.window_days * 86400)
            for inc in recent_incidents:
                mttr = inc.get("mttr_minutes") or 0
                try:
                    ts = datetime.fromisoformat(inc.get("timestamp", ""))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts.timestamp() >= window_start:
                        consumed += mttr
                except (ValueError, TypeError):
                    consumed += mttr

        # Also add from local log
        for ts_str, dur in self._incident_log.get(service, []):
            consumed += dur

        remaining = max(0, total_budget_minutes - consumed)
        remaining_pct = (remaining / total_budget_minutes * 100) if total_budget_minutes > 0 else 0

        # Current burn rate (based on current error rate)
        # Sustainable rate = error_budget_fraction minutes of downtime per minute
        # Current rate = error_rate_pct / 100 minutes of degradation per minute
        sustainable_burn_per_minute = error_budget_fraction
        if current_error_rate_pct > 0:
            current_burn_per_minute = (current_error_rate_pct / 100)
            burn_rate = current_burn_per_minute / max(sustainable_burn_per_minute, 1e-9)
        else:
            # Estimate burn rate from recent consumption trend
            if consumed > 0 and defn.window_days > 0:
                burn_rate = (consumed / (defn.window_days * 24 * 60)) / max(sustainable_burn_per_minute, 1e-9)
            else:
                burn_rate = 0.0

        # Classify burn rate
        if remaining <= 0:
            classification = "exhausted"
        elif burn_rate >= defn.fast_burn_threshold:
            classification = "fast_burn"
        elif burn_rate >= defn.slow_burn_threshold:
            classification = "slow_burn"
        else:
            classification = "normal"

        # Time to exhaustion
        if burn_rate > 1.0 and remaining > 0:
            minutes_to_exhaustion = remaining / (burn_rate * sustainable_burn_per_minute)
            hours_to_exhaustion = round(minutes_to_exhaustion / 60, 1)
        else:
            hours_to_exhaustion = None

        # Policy tier
        if remaining_pct <= 0 or classification == "exhausted":
            policy_tier = "freeze"
            requires_approval = True
            deployment_freeze = True
            alert = (
                f"🔴 {service}: Error budget EXHAUSTED. "
                f"All automation requires human approval. Freeze new deployments."
            )
        elif remaining_pct <= 20 or classification == "fast_burn":
            policy_tier = "restricted"
            requires_approval = True
            deployment_freeze = True
            alert = (
                f"🔴 {service}: Error budget critically low ({remaining_pct:.1f}% remaining). "
                f"Fast burn rate {burn_rate:.1f}x. Manual approval required for all actions."
            )
        elif remaining_pct <= 50 or classification == "slow_burn":
            policy_tier = "conservative"
            requires_approval = False
            deployment_freeze = False
            alert = (
                f"🟡 {service}: Error budget at {remaining_pct:.1f}%. "
                f"Burn rate {burn_rate:.1f}x. Tightened automation policy active."
            )
        else:
            policy_tier = "standard"
            requires_approval = False
            deployment_freeze = False
            alert = (
                f"🟢 {service}: Error budget healthy ({remaining_pct:.1f}% remaining). "
                f"Standard automation policy active."
            )

        status = ErrorBudgetStatus(
            service=service,
            slo_target_pct=defn.slo_target_pct,
            window_days=defn.window_days,
            total_budget_minutes=round(total_budget_minutes, 2),
            consumed_budget_minutes=round(consumed, 2),
            remaining_budget_minutes=round(remaining, 2),
            remaining_budget_pct=round(remaining_pct, 1),
            current_burn_rate=round(burn_rate, 2),
            burn_rate_classification=classification,
            policy_tier=policy_tier,
            requires_human_approval=requires_approval,
            deployment_freeze_recommended=deployment_freeze,
            time_to_budget_exhaustion_hours=hours_to_exhaustion,
            alert_message=alert,
        )

        logger.info(
            f"SLO status: service={service} remaining={remaining_pct:.1f}% "
            f"burn_rate={burn_rate:.1f}x tier={policy_tier}"
        )
        return status

    def to_dict(self, status: ErrorBudgetStatus) -> dict:
        return {
            "service": status.service,
            "slo_target_pct": status.slo_target_pct,
            "window_days": status.window_days,
            "budget": {
                "total_minutes": status.total_budget_minutes,
                "consumed_minutes": status.consumed_budget_minutes,
                "remaining_minutes": status.remaining_budget_minutes,
                "remaining_pct": status.remaining_budget_pct,
            },
            "burn_rate": {
                "current_multiplier": status.current_burn_rate,
                "classification": status.burn_rate_classification,
                "time_to_exhaustion_hours": status.time_to_budget_exhaustion_hours,
            },
            "policy": {
                "tier": status.policy_tier,
                "requires_human_approval": status.requires_human_approval,
                "deployment_freeze_recommended": status.deployment_freeze_recommended,
            },
            "alert_message": status.alert_message,
        }


# Shared singleton
_shared_tracker = SLOErrorBudgetTracker()


def get_slo_tracker() -> SLOErrorBudgetTracker:
    return _shared_tracker
