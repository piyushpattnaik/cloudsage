"""
CloudSage — Economic Impact Model
=====================================================
Transforms CloudSage from an "ops tool" into a "P&L tool".

Every incident gets a live running dollar cost:
  - Revenue at risk per minute = transaction_rate × avg_order_value × error_impact
  - Infrastructure waste per minute = idle compute during incident
  - Projected savings from automation = manual MTTR - automated MTTR

Why this matters:
  "We resolved the incident" is a technical statement.
  "We prevented $47,200 in lost revenue in 2.3 minutes" is a business statement.
  Award judges and executives speak the second language.

Configuration:
  Set service revenue models in config.json under "revenue_models":
  {
    "payment-api": {
      "avg_transactions_per_minute": 1200,
      "avg_order_value_usd": 85,
      "error_impact_factor": 0.7   # fraction of transactions lost per % error rate
    }
  }
  Defaults are used when service-specific config is absent.
"""

import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("CloudSage.EconomicImpact")


# ---------------------------------------------------------------------------
# Default revenue model (used when service config is absent)
# ---------------------------------------------------------------------------
DEFAULT_REVENUE_MODEL = {
    "avg_transactions_per_minute": 100,
    "avg_order_value_usd": 50,
    "error_impact_factor": 0.6,
    "infrastructure_cost_per_minute_usd": 2.5,
}

# Average human SRE MTTR (minutes) by severity — used to compute savings
HUMAN_MTTR_BENCHMARKS = {
    "P1": 45,
    "P2": 90,
    "P3": 240,
    "P4": 1440,
}


@dataclass
class EconomicImpact:
    service: str
    severity: str
    incident_start: str

    # Revenue impact
    revenue_at_risk_per_minute_usd: float
    error_rate_pct: float
    effective_revenue_loss_per_minute_usd: float

    # Infrastructure waste
    infrastructure_waste_per_minute_usd: float

    # Total burn
    total_cost_per_minute_usd: float

    # Automation savings (vs. human MTTR)
    human_mttr_minutes: float
    automated_mttr_minutes: Optional[float]
    revenue_saved_by_automation_usd: Optional[float]
    mttr_improvement_pct: Optional[float]

    # Running totals (updated as incident progresses)
    elapsed_minutes: float
    running_revenue_loss_usd: float
    running_total_cost_usd: float


class EconomicImpactModel:
    """
    Computes real-time economic impact of incidents and quantifies
    the financial value of automated resolution.
    """

    def __init__(self, revenue_models: dict = None):
        """
        Args:
            revenue_models: dict of service_name -> revenue model config
                            Usually loaded from config.json["revenue_models"]
        """
        self._models = revenue_models or {}

    def _get_model(self, service: str) -> dict:
        return self._models.get(service, DEFAULT_REVENUE_MODEL)

    def compute_impact(
        self,
        service: str,
        severity: str,
        error_rate_pct: float,
        incident_start_iso: str = None,
        automated_mttr_minutes: float = None,
    ) -> EconomicImpact:
        """
        Compute economic impact for an active incident.

        Args:
            service: Service name
            severity: P1-P4
            error_rate_pct: Current error rate (0-100)
            incident_start_iso: ISO timestamp when incident started
            automated_mttr_minutes: Actual MTTR if already resolved, else None
        """
        model = self._get_model(service)

        # Revenue impact per minute
        txn_per_min = model.get("avg_transactions_per_minute", 100)
        aov = model.get("avg_order_value_usd", 50)
        impact_factor = model.get("error_impact_factor", 0.6)
        infra_cost = model.get("infrastructure_cost_per_minute_usd", 2.5)

        revenue_at_risk = txn_per_min * aov
        error_fraction = min(1.0, (error_rate_pct / 100) * impact_factor)
        revenue_loss_per_min = revenue_at_risk * error_fraction
        total_per_min = revenue_loss_per_min + infra_cost

        # Elapsed time
        if incident_start_iso:
            try:
                start = datetime.fromisoformat(incident_start_iso)
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                elapsed_minutes = (datetime.now(timezone.utc) - start).total_seconds() / 60
            except (ValueError, TypeError):
                elapsed_minutes = 0.0
        else:
            elapsed_minutes = 0.0

        running_revenue_loss = revenue_loss_per_min * elapsed_minutes
        running_total = total_per_min * elapsed_minutes

        # Automation savings
        human_mttr = HUMAN_MTTR_BENCHMARKS.get(severity, 60)
        revenue_saved = None
        mttr_improvement = None

        if automated_mttr_minutes is not None:
            time_saved = max(0, human_mttr - automated_mttr_minutes)
            revenue_saved = revenue_loss_per_min * time_saved
            mttr_improvement = (
                round((time_saved / human_mttr) * 100, 1) if human_mttr > 0 else 0
            )

        impact = EconomicImpact(
            service=service,
            severity=severity,
            incident_start=incident_start_iso or datetime.now(timezone.utc).isoformat(),
            revenue_at_risk_per_minute_usd=round(revenue_at_risk, 2),
            error_rate_pct=round(error_rate_pct, 2),
            effective_revenue_loss_per_minute_usd=round(revenue_loss_per_min, 2),
            infrastructure_waste_per_minute_usd=round(infra_cost, 2),
            total_cost_per_minute_usd=round(total_per_min, 2),
            human_mttr_minutes=human_mttr,
            automated_mttr_minutes=automated_mttr_minutes,
            revenue_saved_by_automation_usd=round(revenue_saved, 2) if revenue_saved is not None else None,
            mttr_improvement_pct=mttr_improvement,
            elapsed_minutes=round(elapsed_minutes, 2),
            running_revenue_loss_usd=round(running_revenue_loss, 2),
            running_total_cost_usd=round(running_total, 2),
        )

        savings_str = f"${revenue_saved:.2f}" if revenue_saved is not None else "N/A"
        logger.info(
            f"Economic impact: service={service} severity={severity} "
            f"cost_per_min=${total_per_min:.2f} "
            f"running_total=${running_total:.2f} "
            f"savings={savings_str}"
        )
        return impact

    def to_dict(self, impact: EconomicImpact) -> dict:
        return {
            "service": impact.service,
            "severity": impact.severity,
            "revenue_at_risk_per_minute_usd": impact.revenue_at_risk_per_minute_usd,
            "error_rate_pct": impact.error_rate_pct,
            "effective_revenue_loss_per_minute_usd": impact.effective_revenue_loss_per_minute_usd,
            "infrastructure_waste_per_minute_usd": impact.infrastructure_waste_per_minute_usd,
            "total_cost_per_minute_usd": impact.total_cost_per_minute_usd,
            "human_p50_mttr_minutes": impact.human_mttr_minutes,
            "automated_mttr_minutes": impact.automated_mttr_minutes,
            "revenue_saved_by_automation_usd": impact.revenue_saved_by_automation_usd,
            "mttr_improvement_pct": impact.mttr_improvement_pct,
            "elapsed_minutes": impact.elapsed_minutes,
            "running_revenue_loss_usd": impact.running_revenue_loss_usd,
            "running_total_cost_usd": impact.running_total_cost_usd,
            "headline": self._headline(impact),
        }

    @staticmethod
    def _headline(impact: EconomicImpact) -> str:
        """One-line human-readable summary for the dashboard."""
        if impact.revenue_saved_by_automation_usd is not None:
            return (
                f"Automation prevented ${impact.revenue_saved_by_automation_usd:,.0f} "
                f"in lost revenue vs. human P50 MTTR "
                f"({impact.mttr_improvement_pct:.0f}% faster)"
            )
        return (
            f"Incident costing ${impact.total_cost_per_minute_usd:,.2f}/min "
            f"(${impact.running_total_cost_usd:,.0f} so far)"
        )

    def compute_cumulative_savings(self, resolved_incidents: list) -> dict:
        """
        Compute total automation savings across all resolved incidents.
        Used for the 'Total Value Delivered' KPI on the dashboard.
        """
        total_saved = 0
        total_incidents = len(resolved_incidents)
        p1_saved = 0
        by_service = {}

        for inc in resolved_incidents:
            automated_mttr = inc.get("mttr_minutes")
            severity = inc.get("severity", "P4")
            service = inc.get("service", "unknown")
            # Tags store error_rate (persisted by orchestrator); metrics dict is NOT in Cosmos.
            _raw_rate = inc.get("tags", {}).get("error_rate", "")
            try:
                error_rate = float(_raw_rate) if _raw_rate else 5.0
            except (ValueError, TypeError):
                error_rate = 5.0

            if automated_mttr is None:
                continue

            impact = self.compute_impact(
                service=service,
                severity=severity,
                error_rate_pct=error_rate,
                automated_mttr_minutes=automated_mttr,
            )

            saved = impact.revenue_saved_by_automation_usd or 0
            total_saved += saved
            if severity == "P1":
                p1_saved += saved
            by_service[service] = by_service.get(service, 0) + saved

        return {
            "total_revenue_saved_usd": round(total_saved, 2),
            "p1_revenue_saved_usd": round(p1_saved, 2),
            "total_incidents_automated": total_incidents,
            "savings_by_service": {k: round(v, 2) for k, v in by_service.items()},
            "avg_saving_per_incident_usd": round(total_saved / max(total_incidents, 1), 2),
        }
