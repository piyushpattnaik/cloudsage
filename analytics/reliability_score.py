"""
CloudSage — Reliability Score Calculator
Computes a 0-100 reliability score for each service based on incident history.
"""

import logging
from datetime import datetime, timezone, timedelta
from database.cosmos_client import CosmosDBClient

logger = logging.getLogger("ReliabilityScore")

# Weight matrix for reliability score components
WEIGHTS = {
    "uptime": 0.40,
    "mttr": 0.25,
    "p1_penalty": 0.20,
    "deployment_stability": 0.15,
}

SEVERITY_WEIGHTS = {"P1": 10, "P2": 5, "P3": 2, "P4": 1}


class ReliabilityScoreCalculator:
    """Computes the CloudSage Reliability Score from incident history."""

    def __init__(self):
        self.cosmos = CosmosDBClient()

    def compute(self, service: str, days: int = 30) -> dict:
        """
        Compute reliability score for a given service over the specified period.

        Returns:
            dict with reliability_score (0-100) and component breakdown
        """
        incidents = self.cosmos.query_incidents(service=service, days_back=days)
        resolved = [i for i in incidents if i.get("status") == "resolved"]
        total = len(incidents)
        p1_count = sum(1 for i in incidents if i.get("severity") == "P1")
        p2_count = sum(1 for i in incidents if i.get("severity") == "P2")

        # --- Uptime component (penalise per incident weighted by severity) ---
        severity_penalty = sum(
            SEVERITY_WEIGHTS.get(i.get("severity", "P4"), 1) for i in incidents
        )
        uptime_score = max(0, 100 - severity_penalty)

        # --- MTTR component ---
        mttr_minutes = self._average_mttr(resolved)
        mttr_score = self._mttr_to_score(mttr_minutes)

        # --- P1 penalty ---
        p1_score = max(0, 100 - (p1_count * 25))

        # --- Deployment stability (automation success rate) ---
        automation_results = [
            i.get("automation_result", {}).get("status") for i in incidents
            if i.get("automation_result")
        ]
        stability_score = 100
        if automation_results:
            success_rate = automation_results.count("success") / len(automation_results)
            stability_score = round(success_rate * 100)

        # Weighted composite score
        reliability_score = round(
            uptime_score * WEIGHTS["uptime"]
            + mttr_score * WEIGHTS["mttr"]
            + p1_score * WEIGHTS["p1_penalty"]
            + stability_score * WEIGHTS["deployment_stability"]
        )

        result = {
            "service": service,
            "period_days": days,
            "reliability_score": reliability_score,
            "components": {
                "uptime_score": uptime_score,
                "mttr_score": mttr_score,
                "p1_incident_score": p1_score,
                "deployment_stability_score": stability_score,
            },
            "incident_summary": {
                "total": total,
                "p1": p1_count,
                "p2": p2_count,
                "resolved": len(resolved),
            },
            "average_mttr_minutes": mttr_minutes,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(f"Reliability score: service={service} score={reliability_score}")
        return result

    def _average_mttr(self, resolved_incidents: list[dict]) -> float:
        mttr_values = [
            i["mttr_minutes"] for i in resolved_incidents
            if i.get("mttr_minutes") is not None
        ]
        return round(sum(mttr_values) / len(mttr_values), 2) if mttr_values else 0.0

    def _mttr_to_score(self, mttr_minutes: float) -> float:
        """Convert MTTR to a 0-100 score. Lower MTTR = higher score."""
        if mttr_minutes == 0:
            return 100
        if mttr_minutes <= 5:
            return 95
        if mttr_minutes <= 15:
            return 85
        if mttr_minutes <= 30:
            return 70
        if mttr_minutes <= 60:
            return 55
        if mttr_minutes <= 120:
            return 35
        return max(0, 100 - mttr_minutes * 0.5)
