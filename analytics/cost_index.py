"""
CloudSage — Cost Efficiency Index
Computes a 0-1 cost efficiency score for services and the overall cloud estate.
"""

import logging
from datetime import datetime, timezone
from database.cosmos_client import CosmosDBClient

logger = logging.getLogger("CostIndex")


class CostIndexCalculator:
    """
    Computes the CloudSage Cost Efficiency Index.
    
    Index = weighted combination of:
    - Resource utilization efficiency
    - Budget adherence
    - Idle resource elimination rate
    - Reserved/spot instance coverage
    """

    def __init__(self):
        self.cosmos = CosmosDBClient()

    def compute(
        self,
        resource_utilization: list[dict],
        current_spend: float,
        budget: float,
        idle_resources_count: int,
        total_resources: int,
        reserved_instance_coverage: float = 0.0,
    ) -> dict:
        """
        Compute Cost Efficiency Index.

        Args:
            resource_utilization: List of {resource, cpu_percent, memory_percent}
            current_spend: Current month spend in USD
            budget: Monthly budget in USD
            idle_resources_count: Number of identified idle resources
            total_resources: Total managed resources
            reserved_instance_coverage: Fraction of compute covered by RIs (0-1)

        Returns:
            dict with cost_efficiency_index (0-1) and breakdown
        """
        # 1. Utilization efficiency (avg utilization across resources)
        if resource_utilization:
            avg_cpu = sum(r.get("cpu_percent", 0) for r in resource_utilization) / len(resource_utilization)
            avg_mem = sum(r.get("memory_percent", 0) for r in resource_utilization) / len(resource_utilization)
            # Ideal utilization ~70%. Penalty for too low (waste) or too high (risk)
            utilization_score = 1 - abs(avg_cpu - 70) / 100
        else:
            avg_cpu = avg_mem = 0
            utilization_score = 0.5

        # 2. Budget adherence score
        if budget > 0:
            spend_ratio = current_spend / budget
            if spend_ratio <= 1.0:
                budget_score = 1.0 - (spend_ratio * 0.3)  # Small penalty for high spend
            else:
                budget_score = max(0, 1.0 - (spend_ratio - 1.0))  # Heavy penalty for overrun
        else:
            budget_score = 0.5

        # 3. Idle resource elimination score
        if total_resources > 0:
            idle_ratio = idle_resources_count / total_resources
            idle_score = 1 - idle_ratio
        else:
            idle_score = 1.0

        # 4. Reserved instance coverage score (higher = better)
        ri_score = min(1.0, reserved_instance_coverage * 1.5)  # Bonus for high coverage

        # Weighted composite
        cost_efficiency_index = round(
            utilization_score * 0.35
            + budget_score * 0.30
            + idle_score * 0.25
            + ri_score * 0.10,
            3,
        )

        return {
            "cost_efficiency_index": cost_efficiency_index,
            "grade": self._to_grade(cost_efficiency_index),
            "components": {
                "utilization_score": round(utilization_score, 3),
                "budget_adherence_score": round(budget_score, 3),
                "idle_elimination_score": round(idle_score, 3),
                "reserved_instance_score": round(ri_score, 3),
            },
            "summary": {
                "average_cpu_utilization": round(avg_cpu, 1),
                "average_memory_utilization": round(avg_mem, 1),
                "budget_spend_ratio": round(current_spend / budget, 3) if budget else None,
                "idle_resources": idle_resources_count,
                "total_resources": total_resources,
                "reserved_instance_coverage_pct": round(reserved_instance_coverage * 100, 1),
            },
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _to_grade(index: float) -> str:
        if index >= 0.85:
            return "A"
        if index >= 0.70:
            return "B"
        if index >= 0.55:
            return "C"
        if index >= 0.40:
            return "D"
        return "F"
