"""
CloudSage — MTTR Calculator
Mean Time To Resolve analytics with trend analysis.
"""

import logging
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from database.cosmos_client import CosmosDBClient

logger = logging.getLogger("MTTRCalculator")


class MTTRCalculator:
    """Calculates MTTR trends and breakdowns from Cosmos incident history."""

    def __init__(self):
        self.cosmos = CosmosDBClient()

    def compute_mttr(self, service: str = None, days: int = 30) -> dict:
        """Compute MTTR overall and broken down by severity and time period."""
        resolved = self.cosmos.get_mttr_data(service=service, days=days)

        if not resolved:
            return {
                "service": service or "all",
                "period_days": days,
                "overall_mttr_minutes": None,
                "by_severity": {},
                "trend": [],
                "sample_size": 0,
            }

        mttr_values = [r["mttr_minutes"] for r in resolved if r.get("mttr_minutes")]
        overall_mttr = round(sum(mttr_values) / len(mttr_values), 2) if mttr_values else 0

        # MTTR by severity
        by_severity = defaultdict(list)
        for r in resolved:
            if r.get("mttr_minutes"):
                by_severity[r.get("severity", "unknown")].append(r["mttr_minutes"])

        severity_breakdown = {
            sev: round(sum(vals) / len(vals), 2)
            for sev, vals in by_severity.items()
        }

        trend = self._compute_weekly_trend(resolved)

        # FIXED: auto_resolved correctly computed; manual_resolved is everything NOT in auto_resolved.
        # Previous bug: `if not auto_resolved` made all incidents "manual" when auto_resolved was empty.
        auto_resolved = [
            r for r in resolved
            if r.get("automation_result", {}).get("status") == "success"
        ]
        auto_ids = {id(r) for r in auto_resolved}
        manual_resolved = [r for r in resolved if id(r) not in auto_ids]

        auto_mttr_vals = [r["mttr_minutes"] for r in auto_resolved if r.get("mttr_minutes")]
        manual_mttr_vals = [r["mttr_minutes"] for r in manual_resolved if r.get("mttr_minutes")]

        return {
            "service": service or "all",
            "period_days": days,
            "overall_mttr_minutes": overall_mttr,
            "by_severity": severity_breakdown,
            "trend": trend,
            "sample_size": len(mttr_values),
            "automation_impact": {
                "automated_resolution_mttr": (
                    round(sum(auto_mttr_vals) / len(auto_mttr_vals), 2)
                    if auto_mttr_vals else None
                ),
                "manual_resolution_mttr": (
                    round(sum(manual_mttr_vals) / len(manual_mttr_vals), 2)
                    if manual_mttr_vals else None
                ),
                "automation_count": len(auto_resolved),
                "manual_count": len(manual_resolved),
            },
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _compute_weekly_trend(self, resolved: list) -> list:
        """Group resolved incidents by week and compute weekly MTTR."""
        weekly = defaultdict(list)
        for r in resolved:
            if not r.get("mttr_minutes") or not r.get("timestamp"):
                continue
            try:
                ts = datetime.fromisoformat(r["timestamp"])
                week_start = (ts - timedelta(days=ts.weekday())).strftime("%Y-%m-%d")
                weekly[week_start].append(r["mttr_minutes"])
            except (ValueError, TypeError):
                continue

        return [
            {
                "week": week,
                "mttr_minutes": round(sum(vals) / len(vals), 2),
                "incident_count": len(vals),
            }
            for week in sorted(weekly.keys())
            for vals in [weekly[week]]
        ]
