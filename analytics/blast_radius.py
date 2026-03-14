"""
CloudSage — Blast Radius Predictor
=====================================================
Before CloudSage executes ANY automation action, this module answers:
  "If we restart / scale / rollback service X right now, which other
   services will experience degradation, and for how long?"

This transforms CloudSage automation from 'blind execution' to
'informed action' — the single biggest reason SREs distrust automation.

Architecture:
  - Uses the shared ServiceDependencyGraph from causal_engine.py
  - Applies traffic-weighted impact propagation
  - Estimates degradation window from historical MTTR data
  - Returns a structured blast radius report that feeds into PolicyEngine
"""

import logging
from dataclasses import dataclass
from analytics.causal_engine import get_dependency_graph

logger = logging.getLogger("CloudSage.BlastRadius")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImpactedService:
    service: str
    hop_distance: int          # how many edges from the action target
    estimated_impact_pct: float  # 0-100% degradation expected
    estimated_duration_seconds: float
    impact_type: str           # "direct", "cascading", "transient"


@dataclass
class BlastRadiusReport:
    action_target: str
    action_type: str
    impacted_services: list     # list[ImpactedService]
    total_services_affected: int
    max_impact_duration_seconds: float
    risk_level: str             # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    recommendation: str
    safe_to_auto_execute: bool
    estimated_recovery_window_seconds: float


# ---------------------------------------------------------------------------
# Impact decay model
# ---------------------------------------------------------------------------

# How much impact decays per hop (direct dep = 100%, 2 hops = 60%, etc.)
HOP_DECAY = {1: 1.0, 2: 0.6, 3: 0.3, 4: 0.1}

# Estimated disruption duration by action type (seconds)
ACTION_DISRUPTION_SECONDS = {
    "restart_service":      45,    # rolling restart typically takes 30-60s
    "scale_up":             15,    # new pods come up in ~15s
    "scale_down":           30,    # graceful termination
    "rollback_deployment": 120,    # rollback + pod startup
    "enforce_policy":        5,    # near-instantaneous
    "alert_only":            0,    # no disruption
    "notify_teams":          0,
}

# Risk thresholds
RISK_THRESHOLDS = {
    "LOW":      {"max_services": 2,  "max_duration": 30},
    "MEDIUM":   {"max_services": 5,  "max_duration": 120},
    "HIGH":     {"max_services": 10, "max_duration": 300},
    # CRITICAL = anything beyond HIGH
}


class BlastRadiusPredictor:
    """
    Predicts the blast radius of a proposed automation action before execution.
    """

    def __init__(self, historical_mttr_provider=None):
        """
        Args:
            historical_mttr_provider: callable(service) -> avg_mttr_seconds
                                      Used to estimate how long disruption lasts.
                                      Falls back to action-type defaults if None.
        """
        self.graph = get_dependency_graph()
        self._mttr_provider = historical_mttr_provider

    def predict(
        self,
        action_type: str,
        target_service: str,
        current_load_multiplier: float = 1.0,
    ) -> BlastRadiusReport:
        """
        Predict blast radius for a proposed action.

        Args:
            action_type: e.g. "restart_service", "rollback_deployment"
            target_service: The service the action will be applied to
            current_load_multiplier: 1.0 = normal, 2.0 = 2× normal traffic (Black Friday)

        Returns:
            BlastRadiusReport
        """
        logger.info(f"Predicting blast radius: action={action_type} target={target_service}")

        base_disruption = ACTION_DISRUPTION_SECONDS.get(action_type, 60)
        # Scale disruption by current load — higher load = longer recovery
        disruption_seconds = base_disruption * max(1.0, current_load_multiplier * 0.7)

        # Get downstream services from dependency graph
        downstream = self.graph.get_downstream(target_service, max_hops=4)

        impacted = []
        for service, hops in downstream:
            decay = HOP_DECAY.get(hops, 0.05)
            impact_pct = round(100 * decay, 1)
            duration = disruption_seconds * decay

            # Get MTTR-adjusted duration
            if self._mttr_provider:
                try:
                    historical_mttr = self._mttr_provider(service)
                    if historical_mttr:
                        duration = max(duration, historical_mttr * 60 * 0.3)
                except Exception:
                    pass

            impact_type = "direct" if hops == 1 else "cascading" if hops <= 2 else "transient"

            impacted.append(ImpactedService(
                service=service,
                hop_distance=hops,
                estimated_impact_pct=impact_pct,
                estimated_duration_seconds=round(duration, 1),
                impact_type=impact_type,
            ))

        max_duration = max((s.estimated_duration_seconds for s in impacted), default=0)
        max_duration = max(max_duration, disruption_seconds)

        # Compute risk level
        n_affected = len(impacted)
        risk_level = self._compute_risk(n_affected, max_duration)

        # Safe to auto-execute if LOW or MEDIUM risk
        safe_to_auto = risk_level in ("LOW", "MEDIUM")

        # Recovery window = disruption + propagation recovery time
        recovery_window = disruption_seconds + (len(impacted) * 5)

        if action_type == "alert_only":
            recommendation = "Alert-only — no blast radius."
        elif risk_level == "LOW":
            recommendation = (
                f"Low blast radius: {n_affected} downstream service(s) affected "
                f"for ~{disruption_seconds:.0f}s. Auto-execution recommended."
            )
        elif risk_level == "MEDIUM":
            recommendation = (
                f"Moderate blast radius: {n_affected} service(s) affected "
                f"for up to {max_duration:.0f}s. Proceed with monitoring."
            )
        elif risk_level == "HIGH":
            recommendation = (
                f"High blast radius: {n_affected} service(s) including "
                f"{[s.service for s in impacted[:3]]}. "
                f"Human approval strongly recommended."
            )
        else:
            recommendation = (
                f"CRITICAL blast radius: {n_affected} services, up to "
                f"{max_duration:.0f}s disruption. Escalate before executing."
            )

        report = BlastRadiusReport(
            action_target=target_service,
            action_type=action_type,
            impacted_services=[self._to_dict(s) for s in impacted],
            total_services_affected=n_affected,
            max_impact_duration_seconds=round(max_duration, 1),
            risk_level=risk_level,
            recommendation=recommendation,
            safe_to_auto_execute=safe_to_auto,
            estimated_recovery_window_seconds=round(recovery_window, 1),
        )

        logger.info(
            f"Blast radius: risk={risk_level} services_affected={n_affected} "
            f"max_duration={max_duration:.0f}s safe={safe_to_auto}"
        )
        return report

    @staticmethod
    def _compute_risk(n_services: int, max_duration_seconds: float) -> str:
        if n_services == 0 or max_duration_seconds < 5:
            return "LOW"
        if n_services <= RISK_THRESHOLDS["LOW"]["max_services"] and \
                max_duration_seconds <= RISK_THRESHOLDS["LOW"]["max_duration"]:
            return "LOW"
        if n_services <= RISK_THRESHOLDS["MEDIUM"]["max_services"] and \
                max_duration_seconds <= RISK_THRESHOLDS["MEDIUM"]["max_duration"]:
            return "MEDIUM"
        if n_services <= RISK_THRESHOLDS["HIGH"]["max_services"] and \
                max_duration_seconds <= RISK_THRESHOLDS["HIGH"]["max_duration"]:
            return "HIGH"
        return "CRITICAL"

    @staticmethod
    def _to_dict(s: ImpactedService) -> dict:
        return {
            "service": s.service,
            "hop_distance": s.hop_distance,
            "estimated_impact_pct": s.estimated_impact_pct,
            "estimated_duration_seconds": s.estimated_duration_seconds,
            "impact_type": s.impact_type,
        }

    @staticmethod
    def to_report_dict(report: BlastRadiusReport) -> dict:
        return {
            "action_target": report.action_target,
            "action_type": report.action_type,
            "total_services_affected": report.total_services_affected,
            "max_impact_duration_seconds": report.max_impact_duration_seconds,
            "risk_level": report.risk_level,
            "safe_to_auto_execute": report.safe_to_auto_execute,
            "recommendation": report.recommendation,
            "estimated_recovery_window_seconds": report.estimated_recovery_window_seconds,
            "impacted_services": report.impacted_services,
        }
