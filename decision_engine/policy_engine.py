"""
CloudSage — Policy Engine (v4)
Blast-radius-aware and SLO-error-budget-driven policy evaluation.

New in v4:
  - Blast radius risk overrides severity-based auto-approval
  - SLO policy tier (standard / conservative / restricted / freeze) tightens gates
  - Multi-model consensus confidence feeds into approval decisions
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger("PolicyEngine")


class PolicyEngine:
    """
    Enforces policy-based guardrails before any automation executes.
    Decision factors (in priority order):
      1. Hard-blocked actions (always denied)
      2. Environment allowlist
      3. Rate limit
      4. SLO error budget tier
      5. Blast radius risk level
      6. Consensus confidence (P1 only)
      7. Severity-based approval gate
    """

    AUTO_EXECUTE_POLICY = {
        "P1": ["restart_service", "scale_up", "alert_only", "notify_teams", "enforce_policy"],
        "P2": ["scale_up", "alert_only", "notify_teams"],
        "P3": ["alert_only", "notify_teams"],
        "P4": ["alert_only"],
    }

    BLOCKED_ACTIONS = ["delete_resource", "drop_database"]

    ENV_RESTRICTIONS = {
        "production": {
            "allowed_actions": [
                "restart_service", "scale_up", "scale_down", "alert_only",
                "notify_teams", "enforce_policy", "rollback_deployment",
            ],
            "require_approval_for": ["rollback_deployment", "scale_down"],
        },
        "staging": {"allowed_actions": "__all__", "require_approval_for": []},
        "development": {"allowed_actions": "__all__", "require_approval_for": []},
    }

    # Blast radius levels that force human approval regardless of severity
    BLAST_RADIUS_APPROVAL_REQUIRED = {"HIGH", "CRITICAL"}

    # SLO tiers where human approval is required
    SLO_APPROVAL_REQUIRED_TIERS = {"restricted", "freeze"}

    def __init__(self, environment: str = "production"):
        self.environment = environment
        self._action_counters: dict = {}

    def evaluate(self, decision: dict) -> dict:
        """
        Evaluate a decision against all policy layers.
        Returns dict with: approved, action, reason, requires_approval.
        """
        action = decision.get("action", "alert_only")
        severity = decision.get("severity", "P4")
        service = decision.get("service", "unknown")
        blast_risk = decision.get("blast_radius_risk", "LOW")
        blast_safe = decision.get("blast_radius_safe", True)
        slo_tier = decision.get("slo_policy_tier", "standard")
        slo_requires = decision.get("slo_requires_approval", False)
        consensus_confidence = decision.get("consensus_confidence")  # HIGH/MEDIUM/LOW/None

        # ── Layer 1: Hard block ───────────────────────────────────────
        if action in self.BLOCKED_ACTIONS:
            return self._deny(action, f"Action '{action}' is permanently blocked.")

        # ── Layer 2: Environment allowlist ────────────────────────────
        env_policy = self.ENV_RESTRICTIONS.get(self.environment, {})
        allowed = env_policy.get("allowed_actions", "__all__")
        if allowed != "__all__" and action not in allowed:
            return self._deny(action, f"Action '{action}' not permitted in {self.environment}.")

        # ── Layer 3: Rate limit ───────────────────────────────────────
        if not self._check_rate_limit(service, max_per_hour=5):
            return self._deny(action, f"Rate limit exceeded for '{service}'.")

        # ── Layer 4: SLO error budget tier ────────────────────────────
        if slo_tier == "freeze":
            return {
                "approved": False,
                "requires_approval": True,
                "action": action,
                "service": service,
                "severity": severity,
                "reason": (
                    f"SLO error budget EXHAUSTED for '{service}'. "
                    "All automation frozen — human decision required."
                ),
                "policy_layer": "slo_freeze",
            }

        if slo_tier in self.SLO_APPROVAL_REQUIRED_TIERS or slo_requires:
            logger.info(f"SLO tier '{slo_tier}' requires human approval for {service}")
            return self._require_approval(
                action, service, severity,
                f"SLO policy tier '{slo_tier}': error budget low, human approval required.",
                policy_layer="slo_budget",
            )

        # ── Layer 5: Blast radius check ───────────────────────────────
        if blast_risk in self.BLAST_RADIUS_APPROVAL_REQUIRED or not blast_safe:
            logger.warning(
                f"Blast radius {blast_risk} for action '{action}' on '{service}' "
                "— requiring human approval."
            )
            return self._require_approval(
                action, service, severity,
                f"Blast radius risk '{blast_risk}': downstream impact too high for auto-execution.",
                policy_layer="blast_radius",
            )

        # ── Layer 6: Consensus confidence (P1 only) ───────────────────
        # BUG 6 FIX: when the orchestrator overrides the action to "alert_only" after a
        # failed consensus, the policy engine was still firing requires_approval=True.
        # alert_only emits a notification only — there is nothing to approve or deny.
        # Gating it sent engineers a spurious "APPROVAL REQUIRED" Teams message for a
        # passive notification, causing confusion during real P1 incidents.
        if severity == "P1" and consensus_confidence == "LOW" and action != "alert_only":
            return self._require_approval(
                action, service, severity,
                "Multi-model consensus confidence LOW — models disagreed on action. Human decision required.",
                policy_layer="consensus",
            )

        # ── Layer 7: Severity-based gate ──────────────────────────────
        severity_allowed = self.AUTO_EXECUTE_POLICY.get(severity, ["alert_only"])
        requires_approval = action not in severity_allowed

        if action in env_policy.get("require_approval_for", []):
            requires_approval = True

        if requires_approval:
            return self._require_approval(
                action, service, severity,
                f"Severity {severity} requires human approval for '{action}' in {self.environment}.",
                policy_layer="severity",
            )

        self._record_action(service)
        logger.info(f"Policy APPROVED: action={action} service={service} severity={severity}")
        return {
            "approved": True,
            "requires_approval": False,
            "action": action,
            "service": service,
            "severity": severity,
            "reason": "All policy layers passed — approved for automatic execution.",
            "policy_layer": "approved",
        }

    def _deny(self, action: str, reason: str) -> dict:
        logger.warning(f"Policy DENIED: action={action} | {reason}")
        return {
            "approved": False,
            "requires_approval": False,
            "action": "alert_only",
            "reason": reason,
            "policy_layer": "hard_deny",
        }

    def _require_approval(self, action, service, severity, reason, policy_layer="unknown") -> dict:
        logger.info(f"Policy APPROVAL_REQUIRED: {reason}")
        return {
            "approved": False,
            "requires_approval": True,
            "action": action,
            "service": service,
            "severity": severity,
            "reason": reason,
            "policy_layer": policy_layer,
        }

    def _check_rate_limit(self, service: str, max_per_hour: int) -> bool:
        now = datetime.now(timezone.utc)
        history = self._action_counters.get(service, [])
        history = [t for t in history if (now - t).total_seconds() < 3600]
        self._action_counters[service] = history
        return len(history) < max_per_hour

    def _record_action(self, service: str):
        self._action_counters.setdefault(service, [])
        self._action_counters[service].append(datetime.now(timezone.utc))
