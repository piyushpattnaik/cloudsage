"""
CloudSage — Chaos Studio (Free-Tier Stub)

Azure Chaos Studio and AKS are not available on the free tier.
This stub returns an informational response so `main.py chaos` doesn't crash.

When you upgrade to a paid subscription:
  - Enable Chaos Studio in your resource group
  - pip install azure-identity>=1.15.0 requests>=2.31.0
  - Replace this file with the full implementation from git history.
"""

import logging

logger = logging.getLogger("ChaosStudio")


class ChaosStudioClient:
    """Stub — Chaos Studio is unavailable on Azure free tier."""

    def run_healing_validation(self, experiment_type: str = "pod_failure") -> dict:
        logger.warning(
            f"[FREE-TIER STUB] Chaos Studio not available. "
            f"experiment_type={experiment_type} — no-op."
        )
        return {
            "status": "unavailable",
            "reason": (
                "Azure Chaos Studio requires a paid subscription and an AKS cluster. "
                "Upgrade your Azure account to use this feature."
            ),
            "experiment_type": experiment_type,
        }

    def create_experiment(self, *args, **kwargs) -> dict:
        return {"status": "unavailable", "reason": "Free tier — Chaos Studio not available."}

    def start_experiment(self, *args, **kwargs) -> dict:
        return {"status": "unavailable", "reason": "Free tier — Chaos Studio not available."}

    def get_experiment_status(self, *args, **kwargs) -> dict:
        return {"status": "unavailable", "reason": "Free tier — Chaos Studio not available."}
