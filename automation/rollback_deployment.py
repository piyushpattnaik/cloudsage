"""
CloudSage — Rollback Deployment (Free-Tier Stub)

AKS is not available on Azure free tier. This stub logs the rollback intent
and returns a simulated success so the pipeline (Cosmos, SLO, feedback) works.

When you upgrade: replace with the full implementation that:
  1. Lists ReplicaSets owned by the deployment sorted by revision
  2. Patches the Deployment pod template to match the previous ReplicaSet spec
  (equivalent to `kubectl rollout undo deployment/<name>`)
"""

import logging

logger = logging.getLogger("RollbackDeployment")


def rollback_deployment(context: dict) -> dict:
    """
    Simulated deployment rollback — logs intent and returns success.
    """
    service   = context.get("service", "unknown")
    namespace = context.get("namespace", "default")

    logger.info(
        f"[FREE-TIER STUB] rollback_deployment: service={service} "
        f"namespace={namespace} — AKS not available, simulating success."
    )

    return {
        "status": "success",
        "message": (
            f"Rollback recorded for '{service}' in namespace '{namespace}'. "
            "AKS not available on free tier — action logged only."
        ),
        "deployment": service,
        "namespace": namespace,
        "rolled_back_to_revision": "previous",
        "simulated": True,
    }
