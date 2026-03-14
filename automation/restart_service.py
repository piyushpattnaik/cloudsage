"""
CloudSage — Restart Service (Free-Tier Stub)

On Azure free tier, AKS is not available. This stub logs the intended action,
records it in the orchestrator result, and returns a simulated success so the
rest of the pipeline (Cosmos persist, economic model, SLO tracker, feedback
loop) continues to work correctly.

When you upgrade to a paid tier and attach an AKS cluster:
  1. pip install kubernetes>=28.1.0
  2. Set AKS_CLUSTER_NAME and AKS_RESOURCE_GROUP env vars
  3. Replace this file with the full implementation from git history.
"""

import logging
import datetime

logger = logging.getLogger("RestartService")


def restart_service(context: dict) -> dict:
    """
    Simulated rolling restart — logs intent and returns success.

    On a paid tier with AKS, this would patch the Kubernetes deployment
    with a restart annotation to trigger a rolling update.
    """
    service = context.get("service", "unknown")
    namespace = context.get("namespace", "default")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"

    logger.info(
        f"[FREE-TIER STUB] restart_service: deployment={service} "
        f"namespace={namespace} — AKS not available, simulating success."
    )

    return {
        "status": "success",
        "message": (
            f"Restart recorded for '{service}' in namespace '{namespace}'. "
            "AKS not available on free tier — action logged only."
        ),
        "deployment": service,
        "namespace": namespace,
        "restart_timestamp": now,
        "simulated": True,
    }
