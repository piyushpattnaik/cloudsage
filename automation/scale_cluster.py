"""
CloudSage — Scale Cluster (Free-Tier Stub)

AKS is not available on Azure free tier. This stub logs the intended scaling
action and returns a simulated success so the pipeline continues normally.

When you upgrade: replace with the full implementation that calls the
Kubernetes AppsV1Api to patch replica counts.
"""

import logging

logger = logging.getLogger("ScaleCluster")


def scale_cluster(context: dict) -> dict:
    """
    Simulated AKS scaling — logs intent and returns success.
    """
    service   = context.get("service", "unknown")
    namespace = context.get("namespace", "default")
    direction = context.get("direction", "up")
    delta     = context.get("replica_delta", 1)

    logger.info(
        f"[FREE-TIER STUB] scale_cluster: service={service} "
        f"direction={direction} delta={delta} — AKS not available, simulating success."
    )

    # Simulate a plausible before/after replica count
    current = 2
    new = current + delta if direction == "up" else max(1, current - delta)

    return {
        "status": "success",
        "message": (
            f"Scale {direction} recorded for '{service}' in '{namespace}'. "
            "AKS not available on free tier — action logged only."
        ),
        "deployment": service,
        "previous_replicas": current,
        "new_replicas": new,
        "direction": direction,
        "simulated": True,
    }
