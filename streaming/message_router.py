"""
CloudSage — Message Router (Free-Tier: direct in-process routing)

CHANGED FROM SERVICE BUS → DIRECT ORCHESTRATOR CALL
On the free tier, Service Bus is not available. Events are routed directly
to the AgentOrchestrator in-process — no queue or broker needed.

Architecture:
    HTTP Ingestion Server → MessageRouter.route_sync() → AgentOrchestrator

This is simpler and faster than the Service Bus path (no serialisation round-trip,
no queue latency) but sacrifices durability: if the process crashes mid-pipeline,
the event is lost. On a paid tier you'd restore the Service Bus broker for
at-least-once delivery guarantees.

When upgrading:
    - pip install azure-servicebus>=7.11.0
    - Restore _publish_to_service_bus() from git history
    - Change route_sync() back to an async publish + Function trigger
"""

import logging

from config.loader import load_config

logger = logging.getLogger("MessageRouter")

VALID_EVENT_TYPES = {
    "incident_alert",
    "anomaly_signal",
    "cost_spike",
    "deployment_failure",
    "security_alert",
    "predictive_signal",
}


class MessageRouter:
    """
    Validates events and routes them directly to the AgentOrchestrator.
    No Service Bus or Event Hub required.
    """

    def __init__(self):
        self.config = load_config()
        # Lazy-initialise orchestrator to avoid circular imports at module load
        self._orchestrator = None

    def _get_orchestrator(self):
        if self._orchestrator is None:
            import os
            from agents.orchestrator import AgentOrchestrator
            env = os.getenv("ENVIRONMENT", "production")
            self._orchestrator = AgentOrchestrator(environment=env)
            logger.info(f"Orchestrator initialised (environment={env})")
        return self._orchestrator

    def route_sync(self, event_type: str, payload: dict) -> dict:
        """
        Validate the event and run it through the orchestrator synchronously.
        Returns the orchestration result dict.
        """
        if event_type not in VALID_EVENT_TYPES:
            logger.warning(f"Unknown event_type='{event_type}' — dropping message.")
            return {"status": "dropped", "reason": f"Unknown event_type: {event_type}"}

        logger.info(
            f"Routing event: event_type={event_type} "
            f"service={payload.get('service', 'unknown')}"
        )

        orchestrator = self._get_orchestrator()
        return orchestrator.handle_event(event_type, payload)
