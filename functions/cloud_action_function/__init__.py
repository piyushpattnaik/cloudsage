"""
CloudSage — Azure Function: HTTP Event Receiver (Free-Tier)

CHANGED FROM SERVICE BUS TRIGGER → HTTP TRIGGER
On the Azure free (Consumption F1) plan, Service Bus is not included.
This function accepts POST requests directly from Azure Monitor Action Groups,
webhooks, or any HTTP client — no Service Bus queue required.

Endpoint: POST /api/events
Body: JSON payload with event_type and service fields (same schema as before)

To send a test event:
  curl -X POST https://<your-app>.azurewebsites.net/api/events \\
    -H "Content-Type: application/json" \\
    -d '{"event_type":"incident_alert","service":"payment-api","metrics":{"cpu_percent":95}}'

When upgrading to a paid tier with Service Bus:
  - Restore a Service Bus queue trigger instead of the HTTP route
  - Remove the HTTP trigger
"""

import json
import logging
import os
import sys

import azure.functions as func

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.orchestrator import AgentOrchestrator

logger = logging.getLogger("CloudActionFunction")

# Singleton — reused across warm invocations to avoid re-initialising agents
_orchestrator = None


def _get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        environment = os.getenv("ENVIRONMENT", "production")
        _orchestrator = AgentOrchestrator(environment=environment)
        logger.info(f"Orchestrator initialised for environment={environment}")
    return _orchestrator


app = func.FunctionApp()


@app.route(route="events", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def cloud_action_function(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP-triggered function (replaces Service Bus trigger for free-tier compatibility).

    Accepts the same JSON payload shape as the old Service Bus messages.
    Returns 200 with the orchestration result, or 400/500 on error.
    """
    try:
        payload = req.get_json()
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid JSON body: {e}")
        return func.HttpResponse(
            json.dumps({"error": "Request body must be valid JSON", "detail": str(e)}),
            status_code=400,
            mimetype="application/json",
        )

    if not isinstance(payload, dict):
        return func.HttpResponse(
            json.dumps({"error": "Request body must be a JSON object"}),
            status_code=400,
            mimetype="application/json",
        )

    event_type = payload.get("event_type", "incident_alert")
    logger.info(f"HTTP event received: event_type={event_type} service={payload.get('service')}")

    try:
        orchestrator = _get_orchestrator()
        result = orchestrator.handle_event(event_type, payload)
        logger.info(
            f"Orchestration complete: status={result.get('status')} "
            f"incident_id={result.get('incident_id')} "
            f"severity={result.get('severity')}"
        )
        return func.HttpResponse(
            json.dumps(result, default=str),
            status_code=200,
            mimetype="application/json",
        )
    except Exception as e:
        logger.error(f"Orchestration failed: {e}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": "Orchestration failed", "detail": str(e)}),
            status_code=500,
            mimetype="application/json",
        )
