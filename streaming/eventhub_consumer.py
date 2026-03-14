"""
CloudSage — HTTP Ingestion Server (Free-Tier replacement for Event Hub)

CHANGED FROM EVENT HUB → HTTP
Azure Event Hub is not included in the free tier. This module provides a
lightweight HTTP server that accepts the same JSON event payloads and routes
them directly to the AgentOrchestrator — no Event Hub or Service Bus needed.

Usage (local / on a VM or container):
    python -m streaming.eventhub_consumer   # starts server on port 8080
    # or via main.py:
    python main.py consumer

Sending events:
    curl -X POST http://localhost:8080/events \\
      -H "Content-Type: application/json" \\
      -d '{"event_type":"incident_alert","service":"payment-api","metrics":{"cpu_percent":95}}'

On Azure free tier, you can use:
    - Azure Monitor Action Groups → webhook → this endpoint (or the Function HTTP trigger)
    - GitHub Actions → POST on deployment events
    - Any HTTP client for manual event injection

When upgrading to a paid tier with Event Hub:
    - pip install azure-eventhub azure-eventhub-checkpointstoreblob-aio
    - Restore the full EventHubConsumer from git history
"""

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from streaming.message_router import MessageRouter

logger = logging.getLogger("HttpIngestionServer")

VALID_EVENT_TYPES = {
    "incident_alert", "anomaly_signal", "cost_spike",
    "deployment_failure", "security_alert", "predictive_signal",
}


class _EventHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler — routes POST /events to MessageRouter."""

    router: MessageRouter = None  # injected by HttpIngestionServer

    def log_message(self, fmt, *args):
        logger.info(fmt % args)

    def do_POST(self):
        if self.path not in ("/events", "/events/"):
            self._respond(404, {"error": "Not found. POST to /events"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, ValueError) as e:
            self._respond(400, {"error": f"Invalid JSON: {e}"})
            return

        event_type = payload.get("event_type", "unknown")
        if event_type not in VALID_EVENT_TYPES:
            self._respond(400, {"error": f"Unknown event_type '{event_type}'",
                                "valid": sorted(VALID_EVENT_TYPES)})
            return

        # Generate incident_id up front so the dashboard can poll for it
        import uuid as _uuid
        incident_id = payload.get("incident_id") or str(_uuid.uuid4())
        payload["incident_id"] = incident_id

        # Return 202 immediately — process the pipeline in a background thread
        # so the dashboard is not blocked for the full LLM pipeline duration (~40s)
        self._respond(202, {
            "status": "accepted",
            "incident_id": incident_id,
            "message": "Pipeline running — poll /api/incidents for results",
        })

        def _run():
            try:
                self.__class__.router.route_sync(event_type, payload)
            except Exception as e:
                logger.error(f"Background pipeline failed: {e}", exc_info=True)

        import threading as _threading
        _threading.Thread(target=_run, daemon=True).start()

    def do_OPTIONS(self):
        """Handle CORS preflight requests from the dashboard (localhost:3000)."""
        self.send_response(204)
        self._cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/health", "/health/"):
            self._respond(200, {"status": "ok", "server": "CloudSage HTTP Ingestion"})
        else:
            self._respond(404, {"error": "Not found"})

    def _cors_headers(self):
        """Emit permissive CORS headers — dashboard runs on a different port."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _respond(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data)


class HttpIngestionServer:
    """
    Lightweight HTTP server that accepts CloudSage events and routes them
    to the AgentOrchestrator via MessageRouter.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.router = MessageRouter()
        _EventHandler.router = self.router

    def run(self):
        server = ThreadingHTTPServer((self.host, self.port), _EventHandler)
        logger.info(
            f"CloudSage HTTP ingestion server listening on {self.host}:{self.port}\n"
            f"  POST /events   — submit a CloudSage event\n"
            f"  GET  /health   — liveness check"
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped.")
        finally:
            server.server_close()


# Alias kept for backwards-compat with main.py
EventHubConsumer = HttpIngestionServer
