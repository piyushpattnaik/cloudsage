"""
CloudSage — Incident Response Agent (v4)
_reason_with_deployment() calls the API directly per model so multi-model
consensus is fully thread-safe (no shared-state mutation).

Uses _make_retrying_call from base_agent which extracts the server-requested
retryDelay from 429 responses and sleeps for exactly that duration, preventing
the burst-retry loop that exhausted the free-tier per-minute quota.
"""

import openai

from agents.base_agent import BaseAgent, _RETRYABLE, _make_retrying_call, _extract_retry_delay
import logging

_log = logging.getLogger("CloudSage.IncidentAgent")

SYSTEM_PROMPT = """
You are the CloudSage Incident Response Agent — a senior Site Reliability Engineer.
Given real-time telemetry data you must:
1. Confirm whether an incident is occurring.
2. Classify the severity (P1 Critical / P2 High / P3 Medium / P4 Low).
3. Decide the IMMEDIATE mitigation action from: restart_service | scale_up | rollback_deployment | alert_only | escalate
4. Produce a short (<=3 sentence) human-readable summary.

Respond ONLY in valid JSON (no markdown fences):
{
  "incident_confirmed": true,
  "severity": "P1",
  "action": "restart_service",
  "target": "<service_name>",
  "summary": "...",
  "escalate": false
}
"""


class IncidentAgent(BaseAgent):
    def __init__(self):
        super().__init__("IncidentResponseAgent")

    def _reason_with_deployment(
        self, system_prompt: str, user_prompt: str, deployment: str = None, **kwargs
    ) -> str:
        """
        Called by MultiModelConsensus to target a specific model deployment.
        Each consensus call is an independent API call — fully thread-safe.

        Uses _make_retrying_call so that 429 responses sleep for the server-
        requested retryDelay (e.g. 28s) rather than a fixed 8s backoff that
        keeps hitting the per-minute rate limit window.
        """
        if deployment is None:
            return self.reason(system_prompt, user_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        self.logger.info(f"Consensus call: model={deployment}")
        try:
            return _make_retrying_call(
                client=self._openai_client,
                model=deployment,
                messages=messages,
                max_tokens=self.config["openai"].get("max_tokens", 1500),
                max_attempts=self.config["agents"].get("max_retries", 3) + 1,
            )
        except (openai.AuthenticationError, openai.BadRequestError):
            raise   # permanent — surface immediately
        except Exception as e:
            self.logger.warning(f"Consensus model {deployment} exhausted retries: {e}")
            raise

    def run(self, payload: dict) -> dict:
        metrics           = payload.get("metrics", {})
        service           = payload.get("service", "unknown")
        alert_description = payload.get("alert_description", "")
        severity          = self.score_severity(metrics)

        user_prompt = f"""
Service: {service}
Alert: {alert_description}
Pre-scored Severity: {severity}
Metrics snapshot: {metrics}

Determine the mitigation action and respond in JSON only.
"""
        response_text = self.reason(SYSTEM_PROMPT, user_prompt)
        decision = self.safe_parse_json(response_text, {
            "incident_confirmed": True,
            "severity":  severity,
            "action":    "alert_only",
            "target":    service,
            "summary":   response_text[:300],
            "escalate":  severity == "P1",
        })

        decision["service"] = service
        decision["status"]  = "success"
        return decision
