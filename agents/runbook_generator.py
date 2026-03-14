"""
CloudSage — Automatic Runbook Generator
=====================================================
After every resolved incident, CloudSage writes a new runbook entry
and indexes it into Azure AI Search.

Over time, the RAG knowledge base becomes self-improving institutional
memory — the next similar incident retrieves this runbook automatically,
giving the AI *its own past experience* as context.

This is the compound interest of reliability engineering:
  Incident 1: resolves in 4.2 minutes (no runbook)
  Incident 47: resolves in 1.1 minutes (RAG retrieves 5 similar runbooks)
  Incident 200: resolves in 0.3 minutes (runbook is nearly exact match)

Output format: structured Markdown, optimised for RAG chunking and retrieval.
"""

import logging
from datetime import datetime, timezone
from agents.base_agent import BaseAgent

logger = logging.getLogger("CloudSage.RunbookGenerator")

SYSTEM_PROMPT = """
You are the CloudSage Runbook Generator — a technical writer with deep SRE expertise.
Given a resolved incident's full data, write a structured operational runbook.

The runbook must be:
- Written in clear, imperative technical language
- Structured for quick scanning during future incidents  
- Optimised for semantic search retrieval (include relevant technical terms)
- Actionable: every step should be executable by an on-call engineer

Respond with ONLY the runbook in Markdown — no preamble, no JSON wrapper.

Use exactly this structure:
# Runbook: [Descriptive Title]

## Incident Pattern
[1-2 sentences describing what kind of incident this covers]

## Symptoms
- [Observable symptom 1]
- [Observable symptom 2]

## Root Cause
[Technical explanation of what caused the incident]

## Immediate Mitigation (< 5 minutes)
1. [Step 1 — what to do right now]
2. [Step 2]

## Verification
- [How to confirm the fix worked]

## Permanent Fix
[What code/config change prevents recurrence]

## Prevention
- [Monitoring: what alert/threshold to add]
- [Process: what change to deployment/review process]

## Related Services
[Comma-separated list of services that may be involved]

## Tags
[comma-separated keywords for search: service names, error types, symptoms]
"""


class RunbookGenerator(BaseAgent):
    """
    Generates and indexes runbooks from resolved incident data.
    """

    def __init__(self):
        super().__init__("RunbookGenerator")
        # Import here to avoid circular load issues
        from rag.rag_pipeline import RAGPipeline
        self.rag = RAGPipeline()

    def run(self, payload: dict) -> dict:
        """
        Generate a runbook from a resolved incident payload.
        Payload should contain the full orchestrator result dict.
        """
        incident_id = payload.get("incident_id", "unknown")
        service = payload.get("service", "unknown")
        severity = payload.get("severity", "P3")
        description = payload.get("description", "")
        root_cause = payload.get("rca", {}).get("root_cause", "")
        action_taken = payload.get("automation", {}).get("action", "")
        mttr = payload.get("mttr_minutes", 0)
        causal_chain = payload.get("causal_analysis", {})
        blast_radius = payload.get("blast_radius", {})

        user_prompt = f"""
Incident ID: {incident_id}
Service: {service}
Severity: {severity}
Description: {description}
Root Cause (from RCA Agent): {root_cause}
Automated Action Taken: {action_taken}
MTTR: {mttr} minutes

Causal Analysis:
- Root cause service: {causal_chain.get('root_cause_service', 'unknown')}
- Causal path: {' → '.join(causal_chain.get('causal_path', [service]))}
- Methodology: {causal_chain.get('methodology', 'N/A')}

Blast Radius Observed:
- Services affected: {blast_radius.get('total_services_affected', 0)}
- Risk level: {blast_radius.get('risk_level', 'N/A')}
- Impacted: {[s['service'] for s in blast_radius.get('impacted_services', [])]}

Remediation Steps from RCA:
{chr(10).join(payload.get('rca', {}).get('remediation_steps', ['N/A']))}

Prevention Recommendations:
{chr(10).join(payload.get('rca', {}).get('prevention_recommendations', ['N/A']))}

Generate a complete runbook for this incident pattern.
"""

        runbook_markdown = self.reason(SYSTEM_PROMPT, user_prompt, max_tokens=2000)

        # Index the runbook into RAG
        doc = {
            "content": runbook_markdown,
            "source": f"auto-runbook/{service}/{incident_id}",
            "category": "runbook",
            "metadata": {
                "generated_from_incident": incident_id,
                "service": service,
                "severity": severity,
                "mttr_minutes": mttr,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "auto_generated": True,
            },
        }

        indexed_chunks = 0
        try:
            indexed_chunks = self.rag.index_documents_bulk([doc])
            logger.info(
                f"Runbook indexed: incident={incident_id} service={service} "
                f"chunks={indexed_chunks}"
            )
        except Exception as e:
            logger.error(f"Failed to index runbook for {incident_id}: {e}")

        return {
            "status": "success",
            "incident_id": incident_id,
            "service": service,
            "runbook_markdown": runbook_markdown,
            "runbook_source": doc["source"],
            "indexed_chunks": indexed_chunks,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }

    def generate_batch(self, resolved_incidents: list) -> list:
        """
        Backfill runbooks for historical incidents that don't have one yet.
        Call this as a scheduled job to build up the knowledge base.
        """
        results = []
        for incident in resolved_incidents:
            if incident.get("runbook_generated"):
                continue
            try:
                result = self.run(incident)
                results.append(result)
                logger.info(f"Batch runbook: {incident.get('incident_id')} → {result['status']}")
            except Exception as e:
                logger.error(f"Failed runbook for {incident.get('incident_id')}: {e}")
                results.append({
                    "status": "error",
                    "incident_id": incident.get("incident_id"),
                    "error": str(e),
                })
        return results
