"""
CloudSage — Root Cause Analysis Agent
Correlates logs and telemetry, uses RAG to retrieve runbooks, produces structured RCA.
"""

import json
from agents.base_agent import BaseAgent
from rag.rag_pipeline import RAGPipeline

SYSTEM_PROMPT = """
You are the CloudSage Root Cause Analysis Agent — an expert cloud reliability engineer.
You receive an incident description, correlated logs/metrics, and retrieved knowledge base context.

Perform a thorough RCA and respond ONLY in valid JSON (no markdown fences):
{
  "root_cause": "...",
  "affected_components": ["...", "..."],
  "contributing_factors": ["..."],
  "remediation_steps": ["Step 1: ...", "Step 2: ..."],
  "prevention_recommendations": ["..."],
  "confidence": 0.85
}
"""


class RCAAgent(BaseAgent):
    def __init__(self):
        super().__init__("RCAAgent")
        self.rag = RAGPipeline()

    def run(self, payload: dict) -> dict:
        incident_id = payload.get("incident_id", "unknown")
        service = payload.get("service", "unknown")
        description = payload.get("description", "")
        logs = payload.get("logs", [])
        metrics = payload.get("metrics", {})
        # FIXED: orchestrator passes causal_context but it was never extracted or used.
        # The Granger causality result is the highest-quality signal available — the RCA
        # LLM must see it so it can reference the statistically-validated root cause
        # rather than guess from logs alone.
        causal_context = payload.get("causal_context", {})

        query = f"incident {service}: {description}"
        self.logger.info(f"Running RAG retrieval for: {query}")
        retrieved_context = self.rag.retrieve(query, top_k=5)
        rag_available = bool(retrieved_context)
        if not rag_available:
            self.logger.warning(
                "RAG retrieval returned no context (embedding quota or config issue). "
                "RCA proceeding with LLM reasoning + causal engine data only."
            )

        log_sample = "\n".join(logs[-20:]) if logs else "No logs provided."
        context_text = "\n\n".join(
            [f"[{i+1}] {doc['content']}" for i, doc in enumerate(retrieved_context)]
        )

        # Format causal context block only when data is available
        causal_block = ""
        if causal_context.get("root_cause_service"):
            causal_path = " → ".join(causal_context.get("causal_path", [service]))
            causal_block = f"""
--- Causal Inference Engine Result (Granger causality — statistically validated) ---
Root cause service : {causal_context.get('root_cause_service')}
Root cause metric  : {causal_context.get('root_cause_metric')}
Causal path        : {causal_path}
Confidence         : {causal_context.get('overall_confidence', 0):.0%}
Blast radius       : {', '.join(causal_context.get('blast_radius_services', [])) or 'none'}
Methodology        : {causal_context.get('methodology', 'N/A')}

Use this as the primary root cause hypothesis. Validate against logs/metrics and either
confirm it, refine it with specific technical detail, or explain why it should be rejected.
"""

        user_prompt = f"""
Incident ID: {incident_id}
Service: {service}
Description: {description}

--- Recent Logs ---
{log_sample}

--- Metrics ---
{json.dumps(metrics, indent=2)}

--- Retrieved Knowledge Base Context ---
{context_text}
{causal_block}
Perform RCA and respond in JSON only.
"""
        response_text = self.reason(SYSTEM_PROMPT, user_prompt, max_tokens=2000)
        rca = self.safe_parse_json(response_text, {
            "root_cause": "Unable to determine — raw response attached.",
            "affected_components": [service],
            "contributing_factors": [],
            "remediation_steps": [],
            "prevention_recommendations": [],
            "confidence": 0.3,
        })

        rca["incident_id"] = incident_id
        rca["service"] = service
        rca["retrieved_sources"] = [doc.get("source", "") for doc in retrieved_context]
        rca["causal_context_used"] = bool(causal_block)
        rca["rag_context_used"] = rag_available
        rca["status"] = "success" if rag_available else "success_degraded"
        return rca
