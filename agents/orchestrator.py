"""
CloudSage — Multi-Agent Orchestrator (v4 — Award Edition)
==========================================================
Chains every system into one coherent closed loop:

  IncidentAgent (+ Multi-Model Consensus for P1)
    → Causal Inference Engine         (Granger causality, not LLM guessing)
    → Blast Radius Predictor          (simulate downstream impact first)
    → SLO Error Budget Check          (is budget healthy enough to auto-act?)
    → RCAAgent + PredictiveAgent      (parallel)
    → PolicyEngine                    (guards + SLO + blast radius)
    → Adaptive Threshold Evaluation   (dynamic, not static thresholds)
    → ActionMapper                    (kubernetes / scale / rollback)
    → Economic Impact Model           ($/min cost + automation savings)
    → CosmosDB (full structured record)
    → FeedbackLoop (score outcome)
    → RunbookGenerator (write + index runbook if P1/P2)
    → SLO Tracker (record downtime against budget)

New in v4:
  - Multi-model consensus blocks unsafe P1 auto-execution when models disagree
  - Blast radius check can override policy and require human approval
  - Causal analysis replaces "LLM guesses root cause" with Granger causality
  - Economic impact computed on every incident — dashboard shows $/saved
  - SLO error budget drives policy tier (standard / conservative / restricted / freeze)
  - Every resolved P1/P2 auto-generates a runbook and indexes it into RAG
  - Adaptive thresholds learn per-service, per-hour baselines over time
"""

# asyncio removed: _run_parallel replaced with ThreadPoolExecutor.submit() — BUG 7 FIX
import logging
import time
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from agents.incident_agent import IncidentAgent
from agents.rca_agent import RCAAgent
from agents.predictive_agent import PredictiveAgent
from agents.finops_agent import FinOpsAgent
from agents.security_agent import SecurityAgent
from agents.runbook_generator import RunbookGenerator
from agents.consensus import MultiModelConsensus

from decision_engine.policy_engine import PolicyEngine
from decision_engine.action_mapper import ActionMapper

from database.cosmos_client import CosmosDBClient
from database.models import IncidentRecord

from automation.notify_teams import notify_teams

from analytics.causal_engine import get_causal_engine, get_dependency_graph
from analytics.blast_radius import BlastRadiusPredictor, BlastRadiusReport
from analytics.economic_impact import EconomicImpactModel
from analytics.adaptive_thresholds import get_adaptive_threshold_engine
from analytics.slo_tracker import get_slo_tracker

from config.loader import load_config

logger = logging.getLogger("CloudSage.Orchestrator")


# ---------------------------------------------------------------------------
# Pipeline state (for durable checkpoint pattern)
# ---------------------------------------------------------------------------
class PipelineState:
    """
    Holds the full mutable state of one pipeline execution.
    Designed so any step can be resumed from where it left off
    (mimicking Durable Functions checkpoint semantics).
    """

    def __init__(self, event_type: str, payload: dict):
        self.event_type = event_type
        self.payload = payload
        self.incident_id = payload.get("incident_id") or str(uuid.uuid4())
        self.service = payload.get("service", "unknown")

        # Checkpoint flags — step is safe to skip on replay
        self.completed_steps: set = set()

        # Results accumulate here
        self.incident_result: dict = {}
        self.consensus_result: dict = {}
        self.causal_analysis: dict = {}
        self.blast_radius: dict = {}
        self.slo_status: dict = {}
        self.rca_result: dict = {}
        self.pred_result: dict = {}
        self.policy_result: dict = {}
        self.adaptive_evaluation: dict = {}
        self.automation_result: dict = {}
        self.economic_impact: dict = {}
        self.feedback_score: dict = {}
        self.runbook_result: dict = {}

    def mark_done(self, step: str):
        self.completed_steps.add(step)

    def is_done(self, step: str) -> bool:
        return step in self.completed_steps


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
class AgentOrchestrator:
    """
    Central orchestrator — called exclusively from the Azure Function.
    One invocation per event message from Service Bus.
    """

    def __init__(self, environment: str = "production"):
        cfg = load_config()
        self.config = cfg          # BUG 1 FIX: self.config was never assigned; Step 5 crashed with AttributeError on timeout lookup
        self.environment = environment

        # Agents
        self.incident_agent = IncidentAgent()
        self.rca_agent = RCAAgent()
        self.predictive_agent = PredictiveAgent()
        self.finops_agent = FinOpsAgent()
        self.security_agent = SecurityAgent()
        self.runbook_generator = RunbookGenerator()

        # Decision & automation
        self.policy_engine = PolicyEngine(environment=environment)
        self.action_mapper = ActionMapper()

        # Storage
        self.cosmos = CosmosDBClient()

        # Analytics engines
        self.causal_engine = get_causal_engine()
        self.blast_predictor = BlastRadiusPredictor()
        self.economic_model = EconomicImpactModel(
            revenue_models=cfg.get("revenue_models", {})
        )
        self.threshold_engine = get_adaptive_threshold_engine()
        self.slo_tracker = get_slo_tracker()

        # Feedback loop
        self.feedback = FeedbackLoop(self.cosmos)

        # Load service dependency graph from config
        graph = get_dependency_graph()
        for svc, svc_cfg in cfg.get("services", {}).items():
            for dep in svc_cfg.get("dependencies", []):
                graph.add_dependency(upstream=dep, downstream=svc)

        # Load SLO definitions — FIXED: was missing fast_burn_threshold and slow_burn_threshold,
        # so custom thresholds in config.json were silently dropped (e.g. payment-api's 14.4 / 6.0).
        from analytics.slo_tracker import SLODefinition
        for svc, slo_cfg in cfg.get("slo_definitions", {}).items():
            self.slo_tracker._definitions[svc] = SLODefinition(
                service=svc,
                slo_target_pct=slo_cfg.get("slo_target_pct") or slo_cfg.get("target_pct", 99.9),
                window_days=slo_cfg.get("window_days", 30),
                fast_burn_threshold=slo_cfg.get("fast_burn_threshold") or slo_cfg.get("fast_burn_multiplier", 14.4),
                slow_burn_threshold=slo_cfg.get("slow_burn_threshold") or slo_cfg.get("slow_burn_multiplier", 6.0),
            )

        # Multi-model consensus (P1 only).
        # FIXED: "consensus_deployment_2" was a stale Azure-era key that always fell back
        # to its default "gpt-4o", giving the list ["gpt-4o-mini", "gpt-4o", "gpt-4o"].
        # Two identical models in the consensus pool defeats the purpose of multi-model voting.
        # Now uses the three distinct models defined in config (primary, consensus, fallback).
        cfg_openai = cfg.get("openai", {})
        all_deployments = [
            cfg_openai.get("primary_model",    "gpt-4o-mini"),
            cfg_openai.get("consensus_model_2", "gpt-4o"),
            cfg_openai.get("fallback_model",    "gpt-4o-mini"),
        ]
        # Deduplicate while preserving order, so identical models don't get extra votes
        seen = set()
        unique_deployments = [d for d in all_deployments if not (d in seen or seen.add(d))]
        # consensus_stagger_seconds: delay between model calls to avoid
        # simultaneous API burst on free-tier rate limits.
        stagger = cfg.get("agents", {}).get("consensus_stagger_seconds", 2.0)
        self._consensus = MultiModelConsensus(
            reason_fn=self.incident_agent._reason_with_deployment,
            deployments=unique_deployments,
            stagger_seconds=stagger,
        )

        self._executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="orch")

    def __del__(self):
        """Cleanly shut down the thread pool on garbage collection / server teardown."""
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def handle_event(self, event_type: str, payload: dict) -> dict:
        """Route event_type to the correct pipeline. Single entry point."""
        state = PipelineState(event_type, payload)
        state.payload["incident_id"] = state.incident_id
        logger.info(f"Orchestrator: event={event_type} id={state.incident_id}")

        try:
            if event_type in ("incident_alert", "anomaly_signal", "deployment_failure"):
                return self._run_incident_pipeline(state)
            elif event_type == "cost_spike":
                return self._run_finops_pipeline(state)
            elif event_type == "security_alert":
                return self._run_security_pipeline(state)
            elif event_type == "predictive_signal":
                return self._run_predictive_pipeline(state)
            else:
                logger.warning(f"Unknown event_type: {event_type}")
                return {"status": "skipped", "reason": f"No pipeline for {event_type}"}
        except Exception as e:
            logger.error(f"Pipeline fatal error [{state.incident_id}]: {e}", exc_info=True)
            # Re-raise so Service Bus dead-letters after max delivery count
            raise

    # ------------------------------------------------------------------
    # Incident pipeline (the main one — all new features live here)
    # ------------------------------------------------------------------
    def _run_incident_pipeline(self, state: PipelineState) -> dict:
        p = state.payload
        sid = state.incident_id
        service = state.service
        metrics = p.get("metrics", {})
        _pipeline_start = time.time()
        _step_timings: list = []  # [{name, status, duration_ms, detail}]

        def _tick(name: str, detail: str = "") -> float:
            """Record start time for a step; returns start timestamp."""
            return time.time()

        def _tock(name: str, t0: float, detail: str = "", status: str = "complete"):
            """Record completed step into _step_timings."""
            _step_timings.append({
                "name":        name,
                "status":      status,
                "duration_ms": round((time.time() - t0) * 1000),
                "detail":      detail,
            })

        # ── Step 0: Adaptive threshold evaluation ────────────────────
        if not state.is_done("adaptive_eval"):
            logger.info(f"[{sid}] Step 0: Adaptive threshold evaluation")
            _t0 = time.time()
            state.adaptive_evaluation = self.threshold_engine.evaluate_all(service, metrics)
            _tock("adaptive_eval", _t0)
            state.mark_done("adaptive_eval")

        # Feed metrics into causal engine history
        self.causal_engine.record_metrics(service, metrics)

        # ── Step 1: Incident triage (+ multi-model consensus for P1) ─
        if not state.is_done("incident_agent"):
            logger.info(f"[{sid}] Step 1: IncidentAgent")
            state.incident_result = self.incident_agent.execute(p)

            # Multi-model consensus for P1 decisions
            initial_severity = state.incident_result.get("severity", "P4")
            if initial_severity == "P1":
                logger.info(f"[{sid}] P1 detected — running multi-model consensus")
                from agents.incident_agent import SYSTEM_PROMPT as INC_SYSTEM
                user_prompt = (
                    f"Service: {service}\n"
                    f"Alert: {p.get('alert_description', '')}\n"
                    f"Metrics: {metrics}\n"
                    f"Determine mitigation action and respond in JSON only."
                )
                consensus = self._consensus.vote(
                    system_prompt=INC_SYSTEM,
                    user_prompt=user_prompt,
                    parse_fn=lambda t: self.incident_agent.safe_parse_json(t, state.incident_result),
                    fallback=state.incident_result,
                )
                state.consensus_result = MultiModelConsensus.to_dict(consensus)

                # Override incident result with consensus decision
                if consensus.reached_consensus:
                    state.incident_result["severity"] = consensus.agreed_severity
                    state.incident_result["action"] = consensus.agreed_action
                    state.incident_result["consensus_confidence"] = consensus.confidence
                else:
                    # No consensus → force human approval
                    state.incident_result["action"] = "alert_only"
                    state.incident_result["consensus_override"] = True
                    logger.warning(f"[{sid}] No consensus — overriding to alert_only")

            _tock("incident_agent", _pipeline_start,
                  detail=f"{state.incident_result.get('severity','?')} · {state.incident_result.get('action','?')}",
                  status='complete')
            state.mark_done("incident_agent")

        if not state.incident_result.get("incident_confirmed", False):
            logger.info(f"[{sid}] Incident not confirmed — no action.")
            return {"status": "no_incident", "incident_id": sid, "agent_result": state.incident_result}

        severity = state.incident_result.get("severity", "P4")
        proposed_action = state.incident_result.get("action", "alert_only")

        # ── Step 2: Causal Inference ──────────────────────────────────
        if not state.is_done("causal"):
            logger.info(f"[{sid}] Step 2: Causal Inference Engine")
            incident_metric = "error_rate"
            breached = state.adaptive_evaluation
            if breached:
                worst = max(breached.items(), key=lambda x: x[1].get("z_score", 0))
                incident_metric = worst[0]

            causal_chain = self.causal_engine.analyse_incident(
                incident_service=service,
                incident_metric=incident_metric,
                external_metrics=p.get("service_metrics", {}),
            )
            _t0 = time.time()
            state.causal_analysis = self.causal_engine.to_dict(causal_chain)
            _tock("causal", _t0, detail=state.causal_analysis.get('root_cause_service',''))
            state.mark_done("causal")

        # ── Step 3: Blast Radius Prediction ──────────────────────────
        if not state.is_done("blast_radius"):
            logger.info(f"[{sid}] Step 3: Blast Radius Prediction for '{proposed_action}'")
            load_mult = p.get("current_load_multiplier", 1.0)
            br_report = self.blast_predictor.predict(
                action_type=proposed_action,
                target_service=service,
                current_load_multiplier=load_mult,
            )
            _t0 = time.time()
            state.blast_radius = BlastRadiusPredictor.to_report_dict(br_report)
            _tock("blast_radius", _t0, detail=state.blast_radius.get('risk_level',''))
            state.mark_done("blast_radius")

        # ── Step 4: SLO Error Budget Check ───────────────────────────
        if not state.is_done("slo"):
            logger.info(f"[{sid}] Step 4: SLO Error Budget Check")
            recent_incidents = self._fetch_recent_incidents(service)
            slo_status = self.slo_tracker.compute_status(
                service=service,
                recent_incidents=recent_incidents,
                current_error_rate_pct=metrics.get("error_rate", 0),
            )
            _t0 = time.time()
            state.slo_status = self.slo_tracker.to_dict(slo_status)
            _tock("slo", _t0, detail=state.slo_status.get('policy', {}).get('tier',''))
            state.mark_done("slo")

        # ── Step 5: RCA + Predictive (parallel) ──────────────────────
        if not state.is_done("parallel_agents"):
            logger.info(f"[{sid}] Step 5: RCA + Predictive agents (parallel)")
            rca_payload = {
                **p,
                "description": state.incident_result.get("summary", ""),
                "causal_context": state.causal_analysis,
            }
            pred_payload = {
                **p,
                "current_metrics": metrics,
                "time_series": p.get("time_series", []),
            }
            # Previously used asyncio.run here — removed: it raises RuntimeError
            # if an event loop is already running (pytest-asyncio, Azure Function workers).
            # RCA and Predictive agents are fully synchronous — no async needed.
            # Submit directly to the shared ThreadPoolExecutor and wait for both.
            timeout = self.config["agents"]["timeout_seconds"]
            fut_rca  = self._executor.submit(self.rca_agent.execute, rca_payload)
            # Stagger: 1s gap before Predictive to avoid simultaneous LLM burst.
            # Both agents call the LLM; firing at the exact same instant doubles
            # the per-second request rate and can push Gemini free-tier into 429.
            time.sleep(1.0)
            fut_pred = self._executor.submit(self.predictive_agent.execute, pred_payload)
            _t0 = time.time()
            state.rca_result  = fut_rca.result(timeout=timeout)
            state.pred_result = fut_pred.result(timeout=timeout)
            _tock("parallel_agents", _t0, detail=state.rca_result.get('root_cause','')[:60])
            state.mark_done("parallel_agents")

        # ── Step 6: Policy evaluation (blast radius + SLO aware) ─────
        if not state.is_done("policy"):
            logger.info(f"[{sid}] Step 6: PolicyEngine (blast-radius + SLO aware)")
            decision = {
                **state.incident_result,
                "incident_id": sid,
                "blast_radius_risk": state.blast_radius.get("risk_level", "LOW"),
                "blast_radius_safe": state.blast_radius.get("safe_to_auto_execute", True),
                "slo_policy_tier": state.slo_status.get("policy", {}).get("tier", "standard"),
                "slo_requires_approval": state.slo_status.get("policy", {}).get(
                    "requires_human_approval", False
                ),
                "consensus_confidence": state.consensus_result.get("confidence"),
            }
            _t0 = time.time()
            state.policy_result = self.policy_engine.evaluate(decision)
            _tock("policy", _t0, detail=state.policy_result.get('reason','')[:60])
            state.mark_done("policy")

        # ── Step 7: Execute action ────────────────────────────────────
        if not state.is_done("action"):
            logger.info(f"[{sid}] Step 7: Action dispatch")
            if state.policy_result["approved"]:
                state.automation_result = self.action_mapper.dispatch(
                    action=state.policy_result["action"],
                    context={**p, **state.incident_result},
                )
                notify_teams({
                    **p,
                    **state.incident_result,
                    "incident_id": sid,
                    "blast_radius": state.blast_radius,
                    "causal_analysis": state.causal_analysis,
                })
            else:
                state.automation_result = {
                    "status": "policy_denied",
                    "reason": state.policy_result.get("reason"),
                }
                if state.policy_result.get("requires_approval"):
                    notify_teams({
                        **p,
                        "severity": severity,
                        "summary": f"[APPROVAL REQUIRED] {state.incident_result.get('summary', '')}",
                        "action": state.policy_result.get("action", "unknown"),
                        "incident_id": sid,
                        "slo_alert": state.slo_status.get("alert_message", ""),
                        "blast_radius": state.blast_radius,
                    })
            _tock("action", _pipeline_start, detail=state.automation_result.get('status',''))
            state.mark_done("action")

        # ── Step 9: Economic Impact ───────────────────────────────────
        if not state.is_done("economic"):
            logger.info(f"[{sid}] Step 9: Economic Impact Model")
            mttr = None
            if state.automation_result.get("status") == "success":
                # BUG 1 FIX: state.incident_result["timestamp"] is always set by
                # BaseAgent.execute() to the agent completion time (just seconds ago),
                # not the actual incident start. Use the event payload timestamp first
                # (the time the alert was detected), and fall back only if absent.
                ts_str = p.get("timestamp") or state.incident_result.get("timestamp")
                if ts_str:
                    try:
                        start = datetime.fromisoformat(ts_str)
                        if start.tzinfo is None:
                            start = start.replace(tzinfo=timezone.utc)
                        mttr = (datetime.now(timezone.utc) - start).total_seconds() / 60
                    except Exception:
                        mttr = None

            impact = self.economic_model.compute_impact(
                service=service,
                severity=severity,
                error_rate_pct=metrics.get("error_rate", 0),
                incident_start_iso=p.get("timestamp"),
                automated_mttr_minutes=mttr,
            )
            state.economic_impact = self.economic_model.to_dict(impact)
            state.mark_done("economic")

        # ── Step 8: Persist to Cosmos DB ─────────────────────────────
        if not state.is_done("persist"):
            logger.info(f"[{sid}] Step 8: Persisting to Cosmos DB")
            # BUG 2 FIX: IncidentRecord.timestamp defaulted to utcnow() at construction
            # (the moment the dataclass is built, inside Step 8), so record.resolve()
            # computed MTTR as ~0 seconds (record_created → resolved_at = milliseconds).
            # Pass the actual incident start from the payload so MTTR reflects real duration.
            incident_start_ts = (
                p.get("timestamp")
                or state.incident_result.get("timestamp")
                or ""
            )
            record = IncidentRecord(
                id=sid,
                service=service,
                severity=severity,
                description=state.incident_result.get("summary", ""),
                timestamp=incident_start_ts or None,
                action=state.policy_result.get("action", "alert_only"),
                agent=state.incident_result.get("agent"),
                incident_confirmed=True,
                root_cause=state.rca_result.get("root_cause"),
                affected_components=state.rca_result.get("affected_components", []),
                remediation_steps=state.rca_result.get("remediation_steps", []),
                automation_result=state.automation_result,
                policy_approved=state.policy_result.get("approved", False),
                tags={
                    "causal_root": state.causal_analysis.get("root_cause_service", ""),
                    "blast_risk": state.blast_radius.get("risk_level", ""),
                    "slo_tier": state.slo_status.get("policy", {}).get("tier", ""),
                    "consensus": state.consensus_result.get("confidence", "N/A"),
                    # Store error_rate so economics.js can compute real per-incident savings
                    # instead of always defaulting to 5% (metrics dict is not persisted otherwise)
                    "error_rate": str(metrics.get("error_rate", "")),
                },
            )
            if state.automation_result.get("status") == "success":
                record.resolve(f"Automated: {state.policy_result.get('action')}")
            elif state.policy_result.get("requires_approval"):
                # BUG 5 FIX: IncidentRecord.status supports "awaiting_approval" but it
                # was never set. Policy-denied incidents requiring human sign-off were
                # persisted as "open", making the dashboard's approval queue always empty.
                record.status = "awaiting_approval"

            # Merge full pipeline result into the Cosmos document so the dashboard
            # can render consensus, causal analysis, RCA, prediction, economic impact etc.
            enriched = {
                **record.to_dict(),
                "consensus":           state.consensus_result,
                "causal_analysis":     state.causal_analysis,
                "blast_radius":        state.blast_radius,
                "slo_status":          state.slo_status,
                "rca":                 state.rca_result,
                "prediction":          state.pred_result,
                "policy":              state.policy_result,
                "automation":          state.automation_result,
                "economic_impact":     state.economic_impact,
                "adaptive_thresholds": state.adaptive_evaluation,
                "pipeline_steps":      _step_timings,
                "pipeline_total_ms":   round((time.time() - _pipeline_start) * 1000),
            }
            self.cosmos.save_incident(enriched)
            state.mark_done("persist")


        # ── Step 10: Feedback Loop ────────────────────────────────────
        if not state.is_done("feedback"):
            logger.info(f"[{sid}] Step 10: Feedback loop")
            state.feedback_score = self.feedback.score_outcome(
                incident_id=sid,
                service=service,
                automation_status=state.automation_result.get("status"),
                severity=severity,
                mttr_minutes=state.economic_impact.get("automated_mttr_minutes"),
            )
            state.mark_done("feedback")

        # ── Step 11: SLO Downtime Recording ──────────────────────────
        if not state.is_done("slo_record"):
            mttr_min = state.economic_impact.get("automated_mttr_minutes") or 0
            if mttr_min > 0:
                self.slo_tracker.record_downtime(service, mttr_min)
            state.mark_done("slo_record")

        # ── Step 12: Auto Runbook Generation (P1 and P2) ─────────────
        if not state.is_done("runbook") and severity in ("P1", "P2"):
            if state.automation_result.get("status") == "success":
                logger.info(f"[{sid}] Step 12: Auto-generating runbook")
                try:
                    runbook_payload = {
                        **p,
                        "incident_id": sid,
                        "severity": severity,
                        "description": state.incident_result.get("summary", ""),
                        "rca": state.rca_result,
                        "automation": state.automation_result,
                        "causal_analysis": state.causal_analysis,
                        "blast_radius": state.blast_radius,
                        "mttr_minutes": state.economic_impact.get("automated_mttr_minutes"),
                    }
                    state.runbook_result = self.runbook_generator.run(runbook_payload)
                except Exception as e:
                    logger.error(f"Runbook generation failed (non-fatal): {e}")
                    state.runbook_result = {"status": "error", "error": str(e)}
            state.mark_done("runbook")

        logger.info(
            f"[{sid}] Pipeline complete: status={state.automation_result.get('status')} "
            f"severity={severity} savings=${state.economic_impact.get('revenue_saved_by_automation_usd', 0)}"
        )

        return {
            "status": "complete",
            "incident_id": sid,
            "severity": severity,
            "service": service,

            # Core outputs
            "incident": state.incident_result,
            "consensus": state.consensus_result,
            "causal_analysis": state.causal_analysis,
            "blast_radius": state.blast_radius,
            "slo_status": state.slo_status,
            "rca": state.rca_result,
            "prediction": state.pred_result,
            "policy": state.policy_result,
            "automation": state.automation_result,
            "adaptive_thresholds": state.adaptive_evaluation,

            # Business outputs
            "economic_impact": state.economic_impact,
            "feedback_score": state.feedback_score,
            "runbook": state.runbook_result,

            # Convenience
            "mttr_minutes": state.economic_impact.get("automated_mttr_minutes"),
            "revenue_saved_usd": state.economic_impact.get("revenue_saved_by_automation_usd"),
            "headline": state.economic_impact.get("headline", ""),
        }

    # ------------------------------------------------------------------
    # FinOps pipeline
    # ------------------------------------------------------------------
    def _run_finops_pipeline(self, state: PipelineState) -> dict:
        result = self.finops_agent.execute(state.payload)
        # BUG 3 FIX: cosmos_client.save_incident() uses incident.get("id") as the
        # Cosmos document ID. Without an explicit "id", save_incident generates a random
        # UUID, making the record unretrievable via get_incident(state.incident_id, service).
        self.cosmos.save_incident({
            **state.payload,
            "id": state.incident_id,
            "agent": "FinOpsAgent",
            "severity": "P3",
            "action": "alert_only",
            "finops_analysis": result,
        })
        return result

    # ------------------------------------------------------------------
    # Security pipeline
    # ------------------------------------------------------------------
    def _run_security_pipeline(self, state: PipelineState) -> dict:
        result = self.security_agent.execute(state.payload)
        risk = result.get("security_risk_score", 0)
        severity = "P1" if risk >= 70 else "P2" if risk >= 40 else "P3"
        # BUG 3 FIX: same as finops — must pass explicit id so the Cosmos document
        # is retrievable by state.incident_id via get_incident().
        self.cosmos.save_incident({
            **state.payload,
            "id": state.incident_id,
            "agent": "SecurityAgent",
            "severity": severity,
            "action": "alert_only",
            "security_report": result,
        })
        if result.get("critical_threats"):
            notify_teams({
                **state.payload,
                "severity": severity,
                "summary": f"Security: {len(result['critical_threats'])} critical threat(s) detected.",
                "action": "enforce_policy",
                "incident_id": state.incident_id,
            })
        return result

    # ------------------------------------------------------------------
    # Predictive pipeline
    # ------------------------------------------------------------------
    def _run_predictive_pipeline(self, state: PipelineState) -> dict:
        pred_payload = {**state.payload, "current_metrics": state.payload.get("metrics", {})}
        result = self.predictive_agent.execute(pred_payload)
        if result.get("failure_predicted") and result.get("probability", 0) > 0.7:
            self.cosmos.save_incident({
                **state.payload,
                "agent": "PredictiveFailureAgent",
                "severity": "P2",
                "action": "alert_only",
                "prediction": result,
            })
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fetch_recent_incidents(self, service: str) -> list:
        """Fetch last 30 days of incidents for a service from Cosmos."""
        try:
            return self.cosmos.get_incidents_by_service(service, limit=200)
        except Exception as e:
            logger.warning(f"Could not fetch recent incidents for SLO: {e}")
            return []


# ---------------------------------------------------------------------------
# Feedback Loop
# ---------------------------------------------------------------------------
class FeedbackLoop:
    """Scores automation outcomes and persists to Cosmos DB."""

    OUTCOME_SCORES = {
        ("success", "P1"): 10, ("success", "P2"): 7,
        ("success", "P3"): 5,  ("success", "P4"): 3,
        ("error",   "P1"): -10, ("error",  "P2"): -6,
        ("error",   "P3"): -3,  ("error",  "P4"): -1,
        ("policy_denied", "P1"): -2,
        ("skipped", "P1"): -1,
    }

    def __init__(self, cosmos: CosmosDBClient):
        self.cosmos = cosmos

    def score_outcome(
        self,
        incident_id: str,
        service: str,
        automation_status: str,
        severity: str,
        mttr_minutes: float = None,
    ) -> dict:
        base_score = self.OUTCOME_SCORES.get((automation_status, severity), 0)
        mttr_bonus = 0
        if mttr_minutes is not None and automation_status == "success":
            if mttr_minutes <= 2:   mttr_bonus = 5
            elif mttr_minutes <= 5: mttr_bonus = 3
            elif mttr_minutes <= 15: mttr_bonus = 1

        total_score = base_score + mttr_bonus
        record = {
            "id": f"feedback-{incident_id}",
            "service": service,
            "incident_id": incident_id,
            "severity": severity,
            "automation_status": automation_status,
            "base_score": base_score,
            "mttr_bonus": mttr_bonus,
            "total_score": total_score,
            "mttr_minutes": mttr_minutes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "feedback",
        }
        try:
            self.cosmos.save_incident(record)
            logger.info(f"Feedback: incident={incident_id} score={total_score}")
        except Exception as e:
            logger.error(f"Failed to save feedback (non-fatal): {e}")
        return record
