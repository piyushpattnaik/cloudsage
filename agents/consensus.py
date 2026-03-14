"""
CloudSage — Multi-Model Consensus Engine
Runs the same incident through multiple model deployments in parallel and
aggregates their votes. Disagreement triggers a LOW confidence flag so
the PolicyEngine can escalate to human review.

When all consensus models fail (e.g. rate-limited), the engine falls back
to the primary agent's own result with LOW confidence rather than crashing.
"""

import json
import time
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger("CloudSage.Consensus")


@dataclass
class ConsensusResult:
    """Aggregated result from multi-model voting."""
    confidence: str            # HIGH | MEDIUM | LOW
    vote_count: int            # number of models that agreed with the majority
    models_consulted: int      # total models attempted
    agreed_action: str
    agreed_severity: str
    dissenting_models: list = field(default_factory=list)
    individual_results: list  = field(default_factory=list)

    @property
    def reached_consensus(self) -> bool:
        """True when at least one model responded and all agreed."""
        return self.vote_count > 0 and self.vote_count == self.models_consulted


def _safe_parse_json(text: str, fallback) -> Optional[dict]:
    """
    Parse a JSON string returned by an LLM, stripping markdown fences if present.
    Returns fallback on any parse error.
    """
    if not text:
        return fallback
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                pass
    return fallback


class MultiModelConsensus:
    """
    Runs incident classification across multiple model deployments in parallel,
    then aggregates by majority vote on (severity, action).

    stagger_seconds: gap between submitting each model call to the thread pool.
    Prevents all 3 consensus calls from hitting the API at exactly t=0,
    which on Gemini free-tier (15 RPM) triggers a 429 storm and retry cascade.
    Default: 2s gap → 3 models spread over 4s instead of simultaneous burst.
    """

    HIGH_THRESHOLD   = 1.0    # all attempted models agree
    MEDIUM_THRESHOLD = 0.5    # at least half agree

    def __init__(self, reason_fn: Callable, deployments: List,
                 stagger_seconds: float = 2.0):
        """
        Args:
            reason_fn:        Callable matching IncidentAgent._reason_with_deployment.
            deployments:      List of model name strings to consult.
            stagger_seconds:  Delay between submitting each model call.
                              Set to 0 to disable (original parallel behaviour).
        """
        self._reason_fn       = reason_fn
        self._deployments     = deployments
        self._stagger_seconds = stagger_seconds

    def vote(
        self,
        system_prompt:  str,
        user_prompt:    str,
        fallback:       dict,
        parse_fn:       Callable = None,
    ) -> ConsensusResult:
        """
        Call each deployment in parallel, collect JSON results, and vote.

        Falls back gracefully when all models fail (rate limit, network, etc.)
        rather than returning None or crashing the orchestrator.

        Args:
            system_prompt:  System prompt sent to each model.
            user_prompt:    User prompt sent to each model.
            fallback:       Result to use when all consensus calls fail.
                            Also used as the JSON parse fallback per model.
            parse_fn:       Optional callable(text) -> dict to parse each model's
                            response. Defaults to _safe_parse_json(text, fallback).

        Returns:
            ConsensusResult with aggregated vote, confidence, and per-model details.
        """
        if parse_fn is None:
            parse_fn = lambda text: _safe_parse_json(text, fallback)

        individual_results = []

        def _call_one(deployment) -> dict:
            model_name = getattr(deployment, "model", str(deployment))
            logger.info(f"Consensus call: model={model_name}")
            try:
                response_text = self._reason_fn(system_prompt, user_prompt, deployment)
                parsed = parse_fn(response_text)
                return {"model": model_name, "result": parsed, "error": None}
            except Exception as exc:
                logger.warning(f"Consensus model {model_name} exhausted retries: {exc}")
                return {"model": model_name, "result": None, "error": str(exc)}

        # Run all deployments in parallel
        # Stagger model submissions to prevent simultaneous API burst.
        # Gemini free-tier is 15 RPM: 3 simultaneous calls = instant 429 storm.
        # With stagger_seconds=2.0 (default), 3 models spread over 4s instead.
        max_workers = max(1, len(self._deployments))
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="consensus"
        ) as pool:
            futures = {}
            for i, dep in enumerate(self._deployments):
                if i > 0 and self._stagger_seconds > 0:
                    time.sleep(self._stagger_seconds)
                futures[pool.submit(_call_one, dep)] = dep
            for future in as_completed(futures):
                dep = futures[future]
                try:
                    individual_results.append(future.result())
                except Exception as exc:
                    model_name = getattr(dep, "model", str(dep))
                    logger.warning(f"Model {model_name} failed during consensus: {exc}")
                    individual_results.append(
                        {"model": model_name, "result": None, "error": str(exc)}
                    )

        # Filter to successful results only
        successful  = [r for r in individual_results if r.get("result") is not None]
        vote_count  = len(successful)
        n_attempted = len(self._deployments)

        # ── All models failed: fall back to primary result ──────────────────
        if vote_count == 0:
            logger.warning(
                f"All {n_attempted} consensus model(s) failed (rate-limit or network). "
                "Falling back to primary agent result with LOW confidence."
            )
            safe = fallback or {}
            return ConsensusResult(
                confidence="LOW",
                vote_count=0,
                models_consulted=n_attempted,
                agreed_action=safe.get("action", "alert_only"),
                agreed_severity=safe.get("severity", "P3"),
                dissenting_models=[r["model"] for r in individual_results],
                individual_results=individual_results,
            )

        # ── Majority vote on (severity, action) ────────────────────────────
        votes = Counter(
            (
                r["result"].get("severity", "P3"),
                r["result"].get("action",   "alert_only"),
            )
            for r in successful
        )
        (agreed_severity, agreed_action), top_votes = votes.most_common(1)[0]

        agreement_rate = top_votes / n_attempted
        if agreement_rate >= self.HIGH_THRESHOLD:
            confidence = "HIGH"
        elif agreement_rate >= self.MEDIUM_THRESHOLD:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        dissenting = [
            r["model"] for r in successful
            if (
                r["result"].get("severity"),
                r["result"].get("action"),
            ) != (agreed_severity, agreed_action)
        ]
        # Failed models also count as dissenting for transparency
        dissenting += [r["model"] for r in individual_results if r.get("result") is None]

        logger.info(
            f"Consensus result: {confidence} | votes={top_votes}/{n_attempted} | "
            f"severity={agreed_severity} action={agreed_action}"
        )

        return ConsensusResult(
            confidence=confidence,
            vote_count=top_votes,
            models_consulted=n_attempted,
            agreed_action=agreed_action,
            agreed_severity=agreed_severity,
            dissenting_models=dissenting,
            individual_results=individual_results,
        )

    @staticmethod
    def to_dict(consensus: "ConsensusResult") -> dict:
        """
        Serialise a ConsensusResult to a plain dict safe for JSON
        and orchestrator state.

        Guards every individual result against None — a model that was
        rate-limited or errored has result=None and is excluded from the
        individual_results list in the output but still appears in
        dissenting_models so the caller knows it failed.
        """
        valid_results = [
            r for r in (consensus.individual_results or [])
            if r.get("result") is not None
        ]

        return {
            "confidence":         consensus.confidence,
            "vote_count":         consensus.vote_count,
            "models_consulted":   consensus.models_consulted,
            "agreed_action":      consensus.agreed_action,
            "agreed_severity":    consensus.agreed_severity,
            "reached_consensus":  consensus.reached_consensus,
            "dissenting_models":  consensus.dissenting_models or [],
            "individual_results": [
                {
                    "model":      r.get("model"),
                    "severity":   r["result"].get("severity"),
                    "action":     r["result"].get("action"),
                    "confidence": r["result"].get("confidence"),
                }
                for r in valid_results
            ],
        }
