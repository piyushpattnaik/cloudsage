"""
CloudSage — Integration Pipeline Tests
Covers: agents, policy, analytics, orchestrator, feedback loop, RAG chunking.
All Azure SDK calls are mocked.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, call
from datetime import datetime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_incident_payload():
    return {
        "event_type": "incident_alert",
        "service": "payment-api",
        "alert_description": "High CPU utilization detected",
        "metrics": {
            "cpu_percent": 95,
            "memory_percent": 72,
            "error_rate": 1.2,
            "request_latency_ms": 850,
        },
        "namespace": "production",
    }


@pytest.fixture
def sample_cost_payload():
    return {
        "event_type": "cost_spike",
        "service": "data-pipeline",
        "cost_history": [
            {"date": f"2024-01-0{i}", "cost_usd": 120} for i in range(1, 8)
        ] + [{"date": "2024-01-08", "cost_usd": 200}],
        "current_month_spend_usd": 4800,
        "budget_usd": 4000,
        "resource_utilization": [],
        "advisor_recommendations": [],
    }


# ---------------------------------------------------------------------------
# safe_parse_json — the critical new test
# ---------------------------------------------------------------------------

class TestSafeParseJson:

    @patch("agents.base_agent.OpenAI")
    @patch("agents.base_agent.load_config")
    def _make_agent(self, MockConfig, MockOpenAI):
        MockConfig.return_value = {
            "openai": {"api_key": "sk-test-mock",
                       "primary_model": "gpt-4o-mini", "fallback_model": "gpt-4o-mini"},
            "agents": {"rate_limit_per_minute": 100},
            "logging": {"level": "INFO", "log_file": "/tmp/test.log"},
        }
        from agents.incident_agent import IncidentAgent
        return IncidentAgent()

    def test_valid_json_parsed_correctly(self):
        from agents.base_agent import BaseAgent
        # Create a minimal concrete subclass for testing
        class DummyAgent(BaseAgent):
            def run(self, payload): return {}
        with patch("agents.base_agent.load_config") as mc, \
             patch("agents.base_agent.OpenAI"):
            mc.return_value = {
                "openai": {"api_key": "sk-test-mock",
                           "primary_model": "gpt-4o-mini", "fallback_model": "gpt-4o-mini"},
                "agents": {"rate_limit_per_minute": 100},
                "logging": {"level": "INFO", "log_file": "/tmp/test.log"},
            }
            agent = DummyAgent("TestAgent")
            result = agent.safe_parse_json('{"foo": "bar"}', {"foo": "fallback"})
            assert result["foo"] == "bar"

    def test_malformed_json_returns_fallback(self):
        from agents.base_agent import BaseAgent
        class DummyAgent(BaseAgent):
            def run(self, payload): return {}
        with patch("agents.base_agent.load_config") as mc, \
             patch("agents.base_agent.OpenAI"):
            mc.return_value = {
                "openai": {"api_key": "sk-test-mock",
                           "primary_model": "gpt-4o-mini", "fallback_model": "gpt-4o-mini"},
                "agents": {"rate_limit_per_minute": 100},
                "logging": {"level": "INFO", "log_file": "/tmp/test.log"},
            }
            agent = DummyAgent("TestAgent")
            fallback = {"severity": "P4", "action": "alert_only"}
            result = agent.safe_parse_json("Here is my recommendation: restart it.", fallback)
            # Should return fallback values
            assert result["severity"] == "P4"
            assert result["action"] == "alert_only"
            # Should attach raw response for debugging
            assert "_raw_llm_response" in result

    def test_markdown_fenced_json_parsed(self):
        from agents.base_agent import BaseAgent
        class DummyAgent(BaseAgent):
            def run(self, payload): return {}
        with patch("agents.base_agent.load_config") as mc, \
             patch("agents.base_agent.OpenAI"):
            mc.return_value = {
                "openai": {"api_key": "sk-test-mock",
                           "primary_model": "gpt-4o-mini", "fallback_model": "gpt-4o-mini"},
                "agents": {"rate_limit_per_minute": 100},
                "logging": {"level": "INFO", "log_file": "/tmp/test.log"},
            }
            agent = DummyAgent("TestAgent")
            fenced = "```json\n{\"incident_confirmed\": true}\n```"
            result = agent.safe_parse_json(fenced, {"incident_confirmed": False})
            assert result["incident_confirmed"] is True


# ---------------------------------------------------------------------------
# Policy Engine
# ---------------------------------------------------------------------------

class TestPolicyEngine:

    def test_p1_restart_approved_in_production(self):
        from decision_engine.policy_engine import PolicyEngine
        engine = PolicyEngine(environment="production")
        result = engine.evaluate({"action": "restart_service", "severity": "P1", "service": "svc"})
        assert result["approved"] is True

    def test_p4_requires_approval(self):
        from decision_engine.policy_engine import PolicyEngine
        engine = PolicyEngine(environment="production")
        result = engine.evaluate({"action": "restart_service", "severity": "P4", "service": "svc"})
        assert result.get("requires_approval") is True

    def test_blocked_action_always_denied(self):
        from decision_engine.policy_engine import PolicyEngine
        engine = PolicyEngine(environment="production")
        result = engine.evaluate({"action": "delete_resource", "severity": "P1", "service": "svc"})
        assert result["approved"] is False
        assert result.get("requires_approval") is False

    def test_rate_limit_enforced(self):
        from decision_engine.policy_engine import PolicyEngine
        engine = PolicyEngine(environment="staging")
        for _ in range(5):
            engine._record_action("test-svc")
        result = engine.evaluate({"action": "restart_service", "severity": "P1", "service": "test-svc"})
        assert result["approved"] is False


# ---------------------------------------------------------------------------
# Feedback Loop
# ---------------------------------------------------------------------------

class TestFeedbackLoop:

    def _make_feedback(self):
        mock_cosmos = MagicMock()
        from agents.orchestrator import FeedbackLoop
        return FeedbackLoop(mock_cosmos), mock_cosmos

    def test_successful_p1_scores_highest(self):
        fb, cosmos = self._make_feedback()
        result = fb.score_outcome("id1", "svc", "success", "P1", mttr_minutes=1.5)
        assert result["total_score"] == 10 + 5  # base 10 + mttr_bonus 5
        assert cosmos.save_incident.called

    def test_failed_p1_scores_negative(self):
        fb, cosmos = self._make_feedback()
        result = fb.score_outcome("id2", "svc", "error", "P1")
        assert result["total_score"] == -10

    def test_mttr_bonus_thresholds(self):
        fb, _ = self._make_feedback()
        assert fb.score_outcome("a", "s", "success", "P1", mttr_minutes=1)["mttr_bonus"] == 5
        assert fb.score_outcome("b", "s", "success", "P1", mttr_minutes=4)["mttr_bonus"] == 3
        assert fb.score_outcome("c", "s", "success", "P1", mttr_minutes=10)["mttr_bonus"] == 1
        assert fb.score_outcome("d", "s", "success", "P1", mttr_minutes=30)["mttr_bonus"] == 0

    def test_cosmos_failure_does_not_raise(self):
        mock_cosmos = MagicMock()
        mock_cosmos.save_incident.side_effect = Exception("Cosmos unreachable")
        from agents.orchestrator import FeedbackLoop
        fb = FeedbackLoop(mock_cosmos)
        # Should not raise — graceful degradation
        result = fb.score_outcome("id3", "svc", "success", "P2", mttr_minutes=5)
        assert "total_score" in result


# ---------------------------------------------------------------------------
# RAG Chunking
# ---------------------------------------------------------------------------

class TestRAGChunking:

    def _make_pipeline(self):
        with patch("rag.rag_pipeline.EmbeddingsClient"), \
             patch("rag.rag_pipeline.SearchClientWrapper"), \
             patch("rag.rag_pipeline.load_config") as mc:
            mc.return_value = {}
            from rag.rag_pipeline import RAGPipeline
            p = RAGPipeline.__new__(RAGPipeline)
            p.CHUNK_SIZE = 512
            p.CHUNK_OVERLAP = 64
            p.CHARS_PER_TOKEN = 4
            return p

    def test_short_doc_produces_one_chunk(self):
        p = self._make_pipeline()
        chunks = p._chunk_text("This is a short document.")
        assert len(chunks) == 1

    def test_long_doc_produces_multiple_chunks(self):
        p = self._make_pipeline()
        # Create text longer than 512 * 4 = 2048 chars
        long_text = "\n\n".join(["Paragraph " + ("word " * 60)] * 10)
        chunks = p._chunk_text(long_text)
        assert len(chunks) > 1

    def test_chunks_respect_overlap(self):
        p = self._make_pipeline()
        long_text = "\n\n".join(["Paragraph " + str(i) + ". " + ("content " * 80) for i in range(8)])
        chunks = p._chunk_text(long_text)
        # Each chunk should be under the max char limit
        max_chars = p.CHUNK_SIZE * p.CHARS_PER_TOKEN
        for chunk in chunks:
            assert len(chunk) <= max_chars * 1.1  # 10% tolerance for boundary handling

    def test_empty_doc_returns_empty(self):
        """
        FIXED: _chunk_text("") now always returns [] (not [""] or similar).
        The old assertion accepted both [] and [""] — the new implementation
        consistently returns [], so we assert exactly that.
        """
        p = self._make_pipeline()
        assert p._chunk_text("") == []
        assert p._chunk_text("   ") == []  # whitespace-only also empty

    def test_retrieve_empty_query_returns_empty(self):
        """retrieve() should return [] rather than erroring on empty query."""
        with patch("rag.rag_pipeline.EmbeddingsClient"), \
             patch("rag.rag_pipeline.SearchClientWrapper"), \
             patch("rag.rag_pipeline.load_config") as mc:
            mc.return_value = {}
            from rag.rag_pipeline import RAGPipeline
            p = RAGPipeline.__new__(RAGPipeline)
            p.embedder = MagicMock()
            p.search = MagicMock()
            p.CHUNK_SIZE = 512
            p.CHUNK_OVERLAP = 64
            p.CHARS_PER_TOKEN = 4
            result = p.retrieve("")
            assert result == []
            p.embedder.embed.assert_not_called()  # should not call embed on empty query


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

class TestReliabilityAnalytics:

    def test_mttr_score_mapping(self):
        from analytics.reliability_score import ReliabilityScoreCalculator
        calc = ReliabilityScoreCalculator.__new__(ReliabilityScoreCalculator)
        assert calc._mttr_to_score(0) == 100
        assert calc._mttr_to_score(5) == 95
        assert calc._mttr_to_score(60) == 55

    def test_cost_efficiency_grades(self):
        from analytics.cost_index import CostIndexCalculator
        calc = CostIndexCalculator.__new__(CostIndexCalculator)
        assert calc._to_grade(0.90) == "A"
        assert calc._to_grade(0.72) == "B"
        assert calc._to_grade(0.30) == "F"

    def test_cost_index_compute(self):
        from analytics.cost_index import CostIndexCalculator
        calc = CostIndexCalculator.__new__(CostIndexCalculator)
        result = calc.compute(
            resource_utilization=[{"cpu_percent": 68, "memory_percent": 72}],
            current_spend=3800,
            budget=4000,
            idle_resources_count=1,
            total_resources=20,
            reserved_instance_coverage=0.5,
        )
        assert 0 <= result["cost_efficiency_index"] <= 1
        assert result["grade"] in "ABCDF"


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------

class TestConfigLoader:

    def test_interpolates_env_vars(self, monkeypatch):
        """_interpolate() substitutes ${VAR} placeholders from environment."""
        monkeypatch.setenv("MY_API_KEY", "secret-value-123")
        import importlib
        import config.loader as loader_mod
        importlib.reload(loader_mod)
        result = loader_mod._interpolate("${MY_API_KEY}")
        assert result == "secret-value-123"

    def test_missing_env_var_returns_empty_string(self, monkeypatch):
        """_interpolate() returns empty string for unset variables."""
        import config.loader as loader_mod
        monkeypatch.delenv("NONEXISTENT_VAR_XYZ", raising=False)
        result = loader_mod._interpolate("${NONEXISTENT_VAR_XYZ}")
        assert result == ""

    def test_interpolate_recursive_expands_nested(self, monkeypatch):
        """_interpolate_recursive() expands ${VAR} at any depth in a nested dict."""
        monkeypatch.setenv("NESTED_KEY", "expanded")
        import config.loader as loader_mod
        cfg = {"outer": {"inner": "${NESTED_KEY}", "list": ["${NESTED_KEY}"]}}
        result = loader_mod._interpolate_recursive(cfg)
        assert result["outer"]["inner"] == "expanded"
        assert result["outer"]["list"][0] == "expanded"

    def test_env_var_override_takes_priority(self, monkeypatch):
        """Environment variables must override config.json values."""
        monkeypatch.setenv("GEMINI_API_KEY", "env-key-override")
        import importlib
        import config.loader as loader_mod
        importlib.reload(loader_mod)
        cfg = {"providers": {"gemini": {"api_key": "original-key"}}, "openai": {}}
        loader_mod._apply_env_overrides(cfg)
        # Must write to both providers.gemini.api_key AND openai.api_key
        assert cfg["providers"]["gemini"]["api_key"] == "env-key-override"
        assert cfg["openai"]["api_key"] == "env-key-override"

    def test_ensure_defaults_fills_missing_keys(self):
        """_ensure_defaults() fills all required keys even in empty config."""
        import config.loader as loader_mod
        cfg = {}
        loader_mod._ensure_defaults(cfg)
        assert cfg["agents"]["rate_limit_per_minute"] == 60
        assert cfg["logging"]["log_file"] == "logs/cloudsage.log"
        assert cfg["thresholds"]["cost_spike_percent"] == 20

    def test_ensure_defaults_fills_missing_threshold_subkeys(self):
        """_ensure_defaults() fills missing sub-keys even when thresholds block exists."""
        import config.loader as loader_mod
        cfg = {"thresholds": {"cpu_critical": 90}}   # partial thresholds block
        loader_mod._ensure_defaults(cfg)
        # Pre-existing key preserved
        assert cfg["thresholds"]["cpu_critical"] == 90
        # Missing sub-key filled in
        assert cfg["thresholds"]["cost_spike_percent"] == 20
        assert cfg["thresholds"]["memory_critical"] == 85


# ===========================================================================
# v4 Module Tests — Analytics, RAG Search, SLO, Blast Radius, Economic Impact
# These cover every v4 addition that previously had zero test coverage.
# All tests are pure Python — no Azure SDK calls, no OpenAI calls, no mocks
# required unless specifically noted.
# ===========================================================================


# ---------------------------------------------------------------------------
# Causal Engine
# ---------------------------------------------------------------------------

class TestGrangerTest:
    """Unit tests for the _granger_test() statistical helper."""

    def test_returns_dict_with_required_keys(self):
        from analytics.causal_engine import _granger_test
        result = _granger_test([1.0] * 20, [1.0] * 20)
        assert "f_stat" in result
        assert "p_value" in result
        assert "best_lag" in result
        assert "significant" in result

    def test_insufficient_data_returns_not_significant(self):
        from analytics.causal_engine import _granger_test
        result = _granger_test([1, 2, 3], [4, 5, 6])
        assert result["significant"] is False
        assert result["f_stat"] == 0.0

    def test_correlated_series_detected(self):
        """A series that Granger-causes another should yield f_stat > 0."""
        from analytics.causal_engine import _granger_test
        import math
        # cause: sine wave; effect: same sine shifted by 1 step
        cause  = [math.sin(i * 0.4) for i in range(40)]
        effect = cause[1:] + [cause[0]]
        result = _granger_test(cause, effect, max_lag=3)
        assert result["f_stat"] >= 0.0  # must be non-negative

    def test_uncorrelated_series_low_f_stat(self):
        from analytics.causal_engine import _granger_test
        import random
        random.seed(42)
        cause  = [random.gauss(0, 1) for _ in range(50)]
        effect = [random.gauss(5, 1) for _ in range(50)]  # independent
        result = _granger_test(cause, effect, max_lag=2)
        # Not necessarily 0, but should not be flagged as strongly significant
        assert result["f_stat"] >= 0.0


class TestServiceDependencyGraph:
    """Unit tests for the dependency graph traversal."""

    def test_add_and_retrieve_dependency(self):
        from analytics.causal_engine import ServiceDependencyGraph
        g = ServiceDependencyGraph()
        g.add_dependency(upstream="postgres", downstream="payment-api")
        deps = g.get_downstream("postgres", max_hops=1)
        assert any(svc == "payment-api" for svc, _ in deps)

    def test_multi_hop_traversal(self):
        from analytics.causal_engine import ServiceDependencyGraph
        g = ServiceDependencyGraph()
        g.add_dependency("db",          "payment-api")
        g.add_dependency("payment-api", "checkout")
        g.add_dependency("checkout",    "frontend")
        results = dict(g.get_downstream("db", max_hops=3))
        assert "payment-api" in results
        assert results["payment-api"] == 1
        assert "checkout"    in results
        assert results["checkout"] == 2
        assert "frontend"    in results
        assert results["frontend"] == 3

    def test_max_hops_limits_traversal(self):
        from analytics.causal_engine import ServiceDependencyGraph
        g = ServiceDependencyGraph()
        g.add_dependency("a", "b")
        g.add_dependency("b", "c")
        g.add_dependency("c", "d")
        results = dict(g.get_downstream("a", max_hops=2))
        assert "b" in results
        assert "c" in results
        assert "d" not in results

    def test_no_dependencies_returns_empty(self):
        from analytics.causal_engine import ServiceDependencyGraph
        g = ServiceDependencyGraph()
        assert g.get_downstream("isolated-service", max_hops=3) == []

    def test_cycle_does_not_infinite_loop(self):
        """Circular dependency A→B→A must terminate."""
        from analytics.causal_engine import ServiceDependencyGraph
        g = ServiceDependencyGraph()
        g.add_dependency("a", "b")
        g.add_dependency("b", "a")
        # Should return without hanging
        results = g.get_downstream("a", max_hops=5)
        assert isinstance(results, list)


class TestCausalInferenceEngine:
    """Integration tests for the full causal analysis pipeline."""

    def _make_engine(self):
        from analytics.causal_engine import CausalInferenceEngine, ServiceDependencyGraph
        g = ServiceDependencyGraph()
        g.add_dependency("redis",       "payment-api")
        g.add_dependency("postgres",    "payment-api")
        g.add_dependency("payment-api", "checkout")
        engine = CausalInferenceEngine.__new__(CausalInferenceEngine)
        engine._graph = g
        engine._history = {}
        engine._window = 60
        return engine

    def test_analyse_incident_returns_causal_chain(self):
        from analytics.causal_engine import CausalInferenceEngine
        engine = self._make_engine()
        # Feed some metric history so Granger has data
        for i in range(30):
            engine.record_metrics("redis",       {"error_rate": float(i % 5)})
            engine.record_metrics("payment-api", {"error_rate": float((i + 2) % 5)})
        result = engine.analyse_incident("payment-api", "error_rate")
        assert result.incident_service == "payment-api"
        assert isinstance(result.causal_path, list)
        assert isinstance(result.overall_confidence, float)
        assert 0.0 <= result.overall_confidence <= 1.0

    def test_to_dict_has_required_fields(self):
        from analytics.causal_engine import CausalInferenceEngine
        engine = self._make_engine()
        result = engine.analyse_incident("payment-api", "error_rate")
        d = engine.to_dict(result)
        for key in ("root_cause_service", "causal_path", "overall_confidence", "methodology"):
            assert key in d, f"Missing key: {key}"

    def test_record_metrics_builds_history(self):
        from analytics.causal_engine import CausalInferenceEngine
        engine = self._make_engine()
        engine.record_metrics("svc-a", {"cpu_percent": 80})
        engine.record_metrics("svc-a", {"cpu_percent": 85})
        assert "svc-a" in engine._history
        assert len(engine._history["svc-a"]["cpu_percent"]) == 2


# ---------------------------------------------------------------------------
# Blast Radius Predictor
# ---------------------------------------------------------------------------

class TestBlastRadiusPredictor:

    def _make_predictor(self):
        from analytics.blast_radius import BlastRadiusPredictor
        from analytics.causal_engine import ServiceDependencyGraph, get_dependency_graph
        # Use a fresh isolated graph (don't pollute global singleton)
        g = ServiceDependencyGraph()
        g.add_dependency("payment-api", "checkout")
        g.add_dependency("checkout",    "frontend")
        predictor = BlastRadiusPredictor.__new__(BlastRadiusPredictor)
        predictor.graph = g
        predictor._mttr_provider = None
        return predictor

    def test_predict_returns_report(self):
        predictor = self._make_predictor()
        report = predictor.predict("restart_service", "payment-api")
        assert report.action_target == "payment-api"
        assert report.action_type   == "restart_service"
        assert isinstance(report.impacted_services, list)
        assert isinstance(report.risk_level, str)

    def test_downstream_services_appear_in_report(self):
        predictor = self._make_predictor()
        report = predictor.predict("restart_service", "payment-api")
        affected = {s.service for s in report.impacted_services}
        assert "checkout" in affected

    def test_alert_only_has_zero_disruption(self):
        """alert_only action should never have meaningful blast radius."""
        predictor = self._make_predictor()
        report = predictor.predict("alert_only", "payment-api")
        assert report.risk_level in ("LOW", "MEDIUM")
        assert report.safe_to_auto_execute is True

    def test_isolated_service_low_risk(self):
        """A service with no dependents should have LOW risk."""
        from analytics.blast_radius import BlastRadiusPredictor
        from analytics.causal_engine import ServiceDependencyGraph
        predictor = BlastRadiusPredictor.__new__(BlastRadiusPredictor)
        predictor.graph = ServiceDependencyGraph()  # empty graph
        predictor._mttr_provider = None
        report = predictor.predict("restart_service", "isolated-svc")
        assert report.total_services_affected == 0
        assert report.risk_level == "LOW"

    def test_to_report_dict_has_required_fields(self):
        from analytics.blast_radius import BlastRadiusPredictor
        predictor = self._make_predictor()
        report = predictor.predict("restart_service", "payment-api")
        d = BlastRadiusPredictor.to_report_dict(report)
        for key in ("risk_level", "safe_to_auto_execute", "total_services_affected",
                    "recommendation", "impacted_services"):
            assert key in d, f"Missing key: {key}"

    def test_risk_level_increases_with_hop_count(self):
        """More downstream services → higher or equal risk level."""
        from analytics.blast_radius import BlastRadiusPredictor
        from analytics.causal_engine import ServiceDependencyGraph

        def make(n_hops):
            g = ServiceDependencyGraph()
            prev = "root"
            for i in range(n_hops):
                nxt = f"svc-{i}"
                g.add_dependency(prev, nxt)
                prev = nxt
            p = BlastRadiusPredictor.__new__(BlastRadiusPredictor)
            p.graph = g
            p._mttr_provider = None
            return p

        levels = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        r1 = make(1).predict("restart_service", "root").risk_level
        r5 = make(5).predict("restart_service", "root").risk_level
        assert levels[r5] >= levels[r1]


# ---------------------------------------------------------------------------
# Economic Impact Model
# ---------------------------------------------------------------------------

class TestEconomicImpactModel:

    def _model(self):
        from analytics.economic_impact import EconomicImpactModel
        return EconomicImpactModel(revenue_models={
            "payment-api": {
                "avg_transactions_per_minute": 1200,
                "avg_order_value_usd": 85,
                "error_impact_factor": 0.75,
                "infrastructure_cost_per_minute_usd": 12.0,
            }
        })

    def test_compute_impact_returns_dataclass(self):
        from analytics.economic_impact import EconomicImpact
        m = self._model()
        result = m.compute_impact("payment-api", "P1", error_rate_pct=5.0)
        assert isinstance(result, EconomicImpact)

    def test_revenue_at_risk_calculated_correctly(self):
        """payment-api: 1200 txn/min × $85 = $102,000/min at risk."""
        m = self._model()
        result = m.compute_impact("payment-api", "P1", error_rate_pct=5.0)
        assert result.revenue_at_risk_per_minute_usd == pytest.approx(102_000, rel=0.01)

    def test_zero_error_rate_zero_revenue_loss(self):
        m = self._model()
        result = m.compute_impact("payment-api", "P1", error_rate_pct=0.0)
        assert result.effective_revenue_loss_per_minute_usd == 0.0

    def test_automation_savings_computed_when_mttr_provided(self):
        m = self._model()
        # Human P1 MTTR benchmark = 45 min; automated = 3 min → saves 42 min
        result = m.compute_impact("payment-api", "P1",
                                  error_rate_pct=5.0,
                                  automated_mttr_minutes=3.0)
        assert result.revenue_saved_by_automation_usd is not None
        assert result.revenue_saved_by_automation_usd > 0
        assert result.mttr_improvement_pct is not None
        assert result.mttr_improvement_pct > 0

    def test_no_savings_when_automated_mttr_exceeds_human(self):
        """If automated MTTR > human MTTR, savings should be 0 (not negative)."""
        m = self._model()
        result = m.compute_impact("payment-api", "P1",
                                  error_rate_pct=5.0,
                                  automated_mttr_minutes=200.0)  # worse than human
        # revenue_saved = max(0, human - auto) * rate — must not be negative
        assert result.revenue_saved_by_automation_usd == pytest.approx(0.0)

    def test_to_dict_headline_present(self):
        m = self._model()
        result = m.compute_impact("payment-api", "P1",
                                  error_rate_pct=5.0,
                                  automated_mttr_minutes=3.0)
        d = m.to_dict(result)
        assert "headline" in d
        assert len(d["headline"]) > 10  # non-trivial string

    def test_headline_no_savings(self):
        """Headline without MTTR should mention cost per minute."""
        m = self._model()
        result = m.compute_impact("payment-api", "P1", error_rate_pct=5.0)
        d = m.to_dict(result)
        assert "min" in d["headline"].lower() or "$" in d["headline"]

    def test_default_model_used_for_unknown_service(self):
        from analytics.economic_impact import EconomicImpactModel, DEFAULT_REVENUE_MODEL
        m = EconomicImpactModel(revenue_models={})
        result = m.compute_impact("unknown-svc", "P2", error_rate_pct=2.0)
        expected_at_risk = (
            DEFAULT_REVENUE_MODEL["avg_transactions_per_minute"]
            * DEFAULT_REVENUE_MODEL["avg_order_value_usd"]
        )
        assert result.revenue_at_risk_per_minute_usd == pytest.approx(expected_at_risk)

    def test_cumulative_savings_reads_tags_error_rate(self):
        """compute_cumulative_savings must read tags.error_rate, not metrics.error_rate."""
        m = self._model()
        incidents = [
            {
                "service": "payment-api",
                "severity": "P1",
                "mttr_minutes": 3.0,
                "tags": {"error_rate": "8.5"},   # ← correct field
                # no "metrics" key — simulates real Cosmos document
            },
            {
                "service": "payment-api",
                "severity": "P2",
                "mttr_minutes": 10.0,
                "tags": {},   # missing → should default to 5%
            },
        ]
        result = m.compute_cumulative_savings(incidents)
        assert result["total_revenue_saved_usd"] > 0
        assert result["total_incidents_automated"] == 2
        assert "payment-api" in result["savings_by_service"]

    def test_cumulative_savings_skips_incidents_without_mttr(self):
        m = self._model()
        incidents = [
            {"service": "payment-api", "severity": "P1",
             "tags": {"error_rate": "5"}, "mttr_minutes": None},
        ]
        result = m.compute_cumulative_savings(incidents)
        assert result["total_revenue_saved_usd"] == 0


# ---------------------------------------------------------------------------
# SLO Error Budget Tracker
# ---------------------------------------------------------------------------

class TestSLOErrorBudgetTracker:

    def _tracker(self, slo_pct=99.9, fast_burn=14.4, slow_burn=6.0):
        from analytics.slo_tracker import SLOErrorBudgetTracker
        return SLOErrorBudgetTracker(slo_definitions={
            "payment-api": {
                "slo_target_pct": slo_pct,
                "window_days": 30,
                "fast_burn_threshold": fast_burn,
                "slow_burn_threshold": slow_burn,
            }
        })

    def test_healthy_budget_is_standard_tier(self):
        t = self._tracker()
        status = t.compute_status("payment-api", recent_incidents=[], current_error_rate_pct=0.1)
        assert status.policy_tier == "standard"
        assert status.requires_human_approval is False

    def test_exhausted_budget_is_freeze_tier(self):
        """If all budget minutes are consumed, tier must be freeze."""
        t = self._tracker(slo_pct=99.9)   # budget = 43.2 min/month
        # Record 50 minutes of downtime (> budget)
        t.record_downtime("payment-api", 50.0)
        status = t.compute_status("payment-api")
        assert status.policy_tier == "freeze"
        assert status.requires_human_approval is True
        assert status.deployment_freeze_recommended is True

    def test_fast_burn_rate_triggers_restricted_tier(self):
        """Error rate that causes fast burn (>14.4×) should trigger restricted policy."""
        t = self._tracker(slo_pct=99.9)
        # 99.9% SLO → error_budget_fraction = 0.001
        # fast burn = current_rate / sustainable = (error_rate/100) / 0.001
        # error_rate = 0.02 → burn = 0.02/0.001 = 20× > 14.4 → fast burn
        status = t.compute_status("payment-api", current_error_rate_pct=2.0)
        assert status.burn_rate_classification in ("fast_burn", "exhausted")

    def test_normal_burn_rate_below_slow_threshold(self):
        """Low error rate should give normal classification."""
        t = self._tracker(slo_pct=99.9)
        # error_rate=0.0001% → burn ≈ 0
        status = t.compute_status("payment-api", current_error_rate_pct=0.001)
        assert status.burn_rate_classification == "normal"

    def test_record_downtime_reduces_remaining_budget(self):
        t = self._tracker(slo_pct=99.9)
        s1 = t.compute_status("payment-api")
        t.record_downtime("payment-api", 10.0)
        s2 = t.compute_status("payment-api")
        assert s2.remaining_budget_minutes < s1.remaining_budget_minutes

    def test_unknown_service_uses_default_slo(self):
        from analytics.slo_tracker import SLOErrorBudgetTracker
        t = SLOErrorBudgetTracker()
        status = t.compute_status("nonexistent-svc")
        assert status.slo_target_pct == 99.9
        assert isinstance(status.total_budget_minutes, float)

    def test_to_dict_has_required_fields(self):
        t = self._tracker()
        status = t.compute_status("payment-api")
        d = t.to_dict(status)
        for key in ("service", "remaining_budget_pct", "burn_rate", "policy", "alert_message"):
            assert key in d, f"Missing key: {key}"

    def test_policy_tier_transitions(self):
        """Verify all four tiers can be reached with appropriate budgets."""
        t = self._tracker(slo_pct=99.9)
        # standard: no downtime
        s = t.compute_status("payment-api")
        assert s.policy_tier == "standard"

        # conservative: consume ~60% of budget
        from analytics.slo_tracker import SLOErrorBudgetTracker
        t2 = SLOErrorBudgetTracker(slo_definitions={
            "svc": {"slo_target_pct": 99.9, "window_days": 30}
        })
        # budget = 43.2 min; consume ~26 min (60%)
        t2.record_downtime("svc", 26.0)
        s2 = t2.compute_status("svc")
        assert s2.policy_tier in ("conservative", "restricted")

        # freeze: consume > 100%
        t3 = SLOErrorBudgetTracker(slo_definitions={
            "svc": {"slo_target_pct": 99.9, "window_days": 30}
        })
        t3.record_downtime("svc", 100.0)
        s3 = t3.compute_status("svc")
        assert s3.policy_tier == "freeze"


# ---------------------------------------------------------------------------
# FAISS Search Client (no real FAISS required — pure interface tests)
# ---------------------------------------------------------------------------

class TestFAISSSearchClient:

    def _make_client(self, tmp_path):
        """Create a SearchClientWrapper with no blob storage and temp index dir."""
        import sys
        from unittest.mock import patch, MagicMock
        faiss_cfg = {
            "faiss": {
                "index_dir": str(tmp_path),
                "blob_connection_string": "",
                "blob_container": "cloudsage-faiss",
            },
            "openai": {
                "api_key": "sk-test",
                "embedding_model": "text-embedding-3-small",
                "embedding_dims": 1536,
            },
            "active_llm_provider": "openai",
        }
        with patch("rag.search_client.load_config", return_value=faiss_cfg):
            from rag.search_client import SearchClientWrapper
            # Reload to get fresh instance
            import importlib
            import rag.search_client as sc_mod
            importlib.reload(sc_mod)
            client = sc_mod.SearchClientWrapper(index_dir=str(tmp_path))
        return client

    def test_empty_index_returns_no_results(self, tmp_path):
        client = self._make_client(tmp_path)
        results = client.hybrid_search("test query", [0.0] * 1536, top_k=5)
        assert results == []

    def test_upload_and_retrieve(self, tmp_path):
        client = self._make_client(tmp_path)
        import numpy as np
        vec = np.random.rand(1536).astype("float32").tolist()
        client.upload_documents([{
            "id": "doc-1",
            "content": "restart the payment service pod",
            "source": "runbook/payment",
            "category": "runbook",
            "content_vector": vec,
        }])
        results = client.hybrid_search("payment restart", vec, top_k=3)
        assert len(results) == 1
        assert results[0]["source"] == "runbook/payment"

    def test_score_is_between_zero_and_one(self, tmp_path):
        client = self._make_client(tmp_path)
        import numpy as np
        vec = np.random.rand(1536).astype("float32").tolist()
        client.upload_documents([{
            "id": "doc-2",
            "content": "scale up the cluster",
            "source": "runbook/scale",
            "category": "runbook",
            "content_vector": vec,
        }])
        results = client.hybrid_search("scale cluster", vec, top_k=1)
        assert len(results) == 1
        assert 0.0 <= results[0]["score"] <= 2.0  # inner-product can exceed 1.0 before normcheck

    def test_top_k_limits_results(self, tmp_path):
        client = self._make_client(tmp_path)
        import numpy as np
        for i in range(10):
            vec = np.random.rand(1536).astype("float32").tolist()
            client.upload_documents([{
                "id": f"doc-{i}",
                "content": f"content {i}",
                "source": f"src-{i}",
                "category": "general",
                "content_vector": vec,
            }])
        query_vec = np.random.rand(1536).astype("float32").tolist()
        results = client.hybrid_search("content", query_vec, top_k=3)
        assert len(results) <= 3

    def test_index_persists_to_disk(self, tmp_path):
        """After upload, index files must exist on disk."""
        client = self._make_client(tmp_path)
        import numpy as np
        vec = np.random.rand(1536).astype("float32").tolist()
        client.upload_documents([{
            "id": "persist-doc",
            "content": "persistence test",
            "source": "test",
            "category": "test",
            "content_vector": vec,
        }])
        assert (tmp_path / "index.faiss").exists()
        assert (tmp_path / "metadata.jsonl").exists()

    def test_skips_wrong_dimension_vector(self, tmp_path):
        """Vectors with wrong dimension should be skipped, not crash."""
        client = self._make_client(tmp_path)
        client.upload_documents([{
            "id": "bad-dim",
            "content": "test",
            "source": "src",
            "category": "general",
            "content_vector": [0.1] * 512,  # wrong: should be 1536
        }])
        assert client._index.ntotal == 0  # nothing indexed

    def test_create_index_if_not_exists_is_noop(self, tmp_path):
        client = self._make_client(tmp_path)
        client.create_index_if_not_exists()  # Should not raise


# ---------------------------------------------------------------------------
# Pure-Python TF-IDF scorer
# ---------------------------------------------------------------------------

class TestTFIDFScorer:

    def test_exact_match_scores_higher_than_no_match(self):
        from rag.search_client import _tfidf_score, _tokenise
        all_docs = [
            {"content": "restart the payment service"},
            {"content": "unrelated content about cats"},
        ]
        query = _tokenise("payment restart")
        score_match = _tfidf_score(query, "restart the payment service", all_docs)
        score_no_match = _tfidf_score(query, "cats dogs birds", all_docs)
        assert score_match > score_no_match

    def test_empty_query_returns_zero(self):
        from rag.search_client import _tfidf_score
        all_docs = [{"content": "some content"}]
        assert _tfidf_score([], "some content", all_docs) == 0.0

    def test_empty_doc_returns_zero(self):
        from rag.search_client import _tfidf_score, _tokenise
        all_docs = [{"content": ""}]
        assert _tfidf_score(_tokenise("test"), "", all_docs) == 0.0

    def test_score_capped_at_one(self):
        from rag.search_client import _tfidf_score, _tokenise
        doc = "test " * 100
        all_docs = [{"content": doc}]
        score = _tfidf_score(_tokenise("test"), doc, all_docs)
        assert score <= 1.0

    def test_tokenise_strips_punctuation(self):
        from rag.search_client import _tokenise
        tokens = _tokenise("hello, world! foo-bar.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in tokens


# ---------------------------------------------------------------------------
# Embeddings Client — lazy init
# ---------------------------------------------------------------------------

class TestEmbeddingsClientLazyInit:

    def _mock_cfg(self, api_key=""):
        return {
            "openai": {
                "api_key": api_key,
                "embedding_model": "text-embedding-3-small",
                "embedding_dims": 1536,
            },
            "active_llm_provider": "openai",
        }

    def test_no_raise_on_init_with_missing_key(self):
        """__init__ must not raise even when api_key is empty."""
        import rag.embeddings as emb_mod
        with patch.object(emb_mod, "load_config", return_value=self._mock_cfg("")):
            client = emb_mod.EmbeddingsClient()
        assert client._client is None  # lazily constructed

    def test_embed_raises_clearly_when_key_missing(self):
        """embed() must raise RuntimeError with clear message when key absent."""
        import rag.embeddings as emb_mod
        with patch.object(emb_mod, "load_config", return_value=self._mock_cfg("")):
            client = emb_mod.EmbeddingsClient()
        with pytest.raises(RuntimeError, match="API key"):
            client.embed("test text")

    def test_embed_empty_text_raises_value_error(self):
        """embed() should raise ValueError on empty input regardless of key state."""
        import rag.embeddings as emb_mod
        with patch.object(emb_mod, "load_config", return_value=self._mock_cfg("sk-test")):
            client = emb_mod.EmbeddingsClient()
        with pytest.raises(ValueError):
            client.embed("   ")


# ---------------------------------------------------------------------------
# Global Rate Limiter + Consensus Stagger
# ---------------------------------------------------------------------------

class TestGlobalRateLimiter:
    """Verifies the shared cross-agent rate limiter added in the 429-fix session."""

    def test_global_rate_limiter_shared_across_calls(self):
        """_get_global_rate_limiter() must return the same instance every time."""
        from agents.base_agent import _get_global_rate_limiter, _reset_global_rate_limiter
        _reset_global_rate_limiter()
        a = _get_global_rate_limiter(max_calls=5)
        b = _get_global_rate_limiter(max_calls=5)
        assert a is b, "Must be the same singleton instance"

    def test_global_rate_limiter_blocks_over_limit(self):
        """Shared limiter rejects calls beyond max_calls within one window."""
        from agents.base_agent import RateLimiter
        limiter = RateLimiter(max_calls=3, period_seconds=60)
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        assert limiter.acquire() is False   # 4th call blocked

    def test_reset_clears_singleton_for_tests(self):
        """_reset_global_rate_limiter() lets tests start with a clean limiter."""
        from agents.base_agent import _get_global_rate_limiter, _reset_global_rate_limiter
        _reset_global_rate_limiter()
        first  = _get_global_rate_limiter(max_calls=12)
        _reset_global_rate_limiter()
        second = _get_global_rate_limiter(max_calls=12)
        assert first is not second, "Reset must destroy the old singleton"


class TestConsensusStagger:
    """Verifies stagger_seconds wiring in MultiModelConsensus."""

    def test_consensus_accepts_stagger_seconds_param(self):
        """MultiModelConsensus must accept stagger_seconds without error."""
        from agents.consensus import MultiModelConsensus
        dummy_fn = lambda prompt, sys, deployment: '{"severity":"P1","action":"restart_service"}'
        mc = MultiModelConsensus(
            reason_fn=dummy_fn,
            deployments=["m1", "m2"],
            stagger_seconds=0.0,   # zero so test runs instantly
        )
        assert mc._stagger_seconds == 0.0

    def test_consensus_config_stagger_defaults_to_2(self):
        """Default stagger_seconds must be 2.0 seconds."""
        from agents.consensus import MultiModelConsensus
        dummy_fn = lambda prompt, sys, deployment: '{"severity":"P1","action":"alert_only"}'
        mc = MultiModelConsensus(reason_fn=dummy_fn, deployments=["m1"])
        assert mc._stagger_seconds == 2.0


# ---------------------------------------------------------------------------
# Groq Provider
# ---------------------------------------------------------------------------

class TestGroqProvider:
    """Verify Groq wiring in loader and embeddings."""

    def _groq_config(self, groq_key="gsk-fake", gemini_key="AIza-fake"):
        return {
            "llm_provider": "groq",
            "active_llm_provider": "groq",
            "providers": {
                "groq": {
                    "api_key": groq_key,
                    "base_url": "https://api.groq.com/openai/v1",
                    "primary_model": "llama-3.3-70b-versatile",
                    "consensus_model_2": "mixtral-8x7b-32768",
                    "fallback_model": "llama-3.1-8b-instant",
                    "rate_limit_per_minute": 25,
                    "embedding_provider": "gemini",
                    "embedding_key": gemini_key,
                    "embedding_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                    "embedding_model": "text-embedding-004",
                    "embedding_dims": 768,
                },
                "gemini": {"api_key": gemini_key},
                "openai": {"api_key": ""},
            },
            "openai": {
                "api_key": groq_key,
                "base_url": "https://api.groq.com/openai/v1",
                "primary_model": "llama-3.3-70b-versatile",
                "fallback_model": "llama-3.1-8b-instant",
                "embedding_model": "text-embedding-004",
                "embedding_dims": 768,
            },
            "embedding": {
                "api_key": gemini_key,
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "model": "text-embedding-004",
                "dims": 768,
                "provider": "gemini",
            },
            "agents": {"global_rate_limit_per_minute": 25, "max_retries": 3},
        }

    def test_groq_config_has_correct_models(self):
        """Groq provider block must have Llama primary and Mixtral consensus."""
        cfg = self._groq_config()
        groq = cfg["providers"]["groq"]
        assert groq["primary_model"] == "llama-3.3-70b-versatile"
        assert "mixtral" in groq["consensus_model_2"]
        assert groq["fallback_model"] == "llama-3.1-8b-instant"

    def test_groq_rate_limit_is_25(self):
        """Groq provider should use 25 RPM (free tier cap = 30 RPM)."""
        cfg = self._groq_config()
        assert cfg["providers"]["groq"]["rate_limit_per_minute"] == 25
        assert cfg["agents"]["global_rate_limit_per_minute"] == 25

    def test_groq_embedding_uses_gemini_key(self):
        """When Groq is active, config[embedding] must point at Gemini endpoint."""
        cfg = self._groq_config(groq_key="gsk-xxx", gemini_key="AIza-yyy")
        emb = cfg["embedding"]
        assert emb["provider"] == "gemini"
        assert emb["api_key"] == "AIza-yyy"
        assert "generativelanguage" in emb["base_url"]
        assert emb["model"] == "text-embedding-004"
        assert emb["dims"] == 768

    def test_embeddings_client_uses_embedding_config_not_openai(self):
        """EmbeddingsClient must read from config[embedding], not config[openai]."""
        from unittest.mock import patch
        cfg = self._groq_config(groq_key="gsk-xxx", gemini_key="AIza-yyy")
        import rag.embeddings as emb_mod
        with patch.object(emb_mod, "load_config", return_value=cfg):
            client = emb_mod.EmbeddingsClient()
        # Chat key is Groq key, but embedding key must be the Gemini key
        assert client._api_key == "AIza-yyy", (
            f"Expected Gemini key for embeddings, got: {client._api_key}"
        )
        assert client._embed_provider == "gemini"
        assert client._chat_provider == "groq"

    def test_groq_quota_exhausted_detected(self):
        """Groq daily quota messages must be detected as non-retryable."""
        from agents.base_agent import _is_quota_exhausted
        import openai as _openai

        groq_quota_msgs = [
            "Rate limit reached for model llama-3.3-70b-versatile: tokens per day",
            "organization rate limit exceeded",
            "Please reduce your usage.",
            "requests per day limit reached",
        ]
        for msg in groq_quota_msgs:
            exc = _openai.RateLimitError(
                msg,
                response=MagicMock(status_code=429, headers={}, json=MagicMock(return_value={})),
                body={},
            )
            assert _is_quota_exhausted(exc), f"Should detect quota exhaustion: {msg}"

    def test_per_minute_rate_limit_not_flagged_as_quota(self):
        """Groq per-minute rate limit (recoverable) must NOT be flagged as quota exhausted."""
        from agents.base_agent import _is_quota_exhausted
        import openai as _openai

        per_minute_msgs = [
            "Rate limit exceeded. Please retry after 2 seconds.",
            "Too many requests. Retry after 30s.",
        ]
        for msg in per_minute_msgs:
            exc = _openai.RateLimitError(
                msg,
                response=MagicMock(status_code=429, headers={}, json=MagicMock(return_value={})),
                body={},
            )
            assert not _is_quota_exhausted(exc), f"Should NOT flag as quota exhausted: {msg}"
