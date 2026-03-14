"""
CloudSage — Agent Unit Tests
All OpenAI and Azure SDK calls fully mocked — safe to run in CI with no credentials.
"""

import pytest
from unittest.mock import patch, MagicMock


MOCK_CONFIG = {
    "openai": {
        "api_key": "sk-test-mock",
        "primary_model": "gpt-4o-mini",
        "fallback_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "embedding_dims": 1536,
        "max_tokens": 1500,
    },
    "agents": {"rate_limit_per_minute": 100, "timeout_seconds": 30},
    "logging": {"level": "INFO", "log_file": "/tmp/cloudsage-test.log"},
    "anomaly_detection": {"contamination": 0.05},
    "thresholds": {
        "cpu_critical": 90, "cpu_warning": 75,
        "memory_critical": 85, "memory_warning": 70,
        "error_rate_critical": 5.0, "error_rate_warning": 2.0,
    },
}


class TestBaseAgent:

    def test_score_severity_levels(self):
        from agents.base_agent import BaseAgent

        assert BaseAgent.score_severity({"cpu_percent": 95}) == "P1"
        assert BaseAgent.score_severity({"cpu_percent": 80}) == "P2"
        assert BaseAgent.score_severity({"cpu_percent": 65}) == "P3"
        assert BaseAgent.score_severity({"cpu_percent": 30}) == "P4"
        assert BaseAgent.score_severity({"error_rate": 6}) == "P1"
        assert BaseAgent.score_severity({"memory_percent": 88}) == "P1"

    @patch("agents.base_agent.OpenAI")
    @patch("agents.base_agent.load_config")
    def test_reason_fallback_on_primary_failure(self, mock_load_config, MockOpenAI):
        """reason() falls back to the fallback model when the primary call raises."""
        mock_load_config.return_value = MOCK_CONFIG
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        mock_client.chat.completions.create.side_effect = [
            Exception("Primary model unavailable"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Fallback response"))]),
        ]

        from agents.incident_agent import IncidentAgent
        agent = IncidentAgent()
        agent._openai_client = mock_client

        result = agent.reason("system prompt", "user prompt")
        assert result == "Fallback response"
        assert mock_client.chat.completions.create.call_count == 2

    @patch("agents.base_agent.OpenAI")
    @patch("agents.base_agent.load_config")
    def test_both_models_exhausted_raises(self, mock_load_config, MockOpenAI):
        """When both primary and fallback fail, RuntimeError is raised."""
        mock_load_config.return_value = MOCK_CONFIG
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("All models down")

        from agents.incident_agent import IncidentAgent
        agent = IncidentAgent()
        agent._openai_client = mock_client

        # Message changed from "All OpenAI models exhausted" to "All LLM models exhausted"
        with pytest.raises(RuntimeError, match="All LLM models exhausted"):
            agent.reason("system", "user")


class TestPredictiveAgent:

    def test_anomaly_detection_with_normal_data(self):
        from agents.predictive_agent import PredictiveAgent
        agent = PredictiveAgent.__new__(PredictiveAgent)
        agent.config = {"anomaly_detection": {"contamination": 0.05}}

        normal_series = [
            {"cpu_percent": 40 + i % 5, "memory_percent": 55, "error_rate": 0.1,
             "request_latency_ms": 200, "active_connections": 100}
            for i in range(20)
        ]
        result = agent._detect_anomalies(normal_series)
        assert "anomaly_detected" in result
        assert "anomaly_scores" in result
        assert len(result["anomaly_scores"]) == 20

    def test_anomaly_detection_with_spike(self):
        from agents.predictive_agent import PredictiveAgent
        agent = PredictiveAgent.__new__(PredictiveAgent)
        agent.config = {"anomaly_detection": {"contamination": 0.1}}

        series = [
            {"cpu_percent": 40, "memory_percent": 55, "error_rate": 0.1,
             "request_latency_ms": 200, "active_connections": 100}
        ] * 18 + [
            {"cpu_percent": 99, "memory_percent": 99, "error_rate": 15,
             "request_latency_ms": 9000, "active_connections": 5000},
            {"cpu_percent": 98, "memory_percent": 97, "error_rate": 14,
             "request_latency_ms": 8500, "active_connections": 4900},
        ]
        result = agent._detect_anomalies(series)
        assert result["anomaly_detected"] is True

    def test_anomaly_detection_insufficient_data(self):
        from agents.predictive_agent import PredictiveAgent
        agent = PredictiveAgent.__new__(PredictiveAgent)
        agent.config = {"anomaly_detection": {"contamination": 0.05}}
        result = agent._detect_anomalies([{"cpu_percent": 50}])
        assert result["anomaly_detected"] is False

    def test_threshold_breach_detection(self):
        from agents.predictive_agent import PredictiveAgent
        agent = PredictiveAgent.__new__(PredictiveAgent)
        agent.config = MOCK_CONFIG

        breaches = agent._detect_threshold_breaches(
            {"cpu_percent": 95, "memory_percent": 50, "error_rate": 0.1}
        )
        assert "cpu_critical" in breaches
        assert "memory_warning" not in breaches

    def test_no_threshold_breaches_for_normal_metrics(self):
        from agents.predictive_agent import PredictiveAgent
        agent = PredictiveAgent.__new__(PredictiveAgent)
        agent.config = MOCK_CONFIG
        breaches = agent._detect_threshold_breaches(
            {"cpu_percent": 30, "memory_percent": 40, "error_rate": 0.1}
        )
        assert breaches == []


class TestSecurityAgent:

    def test_risk_scoring_empty_alerts(self):
        from agents.security_agent import SecurityAgent
        agent = SecurityAgent.__new__(SecurityAgent)
        assert agent._score_risk([]) == 10

    def test_risk_scoring_high_severity(self):
        from agents.security_agent import SecurityAgent
        agent = SecurityAgent.__new__(SecurityAgent)
        alerts = [{"severity": "High"}, {"severity": "High"}, {"severity": "Medium"}]
        assert agent._score_risk(alerts) == min(100, 30 + 30 + 15)

    def test_risk_score_capped_at_100(self):
        from agents.security_agent import SecurityAgent
        agent = SecurityAgent.__new__(SecurityAgent)
        alerts = [{"severity": "High"}] * 10
        assert agent._score_risk(alerts) == 100


class TestOpenAIRetry:
    """Verify retry behaviour in base_agent._make_retrying_call and reason()."""

    @patch("agents.base_agent.time.sleep")
    @patch("agents.base_agent.OpenAI")
    @patch("agents.base_agent.load_config")
    def test_retries_on_rate_limit_then_succeeds(self, mock_load_config, MockOpenAI, mock_sleep):
        """_make_retrying_call retries RateLimitError and eventually succeeds."""
        mock_load_config.return_value = MOCK_CONFIG
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        import openai as _openai
        ok_response = MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))])

        # Simulate: 429 → 429 → success on third attempt
        mock_client.chat.completions.create.side_effect = [
            _openai.RateLimitError(
                "rate limit",
                response=MagicMock(status_code=429, headers={}, json=MagicMock(return_value={})),
                body={},
            ),
            _openai.RateLimitError(
                "rate limit",
                response=MagicMock(status_code=429, headers={}, json=MagicMock(return_value={})),
                body={},
            ),
            ok_response,
        ]

        from agents.base_agent import _make_retrying_call
        result = _make_retrying_call(
            mock_client, "gpt-4o",
            [{"role": "user", "content": "test"}],
            100,
            max_attempts=4,
        )
        assert result == "ok"
        assert mock_client.chat.completions.create.call_count == 3
        # time.sleep must have been called twice (once per retry)
        assert mock_sleep.call_count == 2

    @patch("agents.base_agent.OpenAI")
    @patch("agents.base_agent.load_config")
    def test_auth_error_not_retried(self, mock_load_config, MockOpenAI):
        """AuthenticationError is permanent — should NOT be retried."""
        mock_load_config.return_value = MOCK_CONFIG
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        import openai as _openai
        mock_client.chat.completions.create.side_effect = _openai.AuthenticationError(
            "Invalid API key",
            response=MagicMock(status_code=401, headers={}), body={}
        )

        from agents.incident_agent import IncidentAgent
        agent = IncidentAgent()
        agent._openai_client = mock_client

        with pytest.raises(_openai.AuthenticationError):
            agent.reason("sys", "user")

        # Exactly one call — no retry
        assert mock_client.chat.completions.create.call_count == 1

    @patch("agents.base_agent.time.sleep")
    @patch("agents.base_agent.OpenAI")
    @patch("agents.base_agent.load_config")
    def test_reason_cascades_to_fallback_after_retry_exhaustion(
        self, mock_load_config, MockOpenAI, mock_sleep
    ):
        """After primary exhausts all retries, reason() tries the fallback model."""
        cfg = {
            **MOCK_CONFIG,
            "openai": {
                **MOCK_CONFIG["openai"],
                "primary_model":  "gpt-4o",
                "fallback_model": "gpt-4o-mini",
            },
        }
        mock_load_config.return_value = cfg
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        import openai as _openai
        ok_response = MagicMock(choices=[MagicMock(message=MagicMock(content="fallback ok"))])

        def side_effect(**kwargs):
            model = kwargs.get("model", "")
            if model == "gpt-4o":
                raise _openai.RateLimitError(
                    "rate limit",
                    response=MagicMock(
                        status_code=429, headers={}, json=MagicMock(return_value={})
                    ),
                    body={},
                )
            return ok_response

        mock_client.chat.completions.create.side_effect = side_effect

        from agents.incident_agent import IncidentAgent
        agent = IncidentAgent()
        agent._openai_client = mock_client

        result = agent.reason("sys", "user")
        assert result == "fallback ok"


    @patch("agents.base_agent.time.sleep")
    @patch("agents.base_agent.OpenAI")
    @patch("agents.base_agent.load_config")
    def test_quota_exhausted_fails_fast_no_retry(self, mock_load_config, MockOpenAI, mock_sleep):
        """Daily quota exhaustion (429 + billing message) must fail immediately.
        No retries — sleeping won't restore the quota until midnight Pacific."""
        mock_load_config.return_value = MOCK_CONFIG
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        import openai as _openai
        quota_error = _openai.RateLimitError(
            "Error code: 429 - You exceeded your current quota, please check your plan and billing.",
            response=MagicMock(status_code=429, headers={}, json=MagicMock(return_value={})),
            body={},
        )
        mock_client.chat.completions.create.side_effect = quota_error

        from agents.base_agent import _make_retrying_call
        with pytest.raises(_openai.RateLimitError):
            _make_retrying_call(
                mock_client, "gemini-2.0-flash",
                [{"role": "user", "content": "test"}],
                100,
                max_attempts=4,
            )

        # Must have given up immediately after the first call — no sleep, no retry
        assert mock_client.chat.completions.create.call_count == 1, (
            "Quota exhaustion must not be retried — only 1 API call expected"
        )
        mock_sleep.assert_not_called()

    @patch("agents.base_agent.time.sleep")
    @patch("agents.base_agent.OpenAI")
    @patch("agents.base_agent.load_config")
    def test_rate_limit_still_retried(self, mock_load_config, MockOpenAI, mock_sleep):
        """Plain per-minute rate limit (no billing message) IS retried as before."""
        mock_load_config.return_value = MOCK_CONFIG
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        import openai as _openai
        ok_response = MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))])
        rate_error = _openai.RateLimitError(
            "Rate limit exceeded. Please retry.",  # no billing phrase
            response=MagicMock(status_code=429, headers={}, json=MagicMock(return_value={})),
            body={},
        )
        mock_client.chat.completions.create.side_effect = [rate_error, ok_response]

        from agents.base_agent import _make_retrying_call
        result = _make_retrying_call(
            mock_client, "gemini-2.0-flash",
            [{"role": "user", "content": "test"}],
            100,
            max_attempts=4,
        )
        assert result == "ok"
        assert mock_client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once()  # slept once between the two attempts


class TestRateLimiter:

    def test_rate_limiter_allows_within_limit(self):
        from agents.base_agent import RateLimiter
        limiter = RateLimiter(max_calls=5, period_seconds=60)
        results = [limiter.acquire() for _ in range(5)]
        assert all(results)

    def test_rate_limiter_blocks_over_limit(self):
        from agents.base_agent import RateLimiter
        limiter = RateLimiter(max_calls=3, period_seconds=60)
        for _ in range(3):
            limiter.acquire()
        assert limiter.acquire() is False

    def test_rate_limiter_independent_instances(self):
        """Two separate limiters must not share state."""
        from agents.base_agent import RateLimiter
        a = RateLimiter(max_calls=2, period_seconds=60)
        b = RateLimiter(max_calls=2, period_seconds=60)
        a.acquire()
        a.acquire()
        assert a.acquire() is False
        assert b.acquire() is True
