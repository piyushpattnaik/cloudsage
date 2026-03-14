"""
CloudSage Base Agent
Provides shared capabilities: OpenAI reasoning with retry, global rate limiting,
structured logging, and model fallback.

KEY FIX — Gemini free-tier 429 storm prevention:
  Root cause: 4 simultaneous LLM calls fired at t=0 (IncidentAgent + 3 consensus
  models) then 2 more at t~3s (RCA + Predictive). All 4 get 429 simultaneously,
  sleep the same duration, then retry simultaneously → another 429 burst → loop.
  The API overview showed ~55 errors (429 TooManyRequests) in a single day.

  Three-part fix applied here:
    1. GLOBAL shared rate limiter: one token bucket across ALL agent instances
       rather than one per agent. Each call to reason() or _make_retrying_call
       acquires a token from the shared 12 RPM bucket before hitting the API.
    2. JITTER: ±0.5–4s random noise added to every retry delay so parallel
       retriers no longer all sleep the exact same duration and fire together.
    3. IMPROVED _extract_retry_delay: checks Gemini JSON body, OpenAI message
       text, Retry-After header, and x-ratelimit-reset-requests header in order.
       Default raised from 5s → 30s (Gemini's actual window is 28-30s).

  Additional fixes in consensus.py and orchestrator.py:
    4. Consensus stagger: N-second gap between model calls so 3 consensus calls
       don't all hit the API at t=0.
    5. Orchestrator stagger: 1s delay before submitting Predictive after RCA.

Retry strategy:
  - Configurable via config["agents"]["max_retries"] (default 3 = 4 attempts).
  - On RateLimitError: server-requested delay + jitter.
  - On other errors: exponential backoff capped at 30s + jitter.
  - Does NOT retry AuthenticationError, BadRequestError, or NotFoundError.
  - Falls back from primary_model to fallback_model if all attempts fail.
"""

import json
import re
import time
import random
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import openai
from openai import OpenAI

from config.loader import load_config

logger = logging.getLogger("CloudSage.BaseAgent")


# ---------------------------------------------------------------------------
# Structured Logger
# ---------------------------------------------------------------------------
def get_structured_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        config = load_config()
        formatter = logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"agent": "%(name)s", "message": "%(message)s"}'
        )
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)

        import pathlib
        log_path = pathlib.Path(config["logging"]["log_file"])
        log_path.parent.mkdir(exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        log.addHandler(fh)

        level = getattr(logging, config["logging"]["level"], logging.INFO)
        log.setLevel(level)
    return log


# ---------------------------------------------------------------------------
# Rate Limiter  (global singleton shared across ALL agent instances)
# ---------------------------------------------------------------------------
class RateLimiter:
    """
    Sliding-window token-bucket rate limiter.

    A single shared instance (_GLOBAL_RATE_LIMITER below) is used by every
    agent so that parallel agent calls — consensus models, RCA + Predictive
    running simultaneously — are counted against ONE global ceiling rather than
    each having their own independent bucket that never fills up.

    Gemini free-tier: 15 RPM hard limit. We default to 12 RPM (80% headroom)
    to stay clear of the limit and avoid cascading 429 storms.
    """

    def __init__(self, max_calls: int, period_seconds: int = 60):
        self.max_calls = max_calls
        self.period    = period_seconds
        self._calls: list = []
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        now = time.time()
        with self._lock:
            self._calls = [t for t in self._calls if now - t < self.period]
            if len(self._calls) < self.max_calls:
                self._calls.append(now)
                return True
            return False

    def wait_and_acquire(self):
        """Block until a token is available. Polls every 0.5s to reduce latency."""
        while not self.acquire():
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# Global rate limiter — one shared instance for ALL agents
# ---------------------------------------------------------------------------
_GLOBAL_RATE_LIMITER: "RateLimiter | None" = None
_GLOBAL_RATE_LIMITER_LOCK = threading.Lock()


def _get_global_rate_limiter(max_calls: int = None) -> RateLimiter:
    """
    Return the process-wide shared rate limiter, creating it on first call.
    All agents share a single instance so combined RPM never exceeds the ceiling.

    max_calls defaults to config["agents"]["global_rate_limit_per_minute"] which
    _resolve_provider() in loader.py sets based on the active provider:
      groq   → 25 RPM  (free tier hard limit = 30 RPM)
      gemini → 12 RPM  (free tier hard limit = 15 RPM)
      openai → 60 RPM  (standard tier)
    """
    global _GLOBAL_RATE_LIMITER
    if _GLOBAL_RATE_LIMITER is None:
        with _GLOBAL_RATE_LIMITER_LOCK:
            if _GLOBAL_RATE_LIMITER is None:
                if max_calls is None:
                    try:
                        from config.loader import load_config
                        cfg = load_config()
                        max_calls = cfg.get("agents", {}).get("global_rate_limit_per_minute", 25)
                        provider  = cfg.get("active_llm_provider", "unknown")
                    except Exception:
                        max_calls = 25
                        provider  = "unknown"
                else:
                    provider = "unknown"
                _GLOBAL_RATE_LIMITER = RateLimiter(max_calls=max_calls)
                logger.info(
                    f"Global LLM rate limiter: {max_calls} RPM "
                    f"(provider={provider})"
                )
    return _GLOBAL_RATE_LIMITER


def _reset_global_rate_limiter():
    """Reset the global limiter. Use in tests to avoid state leakage."""
    global _GLOBAL_RATE_LIMITER
    _GLOBAL_RATE_LIMITER = None


# ---------------------------------------------------------------------------
# OpenAI client factory
# ---------------------------------------------------------------------------
def _build_openai_client(config: dict) -> OpenAI:
    """
    Build an OpenAI-compatible client.
    Points at the provider's base_url (e.g. Gemini's OpenAI-compatible endpoint)
    when configured, otherwise uses the default OpenAI endpoint.

    IMPORTANT: max_retries=0 disables the SDK's built-in retry loop.
    Without this, each CloudSage attempt fires 3 requests (1 original +
    2 SDK retries), multiplying quota consumption by 3x and causing
    premature daily quota exhaustion on Gemini free tier.
    CloudSage's own retry logic in _make_retrying_call() handles all retries.
    """
    api_key  = config["openai"].get("api_key") or "missing-key"
    base_url = (config["openai"].get("base_url") or "").strip() or None

    if base_url:
        logger.info(f"OpenAI client -> custom endpoint: {base_url}")
        return OpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    logger.info("OpenAI client -> default endpoint: https://api.openai.com/v1")
    return OpenAI(api_key=api_key, max_retries=0)


# ---------------------------------------------------------------------------
# Retryable errors
# ---------------------------------------------------------------------------
_RETRYABLE = (openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError)

# Phrases in 429 error messages that mean DAILY QUOTA EXHAUSTED (not per-minute rate limit).
# Retrying will never help — quota won't recover until the daily reset.
# Fail fast on these so we try the next model immediately instead of
# wasting 4 x 33s = 2+ minutes on hopeless retries.
_QUOTA_EXHAUSTED_PHRASES = (
    # Gemini — daily quota exhausted (not per-minute rate limit)
    "check your plan and billing",
    "exceeded your current quota",
    "quota exceeded",
    "billing",
    # Groq — daily/org quota exhausted
    "rate limit reached for model",
    "organization rate limit",
    "please reduce your usage",
    "tokens per day",
    "requests per day",
)


def _is_quota_exhausted(exc: Exception) -> bool:
    """Return True if this 429 is a hard daily quota error, not a per-minute rate limit."""
    msg = str(exc).lower()
    return any(phrase in msg for phrase in _QUOTA_EXHAUSTED_PHRASES)



# ---------------------------------------------------------------------------
# Retry-delay extractor  (enhanced for Gemini free-tier)
# ---------------------------------------------------------------------------
def _extract_retry_delay(exc: Exception, default: float = 30.0) -> float:
    """
    Extract the server-requested retry delay and add random jitter.

    Checks in priority order:
      1. Gemini JSON body: error.details[].retryDelay (e.g. "28s")
      2. OpenAI message text: "Please retry after N seconds"
      3. Retry-After HTTP header (integer seconds)
      4. x-ratelimit-reset-requests header (milliseconds)
      5. Falls back to `default` (30s — Gemini's actual window is 28-30s)

    Jitter (0.5–4s random) is added to ALL paths so that parallel callers
    that all received 429 at the same moment don't all retry simultaneously.
    Caps total at 90s.
    """
    jitter = random.uniform(0.5, 4.0)

    try:
        body = {}
        if hasattr(exc, "response") and exc.response is not None:
            try:
                body = exc.response.json()
            except Exception:
                pass

        # 1. Gemini body: error.details[].retryDelay
        for detail in body.get("error", {}).get("details", []):
            if "RetryInfo" in detail.get("@type", ""):
                delay_str = detail.get("retryDelay", "")
                if delay_str:
                    seconds = float(delay_str.replace("s", "").strip())
                    return min(seconds + 1.0 + jitter, 90.0)

        # 2. OpenAI message: "Please retry after N seconds"
        msg = body.get("error", {}).get("message", "")
        m = re.search(r"retry after (\d+\.?\d*)\s*s", msg, re.IGNORECASE)
        if m:
            return min(float(m.group(1)) + 1.0 + jitter, 90.0)

        # 3. Retry-After header
        if hasattr(exc, "response") and exc.response is not None:
            try:
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after:
                    return min(float(retry_after) + jitter, 90.0)
            except Exception:
                pass

        # 4. x-ratelimit-reset-requests header (milliseconds)
        if hasattr(exc, "response") and exc.response is not None:
            try:
                reset_ms = exc.response.headers.get("x-ratelimit-reset-requests")
                if reset_ms:
                    return min(float(reset_ms) / 1000.0 + jitter, 90.0)
            except Exception:
                pass

    except Exception:
        pass

    return default + jitter


# ---------------------------------------------------------------------------
# Retryable API call  (module-level so tests can patch it)
# ---------------------------------------------------------------------------
def _make_retrying_call(
    client: OpenAI,
    model: str,
    messages: list,
    max_tokens: int,
    max_attempts: int = 4,
) -> str:
    """
    Single OpenAI chat completion with smart retry + global rate limiting.

    Acquires a token from the global rate limiter before every API call so
    all agents combined never exceed the configured RPM ceiling.

    On RateLimitError: waits the server-requested delay + jitter.
    On other errors:  exponential backoff capped at 30s + jitter.
    Never retries AuthenticationError, BadRequestError, or NotFoundError.
    """
    last_exc = None
    _limiter = _get_global_rate_limiter()

    for attempt in range(1, max_attempts + 1):
        # Acquire global rate-limit token BEFORE hitting the API
        _limiter.wait_and_acquire()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()

        except (openai.AuthenticationError, openai.BadRequestError):
            raise   # permanent — never retry

        except openai.NotFoundError:
            raise   # model doesn't exist — never retry

        except _RETRYABLE as exc:
            last_exc = exc
            # Quota exhausted (daily limit) — retrying will never work today.
            # Raise immediately so the caller cascades to the next model
            # instead of burning 4 x 33s waiting for a reset that won't come.
            if _is_quota_exhausted(exc):
                logger.warning(
                    f"Daily quota exhausted for model={model} — "
                    f"failing fast (retries won't help until quota resets). "
                    f"Error: {str(exc)[:120]}"
                )
                raise
            if attempt == max_attempts:
                break
            delay = _extract_retry_delay(exc)
            logger.warning(
                f"Retrying call to model={model} in {delay:.1f}s "
                f"(attempt {attempt}/{max_attempts}) — {exc.__class__.__name__}: "
                f"{str(exc)[:120]}"
            )
            time.sleep(delay)

        except Exception as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            delay = min(2.0 ** attempt, 30.0) + random.uniform(0.5, 3.0)
            logger.warning(
                f"Retrying call to model={model} in {delay:.1f}s "
                f"(attempt {attempt}/{max_attempts}) — {exc.__class__.__name__}: "
                f"{str(exc)[:120]}"
            )
            time.sleep(delay)

    raise last_exc


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------
class BaseAgent(ABC):
    """
    Abstract base for all CloudSage agents.

    Provides:
    - LLM API with global rate limiting + smart retry + primary -> fallback cascade
    - Structured JSON logging to file + stdout
    - safe_parse_json() for robust LLM response handling
    - Standardised execute() / run() interface
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.config     = load_config()
        self.logger     = get_structured_logger(agent_name)

        # Initialise global limiter with configured RPM on first agent startup
        global_rpm = self.config["agents"].get("global_rate_limit_per_minute", 12)
        _get_global_rate_limiter(max_calls=global_rpm)

        self._openai_client = _build_openai_client(self.config)
        self.primary_model  = self.config["openai"]["primary_model"]
        self.fallback_model = self.config["openai"]["fallback_model"]

    # ------------------------------------------------------------------
    # Internal: get a client for a specific base_url (used by consensus)
    # ------------------------------------------------------------------
    def _get_client(self, base_url: str = None) -> OpenAI:
        """Return an OpenAI client for the given base_url, or the default client.
        max_retries=0 ensures SDK never doubles CloudSage's own retry logic."""
        if base_url:
            api_key = self.config["openai"].get("api_key") or "missing-key"
            return OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        return self._openai_client

    # ------------------------------------------------------------------
    # OpenAI reasoning: global rate limit → retry per model → cascade
    # ------------------------------------------------------------------
    def reason(self, system_prompt: str, user_prompt: str, max_tokens: int = 1500) -> str:
        """
        Call the configured LLM provider.
        Each model gets up to max_retries+1 attempts with smart retry delay.
        Falls through to fallback_model if primary exhausts all retries.
        Raises RuntimeError if both models fail (caught by execute()).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        models = list(dict.fromkeys([self.primary_model, self.fallback_model]))
        max_attempts = self.config["agents"].get("max_retries", 3) + 1

        for model in models:
            try:
                self.logger.info(f"Calling LLM model={model} agent={self.agent_name}")
                return _make_retrying_call(
                    self._openai_client, model, messages, max_tokens,
                    max_attempts=max_attempts,
                )
            except _RETRYABLE as e:
                self.logger.warning(
                    f"Model {model} exhausted retries ({e.__class__.__name__}). "
                    "Trying next model..."
                )
            except (openai.AuthenticationError, openai.BadRequestError):
                raise
            except Exception as e:
                self.logger.warning(f"Model {model} failed ({e}). Trying next model...")

        raise RuntimeError(
            f"[{self.agent_name}] All LLM models exhausted after retries. "
            f"Models tried: {models}"
        )

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------
    def safe_parse_json(self, text: str, fallback: dict) -> dict:
        """
        Parse LLM response as JSON with graceful fallback.
        Handles: valid JSON, JSON in markdown fences, and plain text.
        """
        if not text:
            return fallback
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            self.logger.warning(
                f"LLM returned non-JSON; using fallback. "
                f"Raw (first 200 chars): {text[:200]}"
            )
            return {**fallback, "_raw_llm_response": text[:500]}

    # ------------------------------------------------------------------
    # Severity scoring
    # ------------------------------------------------------------------
    @staticmethod
    def score_severity(metrics: dict) -> str:
        """Derive P1/P2/P3/P4 severity from metric thresholds."""
        cpu        = metrics.get("cpu_percent", 0)
        mem        = metrics.get("memory_percent", 0)
        error_rate = metrics.get("error_rate", 0)

        if cpu >= 90 or mem >= 85 or error_rate >= 5:
            return "P1"
        if cpu >= 75 or mem >= 70 or error_rate >= 2:
            return "P2"
        if cpu >= 60 or mem >= 60:
            return "P3"
        return "P4"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def execute(self, payload: dict) -> dict:
        """Rate-limited entry point called by the orchestrator."""
        start = time.time()
        self.logger.info(
            f"Agent {self.agent_name} executing | payload_keys={list(payload.keys())}"
        )
        try:
            result = self.run(payload)
            result["agent"]             = self.agent_name
            result["execution_time_ms"] = round((time.time() - start) * 1000, 2)
            result["timestamp"]         = datetime.now(timezone.utc).isoformat()
            self.logger.info(
                f"Agent {self.agent_name} completed | "
                f"duration={result['execution_time_ms']}ms"
            )
            return result
        except Exception as e:
            self.logger.error(f"Agent {self.agent_name} failed: {e}")
            return {
                "agent":     self.agent_name,
                "status":    "error",
                "error":     str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    @abstractmethod
    def run(self, payload: dict) -> dict:
        """Implement agent-specific logic here."""
        ...
