"""
CloudSage — Config Loader (v3)

Primary source of truth: config/config.json
Keys may be plain strings OR ${VAR_NAME} placeholders — both work.
Environment variables can override any config value (useful for CI/CD secrets).

Resolution order (highest priority first):
  1. Environment variables (via ENV_VAR_MAP)
  2. config.json plain values
  3. ${VAR_NAME} placeholders — expanded from environment variables
  4. _ensure_defaults() fallbacks

ENV_VAR_MAP (env var → dotted config path written directly):
  GEMINI_API_KEY             → openai.api_key  (and providers.gemini.api_key)
  OPENAI_API_KEY             → openai.api_key  (and providers.openai.api_key)
  COSMOS_DB_ENDPOINT         → cosmos_db.endpoint
  COSMOS_DB_KEY              → cosmos_db.key
  TEAMS_WEBHOOK_URL          → teams.webhook_url
  AZURE_STORAGE_CONNECTION_STRING → faiss.blob_connection_string
  LLM_PROVIDER               → llm_provider
  GROQ_API_KEY               → providers.groq.api_key
  GROQ_EMBEDDING_KEY         → providers.groq.embedding_key  (Gemini key used for embeddings with Groq)
  ENVIRONMENT                → environment
"""

import json
import os
import re
import logging
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger("CloudSage.Config")

# ---------------------------------------------------------------------------
# ENV_VAR_MAP  —  environment variable name → dotted config path(s)
# Each env var can write to one or more paths (list).
# ---------------------------------------------------------------------------
ENV_VAR_MAP: dict = {
    "OPENAI_API_KEY":                  ["openai.api_key", "providers.openai.api_key"],
    "GEMINI_API_KEY":                  ["openai.api_key", "providers.gemini.api_key"],
    "COSMOS_DB_ENDPOINT":              ["cosmos_db.endpoint"],
    "COSMOS_DB_KEY":                   ["cosmos_db.key"],
    "TEAMS_WEBHOOK_URL":               ["teams.webhook_url"],
    "AZURE_STORAGE_CONNECTION_STRING": ["faiss.blob_connection_string"],
    "LLM_PROVIDER":                    ["llm_provider"],
    "GROQ_API_KEY":                    ["providers.groq.api_key"],
    "GROQ_EMBEDDING_KEY":              ["providers.groq.embedding_key"],
    "ENVIRONMENT":                     ["environment"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interpolate(value: str) -> str:
    """Replace ${VAR_NAME} placeholders with environment variable values."""
    def replacer(match):
        val = os.environ.get(match.group(1))
        if val is None:
            logger.debug(f"Env var '{match.group(1)}' not set — using empty string.")
            return ""
        return val
    return re.sub(r'\$\{([^}]+)\}', replacer, value)


def _interpolate_recursive(obj):
    """Walk the entire config tree, expanding ${VAR} placeholders in string values."""
    if isinstance(obj, dict):
        return {k: _interpolate_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_recursive(i) for i in obj]
    if isinstance(obj, str):
        return _interpolate(obj)
    return obj


def _strip_comments(obj):
    """Recursively remove keys prefixed with '_' (JSON comments convention)."""
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_comments(i) for i in obj]
    return obj


def _set_nested(d: dict, dotted_path: str, value: str):
    """Set a value in a nested dict using a dotted key path, creating parents as needed."""
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def _get_nested(d: dict, dotted_path: str):
    """Get a value from a nested dict using a dotted key path. Returns None if missing."""
    parts = dotted_path.split(".")
    cur = d
    for part in parts:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _apply_env_overrides(config: dict):
    """
    Apply environment variable overrides. Env vars take priority over config.json.
    Each entry in ENV_VAR_MAP can write to multiple dotted paths so that both the
    provider-specific block (providers.gemini.api_key) and the active-provider
    shortcut (openai.api_key) are always kept in sync.
    """
    for env_var, dotted_paths in ENV_VAR_MAP.items():
        val = os.environ.get(env_var)
        if val:
            for path in dotted_paths:
                _set_nested(config, path, val)


def _resolve_provider(config: dict):
    """
    Copy the active provider's block into config["openai"] so all agents
    access LLM settings uniformly at config["openai"] regardless of provider.

    Also handles the special case of Groq, which has no embedding API.
    When provider=groq, embedding settings are taken from groq.embedding_*
    keys (which default to Gemini's free embedding endpoint) and stored in
    config["embedding"] so EmbeddingsClient uses the right endpoint/key.

    Sets config["agents"]["global_rate_limit_per_minute"] based on the
    provider's rate_limit_per_minute so the global limiter auto-adjusts:
      groq   → 25 RPM (free tier limit = 30)
      gemini → 12 RPM (free tier limit = 15)
      openai → 60 RPM (standard)
    """
    provider = config.get("llm_provider", "openai")
    provider_cfg = config.get("providers", {}).get(provider, {})
    if provider_cfg:
        # Merge: provider block wins over any stale values already in openai{}
        config["openai"] = {**config.get("openai", {}), **provider_cfg}
    config["active_llm_provider"] = provider

    # Set rate limit based on the active provider
    rpm = provider_cfg.get("rate_limit_per_minute")
    if rpm:
        config["agents"]["global_rate_limit_per_minute"] = rpm

    # Resolve embedding config — Groq uses a separate embedding provider
    if provider == "groq":
        emb_key      = provider_cfg.get("embedding_key") or ""
        emb_base_url = provider_cfg.get("embedding_base_url") or ""
        emb_model    = provider_cfg.get("embedding_model", "text-embedding-004")
        emb_dims     = provider_cfg.get("embedding_dims", 768)
        config["embedding"] = {
            "api_key":   emb_key,
            "base_url":  emb_base_url,
            "model":     emb_model,
            "dims":      emb_dims,
            "provider":  provider_cfg.get("embedding_provider", "gemini"),
        }
        logger.info(
            f"Groq embedding fallback: provider={config['embedding']['provider']} "
            f"model={emb_model} endpoint={emb_base_url or 'default'}"
        )
    else:
        # For OpenAI and Gemini, embedding settings come from the chat provider block
        config["embedding"] = {
            "api_key":  config["openai"].get("api_key", ""),
            "base_url": config["openai"].get("base_url", ""),
            "model":    config["openai"].get("embedding_model", "text-embedding-3-small"),
            "dims":     config["openai"].get("embedding_dims", 1536),
            "provider": provider,
        }

    logger.info(
        f"LLM provider: '{provider}' | "
        f"model={config['openai'].get('primary_model', 'unknown')} | "
        f"endpoint={config['openai'].get('base_url') or 'https://api.openai.com/v1'} | "
        f"rate_limit={config['agents']['global_rate_limit_per_minute']} RPM"
    )


def _ensure_defaults(config: dict):
    """
    Guarantee every key accessed with [] indexing exists with a safe fallback.
    Uses per-key setdefault on nested dicts (not top-level setdefault) so that
    partially-configured sections still get their missing sub-keys filled in.
    """
    # openai / LLM
    config.setdefault("openai", {})
    config["openai"].setdefault("api_key", "")
    config["openai"].setdefault("primary_model", "gpt-4o-mini")
    config["openai"].setdefault("fallback_model", "gpt-4o-mini")
    config["openai"].setdefault("embedding_model", "text-embedding-3-small")
    config["openai"].setdefault("embedding_dims", 1536)
    config["openai"].setdefault("max_tokens", 1500)

    # providers (default stubs so _resolve_provider always has something to merge)
    config.setdefault("providers", {})
    config["providers"].setdefault("openai", {
        "api_key":       config["openai"].get("api_key", ""),
        "primary_model": "gpt-4o-mini",
        "fallback_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "embedding_dims": 1536,
    })
    config["providers"].setdefault("gemini", {
        "api_key":       "",
        "base_url":      "https://generativelanguage.googleapis.com/v1beta/openai/",
        "primary_model": "gemini-2.0-flash",
        "fallback_model": "gemini-2.0-flash-lite",
        "embedding_model": "text-embedding-004",
        "embedding_dims": 768,
        "rate_limit_per_minute": 12,
    })
    config["providers"].setdefault("groq", {
        "api_key":              "",
        "base_url":             "https://api.groq.com/openai/v1",
        "primary_model":        "llama-3.3-70b-versatile",
        "consensus_model_2":    "mixtral-8x7b-32768",
        "fallback_model":       "llama-3.1-8b-instant",
        "max_tokens":           1500,
        "rate_limit_per_minute": 25,
        # Groq has no embedding API — embeddings use a separate provider
        "embedding_provider":   "gemini",
        "embedding_key":        "",
        "embedding_base_url":   "https://generativelanguage.googleapis.com/v1beta/openai/",
        "embedding_model":      "text-embedding-004",
        "embedding_dims":       768,
    })

    config.setdefault("llm_provider", "openai")

    # agents
    config.setdefault("agents", {})
    config["agents"].setdefault("rate_limit_per_minute", 60)
    config["agents"].setdefault("max_retries", 3)
    config["agents"].setdefault("retry_delay_seconds", 5)
    config["agents"].setdefault("timeout_seconds", 30)
    # Rate limit is set per-provider in _resolve_provider() below.
    # Defaults here are overridden once the active provider is known.
    config["agents"].setdefault("global_rate_limit_per_minute", 25)   # set correctly per provider
    config["agents"].setdefault("consensus_stagger_seconds", 2.0)      # gap between consensus model calls

    # logging
    config.setdefault("logging", {})
    config["logging"].setdefault("level", "INFO")
    config["logging"].setdefault("log_file", "logs/cloudsage.log")

    # cosmos db
    config.setdefault("cosmos_db", {})
    config["cosmos_db"].setdefault("database", "cloudsage")
    config["cosmos_db"].setdefault("container", "incidents")

    # faiss
    config.setdefault("faiss", {})
    config["faiss"].setdefault("blob_connection_string", "")
    config["faiss"].setdefault("blob_container", "cloudsage-faiss")
    config["faiss"].setdefault("index_dir", "data/faiss_index")

    # teams
    config.setdefault("teams", {})
    config["teams"].setdefault("webhook_url", "")

    # thresholds — use per-key setdefault so partial config.json sections are filled in
    config.setdefault("thresholds", {})
    t = config["thresholds"]
    t.setdefault("cpu_critical", 90)
    t.setdefault("cpu_warning", 75)
    t.setdefault("memory_critical", 85)
    t.setdefault("memory_warning", 70)
    t.setdefault("error_rate_critical", 5.0)
    t.setdefault("error_rate_warning", 2.0)
    t.setdefault("cost_spike_percent", 20)          # used by FinOpsAgent._detect_cost_spike

    config.setdefault("anomaly_detection", {})
    config["anomaly_detection"].setdefault("contamination", 0.05)

    config.setdefault("slo_definitions", {})
    config.setdefault("services", {})
    config.setdefault("revenue_models", {})


def _validate(config: dict):
    """Warn about critical missing values without blocking startup."""
    if not config.get("cosmos_db", {}).get("endpoint"):
        logger.warning(
            "cosmos_db.endpoint is not set. Cosmos DB persistence will fail. "
            "Set it in config/config.json or via COSMOS_DB_ENDPOINT env var."
        )
    if not config.get("openai", {}).get("api_key"):
        logger.warning(
            "API key is not set. All LLM calls will fail. "
            "Set providers.openai.api_key / providers.gemini.api_key / providers.groq.api_key "
            "in config/config.json or via OPENAI_API_KEY / GEMINI_API_KEY / GROQ_API_KEY env var."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_config() -> dict:
    """
    Load, process, and cache the CloudSage configuration.
    Call reload_config() to invalidate the cache (e.g. in tests).
    """
    config_path = Path(__file__).parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Copy config/config.json.example to config/config.json and fill in your keys."
        )

    with open(config_path) as f:
        raw = json.load(f)

    config = _strip_comments(raw)
    config = _interpolate_recursive(config)   # expand ${VAR} placeholders first
    _apply_env_overrides(config)              # then env vars override individual keys
    _ensure_defaults(config)                  # fill any still-missing keys
    _resolve_provider(config)                 # copy active provider → openai{}
    _validate(config)
    return config


def reload_config() -> dict:
    """Invalidate the lru_cache and reload config from disk. Useful in tests."""
    load_config.cache_clear()
    return load_config()
