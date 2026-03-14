"""
CloudSage — Embeddings Client

Provider support:
  - OpenAI: text-embedding-3-small (1536 dims) via standard OpenAI endpoint
  - Gemini: text-embedding-004 (768 dims) via NATIVE Gemini REST API
            (NOT the OpenAI-compat endpoint — that route returns 404 for embed)
  - Groq:   No embedding API — falls back to Gemini native embed endpoint.
            Set GROQ_EMBEDDING_KEY=<gemini-key> or GEMINI_API_KEY.

Configuration is read from config["embedding"] (set by loader.py).
Lazy init: __init__ never raises — errors deferred to first embed() call.
"""

import hashlib
import json
import logging
import urllib.request
import urllib.error

from openai import OpenAI
from config.loader import load_config

logger = logging.getLogger("CloudSage.Embeddings")

_MAX_CHARS = 32_000

# Providers that use the native REST embed API (not OpenAI-compat)
_NATIVE_REST_PROVIDERS = {"gemini"}


class EmbeddingsClient:
    """
    Embeddings with in-process MD5 caching.

    For Gemini: calls https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent
    For OpenAI: calls standard OpenAI embeddings endpoint
    For Groq:   delegates to Gemini native embed (Groq has no embedding API)
    """

    def __init__(self):
        config   = load_config()
        emb_cfg  = config.get("embedding", {})

        self._api_key        = emb_cfg.get("api_key", "")
        self._model          = emb_cfg.get("model", "text-embedding-3-small")
        self._dims           = emb_cfg.get("dims", 1536)
        self._base_url       = (emb_cfg.get("base_url") or "").strip() or None
        self._chat_provider  = config.get("active_llm_provider", "openai")
        self._embed_provider = emb_cfg.get("provider", self._chat_provider)

        # Groq fallback: no native embedding → use Gemini key
        if self._chat_provider == "groq" and not self._api_key:
            import os
            fallback = (
                os.environ.get("GROQ_EMBEDDING_KEY")
                or os.environ.get("GEMINI_API_KEY")
                or config.get("providers", {}).get("gemini", {}).get("api_key", "")
            )
            if fallback:
                self._api_key = fallback
                self._embed_provider = "gemini"
                logger.info(
                    "Groq active — using Gemini native embed endpoint "
                    "(set GROQ_EMBEDDING_KEY to use a dedicated key)"
                )

        # Use native REST for Gemini regardless of base_url config
        self._use_native = self._embed_provider in _NATIVE_REST_PROVIDERS

        self._openai_client: OpenAI = None
        self._cache: dict = {}

        if self._api_key:
            logger.info(
                f"EmbeddingsClient — chat={self._chat_provider} "
                f"embed={self._embed_provider} model={self._model} "
                f"native_rest={self._use_native}"
            )
        else:
            logger.warning(
                f"EmbeddingsClient: no API key set for embed_provider={self._embed_provider}. "
                "Embedding calls will fail. "
                "For Groq: set GROQ_EMBEDDING_KEY or GEMINI_API_KEY."
            )

    @property
    def dims(self) -> int:
        return self._dims

    # ── Native Gemini REST embed ───────────────────────────────────────────────
    def _gemini_embed(self, text: str) -> list:
        """
        Call https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent
        This is the ONLY endpoint that works for text-embedding-004.
        The OpenAI-compat /v1beta/openai/embeddings route does NOT support embed models.
        """
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:embedContent?key={self._api_key}"
        )
        body = json.dumps({
            "model": f"models/{self._model}",
            "content": {"parts": [{"text": text[:_MAX_CHARS]}]},
        }).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                return data["embedding"]["values"]
        except urllib.error.HTTPError as e:
            body_text = e.read().decode(errors="replace")
            raise RuntimeError(
                f"Gemini embed HTTP {e.code}: {body_text}"
            ) from e

    def _gemini_embed_batch(self, texts: list) -> list:
        """Batch embed via Gemini batchEmbedContents endpoint."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:batchEmbedContents?key={self._api_key}"
        )
        requests_body = [
            {
                "model": f"models/{self._model}",
                "content": {"parts": [{"text": t[:_MAX_CHARS]}]},
            }
            for t in texts
        ]
        body = json.dumps({"requests": requests_body}).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return [item["values"] for item in data.get("embeddings", [])]
        except urllib.error.HTTPError as e:
            body_text = e.read().decode(errors="replace")
            raise RuntimeError(
                f"Gemini batch embed HTTP {e.code}: {body_text}"
            ) from e

    # ── OpenAI-compatible embed ────────────────────────────────────────────────
    def _get_openai_client(self) -> OpenAI:
        if self._openai_client is None:
            if not self._api_key:
                raise RuntimeError(
                    f"Embedding API key not set for provider '{self._embed_provider}'. "
                    "Set OPENAI_API_KEY / GEMINI_API_KEY / GROQ_API_KEY."
                )
            self._openai_client = (
                OpenAI(api_key=self._api_key, base_url=self._base_url, max_retries=0)
                if self._base_url
                else OpenAI(api_key=self._api_key, max_retries=0)
            )
        return self._openai_client

    # ── Public API ────────────────────────────────────────────────────────────
    def embed(self, text: str) -> list:
        """Return embedding vector for text. Cached by MD5."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self._api_key:
            raise RuntimeError(
                "Groq has no embedding API. "
                "Set GROQ_EMBEDDING_KEY=<gemini-key> or GEMINI_API_KEY."
                if self._chat_provider == "groq"
                else f"Embedding API key not set for provider '{self._embed_provider}'."
            )

        if self._use_native:
            vector = self._gemini_embed(text)
        else:
            resp = self._get_openai_client().embeddings.create(
                model=self._model, input=text[:_MAX_CHARS]
            )
            vector = resp.data[0].embedding

        self._cache[cache_key] = vector
        return vector

    def embed_batch(self, texts: list) -> list:
        """Embed a list of texts, skipping empty strings."""
        clean = [t[:_MAX_CHARS] for t in texts if t and t.strip()]
        if not clean:
            return []

        if not self._api_key:
            raise RuntimeError("Embedding API key not set.")

        if self._use_native:
            return self._gemini_embed_batch(clean)
        else:
            resp = self._get_openai_client().embeddings.create(
                model=self._model, input=clean
            )
            return [item.embedding for item in resp.data]
