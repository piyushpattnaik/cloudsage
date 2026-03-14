"""
CloudSage — FAISS Vector Search Client
Replaces Azure AI Search with a local FAISS index + pure-Python TF-IDF keyword scorer.

Architecture:
  - FAISS IndexFlatIP (inner-product / cosine with normalised vectors) for semantic search
  - Pure-Python TF-IDF keyword scorer for lexical matching
  - Hybrid score = 0.7 × vector_score + 0.3 × keyword_score
  - Index persisted to disk AND optionally synced to Azure Blob Storage
  - Thread-safe: a single RLock guards all FAISS read/write operations

Embedding dimensions:
  Read from config["openai"]["embedding_dims"] at runtime so the index
  automatically matches the active provider (768 for Gemini text-embedding-004,
  1536 for OpenAI text-embedding-3-small). The index is recreated when dims change.
"""

import json
import logging
import math
import re
import threading
import uuid
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError as exc:
    raise ImportError("faiss-cpu is required: pip install faiss-cpu") from exc

from config.loader import load_config

logger = logging.getLogger("CloudSage.FAISSSearch")

# Hybrid search weights
_VECTOR_WEIGHT  = 0.7
_KEYWORD_WEIGHT = 0.3

# Local file names
_INDEX_FILE = "index.faiss"
_META_FILE  = "metadata.jsonl"

# Default local path
_DEFAULT_INDEX_DIR = Path("data/faiss_index")


# ---------------------------------------------------------------------------
# Blob Storage helper  (optional — degrades gracefully when unavailable)
# ---------------------------------------------------------------------------

def _get_blob_client(cfg: dict):
    conn_str  = cfg.get("faiss", {}).get("blob_connection_string", "")
    container = cfg.get("faiss", {}).get("blob_container", "cloudsage-faiss")
    if not conn_str:
        return None, None
    try:
        from azure.storage.blob import BlobServiceClient
        client = BlobServiceClient.from_connection_string(conn_str)
        try:
            client.create_container(container)
        except Exception:
            pass
        return client, container
    except ImportError:
        logger.warning(
            "azure-storage-blob not installed — FAISS index will be local-only. "
            "Install it: pip install azure-storage-blob"
        )
        return None, None
    except Exception as e:
        logger.warning(f"Blob Storage unavailable — falling back to local FAISS: {e}")
        return None, None


class SearchClientWrapper:
    """
    FAISS-backed vector store with hybrid keyword+semantic search.
    Embedding dimensions are read from config at construction time so the
    index always matches the active provider (Gemini=768, OpenAI=1536).
    """

    def __init__(self, index_dir: str = None):
        cfg = load_config()

        # Read dims from config — supports both Gemini (768) and OpenAI (1536)
        self._dims = cfg["openai"].get("embedding_dims", 1536)
        provider   = cfg.get("active_llm_provider", "openai")
        model      = cfg["openai"].get("embedding_model", "text-embedding-3-small")

        logger.info(
            f"FAISSSearch: provider={provider} model={model} embedding_dims={self._dims}"
        )

        self._index_dir = Path(
            index_dir or cfg.get("faiss", {}).get("index_dir", str(_DEFAULT_INDEX_DIR))
        )
        self._index_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._index: faiss.IndexFlatIP = None
        self._docs: list = []

        self._blob_client, self._blob_container = _get_blob_client(cfg)
        self._load_or_create()

        logger.info(
            f"FAISSSearch ready — {self._index.ntotal} vectors | "
            f"dims={self._dims} | blob_backed={self._blob_client is not None}"
        )

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def _download_from_blob(self) -> bool:
        if not self._blob_client:
            return False
        try:
            for filename in (_INDEX_FILE, _META_FILE):
                blob = self._blob_client.get_blob_client(
                    container=self._blob_container, blob=filename
                )
                local_path = self._index_dir / filename
                with open(local_path, "wb") as f:
                    f.write(blob.download_blob().readall())
            logger.info("FAISS index downloaded from Blob Storage")
            return True
        except Exception as e:
            logger.info(f"No existing FAISS index in blob (will create): {e}")
            return False

    def _upload_to_blob(self):
        if not self._blob_client:
            return
        try:
            for filename in (_INDEX_FILE, _META_FILE):
                local_path = self._index_dir / filename
                if not local_path.exists():
                    continue
                blob = self._blob_client.get_blob_client(
                    container=self._blob_container, blob=filename
                )
                with open(local_path, "rb") as f:
                    blob.upload_blob(f, overwrite=True)
            logger.info("FAISS index uploaded to Blob Storage")
        except Exception as e:
            logger.error(f"Failed to upload FAISS index to blob: {e}")

    def _load_or_create(self):
        index_path = self._index_dir / _INDEX_FILE
        meta_path  = self._index_dir / _META_FILE

        # Try local first
        if index_path.exists() and meta_path.exists():
            loaded = faiss.read_index(str(index_path))
            # Reject index if dims don't match current provider config
            if loaded.d == self._dims:
                self._index = loaded
                self._docs  = [
                    json.loads(line)
                    for line in meta_path.read_text().splitlines()
                    if line.strip()
                ]
                logger.info(f"Loaded local FAISS index: {self._index.ntotal} vectors")
                return
            else:
                logger.warning(
                    f"Existing FAISS index has dims={loaded.d} but config requires "
                    f"dims={self._dims}. Discarding stale index and creating fresh one. "
                    "Re-index your documents with: python main.py index --dir <path>"
                )

        # Try blob (cold start recovery)
        if self._download_from_blob():
            if index_path.exists() and meta_path.exists():
                loaded = faiss.read_index(str(index_path))
                if loaded.d == self._dims:
                    self._index = loaded
                    self._docs  = [
                        json.loads(line)
                        for line in meta_path.read_text().splitlines()
                        if line.strip()
                    ]
                    logger.info(f"Loaded FAISS index from blob: {self._index.ntotal} vectors")
                    return

        # Fresh start
        self._index = faiss.IndexFlatIP(self._dims)
        self._docs  = []
        logger.info(f"Created new empty FAISS index (dims={self._dims})")

    def _persist(self):
        faiss.write_index(self._index, str(self._index_dir / _INDEX_FILE))
        (self._index_dir / _META_FILE).write_text(
            "\n".join(json.dumps(d) for d in self._docs)
        )
        self._upload_to_blob()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def upload_documents(self, documents: list):
        """
        Add documents to the FAISS index.
        Each dict must have: content, source, category, content_vector, id (optional).
        """
        if not documents:
            return

        vectors = []
        metas   = []
        for doc in documents:
            vec = doc.get("content_vector")
            if vec is None or len(vec) != self._dims:
                logger.warning(
                    f"Skipping doc with missing/wrong-dim vector: {doc.get('id')} "
                    f"(expected {self._dims} dims, got {len(vec) if vec else 'None'})"
                )
                continue
            arr = np.array(vec, dtype="float32")
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr /= norm
            vectors.append(arr)
            metas.append({
                "id":       doc.get("id", str(uuid.uuid4())),
                "content":  doc.get("content", ""),
                "source":   doc.get("source", ""),
                "category": doc.get("category", "general"),
            })

        if not vectors:
            return

        matrix = np.vstack(vectors)
        with self._lock:
            self._index.add(matrix)
            self._docs.extend(metas)
            self._persist()

        logger.info(f"Indexed {len(vectors)} chunk(s) — total: {self._index.ntotal}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def hybrid_search(self, query_text: str, query_vector: list, top_k: int = 5) -> list:
        """
        Hybrid vector + keyword search.
        Returns up to top_k results sorted by combined score descending.
        """
        with self._lock:
            n_docs = self._index.ntotal
            if n_docs == 0:
                return []

            k = min(top_k * 3, n_docs)

            arr = np.array(query_vector, dtype="float32")
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr /= norm
            arr = arr.reshape(1, -1)

            scores, indices = self._index.search(arr, k)
            scores  = scores[0].tolist()
            indices = indices[0].tolist()

            query_terms = _tokenise(query_text)
            keyword_scores = [
                _tfidf_score(query_terms, self._docs[i]["content"], self._docs)
                for i in indices
                if 0 <= i < len(self._docs)
            ]

            results = []
            for rank, (vec_score, idx) in enumerate(zip(scores, indices)):
                if idx < 0 or idx >= len(self._docs):
                    continue
                kw_score = keyword_scores[rank] if rank < len(keyword_scores) else 0.0
                combined = _VECTOR_WEIGHT * float(vec_score) + _KEYWORD_WEIGHT * kw_score
                doc = self._docs[idx]
                results.append({
                    "content":  doc["content"],
                    "source":   doc["source"],
                    "category": doc["category"],
                    "score":    round(combined, 4),
                })

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

    def create_index_if_not_exists(self):
        pass  # FAISS index is created lazily in __init__


# ---------------------------------------------------------------------------
# Pure-Python keyword scoring  (TF-IDF approximation)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list:
    return [t for t in re.split(r"\W+", text.lower()) if t]


def _tfidf_score(query_terms: list, doc_text: str, all_docs: list) -> float:
    if not query_terms or not doc_text:
        return 0.0

    N          = len(all_docs)
    doc_tokens = _tokenise(doc_text)
    doc_len    = len(doc_tokens) or 1
    doc_freq   = {}
    for t in doc_tokens:
        doc_freq[t] = doc_freq.get(t, 0) + 1

    score = 0.0
    for term in query_terms:
        tf = doc_freq.get(term, 0) / doc_len
        if tf == 0:
            continue
        df  = sum(1 for d in all_docs if term in d["content"].lower())
        idf = math.log((N + 1) / (df + 1)) + 1
        score += tf * idf

    return min(1.0, score / len(query_terms))
