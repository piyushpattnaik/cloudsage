"""
CloudSage — RAG Pipeline
Query -> Embed -> Hybrid Search -> Retrieve -> Provide to LLM Context

Includes document chunking to handle long runbooks/SOPs correctly.
"""

import re
from config.loader import load_config
from rag.embeddings import EmbeddingsClient
from rag.search_client import SearchClientWrapper


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline with chunking support.

    Workflow:
    1. Embed the user query
    2. Run hybrid search (keyword + vector) against Azure AI Search
    3. Return ranked document chunks for LLM context injection
    """

    CHUNK_SIZE = 512        # approximate tokens
    CHUNK_OVERLAP = 64      # overlap tokens to preserve context across boundaries
    CHARS_PER_TOKEN = 4     # rough approximation for splitting

    def __init__(self):
        self.embedder = EmbeddingsClient()
        self.search = SearchClientWrapper()
        self.config = load_config()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5) -> list:
        """
        Main retrieval method. Returns top_k relevant document chunks.

        Degrades gracefully on embedding failures (quota exhaustion, 404, network
        errors) — returns [] so RCA can still run using the LLM + causal context
        rather than crashing the entire agent pipeline.
        """
        if not query or not query.strip():
            return []
        try:
            query_vector = self.embedder.embed(query)
        except Exception as exc:
            # Non-fatal: log and continue without RAG context.
            # Common causes: Gemini embedding quota exhausted, wrong API version,
            # key not configured. RCA will still run via LLM + causal engine.
            import logging
            logging.getLogger("CloudSage.RAGPipeline").warning(
                f"Embedding failed — RAG context unavailable for this query. "
                f"RCA will proceed without knowledge base retrieval. "
                f"Error: {exc}"
            )
            return []
        try:
            return self.search.hybrid_search(
                query_text=query, query_vector=query_vector, top_k=top_k
            )
        except Exception as exc:
            import logging
            logging.getLogger("CloudSage.RAGPipeline").warning(
                f"Hybrid search failed — returning empty context. Error: {exc}"
            )
            return []

    def build_context_string(self, query: str, top_k: int = 5) -> str:
        """Retrieve docs and format as a context string for LLM injection."""
        docs = self.retrieve(query, top_k=top_k)
        if not docs:
            return "No relevant knowledge base entries found."
        parts = [
            f"[Source {i+1}] ({doc['category']}) {doc['source']}\n{doc['content']}"
            for i, doc in enumerate(docs)
        ]
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    def _chunk_text(self, text: str) -> list:
        """
        Split text into overlapping chunks of ~CHUNK_SIZE tokens.
        Respects paragraph and sentence boundaries where possible.
        Returns [] for empty/whitespace-only input.
        """
        if not text or not text.strip():
            return []

        max_chars = self.CHUNK_SIZE * self.CHARS_PER_TOKEN
        overlap_chars = self.CHUNK_OVERLAP * self.CHARS_PER_TOKEN

        paragraphs = re.split(r'\n\n+', text.strip())
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 > max_chars and current:
                chunks.append(current.strip())
                # Start next chunk with overlap from end of previous
                overlap_start = max(0, len(current) - overlap_chars)
                current = current[overlap_start:] + "\n\n" + para
            else:
                current = (current + "\n\n" + para).lstrip()

            # Hard-split if a single para exceeds max
            while len(current) > max_chars:
                chunks.append(current[:max_chars].strip())
                current = current[max_chars - overlap_chars:]

        if current.strip():
            chunks.append(current.strip())

        return chunks

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_document(self, content: str, source: str, category: str = "general", doc_id: str = None) -> int:
        """Index a single document with chunking. Returns number of chunks created."""
        chunks = self._chunk_text(content)
        if not chunks:
            return 0
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id or abs(hash(content)) & 0xFFFFFF}-chunk-{i}"
            vector = self.embedder.embed(chunk)
            documents.append({
                "id": chunk_id,
                "content": chunk,
                "source": source,
                "category": category,
                "content_vector": vector,
            })
        self.search.upload_documents(documents)
        return len(documents)

    def index_documents_bulk(self, documents: list) -> int:
        """
        Bulk index a list of documents with chunking.
        Each dict must have: content, source, category.
        Returns total number of chunks indexed.
        """
        all_chunks = []
        for doc in documents:
            chunks = self._chunk_text(doc.get("content", ""))
            base_id = abs(hash(doc.get("content", ""))) & 0xFFFFFF
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "content": chunk,
                    "source": doc.get("source", ""),
                    "category": doc.get("category", "general"),
                    "id": f"{base_id}-chunk-{i}",
                })

        if not all_chunks:
            return 0

        # Embed and upload in batches of 100
        # BUG 2 FIX: embed_batch skips empty/whitespace strings, returning fewer vectors
        # than inputs. If any chunk has empty content, zip(batch, vectors) silently
        # assigns the wrong vector to every subsequent chunk in the batch.
        # Fix: filter out empty chunks BEFORE building the embed batch so texts and
        # batch always correspond 1:1.
        batch_size = 100
        for start in range(0, len(all_chunks), batch_size):
            batch = all_chunks[start:start + batch_size]
            # Remove empty-content chunks so texts[] and non_empty_batch[] are aligned
            non_empty_batch = [c for c in batch if c.get("content") and c["content"].strip()]
            if not non_empty_batch:
                continue
            texts = [c["content"] for c in non_empty_batch]
            vectors = self.embedder.embed_batch(texts)
            for chunk, vec in zip(non_empty_batch, vectors):
                chunk["content_vector"] = vec
            self.search.upload_documents(non_empty_batch)

        return len(all_chunks)
