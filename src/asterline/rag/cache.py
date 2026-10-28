"""Thread-safe documentation query cache for public, user-independent answers.

Rules:
1. Cache only answers whose correctness is independent of user identity and conversation-specific context.
2. Keyed strictly by corpus SHA-256 hash + normalized query string. If corpus changes, cache invalidates automatically.
3. NEVER cache memory-dependent queries, personalized answers, or private user data.
4. Avoid caching transient failures, fallbacks, or abstentions resulting from transient exceptions.
5. Thread-safe with bounded capacity.
"""

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional
from asterline.config.settings import get_settings
from asterline.graph.state import Citation, RetrievedChunk
from asterline.rag.ingestion import compute_file_sha256


class DocumentationAnswerCache:
    """Thread-safe, corpus-versioned cache for public factual documentation answers."""

    def __init__(self, maxsize: int = 128, corpus_hash: Optional[str] = None):
        self.maxsize = maxsize
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        if corpus_hash is not None:
            self.corpus_hash = corpus_hash
        else:
            settings = get_settings()
            corpus_path = settings.resolve_path(settings.corpus_path)
            if corpus_path.exists():
                self.corpus_hash = compute_file_sha256(corpus_path)
            else:
                self.corpus_hash = "UNKNOWN_CORPUS_HASH"

    def _make_key(self, query: str) -> str:
        norm_query = " ".join(query.strip().lower().split())
        return f"{self.corpus_hash}:{norm_query}"

    def get(self, query: str, user_memories: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached answer only if the query has no active user-specific memories."""
        # Never serve cached public answers if user has active personal context
        if user_memories:
            return None

        key = self._make_key(query)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                val = self._entries[key]
                return {
                    "draft_response": val["draft_response"],
                    "citations": list(val["citations"]),
                    "retrieved_chunks": list(val["retrieved_chunks"]),
                    "is_fallback": False,
                    "llm_metadata": dict(val.get("llm_metadata", {})),
                    "is_cached": True,
                }
            return None

    def set(
        self,
        query: str,
        result: Dict[str, Any],
        user_memories: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Store verified, non-fallback, non-memory-dependent documentation result."""
        # Policy: Never cache if user memories were involved
        if user_memories:
            return

        # Policy: Never cache transient fallbacks or provider failures
        if result.get("is_fallback", False):
            return

        draft = result.get("draft_response", "")
        # Policy: Avoid caching transient empty responses
        if not draft or not draft.strip():
            return

        # Policy: Never cache if citations are invalid or empty for an affirmative factual claim
        citations = result.get("citations", [])
        if not citations and "does not contain" not in draft.lower():
            return

        key = self._make_key(query)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
            self._entries[key] = {
                "draft_response": draft,
                "citations": citations,
                "retrieved_chunks": result.get("retrieved_chunks", []),
                "llm_metadata": result.get("llm_metadata", {}),
            }
            if len(self._entries) > self.maxsize:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
