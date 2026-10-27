"""Local hybrid retrieval engine for Asterline documentation.

Combines:
1. Local dense vector retrieval (FastEmbed BAAI/bge-small-en-v1.5 + ChromaDB)
2. Exact keyword and technical identifier retrieval (Lexical BM25 index)
3. Reciprocal Rank Fusion (RRF) combining dense semantic and lexical matches.

Zero external API dependencies for embedding generation or vector search.
"""

from collections import Counter
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set
import chromadb
from fastembed import TextEmbedding
from asterline.graph.state import RetrievedChunk
from asterline.rag.metadata import (
    CorpusChunk,
    from_chroma_result,
    to_chroma_metadata,
    to_retrieved_chunk,
)


# Tokenizer preserving technical terms, incident numbers, paths, and version strings
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-_./:][A-Za-z0-9]+)*")


def tokenize_technical_text(text: str) -> List[str]:
    """Tokenize text into lowercase terms while preserving technical identifiers."""
    if not text:
        return []
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


class LocalBM25Index:
    """In-memory BM25 lexical index tuned for technical documentation and exact identifiers."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[CorpusChunk] = []
        self.doc_len: List[int] = []
        self.avg_dl: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Counter] = []
        self.num_docs: int = 0

    def fit(self, chunks: List[CorpusChunk]) -> None:
        """Build BM25 index over the provided corpus chunks."""
        self.chunks = chunks
        self.num_docs = len(chunks)
        self.term_freqs = []
        self.doc_len = []
        self.doc_freqs = {}

        if self.num_docs == 0:
            self.avg_dl = 0.0
            return

        total_length = 0
        for chunk in chunks:
            # Index both chunk text and title for high identifier precision
            full_content = f"{chunk.title} {chunk.chunk_id} {chunk.text}"
            tokens = tokenize_technical_text(full_content)
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            self.doc_len.append(len(tokens))
            total_length += len(tokens)

            for token in tf.keys():
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_dl = total_length / self.num_docs

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Search BM25 index and return top_k ranked chunks."""
        query_tokens = tokenize_technical_text(query)
        if not query_tokens or self.num_docs == 0:
            return []

        scores: List[float] = [0.0] * self.num_docs

        for token in query_tokens:
            if token not in self.doc_freqs:
                continue

            # Standard BM25 IDF with smoothing
            df = self.doc_freqs[token]
            idf = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

            for doc_idx, tf_counter in enumerate(self.term_freqs):
                if token not in tf_counter:
                    continue
                tf = tf_counter[token]
                dl = self.doc_len[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
                scores[doc_idx] += idf * (numerator / denominator)

        # Apply metadata filters if provided
        ranked_indices = sorted(
            range(self.num_docs), key=lambda idx: scores[idx], reverse=True
        )

        results: List[RetrievedChunk] = []
        for idx in ranked_indices:
            if scores[idx] <= 0.0:
                continue
            chunk = self.chunks[idx]
            if metadata_filter:
                match = all(
                    getattr(chunk, k, None) == v for k, v in metadata_filter.items()
                )
                if not match:
                    continue

            results.append(to_retrieved_chunk(chunk, score=round(scores[idx], 4)))
            if len(results) >= top_k:
                break

        return results


class EmbeddingEngine:
    """Local FastEmbed wrapper with model caching."""

    _instance: Optional["EmbeddingEngine"] = None
    _model: Optional[TextEmbedding] = None

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        if EmbeddingEngine._model is None:
            EmbeddingEngine._model = TextEmbedding(model_name=self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embedding vectors for documents."""
        if not texts:
            return []
        assert EmbeddingEngine._model is not None
        generator = EmbeddingEngine._model.embed(texts)
        return [emb.tolist() for emb in generator]

    def embed_query(self, query: str) -> List[float]:
        """Generate dense embedding vector for a single query."""
        if not query or not query.strip():
            return []
        assert EmbeddingEngine._model is not None
        generator = EmbeddingEngine._model.embed([query])
        return list(generator)[0].tolist()


class HybridRetriever:
    """Coordinates local ChromaDB vector retrieval, BM25 exact search, and RRF fusion."""

    def __init__(
        self,
        corpus_chunks: List[CorpusChunk],
        chroma_persist_dir: Optional[Path] = None,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        collection_name: str = "asterline_docs",
        chroma_persist_directory: Optional[Path] = None,
    ):
        self.corpus_chunks = corpus_chunks
        self.chunks_by_id: Dict[str, CorpusChunk] = {c.chunk_id: c for c in corpus_chunks}
        target_dir = chroma_persist_dir or chroma_persist_directory or Path("data/chroma")
        self.chroma_dir = Path(target_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Initialize Local BM25 Index
        self.bm25_index = LocalBM25Index()
        self.bm25_index.fit(corpus_chunks)

        # Initialize FastEmbed Engine
        self.embedder = EmbeddingEngine(model_name=embedding_model_name)

        # Initialize ChromaDB Client
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Synchronize index idempotently
        self.sync_index()

    def sync_index(self, force_rebuild: bool = False) -> None:
        """Ensure all corpus chunks are indexed in ChromaDB without creating duplicates."""
        existing_count = self.collection.count()
        chunk_count = len(self.corpus_chunks)

        if not force_rebuild and existing_count == chunk_count:
            # Verify IDs match
            existing_ids = set(self.collection.get()["ids"])
            expected_ids = set(self.chunks_by_id.keys())
            if existing_ids == expected_ids:
                return  # Index is already up-to-date and complete

        if force_rebuild:
            # Delete and recreate
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        # Batch embed and populate
        ids = [c.chunk_id for c in self.corpus_chunks]
        documents = [c.text for c in self.corpus_chunks]
        metadatas = [to_chroma_metadata(c) for c in self.corpus_chunks]
        embeddings = self.embedder.embed_documents(documents)

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def dense_search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Perform dense semantic vector retrieval via ChromaDB."""
        if not query or not query.strip():
            return []

        query_emb = self.embedder.embed_query(query)
        if not query_emb:
            return []

        where_filter = metadata_filter if metadata_filter else None

        chroma_res = self.collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, len(self.corpus_chunks)),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        results: List[RetrievedChunk] = []
        if not chroma_res or not chroma_res["ids"] or not chroma_res["ids"][0]:
            return results

        ids = chroma_res["ids"][0]
        docs = chroma_res["documents"][0] if chroma_res["documents"] else [""] * len(ids)
        metas = chroma_res["metadatas"][0] if chroma_res["metadatas"] else [{}] * len(ids)
        dists = chroma_res["distances"][0] if chroma_res["distances"] else [0.0] * len(ids)

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
            # Cosine distance to similarity: sim = 1.0 - dist
            similarity = round(max(0.0, 1.0 - dist), 4)
            results.append(from_chroma_result(chunk_id, doc, meta, score=similarity))

        return results

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Perform exact keyword and technical identifier retrieval via local BM25."""
        return self.bm25_index.search(
            query=query, top_k=top_k, metadata_filter=metadata_filter
        )

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.6,
        keyword_weight: float = 0.4,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Perform hybrid retrieval using Reciprocal Rank Fusion (RRF).

        Formula:
            RRF_score(chunk) = (dense_weight / (60 + rank_dense)) + (keyword_weight / (60 + rank_keyword))
        """
        if not query or not query.strip():
            return []

        # Retrieve a broader candidate pool for fusion (up to 20 candidates per channel)
        candidate_k = min(max(top_k * 3, 15), len(self.corpus_chunks))

        dense_results = self.dense_search(
            query=query, top_k=candidate_k, metadata_filter=metadata_filter
        )
        keyword_results = self.keyword_search(
            query=query, top_k=candidate_k, metadata_filter=metadata_filter
        )

        rrf_constant = 60.0
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}

        # Accumulate dense ranks
        for rank, chunk in enumerate(dense_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (
                dense_weight / (rrf_constant + rank)
            )

        # Accumulate keyword ranks
        for rank, chunk in enumerate(keyword_results, start=1):
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (
                keyword_weight / (rrf_constant + rank)
            )

        # Sort by RRF score descending
        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        final_results: List[RetrievedChunk] = []
        for cid in sorted_ids[:top_k]:
            base_chunk = chunk_map[cid]
            final_results.append(
                RetrievedChunk(
                    chunk_id=base_chunk.chunk_id,
                    doc_id=base_chunk.doc_id,
                    title=base_chunk.title,
                    category=base_chunk.category,
                    owner=base_chunk.owner,
                    source_url=base_chunk.source_url,
                    published=base_chunk.published,
                    version=base_chunk.version,
                    text=base_chunk.text,
                    score=round(scores[cid], 5),
                )
            )

        return final_results
