"""RAG and document retrieval package for Asterline Research Assistant."""

from asterline.rag.ingestion import (
    compute_file_sha256,
    load_and_validate_corpus,
)
from asterline.rag.metadata import (
    CorpusChunk,
    IngestionReport,
    from_chroma_result,
    to_chroma_metadata,
    to_retrieved_chunk,
)
from asterline.rag.retriever import (
    EmbeddingEngine,
    HybridRetriever,
    LocalBM25Index,
)

__all__ = [
    "CorpusChunk",
    "IngestionReport",
    "compute_file_sha256",
    "load_and_validate_corpus",
    "to_chroma_metadata",
    "to_retrieved_chunk",
    "from_chroma_result",
    "EmbeddingEngine",
    "LocalBM25Index",
    "HybridRetriever",
]
