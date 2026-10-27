"""Metadata schemas, models, and converters for Asterline RAG pipeline.

Preserves exact corpus fields, handles scalar normalization for ChromaDB,
and maps retrieved results cleanly into the application's RetrievedChunk schema.
"""

from typing import Any, Dict, List, Union
from pydantic import BaseModel, Field
from asterline.graph.state import RetrievedChunk


class CorpusChunk(BaseModel):
    """Pydantic model representing an authoritative documentation chunk from corpus.jsonl."""

    chunk_id: str = Field(description="Unique stable identifier, e.g. 'company:0' or 'pricing:1'")
    doc_id: str = Field(description="Parent document identifier, e.g. 'company' or 'pricing'")
    title: str = Field(description="Human-readable title, e.g. 'Company overview'")
    category: str = Field(description="Document category: 'onboarding', 'pricing', 'product', 'release-notes', 'engineering', 'incident', 'policy'")
    owner: str = Field(description="Document owner, e.g. 'Asterline Documentation'")
    source_url: str = Field(description="Internal source URI, e.g. 'kb://asterline/pricing/pricing'")
    published: str = Field(description="Publication date string in YYYYMMDD format, e.g. '20260310'")
    version: str = Field(description="Document or product version, e.g. '1', '2.3', '3'")
    text: str = Field(description="Authoritative source text chunk")


class IngestionReport(BaseModel):
    """Summary metrics and validation status for corpus ingestion."""

    total_records: int
    valid_records: int
    rejected_records: int
    unique_chunk_ids: int
    document_ids: List[str]
    corpus_sha256: str
    errors: List[str] = Field(default_factory=list)


def to_chroma_metadata(chunk: CorpusChunk) -> Dict[str, Union[str, int, float, bool]]:
    """Convert CorpusChunk metadata into ChromaDB-compatible scalar values."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "category": chunk.category,
        "owner": chunk.owner,
        "source_url": chunk.source_url,
        "published": chunk.published,
        "version": chunk.version,
    }


def to_retrieved_chunk(chunk: CorpusChunk, score: float = 0.0) -> RetrievedChunk:
    """Convert a CorpusChunk into the canonical application RetrievedChunk model."""
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        title=chunk.title,
        category=chunk.category,
        owner=chunk.owner,
        source_url=chunk.source_url,
        published=chunk.published,
        version=chunk.version,
        text=chunk.text,
        score=score,
    )


def from_chroma_result(
    chunk_id: str,
    document: str,
    metadata: Dict[str, Any],
    score: float = 0.0,
) -> RetrievedChunk:
    """Reconstruct a RetrievedChunk from ChromaDB query output."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=str(metadata.get("doc_id", "")),
        title=str(metadata.get("title", "")),
        category=str(metadata.get("category", "")),
        owner=str(metadata.get("owner", "")),
        source_url=str(metadata.get("source_url", "")),
        published=str(metadata.get("published", "")),
        version=str(metadata.get("version", "")),
        text=document,
        score=score,
    )
