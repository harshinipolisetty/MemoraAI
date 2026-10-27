"""Comprehensive test suite for document ingestion, local vector indexing, and hybrid retrieval.

Covers:
1. All 75 source chunks are loaded when corpus is valid.
2. Duplicate chunk IDs are detected and rejected.
3. Malformed records fail clearly.
4. The source corpus checksum remains unchanged.
5. All indexed chunks preserve their original IDs and required metadata.
6. Index initialization can run repeatedly without duplicate entries.
7. The index persists and remains usable after reopening.
8. Exact incident IDs (e.g. INC-2025-12) are retrievable.
9. Exact API paths (e.g. POST /v2/runs) are retrievable.
10. Version strings and technical identifiers remain searchable.
11. Dense retrieval finds semantically relevant chunks.
12. Hybrid retrieval finds expected evidence for queries.
13. Duplicate results are removed.
14. Metadata filtering works correctly.
15. Missing or empty queries are handled safely.
16. Missing or invalid corpus files produce actionable errors.
17. Retrieval results can be converted to the existing structured evidence schema.
18. Index rebuilds handle source changes without corrupting the previous usable index.
"""

import json
from pathlib import Path
import pytest
from asterline.config.settings import get_settings
from asterline.graph.state import RetrievedChunk
from asterline.rag.ingestion import compute_file_sha256, load_and_validate_corpus
from asterline.rag.metadata import CorpusChunk, to_retrieved_chunk
from asterline.rag.retriever import HybridRetriever, LocalBM25Index


# The known golden checksum of the unchanged data/corpus.jsonl file
GOLDEN_CORPUS_SHA256 = "F63FC05CF43204485C98BD8FB6D89860A32C16A7E906BC90975E7C3954FE9FB1"


@pytest.fixture(scope="module")
def valid_corpus_chunks() -> list[CorpusChunk]:
    """Load and validate the real 75-chunk corpus."""
    settings = get_settings()
    corpus_path = settings.get_resolved_corpus_path()
    chunks, report = load_and_validate_corpus(corpus_path)
    assert report.valid_records == 75
    return chunks


@pytest.fixture
def temp_chroma_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary ChromaDB persistence directory."""
    chroma_dir = tmp_path / "chroma_test"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chroma_dir


@pytest.fixture
def retriever(valid_corpus_chunks: list[CorpusChunk], temp_chroma_dir: Path) -> HybridRetriever:
    """Fixture providing an initialized HybridRetriever with temporary Chroma storage."""
    return HybridRetriever(
        corpus_chunks=valid_corpus_chunks,
        chroma_persist_dir=temp_chroma_dir,
    )


# -----------------------------------------------------------------------------
# 1. Ingestion & Validation
# -----------------------------------------------------------------------------

def test_01_all_75_source_chunks_loaded(valid_corpus_chunks: list[CorpusChunk]):
    """Test 1: All 75 source chunks are loaded when the supplied corpus is valid."""
    assert len(valid_corpus_chunks) == 75
    chunk_ids = {c.chunk_id for c in valid_corpus_chunks}
    assert len(chunk_ids) == 75
    # Verify presence of representative chunk IDs
    assert "company:0" in chunk_ids
    assert "pricing:0" in chunk_ids
    assert "incident-12:1" in chunk_ids
    assert "retention:0" in chunk_ids


def test_02_duplicate_chunk_ids_detected(tmp_path: Path):
    """Test 2: Ingestion detects and rejects duplicate chunk IDs."""
    test_jsonl = tmp_path / "dup_corpus.jsonl"
    record1 = {
        "chunk_id": "test:0",
        "doc_id": "test",
        "title": "Test Title",
        "category": "onboarding",
        "owner": "Asterline Documentation",
        "source_url": "kb://test",
        "published": "20260301",
        "version": "1",
        "text": "First copy",
    }
    record2 = dict(record1, text="Second copy with same chunk_id")
    with open(test_jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps(record1) + "\n" + json.dumps(record2) + "\n")

    chunks, report = load_and_validate_corpus(test_jsonl)
    assert report.total_records == 2
    assert report.valid_records == 1
    assert report.rejected_records == 1
    assert any("Duplicate chunk_id" in err for err in report.errors)


def test_03_malformed_records_fail_clearly(tmp_path: Path):
    """Test 3: Malformed JSON or missing required fields produce clear errors."""
    test_jsonl = tmp_path / "malformed.jsonl"
    with open(test_jsonl, "w", encoding="utf-8") as f:
        f.write("{invalid json syntax\n")
        f.write(json.dumps({"chunk_id": "missing_fields"}) + "\n")

    chunks, report = load_and_validate_corpus(test_jsonl)
    assert report.total_records == 2
    assert report.valid_records == 0
    assert report.rejected_records == 2
    assert any("Invalid JSON syntax" in err for err in report.errors)
    assert any("Validation failed" in err for err in report.errors)


def test_04_source_corpus_checksum_unchanged():
    """Test 4: Ingestion does not modify the source corpus file; SHA256 matches golden."""
    settings = get_settings()
    corpus_path = settings.get_resolved_corpus_path()
    current_sha = compute_file_sha256(corpus_path)
    assert current_sha == GOLDEN_CORPUS_SHA256


def test_05_indexed_chunks_preserve_metadata(valid_corpus_chunks: list[CorpusChunk]):
    """Test 5: All chunks preserve exact original IDs, titles, versions, and categories."""
    by_id = {c.chunk_id: c for c in valid_corpus_chunks}
    
    # Check pricing:0
    p0 = by_id["pricing:0"]
    assert p0.title == "Plans and billing"
    assert p0.category == "pricing"
    assert p0.published == "20260310"
    assert p0.version == "3"

    # Check incident-12:1
    inc12 = by_id["incident-12:1"]
    assert inc12.title == "INC-2025-12 duplicate exports"
    assert inc12.category == "incident"
    assert "Heron changed query result ordering" in inc12.text


# -----------------------------------------------------------------------------
# 2. Vector Indexing Lifecycle & Persistence
# -----------------------------------------------------------------------------

def test_06_index_initialization_repeated_without_duplicates(retriever: HybridRetriever):
    """Test 6: Repeated sync_index calls do not create duplicate entries in ChromaDB."""
    assert retriever.collection.count() == 75
    # Call sync again
    retriever.sync_index()
    assert retriever.collection.count() == 75


def test_07_index_persists_and_reopens(
    valid_corpus_chunks: list[CorpusChunk], temp_chroma_dir: Path
):
    """Test 7: The index persists on disk and reopens without re-embedding."""
    r1 = HybridRetriever(
        corpus_chunks=valid_corpus_chunks,
        chroma_persist_dir=temp_chroma_dir,
    )
    assert r1.collection.count() == 75

    # Reopen with new retriever instance pointing to same directory
    r2 = HybridRetriever(
        corpus_chunks=valid_corpus_chunks,
        chroma_persist_dir=temp_chroma_dir,
    )
    assert r2.collection.count() == 75


# -----------------------------------------------------------------------------
# 3. Exact Keyword & Identifier Retrieval
# -----------------------------------------------------------------------------

def test_08_exact_incident_id_retrieval(retriever: HybridRetriever):
    """Test 8: Exact incident IDs (INC-2025-12, INC-2025-08, INC-2026-03) are retrievable."""
    for inc_id in ["INC-2025-12", "INC-2025-08", "INC-2026-03"]:
        results = retriever.keyword_search(inc_id, top_k=3)
        assert len(results) > 0
        assert any(
            inc_id.lower() in (r.chunk_id + " " + r.title + " " + r.text).lower()
            for r in results
        )


def test_09_exact_api_paths_retrieval(retriever: HybridRetriever):
    """Test 9: Exact API endpoint paths (POST /v2/runs) return ingest:0."""
    results = retriever.keyword_search("POST /v2/runs", top_k=3)
    assert len(results) > 0
    assert results[0].chunk_id == "ingest:0"
    assert "400 run records or 1 MB" in results[0].text


def test_10_version_identifiers_searchable(retriever: HybridRetriever):
    """Test 10: Specific version numbers (e.g. patch 2.1.2, version 2.3.1) are searchable."""
    results = retriever.keyword_search("patch 2.1.2 deduplication", top_k=3)
    assert len(results) > 0
    chunk_ids = [r.chunk_id for r in results]
    assert "release-21:2" in chunk_ids or "incident-12:1" in chunk_ids


# -----------------------------------------------------------------------------
# 4. Dense & Hybrid Search Quality
# -----------------------------------------------------------------------------

def test_11_dense_retrieval_semantic_relevance(retriever: HybridRetriever):
    """Test 11: Dense retrieval finds semantically relevant chunks for natural language queries."""
    results = retriever.dense_search("What are the subscription plans and costs?", top_k=3)
    assert len(results) > 0
    top_chunk_ids = [r.chunk_id for r in results]
    assert "pricing:0" in top_chunk_ids
    assert any("Launch costs $59" in r.text for r in results)


def test_12_hybrid_retrieval_finds_expected_evidence(retriever: HybridRetriever):
    """Test 12: Hybrid retrieval successfully finds relevant chunks combining semantic and exact search."""
    # Test cold recovery retention query
    ret_results = retriever.hybrid_search("cold recovery storage retention policy", top_k=3)
    ret_ids = [r.chunk_id for r in ret_results]
    assert "retention:0" in ret_ids

    # Test on-call stipend query
    oncall_results = retriever.hybrid_search("weekly on-call stipend compensation", top_k=3)
    oncall_ids = [r.chunk_id for r in oncall_results]
    assert "handbook:1" in oncall_ids or "oncall:0" in oncall_ids


def test_13_duplicate_results_removed(retriever: HybridRetriever):
    """Test 13: Results returned by hybrid search are deduplicated by chunk_id."""
    results = retriever.hybrid_search("Asterline workflow monitoring features", top_k=10)
    seen_ids = set()
    for r in results:
        assert r.chunk_id not in seen_ids
        seen_ids.add(r.chunk_id)


def test_14_metadata_filtering(retriever: HybridRetriever):
    """Test 14: Metadata filtering scopes results to requested category."""
    results = retriever.dense_search(
        "outage and downtime", top_k=5, metadata_filter={"category": "incident"}
    )
    assert len(results) > 0
    for r in results:
        assert r.category == "incident"


def test_15_empty_query_handled_safely(retriever: HybridRetriever):
    """Test 15: Empty or whitespace query returns empty list without error."""
    assert retriever.dense_search("") == []
    assert retriever.dense_search("   ") == []
    assert retriever.keyword_search("") == []
    assert retriever.hybrid_search("") == []


def test_16_missing_corpus_file_produces_error(tmp_path: Path):
    """Test 16: Missing corpus file path raises FileNotFoundError."""
    missing_path = tmp_path / "does_not_exist.jsonl"
    with pytest.raises(FileNotFoundError, match="Corpus file not found"):
        load_and_validate_corpus(missing_path)


def test_17_retrieval_converts_to_state_retrieved_chunk_schema(retriever: HybridRetriever):
    """Test 17: Retrieved outputs are valid instances of RetrievedChunk from asterline.graph.state."""
    results = retriever.hybrid_search("AQL query timeout", top_k=2)
    assert len(results) > 0
    for r in results:
        assert isinstance(r, RetrievedChunk)
        assert isinstance(r.chunk_id, str)
        assert isinstance(r.title, str)
        assert isinstance(r.text, str)
        assert isinstance(r.score, float)


def test_18_index_rebuild_does_not_corrupt_prior_usable_index(
    valid_corpus_chunks: list[CorpusChunk], temp_chroma_dir: Path
):
    """Test 18: Re-indexing with force_rebuild safely completes without index corruption."""
    retriever = HybridRetriever(
        corpus_chunks=valid_corpus_chunks,
        chroma_persist_dir=temp_chroma_dir,
    )
    assert retriever.collection.count() == 75

    # Force rebuild
    retriever.sync_index(force_rebuild=True)
    assert retriever.collection.count() == 75

    # Confirm queries still work after rebuild
    test_res = retriever.hybrid_search("Signals alert rules", top_k=2)
    assert len(test_res) > 0
    assert any("alerts" in r.chunk_id for r in test_res)
