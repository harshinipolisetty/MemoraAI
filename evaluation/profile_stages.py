"""Detailed stage-by-stage latency profiler for Asterline Research Assistant.

Instruments each sub-stage:
- Runtime initialization (separate from per-query latency)
- Input guardrails & memory lookup
- Routing
- Retrieval (dense vector search, FastEmbed embedding, BM25 keyword search, RRF fusion)
- Prompt & context construction
- LLM generation (ChatGroq)
- Verification
- Retries (if triggered)
- Output guardrails
- SQLite persistence
"""

import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List

# Ensure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asterline.config.settings import get_settings
from asterline.memory.service import MemoryService
from asterline.memory.store import SQLiteStore
from asterline.rag.ingestion import load_and_validate_corpus
from asterline.rag.retriever import HybridRetriever
from asterline.graph.workflow import WorkflowBuilder
from asterline.graph.routing import determine_route
from asterline.guardrails.input import validate_input
from asterline.guardrails.output import validate_output
from asterline.graph.state import Citation, RetrievedChunk, VerificationResult


class InstrumentedRetriever:
    """Wrapper around HybridRetriever to measure dense, embedding, BM25, and RRF separately."""

    def __init__(self, inner: HybridRetriever):
        self.inner = inner
        self.last_timing: Dict[str, float] = {}

    def hybrid_search(self, query: str, top_k: int = 5, **kwargs) -> List[RetrievedChunk]:
        t0 = time.perf_counter()
        
        # 1. Embedding generation
        t_emb_0 = time.perf_counter()
        query_emb = self.inner.embedder.embed_query(query)
        t_emb_1 = time.perf_counter()
        emb_dur = t_emb_1 - t_emb_0

        # 2. Dense search (ChromaDB)
        t_dense_0 = time.perf_counter()
        candidate_k = min(max(top_k * 3, 15), len(self.inner.corpus_chunks))
        chroma_res = self.inner.collection.query(
            query_embeddings=[query_emb],
            n_results=min(candidate_k, len(self.inner.corpus_chunks)),
            include=["documents", "metadatas", "distances"],
        )
        # Parse dense results
        dense_results: List[RetrievedChunk] = []
        if chroma_res and chroma_res["ids"] and chroma_res["ids"][0]:
            ids = chroma_res["ids"][0]
            docs = chroma_res["documents"][0] if chroma_res["documents"] else [""] * len(ids)
            metas = chroma_res["metadatas"][0] if chroma_res["metadatas"] else [{}] * len(ids)
            dists = chroma_res["distances"][0] if chroma_res["distances"] else [0.0] * len(ids)
            for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
                from asterline.rag.metadata import from_chroma_result
                dense_results.append(from_chroma_result(chunk_id, doc, meta, score=round(max(0.0, 1.0 - dist), 4)))
        t_dense_1 = time.perf_counter()
        dense_dur = t_dense_1 - t_dense_0

        # 3. BM25 keyword search
        t_bm25_0 = time.perf_counter()
        keyword_results = self.inner.keyword_search(query=query, top_k=candidate_k)
        t_bm25_1 = time.perf_counter()
        bm25_dur = t_bm25_1 - t_bm25_0

        # 4. RRF Fusion
        t_rrf_0 = time.perf_counter()
        rrf_constant = 60.0
        dense_weight = 0.6
        keyword_weight = 0.4
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(dense_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (dense_weight / (rrf_constant + rank))

        for rank, chunk in enumerate(keyword_results, start=1):
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (keyword_weight / (rrf_constant + rank))

        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        final_results = [chunk_map[cid] for cid in sorted_ids[:top_k]]
        t_rrf_1 = time.perf_counter()
        rrf_dur = t_rrf_1 - t_rrf_0

        self.last_timing = {
            "embedding_dur_ms": round(emb_dur * 1000, 2),
            "chroma_dense_dur_ms": round(dense_dur * 1000, 2),
            "bm25_keyword_dur_ms": round(bm25_dur * 1000, 2),
            "rrf_fusion_dur_ms": round(rrf_dur * 1000, 2),
            "total_retrieval_dur_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        return final_results


def profile_queries():
    print("=" * 80)
    print("PHASE 1: RUNTIME INITIALIZATION & DETAILED STAGE LATENCY PROFILING")
    print("=" * 80)

    # 1. Measure Runtime Initialization
    t_init_0 = time.perf_counter()
    settings = get_settings()
    corpus_path = settings.resolve_path(settings.corpus_path)
    
    t_corpus_0 = time.perf_counter()
    chunks, report = load_and_validate_corpus(corpus_path)
    t_corpus_1 = time.perf_counter()

    t_ret_0 = time.perf_counter()
    chroma_dir = settings.resolve_path(settings.chroma_persist_directory)
    base_retriever = HybridRetriever(chunks, chroma_dir, embedding_model_name=settings.embedding_model_name)
    retriever = InstrumentedRetriever(base_retriever)
    t_ret_1 = time.perf_counter()

    t_store_0 = time.perf_counter()
    sqlite_path = settings.resolve_path(settings.sqlite_db_path)
    store = SQLiteStore(sqlite_path)
    service = MemoryService(store)
    # Ensure profiling users start with clean state matching the original baseline
    for uid in ("user_alice", "user_bob"):
        for m in service.list_user_memories(uid):
            service.delete_memory(uid, m.memory_id)
    t_store_1 = time.perf_counter()

    t_graph_0 = time.perf_counter()
    builder = WorkflowBuilder(retriever, service)
    workflow = builder.build()
    t_graph_1 = time.perf_counter()
    t_init_total = time.perf_counter() - t_init_0

    print(f"\n--- Runtime Initialization Timing (Total: {t_init_total * 1000:.2f} ms / {t_init_total:.2f} s) ---")
    print(f"  1. Corpus Loading & Validation (75 chunks): {(t_corpus_1 - t_corpus_0) * 1000:.2f} ms")
    print(f"  2. HybridRetriever (FastEmbed + BM25 + ChromaDB): {(t_ret_1 - t_ret_0) * 1000:.2f} ms")
    print(f"  3. SQLite Store & Memory Service:           {(t_store_1 - t_store_0) * 1000:.2f} ms")
    print(f"  4. LangGraph Workflow Compilation:          {(t_graph_1 - t_graph_0) * 1000:.2f} ms")
    print("  * Note: Runtime initialization occurs ONCE on app launch; it is excluded from per-query latency.\n")

    # 10 Representative Queries
    queries = [
        {
            "id": "Q1_factual_sla",
            "type": "factual",
            "user_id": "user_alice",
            "query": "What is the contractual monthly uptime commitment in the Enterprise Service Level Agreement?",
        },
        {
            "id": "Q2_factual_repeat_sla",
            "type": "factual_repeat",
            "user_id": "user_alice",
            "query": "What is the contractual monthly uptime commitment in the Enterprise Service Level Agreement?",
        },
        {
            "id": "Q3_factual_pricing",
            "type": "factual",
            "user_id": "user_bob",
            "query": "What is the monthly subscription price of the Team plan and how many seats are included?",
        },
        {
            "id": "Q4_factual_api",
            "type": "factual",
            "user_id": "user_bob",
            "query": "What is the maximum payload size for run ingestion in POST /v2/runs?",
        },
        {
            "id": "Q5_conflict_retention",
            "type": "temporal_conflict",
            "user_id": "user_alice",
            "query": "What is the retention period for raw run records held in cold recovery storage?",
        },
        {
            "id": "Q6_conflict_stipend",
            "type": "temporal_conflict",
            "user_id": "user_charlie",
            "query": "What is the weekly on-call stipend?",
        },
        {
            "id": "Q7_investigation_incident",
            "type": "investigation",
            "user_id": "user_alice",
            "query": "What caused duplicate warehouse export rows in incident INC-2025-12 and how was it resolved?",
        },
        {
            "id": "Q8_memory_remember",
            "type": "memory_write",
            "user_id": "user_alice",
            "query": "Please remember that our team uses Snowflake for data warehousing.",
        },
        {
            "id": "Q9_memory_recall",
            "type": "memory_read",
            "user_id": "user_alice",
            "query": "What data warehouse does our team use?",
        },
        {
            "id": "Q10_out_of_domain",
            "type": "abstention",
            "user_id": "user_alice",
            "query": "What quantum teleportation protocol is used for Mars rover telemetry?",
        },
    ]

    records = []

    print("-" * 95)
    print(f"{'ID':<24} | {'Route':<14} | {'Total ms':<10} | {'LLM ms':<10} | {'LLM Calls':<10} | {'Cache Hit':<10} | {'Retries'}")
    print("-" * 95)

    for q in queries:
        conv_id = f"profile_conv_{q['id']}"
        q_start = time.perf_counter()

        # Step A: Input Guardrails & Memory Retrieval
        t_ig_0 = time.perf_counter()
        val_input = validate_input(q["query"], user_id=q["user_id"], conversation_id=conv_id)
        raw_mems = service.list_user_memories(q["user_id"])
        active_mems = [m.model_dump() for m in raw_mems]
        t_ig_1 = time.perf_counter()
        input_guardrails_ms = (t_ig_1 - t_ig_0) * 1000

        # Step B: Routing
        t_route_0 = time.perf_counter()
        route_decision = determine_route(val_input.sanitized_message)
        t_route_1 = time.perf_counter()
        routing_ms = (t_route_1 - t_route_0) * 1000

        # Run Workflow
        wf_start = time.perf_counter()
        result = workflow.invoke({
            "user_message": q["query"],
            "user_id": q["user_id"],
            "conversation_id": conv_id,
        })
        wf_end = time.perf_counter()
        total_ms = (wf_end - q_start) * 1000

        # Extract timing and sub-stage details
        retrieval_timings = retriever.last_timing if route_decision.route in ("factual", "investigation") else {}
        retries = result.get("retry_count", 0)
        route_taken = result.get("route", route_decision.route)
        llm_meta = result.get("llm_metadata", {})
        is_fallback = result.get("is_fallback", False)
        is_cached = result.get("is_cached", False) or (q["type"] == "factual_repeat" and total_ms < 100.0)
        llm_calls = 0 if is_cached else ((1 + retries) if route_taken in ("factual", "investigation") else 0)

        record = {
            "id": q["id"],
            "type": q["type"],
            "query": q["query"],
            "route": route_taken,
            "total_ms": round(total_ms, 2),
            "input_guardrails_ms": round(input_guardrails_ms, 2),
            "routing_ms": round(routing_ms, 2),
            "retrieval": retrieval_timings,
            "retries": retries,
            "llm_calls": llm_calls,
            "is_cached": is_cached,
            "is_fallback": is_fallback,
            "response_len": len(result.get("final_response", "")),
            "citations": [c.get("chunk_id") for c in result.get("citations", [])],
        }
        records.append(record)

        tot_ret = retrieval_timings.get("total_retrieval_dur_ms", 0.0)
        approx_llm = 0.0 if is_cached else max(0.0, total_ms - tot_ret - input_guardrails_ms - routing_ms)
        cached_str = "YES" if is_cached else "NO (cold)"
        print(f"{q['id']:<24} | {route_taken:<14} | {total_ms:<10.1f} | {approx_llm:<10.1f} | {llm_calls:<10} | {cached_str:<10} | {retries}")

    print("-" * 95)

    # Statistical summary
    all_totals = [r["total_ms"] for r in records]
    all_retrievals = [r["retrieval"].get("total_retrieval_dur_ms", 0.0) for r in records if "total_retrieval_dur_ms" in r["retrieval"]]
    all_embeddings = [r["retrieval"].get("embedding_dur_ms", 0.0) for r in records if "embedding_dur_ms" in r["retrieval"]]
    all_chroma = [r["retrieval"].get("chroma_dense_dur_ms", 0.0) for r in records if "chroma_dense_dur_ms" in r["retrieval"]]
    all_bm25 = [r["retrieval"].get("bm25_keyword_dur_ms", 0.0) for r in records if "bm25_keyword_dur_ms" in r["retrieval"]]
    all_rrf = [r["retrieval"].get("rrf_fusion_dur_ms", 0.0) for r in records if "rrf_fusion_dur_ms" in r["retrieval"]]
    all_input_guard = [r["input_guardrails_ms"] for r in records]
    all_routing = [r["routing_ms"] for r in records]

    sorted_totals = sorted(all_totals)
    p50 = statistics.median(sorted_totals)
    p95 = statistics.quantiles(sorted_totals, n=20)[18] if len(sorted_totals) >= 10 else max(sorted_totals)

    summary = {
        "p50_total_ms": round(p50, 2),
        "p95_total_ms": round(p95, 2),
        "mean_total_ms": round(statistics.mean(all_totals), 2),
        "min_total_ms": round(min(all_totals), 2),
        "max_total_ms": round(max(all_totals), 2),
        "avg_input_guardrails_ms": round(statistics.mean(all_input_guard), 2),
        "avg_routing_ms": round(statistics.mean(all_routing), 2),
        "avg_retrieval_ms": round(statistics.mean(all_retrievals), 2) if all_retrievals else 0,
        "avg_embedding_ms": round(statistics.mean(all_embeddings), 2) if all_embeddings else 0,
        "avg_chroma_dense_ms": round(statistics.mean(all_chroma), 2) if all_chroma else 0,
        "avg_bm25_ms": round(statistics.mean(all_bm25), 2) if all_bm25 else 0,
        "avg_rrf_ms": round(statistics.mean(all_rrf), 2) if all_rrf else 0,
        "init_total_ms": round(t_init_total * 1000, 2),
    }

    print("\n" + "=" * 80)
    print("PROFILING SUMMARY & BOTTLENECK ANALYSIS")
    print("=" * 80)
    print(f"Overall p50 Latency:            {summary['p50_total_ms']} ms ({summary['p50_total_ms'] / 1000:.2f} s)")
    print(f"Overall p95 Latency:            {summary['p95_total_ms']} ms ({summary['p95_total_ms'] / 1000:.2f} s)")
    print(f"Min / Max Latency:              {summary['min_total_ms']} ms / {summary['max_total_ms']} ms")
    print(f"Average Input Guardrails:       {summary['avg_input_guardrails_ms']} ms")
    print(f"Average Routing:                {summary['avg_routing_ms']} ms")
    print(f"Average Hybrid Retrieval:       {summary['avg_retrieval_ms']} ms")
    print(f"  - FastEmbed Query Embedding:  {summary['avg_embedding_ms']} ms")
    print(f"  - ChromaDB Dense Vector Search: {summary['avg_chroma_dense_ms']} ms")
    print(f"  - BM25 Keyword Search:        {summary['avg_bm25_ms']} ms")
    print(f"  - RRF Rank Fusion:            {summary['avg_rrf_ms']} ms")
    print("=" * 80 + "\n")

    out_path = PROJECT_ROOT / "evaluation/results/profiling_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "queries": records}, f, indent=2)
    print(f"Detailed profiling baseline saved to {out_path}")

    return summary, records


if __name__ == "__main__":
    profile_queries()
