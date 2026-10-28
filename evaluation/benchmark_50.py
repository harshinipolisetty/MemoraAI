"""50-Query Latency Benchmark Suite across 5 distinct query categories.

Categories (10 requests each = 50 total):
1. Simple questions: Greetings, bot identity, polite conversational follow-ups.
2. Memory questions: Remember workspace facts, personal context recall, list memories, forget.
3. Factual documentation: Single-topic pricing, limits, APIs, retention, and SLAs.
4. Conflicting policies: Multi-version comparisons, incidents (INC-*), temporal supersessions.
5. Unanswerable questions: Out-of-domain queries and missing documentation requiring evidence-based abstention.

Measures Cold-Start vs. Warm-Request latency, p50 (median), and p95 separately.
"""

import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List
import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asterline.config.settings import get_settings
from asterline.graph.workflow import WorkflowBuilder
from asterline.memory.service import MemoryService
from asterline.memory.store import SQLiteStore
from asterline.rag.cache import DocumentationAnswerCache
from asterline.rag.ingestion import load_and_validate_corpus
from asterline.rag.retriever import HybridRetriever

CATEGORIES: Dict[str, List[Dict[str, Any]]] = {
    "simple_questions": [
        {"id": "simp_01", "query": "Hello! How are you today?"},
        {"id": "simp_02", "query": "Hi Asterline assistant"},
        {"id": "simp_03", "query": "Who are you and what can you do?"},
        {"id": "simp_04", "query": "Good morning!"},
        {"id": "simp_05", "query": "What is your purpose?"},
        {"id": "simp_06", "query": "How can you help my engineering team?"},
        {"id": "simp_07", "query": "Thank you so much!"},
        {"id": "simp_08", "query": "Thanks for the help"},
        {"id": "simp_09", "query": "Understood, got it"},
        {"id": "simp_10", "query": "Goodbye, have a great day!"},
    ],
    "memory_questions": [
        {"id": "mem_01", "query": "Please remember that our team uses Snowflake for warehouse exports."},
        {"id": "mem_02", "query": "Remember that our deployment region is eu-west-1."},
        {"id": "mem_03", "query": "Remember that our workspace is on the Team tier."},
        {"id": "mem_04", "query": "Take note that Alice is the lead DevOps engineer for our project."},
        {"id": "mem_05", "query": "What do you remember about me?"},
        {"id": "mem_06", "query": "What data warehouse does our team use?"},
        {"id": "mem_07", "query": "Which region is our deployment in?"},
        {"id": "mem_08", "query": "What tier is our workspace on?"},
        {"id": "mem_09", "query": "What is Alice's role on our team?"},
        {"id": "mem_10", "query": "Please forget our deployment region."},
    ],
    "factual_documentation": [
        {"id": "fact_01", "query": "What is the monthly cost of the Launch plan in Asterline?"},
        {"id": "fact_02", "query": "What is the maximum ingestion payload size for the /v2/runs endpoint?"},
        {"id": "fact_03", "query": "How many active alert rules are allowed on the Launch tier?"},
        {"id": "fact_04", "query": "What is the default retention period for raw trace data on the Free tier?"},
        {"id": "fact_05", "query": "What SLA uptime percentage is guaranteed for Enterprise tier customers?"},
        {"id": "fact_06", "query": "Which compression formats are supported for trace ingestion payloads?"},
        {"id": "fact_07", "query": "What rate limit applies to the /v2/query API endpoint per workspace?"},
        {"id": "fact_08", "query": "How many team members can be added to an Asterline Team tier workspace?"},
        {"id": "fact_09", "query": "What is the maximum execution duration for an Asterline query in AQL?"},
        {"id": "fact_10", "query": "Does Asterline support SAML single sign-on for Enterprise workspaces?"},
    ],
    "conflicting_policies": [
        {"id": "conf_01", "query": "What caused the duplicate export rows in incident INC-2025-12 and how was it resolved?"},
        {"id": "conf_02", "query": "Explain what happened in incident INC-2025-08 and the patch fix applied in release 2.1."},
        {"id": "conf_03", "query": "What is the current raw trace data retention policy in policy-v3 compared to policy-v1?"},
        {"id": "conf_04", "query": "What is the current weekend on-call stipend policy compared to the previous compensation guidelines?"},
        {"id": "conf_05", "query": "Compare the AQL query timeout in v2.4 versus the older limit in v2.0."},
        {"id": "conf_06", "query": "What is the current status of the APAC region data center launch according to the latest rollout updates?"},
        {"id": "conf_07", "query": "Why did Signals alert evaluations fail during the NTP clock skew incident?"},
        {"id": "conf_08", "query": "How did the Kafka ingestion buffer overflow in INC-2025-04 differ from earlier memory issues?"},
        {"id": "conf_09", "query": "How did the Enterprise SLA evolve between the 2024 terms and the 2025 updated master agreement?"},
        {"id": "conf_10", "query": "Compare the webhook retry backoff schedule across version 1.2 and version 2.0."},
    ],
    "unanswerable_questions": [
        {"id": "unans_01", "query": "What is Asterline's pricing discount for nonprofit quantum computing organizations?"},
        {"id": "unans_02", "query": "How do I configure Asterline's automated Terraform provider on Kubernetes 1.32?"},
        {"id": "unans_03", "query": "What is the return policy for Asterline hardware telemetry sensors?"},
        {"id": "unans_04", "query": "Which Asterline office in Sydney offers in-person customer support?"},
        {"id": "unans_05", "query": "What key rotation policy applies to webhook signing keys in Asterline?"},
        {"id": "unans_06", "query": "How do I deploy an on-premise Asterline cluster on IBM mainframe architecture?"},
        {"id": "unans_07", "query": "What is the warranty coverage for Asterline IoT edge gateway devices?"},
        {"id": "unans_08", "query": "Does Asterline offer a direct fiber optic interconnection at the Frankfurt exchange?"},
        {"id": "unans_09", "query": "How can I integrate Asterline with SAP R/3 accounting software?"},
        {"id": "unans_10", "query": "What was the resolution of the non-existent incident INC-2024-99?"},
    ],
}


def run_benchmark_pass(
    app: Any,
    pass_name: str,
) -> List[Dict[str, Any]]:
    """Execute all 50 queries sequentially and measure execution latency."""
    results = []
    print(f"\n--- Starting {pass_name.upper()} Benchmark Pass (50 queries) ---")

    for cat_name, queries in CATEGORIES.items():
        user_id = f"user_{cat_name}"
        for item in queries:
            qid = item["id"]
            query = item["query"]
            conv_id = f"conv_{pass_name}_{qid}"

            t_start = time.perf_counter()
            state = {
                "user_id": user_id,
                "conversation_id": conv_id,
                "user_message": query,
            }
            try:
                res = app.invoke(state)
                t_end = time.perf_counter()
                latency_ms = round((t_end - t_start) * 1000.0, 2)
                route = res.get("route", "unknown")
                llm_meta = res.get("llm_metadata", {})
                is_cached = llm_meta.get("cached", False)
                llm_calls = llm_meta.get("llm_calls", 1 if route in ("factual", "investigation") and not is_cached else 0)
                citations_count = len(res.get("citations", []))
                status = "PASS" if not res.get("error") else "ERROR"

                record = {
                    "id": qid,
                    "category": cat_name,
                    "query": query,
                    "latency_ms": latency_ms,
                    "route": route,
                    "is_cached": is_cached,
                    "llm_calls": llm_calls,
                    "citations_count": citations_count,
                    "status": status,
                    "answer_preview": res.get("final_response", "")[:120].replace("\n", " "),
                }
            except Exception as e:
                t_end = time.perf_perf_counter()
                latency_ms = round((t_end - t_start) * 1000.0, 2)
                record = {
                    "id": qid,
                    "category": cat_name,
                    "query": query,
                    "latency_ms": latency_ms,
                    "route": "error",
                    "is_cached": False,
                    "llm_calls": 0,
                    "citations_count": 0,
                    "status": f"EXCEPTION: {str(e)[:80]}",
                    "answer_preview": "",
                }

            cached_str = "[CACHE]" if record.get("is_cached") else ""
            print(f"  [{record['category'][:8]}] {record['id']} ({record['route']}) -> {record['latency_ms']:>7.1f} ms {cached_str}")
            results.append(record)

    return results


def compute_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics (min, median/p50, p95, max, mean, % < 2s)."""
    by_category: Dict[str, List[float]] = {}
    all_latencies = [r["latency_ms"] for r in records]

    for r in records:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r["latency_ms"])

    def calc_stats(lats: List[float]) -> Dict[str, float]:
        if not lats:
            return {}
        arr = np.array(lats)
        return {
            "count": len(lats),
            "min_ms": round(float(np.min(arr)), 1),
            "p50_median_ms": round(float(np.median(arr)), 1),
            "p95_ms": round(float(np.percentile(arr, 95)), 1),
            "max_ms": round(float(np.max(arr)), 1),
            "mean_ms": round(float(np.mean(arr)), 1),
            "pct_under_2s": round(float(np.mean(arr < 2000.0) * 100.0), 1),
        }

    cat_stats = {cat: calc_stats(lats) for cat, lats in by_category.items()}
    overall_stats = calc_stats(all_latencies)

    return {
        "overall": overall_stats,
        "by_category": cat_stats,
    }


def main():
    settings = get_settings()
    corpus_path = settings.get_resolved_corpus_path()
    chunks, _ = load_and_validate_corpus(corpus_path)

    # Isolated benchmark SQLite database
    db_path = Path("data/benchmark_50.db")
    if db_path.exists():
        db_path.unlink()
    store = SQLiteStore(db_path)
    service = MemoryService(store)

    retriever = HybridRetriever(
        corpus_chunks=chunks,
        chroma_persist_dir=Path("data/chroma"),
    )

    builder = WorkflowBuilder(retriever=retriever, memory_service=service)
    app = builder.build()

    # Clear cache before cold pass
    builder.doc_agent.cache.clear()

    # Pass 1: Cold Start
    cold_results = run_benchmark_pass(app, pass_name="cold_start")
    cold_stats = compute_statistics(cold_results)

    # Pass 2: Warm Requests
    warm_results = run_benchmark_pass(app, pass_name="warm_request")
    warm_stats = compute_statistics(warm_results)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cold_start": {
            "statistics": cold_stats,
            "queries": cold_results,
        },
        "warm_request": {
            "statistics": warm_stats,
            "queries": warm_results,
        },
    }

    out_path = Path("evaluation/results/latency_benchmark_50.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("50-QUERY BENCHMARK SUMMARY (10 requests per category)")
    print("=" * 80)
    print(f"{'Category':<24} | {'Cold p50':>10} | {'Cold p95':>10} | {'Warm p50':>10} | {'Warm p95':>10} | {'Warm %<2s':>9}")
    print("-" * 80)
    for cat in CATEGORIES.keys():
        c_p50 = f"{cold_stats['by_category'][cat]['p50_median_ms']:.1f} ms"
        c_p95 = f"{cold_stats['by_category'][cat]['p95_ms']:.1f} ms"
        w_p50 = f"{warm_stats['by_category'][cat]['p50_median_ms']:.1f} ms"
        w_p95 = f"{warm_stats['by_category'][cat]['p95_ms']:.1f} ms"
        w_pct = f"{warm_stats['by_category'][cat]['pct_under_2s']:.1f}%"
        print(f"{cat:<24} | {c_p50:>10} | {c_p95:>10} | {w_p50:>10} | {w_p95:>10} | {w_pct:>9}")
    print("-" * 80)
    o_c_p50 = f"{cold_stats['overall']['p50_median_ms']:.1f} ms"
    o_c_p95 = f"{cold_stats['overall']['p95_ms']:.1f} ms"
    o_w_p50 = f"{warm_stats['overall']['p50_median_ms']:.1f} ms"
    o_w_p95 = f"{warm_stats['overall']['p95_ms']:.1f} ms"
    o_w_pct = f"{warm_stats['overall']['pct_under_2s']:.1f}%"
    print(f"{'OVERALL (50 queries)':<24} | {o_c_p50:>10} | {o_c_p95:>10} | {o_w_p50:>10} | {o_w_p95:>10} | {o_w_pct:>9}")
    print("=" * 80)
    print(f"Detailed results written to: {out_path}")


if __name__ == "__main__":
    main()
