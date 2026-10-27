"""Reproducible evaluation runner for Asterline Research Assistant.

Executes all evaluation benchmarks offline without requiring paid API keys:
1. Local Hybrid Retrieval (Dense vs Keyword vs Hybrid: Hit Rate@k, Recall@k, MRR)
2. Grounding & Citation Fidelity (Citation Validity, Grounding Proxy, Abstention, Temporal Supersession)
3. Conditional Intent Routing (Accuracy across 5 workflow routes)
4. Persistent Memory & User Isolation (Persistence, Alice/Bob/Charlie isolation, Forget purge, Policy safety)

Usage:
    python -m evaluation.run_evals
"""

from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List

# Ensure src is on Python path for standalone script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asterline.config.settings import get_settings
from asterline.graph.workflow import WorkflowBuilder
from asterline.memory.service import MemoryService
from asterline.memory.store import SQLiteStore
from asterline.rag.ingestion import load_and_validate_corpus
from asterline.rag.retriever import HybridRetriever
from evaluation.evaluators import (

    GroundingEvaluator,
    MemoryEvaluator,
    RetrievalEvaluator,
    RoutingEvaluator,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSON lines dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def run_all_evaluations() -> Dict[str, Any]:
    """Execute complete offline evaluation suite and return structured report."""
    settings = get_settings()
    corpus_path = settings.get_resolved_corpus_path()
    chunks, report = load_and_validate_corpus(corpus_path)

    # Initialize retriever
    chroma_dir = settings.get_resolved_chroma_path()
    retriever = HybridRetriever(
        corpus_chunks=chunks,
        chroma_persist_dir=chroma_dir,
        embedding_model_name=settings.embedding_model_name,
    )

    # 1. Retrieval Benchmark
    research_path = settings.resolve_path(Path("evaluation/datasets/research.jsonl"))
    research_cases = load_jsonl(research_path)
    retrieval_evaluator = RetrievalEvaluator(retriever)
    retrieval_results = retrieval_evaluator.evaluate(research_cases, top_k=5)

    # 2. End-to-End Grounding & Citation Benchmark
    temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    temp_db_path = Path(temp_dir.name) / "eval_memory.db"
    store = SQLiteStore(temp_db_path)
    memory_service = MemoryService(store)
    workflow_builder = WorkflowBuilder(retriever=retriever, memory_service=memory_service)
    workflow_app = workflow_builder.build()

    grounding_evaluator = GroundingEvaluator(workflow_app, corpus_chunks=chunks)
    grounding_results = grounding_evaluator.evaluate(research_cases)

    # 3. Routing Benchmark
    routing_path = settings.resolve_path(Path("evaluation/datasets/routing.jsonl"))
    routing_cases = load_jsonl(routing_path)
    routing_evaluator = RoutingEvaluator()
    routing_results = routing_evaluator.evaluate(routing_cases)

    # 4. Memory Persistence & Isolation Benchmark
    memory_path = settings.resolve_path(Path("evaluation/datasets/memory.jsonl"))
    memory_scenarios = load_jsonl(memory_path)
    # Use isolated fresh database for clean scenario execution
    isolated_db_path = Path(temp_dir.name) / "isolated_eval_memory.db"
    isolated_store = SQLiteStore(isolated_db_path)
    isolated_service = MemoryService(isolated_store)
    memory_evaluator = MemoryEvaluator(isolated_service)
    memory_results = memory_evaluator.evaluate(memory_scenarios)

    # Compile Final Report
    final_report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "corpus_integrity": {
            "total_chunks": report.total_records,
            "sha256": report.corpus_sha256,
        },
        "retrieval_benchmark": retrieval_results,
        "grounding_and_citation_benchmark": {
            "total_queries": grounding_results["num_cases"],
            "standard_queries": grounding_results["standard_queries_count"],
            "abstention_queries": grounding_results["abstention_queries_count"],
            "citation_validity_rate": grounding_results["citation_validity_rate"],
            "grounded_response_rate": grounding_results["grounded_response_rate"],
            "correct_abstention_rate": grounding_results["correct_abstention_rate"],
            "temporal_supersession_accuracy": grounding_results["temporal_supersession_accuracy"],
            "total_citations_evaluated": grounding_results["total_citations_evaluated"],
            "valid_citations_count": grounding_results["valid_citations_count"],
        },
        "routing_benchmark": routing_results,
        "memory_benchmark": memory_results,
    }

    # Save to disk
    results_dir = settings.resolve_path(Path("evaluation/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    report_file = results_dir / "evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    return final_report


def print_summary_table(report: Dict[str, Any]) -> None:
    """Print readable human-formatted summary table to console."""
    print("\n" + "=" * 70)
    print("ASTERLINE RESEARCH ASSISTANT — EVALUATION SUMMARY REPORT")
    print("=" * 70)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Corpus Chunks: {report['corpus_integrity']['total_chunks']} (SHA256: {report['corpus_integrity']['sha256'][:16]}...)")

    print("\n[1] RETRIEVAL BENCHMARK (Top-K = 5)")
    print("-" * 50)
    modes = report["retrieval_benchmark"]["modes"]
    print(f"{'Mode':<12} | {'Hit Rate@5':<12} | {'Recall@5':<12} | {'MRR':<10}")
    print("-" * 50)
    for m, vals in modes.items():
        print(f"{m.capitalize():<12} | {vals['hit_rate']:<12.4f} | {vals['recall']:<12.4f} | {vals['mrr']:<10.4f}")

    print("\n[2] GROUNDING & CITATION BENCHMARK")
    print("-" * 50)
    gb = report["grounding_and_citation_benchmark"]
    print(f"Total Research Cases:            {gb['total_queries']}")
    print(f"Citation Validity Rate:          {gb['citation_validity_rate'] * 100:.2f}% ({gb['valid_citations_count']}/{gb['total_citations_evaluated']})")
    print(f"Grounded Response Rate:          {gb['grounded_response_rate'] * 100:.2f}%")
    print(f"Correct Abstention Rate:         {gb['correct_abstention_rate'] * 100:.2f}%")
    print(f"Temporal Supersession Accuracy:  {gb['temporal_supersession_accuracy'] * 100:.2f}%")

    print("\n[3] INTENT ROUTING BENCHMARK")
    print("-" * 50)
    rb = report["routing_benchmark"]
    print(f"Total Routing Queries:           {rb['total_queries']}")
    print(f"Overall Correct Routing Rate:    {rb['correct_routing_rate'] * 100:.2f}%")
    for r, acc in rb["per_route_accuracy"].items():
        print(f"  - Route '{r}': {acc * 100:.1f}%")

    print("\n[4] PERSISTENT MEMORY & ISOLATION BENCHMARK")
    print("-" * 50)
    mb = report["memory_benchmark"]
    print(f"Total Scenarios Evaluated:       {mb['total_scenarios']}")
    print(f"Operation Success Rate:          {mb['memory_operation_success_rate'] * 100:.2f}%")
    print(f"Cross-User Isolation Leaks:      {mb['cross_user_isolation_violations']} (Expected: 0)")
    print(f"Forget Permanent Purge Rate:     {mb['forget_purge_rate'] * 100:.2f}%")
    print(f"Policy Rejection Accuracy:       {mb['policy_rejection_accuracy'] * 100:.2f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    rep = run_all_evaluations()
    print_summary_table(rep)
