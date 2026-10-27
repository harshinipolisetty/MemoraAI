"""Benchmark runner evaluating Dense, Keyword, and Hybrid retrieval modes.

Computes Hit Rate@k, Recall@k, and Mean Reciprocal Rank (MRR) on evaluation/datasets/research.jsonl.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure src is on Python path for standalone script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asterline.config.settings import get_settings
from asterline.rag.ingestion import load_and_validate_corpus
from asterline.rag.retriever import HybridRetriever



def evaluate_retrieval(top_k: int = 5) -> Dict[str, Any]:
    """Run retrieval benchmark and return comparative metrics."""
    settings = get_settings()
    corpus_path = settings.get_resolved_corpus_path()
    chunks, _ = load_and_validate_corpus(corpus_path)

    chroma_dir = settings.get_resolved_chroma_path()
    retriever = HybridRetriever(
        corpus_chunks=chunks,
        chroma_persist_dir=chroma_dir,
        embedding_model_name=settings.embedding_model_name,
    )

    dataset_path = settings.resolve_path(Path("evaluation/datasets/research.jsonl"))
    if not dataset_path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {dataset_path}")

    cases: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line.strip()))

    modes = ["dense", "keyword", "hybrid"]
    metrics: Dict[str, Dict[str, float]] = {
        m: {"hit_rate": 0.0, "recall": 0.0, "mrr": 0.0} for m in modes
    }

    num_cases = len(cases)
    if num_cases == 0:
        return {"error": "Empty dataset"}

    for case in cases:
        query = case["query"]
        expected_ids = set(case["expected_chunk_ids"])

        for mode in modes:
            if mode == "dense":
                results = retriever.dense_search(query, top_k=top_k)
            elif mode == "keyword":
                results = retriever.keyword_search(query, top_k=top_k)
            else:
                results = retriever.hybrid_search(query, top_k=top_k)

            retrieved_ids = [r.chunk_id for r in results]

            # 1. Hit Rate: At least one expected chunk found in top_k
            hits = [cid for cid in retrieved_ids if cid in expected_ids]
            if hits:
                metrics[mode]["hit_rate"] += 1.0

            # 2. Recall@k: Fraction of expected chunks found
            case_recall = len(hits) / len(expected_ids) if expected_ids else 0.0
            metrics[mode]["recall"] += case_recall

            # 3. MRR: Reciprocal rank of first matching chunk
            reciprocal_rank = 0.0
            for rank, cid in enumerate(retrieved_ids, start=1):
                if cid in expected_ids:
                    reciprocal_rank = 1.0 / rank
                    break
            metrics[mode]["mrr"] += reciprocal_rank

    # Average metrics
    summary: Dict[str, Any] = {"num_queries": num_cases, "top_k": top_k, "modes": {}}
    for m in modes:
        summary["modes"][m] = {
            "hit_rate": round(metrics[m]["hit_rate"] / num_cases, 4),
            "recall": round(metrics[m]["recall"] / num_cases, 4),
            "mrr": round(metrics[m]["mrr"] / num_cases, 4),
        }

    return summary


if __name__ == "__main__":
    results = evaluate_retrieval(top_k=5)
    print(json.dumps(results, indent=2))
