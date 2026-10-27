"""Evaluators for Asterline Research Assistant.

Provides modular evaluation classes for:
1. RetrievalEvaluator: Dense, Keyword, and Hybrid retrieval metrics (Hit Rate, Recall, MRR).
2. GroundingEvaluator: Citation validity, factual grounding proxy, abstention accuracy, supersession.
3. RoutingEvaluator: Intent classification accuracy and expected node execution.
4. MemoryEvaluator: Persistence across sessions, Alice/Bob/Charlie isolation, forget purge, policy safety.
"""

from typing import Any, Dict, List, Set
from asterline.agents.conflict_resolution import ConflictResolutionEngine
from asterline.graph.routing import determine_route
from asterline.graph.workflow import WorkflowBuilder
from asterline.memory.service import MemoryService
from asterline.memory.store import SQLiteStore
from asterline.rag.retriever import HybridRetriever


class RetrievalEvaluator:
    """Evaluates Dense, Keyword, and Hybrid retrieval modes against ground-truth chunk IDs."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def evaluate(self, cases: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
        """Compute Hit Rate@k, Recall@k, and MRR across modes for cases with expected chunks."""
        eval_cases = [c for c in cases if c.get("expected_chunk_ids")]
        num_cases = len(eval_cases)
        if num_cases == 0:
            return {"error": "No cases with expected_chunk_ids provided."}

        modes = ["dense", "keyword", "hybrid"]
        metrics: Dict[str, Dict[str, float]] = {
            m: {"hits": 0.0, "total_recall": 0.0, "mrr_sum": 0.0} for m in modes
        }

        for case in eval_cases:
            query = case["query"]
            expected_ids = set(case["expected_chunk_ids"])

            for mode in modes:
                if mode == "dense":
                    results = self.retriever.dense_search(query, top_k=top_k)
                elif mode == "keyword":
                    results = self.retriever.keyword_search(query, top_k=top_k)
                else:
                    results = self.retriever.hybrid_search(query, top_k=top_k)

                retrieved_ids = [r.chunk_id for r in results]
                matching = [cid for cid in retrieved_ids if cid in expected_ids]

                # Hit Rate (at least 1 matching chunk in top-k)
                if matching:
                    metrics[mode]["hits"] += 1.0

                # Recall@k (fraction of expected chunks found)
                case_recall = len(matching) / len(expected_ids) if expected_ids else 0.0
                metrics[mode]["total_recall"] += case_recall

                # MRR (reciprocal rank of first matching chunk)
                mrr_score = 0.0
                for rank, cid in enumerate(retrieved_ids, start=1):
                    if cid in expected_ids:
                        mrr_score = 1.0 / rank
                        break
                metrics[mode]["mrr_sum"] += mrr_score

        summary: Dict[str, Any] = {"num_cases": num_cases, "top_k": top_k, "modes": {}}
        for m in modes:
            summary["modes"][m] = {
                "hit_rate": round(metrics[m]["hits"] / num_cases, 4),
                "recall": round(metrics[m]["total_recall"] / num_cases, 4),
                "mrr": round(metrics[m]["mrr_sum"] / num_cases, 4),
            }
        return summary


class GroundingEvaluator:
    """Evaluates end-to-end assistant responses for citation validity, grounding proxy, and abstention."""

    def __init__(self, workflow_app: Any, corpus_chunks: List[Any]):
        self.app = workflow_app
        self.chunk_map = {c.chunk_id: c for c in corpus_chunks}

    def evaluate(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute end-to-end workflow on research cases and evaluate factual citations and abstentions."""
        total_cases = len(cases)
        total_citations = 0
        valid_citations = 0

        grounded_passes = 0
        abstention_targets = 0
        correct_abstentions = 0

        supersession_targets = 0
        correct_supersessions = 0

        case_results: List[Dict[str, Any]] = []

        for case in cases:
            query = case["query"]
            should_abstain = case.get("should_abstain", False)
            expected_cids = set(case.get("expected_chunk_ids", []))
            category = case.get("category", "general")

            initial_state = {
                "user_id": "eval_user",
                "conversation_id": f"conv_{case['id']}",
                "user_message": query,
            }

            result = self.app.invoke(initial_state)
            citations = result.get("citations", [])
            response_text = result.get("final_response", "")
            verification = result.get("verification", {})
            verdict = verification.get("verdict", "pass") if verification else "pass"

            # 1. Citation Validity (Every cited chunk exists in corpus with exact matching title)
            case_valid_cits = 0
            for cit in citations:
                total_citations += 1
                cid = cit.get("chunk_id", "")
                title = cit.get("title", "")
                if cid in self.chunk_map and self.chunk_map[cid].title == title:
                    valid_citations += 1
                    case_valid_cits += 1

            # 2. Grounded Pass Proxy
            if verdict == "pass" and not should_abstain:
                grounded_passes += 1

            # 3. Abstention Check
            is_abstention_correct = False
            if should_abstain:
                abstention_targets += 1
                # Must acknowledge lack of documentation and NOT fabricate citations
                lowered = response_text.lower()
                acknowledged = any(
                    phrase in lowered
                    for phrase in ("does not contain", "no authoritative", "not documented", "cannot answer", "verification note")
                )
                if acknowledged and len(citations) == 0:
                    correct_abstentions += 1
                    is_abstention_correct = True

            # 4. Temporal Supersession Check
            is_supersession_correct = False
            if category == "policy_supersession":
                supersession_targets += 1
                cited_cids = {c.get("chunk_id") for c in citations}
                # Must cite the newer controlling chunk
                if any(cid in cited_cids for cid in expected_cids):
                    correct_supersessions += 1
                    is_supersession_correct = True

            case_results.append({
                "id": case["id"],
                "category": category,
                "verdict": verdict,
                "citations_count": len(citations),
                "is_abstention_correct": is_abstention_correct if should_abstain else None,
                "is_supersession_correct": is_supersession_correct if category == "policy_supersession" else None,
            })

        standard_cases_count = total_cases - abstention_targets
        return {
            "num_cases": total_cases,
            "standard_queries_count": standard_cases_count,
            "abstention_queries_count": abstention_targets,
            "citation_validity_rate": round(valid_citations / total_citations, 4) if total_citations > 0 else 1.0,
            "grounded_response_rate": round(grounded_passes / standard_cases_count, 4) if standard_cases_count > 0 else 1.0,
            "correct_abstention_rate": round(correct_abstentions / abstention_targets, 4) if abstention_targets > 0 else 1.0,
            "temporal_supersession_accuracy": round(correct_supersessions / supersession_targets, 4) if supersession_targets > 0 else 1.0,
            "total_citations_evaluated": total_citations,
            "valid_citations_count": valid_citations,
            "details": case_results,
        }


class RoutingEvaluator:
    """Evaluates conditional router intent classification and execution path."""

    def evaluate(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute routing accuracy and per-route confusion metrics."""
        total = len(cases)
        if total == 0:
            return {"error": "Empty routing cases"}

        correct = 0
        per_route: Dict[str, Dict[str, int]] = {}

        for case in cases:
            query = case["query"]
            expected = case["expected_route"]

            if expected not in per_route:
                per_route[expected] = {"total": 0, "correct": 0}
            per_route[expected]["total"] += 1

            decision = determine_route(query)
            if decision.route == expected:
                correct += 1
                per_route[expected]["correct"] += 1

        summary = {
            "total_queries": total,
            "correct_routing_rate": round(correct / total, 4),
            "per_route_accuracy": {
                r: round(stats["correct"] / stats["total"], 4) for r, stats in per_route.items()
            },
        }
        return summary


class MemoryEvaluator:
    """Evaluates multi-turn user memory persistence, cross-user isolation, and safety policies."""

    def __init__(self, memory_service: MemoryService):
        self.service = memory_service

    def evaluate(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute memory scenarios covering persistence, isolation, forget, and safety."""
        total = len(scenarios)
        successful_operations = 0
        isolation_violations = 0
        policy_rejections_correct = 0
        policy_rejections_total = 0
        forget_purged_count = 0
        forget_total = 0

        for sc in scenarios:
            action = sc.get("action")
            uid = sc["user_id"]

            if action == "create":
                expected_status = sc.get("expected_status")
                item, decision = self.service.create_memory(
                    user_id=uid,
                    content=sc["content"],
                    category=sc.get("category", "general"),
                )

                if expected_status == "accepted":
                    if decision.allowed and item is not None:
                        successful_operations += 1
                        # Check isolation if requested
                        other_users = sc.get("isolation_check_users", [])
                        for other_u in other_users:
                            other_memories = self.service.list_user_memories(other_u)
                            if any(item.memory_id == m.memory_id for m in other_memories):
                                isolation_violations += 1

                elif expected_status == "rejected":
                    policy_rejections_total += 1
                    if not decision.allowed:
                        expected_reason = sc.get("expected_reason")
                        if not expected_reason or decision.reason_code == expected_reason:
                            policy_rejections_correct += 1
                            successful_operations += 1

            elif action == "forget":
                forget_total += 1
                target_str = sc.get("target_content", "")
                active_mems = self.service.list_user_memories(uid)
                target_mem = next((m for m in active_mems if target_str.lower() in m.content.lower()), None)
                if target_mem:
                    deleted = self.service.delete_memory(uid, target_mem.memory_id)
                    # Verify purged at persistence layer
                    is_active = self.service.is_memory_active(uid, target_mem.memory_id)
                    if deleted and not is_active:
                        forget_purged_count += 1
                        successful_operations += 1

            elif action == "update":
                initial_mem, _ = self.service.create_memory(uid, sc["initial_content"], "preference")
                if initial_mem:
                    updated_mem, decision = self.service.update_memory(uid, initial_mem.memory_id, sc["new_content"])
                    if updated_mem and decision.allowed:
                        successful_operations += 1

        return {
            "total_scenarios": total,
            "memory_operation_success_rate": round(successful_operations / total, 4) if total > 0 else 1.0,
            "cross_user_isolation_violations": isolation_violations,
            "forget_purge_rate": round(forget_purged_count / forget_total, 4) if forget_total > 0 else 1.0,
            "policy_rejection_accuracy": (
                round(policy_rejections_correct / policy_rejections_total, 4)
                if policy_rejections_total > 0 else 1.0
            ),
        }
