"""Investigation Research Agent for Asterline Labs.

Handles complex multi-document investigations, incident root-cause analysis,
historical changelog comparisons, and conflicting policy reconciliations.
Integrates ConflictResolutionEngine to evaluate temporal supersession.
"""

from typing import Any, Dict, List, Optional
from asterline.agents.conflict_resolution import ConflictResolutionEngine
from asterline.agents.llm import LLMClient
from asterline.graph.state import Citation, RetrievedChunk
from asterline.guardrails.input import wrap_user_prompt
from asterline.rag.retriever import HybridRetriever


class InvestigationAgent:
    """Investigates complex multi-document inquiries and resolves temporal conflicts."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: Optional[LLMClient] = None,
    ):
        self.retriever = retriever
        self.llm = llm_client or LLMClient()

    def investigate(
        self,
        query: str,
        user_memories: Optional[List[Dict[str, Any]]] = None,
        feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute deep investigation across multiple documents with conflict resolution."""
        # 1. Retrieve candidate chunks with a wider Top-K for multi-document synthesis
        retrieved = self.retriever.hybrid_search(query, top_k=6)

        if not retrieved:
            return {
                "draft_response": "Investigation concluded: No authoritative company records were found relating to this inquiry.",
                "citations": [],
                "retrieved_chunks": [],
            }

        # 2. Run Conflict & Temporal Resolution Engine
        resolution = ConflictResolutionEngine.resolve(retrieved, query=query)

        # 3. Format structured multi-document evidence
        evidence_lines = []
        for c in retrieved:
            is_controlling = c.chunk_id in resolution.controlling_chunk_ids
            is_superseded = c.chunk_id in resolution.superseded_chunk_ids
            status_tag = " [CONTROLLING/CURRENT]" if is_controlling else (" [SUPERSEDED/HISTORICAL]" if is_superseded else "")
            evidence_lines.append(
                f"[{c.chunk_id}]{status_tag} {c.title} (Published: {c.published}, Version: {c.version}):\n{c.text}\n"
            )
        evidence_block = "\n".join(evidence_lines)

        resolution_note = (
            f"\nTEMPORAL RESOLUTION ANALYSIS (Tier {resolution.tier}):\n"
            f"- Controlling Evidence: {resolution.controlling_chunk_ids}\n"
            f"- Superseded/Historical: {resolution.superseded_chunk_ids}\n"
            f"- Resolution Rationale: {resolution.reasoning}\n"
        )

        revision_str = ""
        if feedback:
            revision_str = f"\nVERIFIER FEEDBACK FROM PREVIOUS ATTEMPT (Please fix):\n{feedback}\n"

        system_prompt = (
            "You are the Investigation Research Agent for Asterline Labs.\n"
            "Synthesize a thorough, multi-document investigation report.\n"
            "Rules:\n"
            "1. State the factual findings directly supported by evidence.\n"
            "2. Explain chronological timelines, root causes, or version evolutions.\n"
            "3. Explicitly acknowledge superseded policies vs current policies.\n"
            "4. Cite supporting chunk IDs in square brackets, e.g. [incident-12:1] [release-21:2].\n"
            "5. Acknowledge when a conflict cannot be resolved or an announcement has not occurred."
        )

        user_content = (
            f"Retrieved Investigation Evidence:\n{evidence_block}\n"
            f"{resolution_note}\n"
            f"{revision_str}\n"
            f"Investigation Query:\n{wrap_user_prompt(query)}"
        )

        # 4. Generate draft answer
        if not self.llm.settings.llm_api_key or self.llm.mock_synthesizer is not None:
            # Deterministic synthesis for test reproducibility and offline environments
            draft_parts = [
                f"Investigation summary for: '{query}'",
                resolution.reasoning,
            ]
            for cid in resolution.controlling_chunk_ids:
                matching = next((c for c in retrieved if c.chunk_id == cid), None)
                if matching:
                    draft_parts.append(f"According to {matching.title} [{matching.chunk_id}]: {matching.text}")
            draft = "\n\n".join(draft_parts)
        else:
            draft = self.llm.generate(prompt=user_content, system_prompt=system_prompt)

        # 5. Extract Citations
        citations: List[Citation] = []
        for c in retrieved:
            if f"[{c.chunk_id}]" in draft or c.chunk_id in draft or c.chunk_id in resolution.controlling_chunk_ids:
                citations.append(
                    Citation(
                        chunk_id=c.chunk_id,
                        doc_id=c.doc_id,
                        title=c.title,
                        source_url=c.source_url,
                    )
                )

        return {
            "draft_response": draft,
            "citations": citations,
            "retrieved_chunks": retrieved,
            "resolution": resolution,
        }
