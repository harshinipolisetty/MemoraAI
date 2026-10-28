"""Investigation Research Agent for Asterline Labs.

Handles complex multi-document investigations, incident root-cause analysis,
historical changelog comparisons, and conflicting policy reconciliations.
Integrates ConflictResolutionEngine to evaluate temporal supersession.
"""

import re
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
        max_tokens: int = 1500,
    ):
        self.retriever = retriever
        self.llm = llm_client or LLMClient()
        self.max_tokens = max_tokens

    def investigate(
        self,
        query: str,
        user_memories: Optional[List[Dict[str, Any]]] = None,
        feedback: Optional[str] = None,
        existing_chunks: Optional[List[RetrievedChunk]] = None,
    ) -> Dict[str, Any]:
        """Execute deep investigation across multiple documents with conflict resolution."""
        # 1. Retrieve candidate chunks with a wider Top-K for multi-document synthesis or reuse preserved evidence
        if existing_chunks is not None:
            retrieved = existing_chunks
        else:
            retrieved = self.retriever.hybrid_search(query, top_k=6)

        if not retrieved:
            return {
                "draft_response": "Investigation concluded: No authoritative company records were found relating to this inquiry.",
                "citations": [],
                "retrieved_chunks": [],
                "is_fallback": False,
                "llm_metadata": {"reason": "no_chunks_found"},
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
            "Synthesize a thorough, multi-document investigation report grounded strictly in the provided evidence.\n"
            "Rules:\n"
            "1. State factual findings clearly, citing supporting chunk IDs in square brackets (e.g. [incident-12:1] [release-21:2]). When analyzing an incident, always cite the specific incident report chunk (e.g., [incident-12:1]) explaining the root cause and any release chunk documenting the fix.\n"
            "2. Explain chronological timelines, root causes, and version evolutions in detail.\n"
            "3. Explicitly acknowledge superseded policies vs current/controlling policies based on the temporal analysis.\n"
            "4. Acknowledge when a conflict cannot be resolved or an announcement has not occurred.\n"
            "5. Never invent chunk IDs or unsupported factual claims."
        )

        user_content = (
            f"Retrieved Investigation Evidence:\n{evidence_block}\n"
            f"{resolution_note}\n"
            f"{revision_str}\n"
            f"Investigation Query:\n{wrap_user_prompt(query)}"
        )

        # 4. Generate draft answer with explicit fallback tracking
        is_fallback = False
        llm_metadata: Dict[str, Any] = {}

        if not self.llm.settings.llm_api_key and self.llm.mock_synthesizer is None:
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
            is_fallback = True
            llm_metadata = {
                "provider": self.llm.settings.llm_provider,
                "model": self.llm.settings.llm_model,
                "is_fallback": True,
                "fallback_reason": "No LLM API key configured (offline deterministic fallback)",
                "error": None,
            }
        else:
            llm_res = self.llm.generate_with_metadata(
                prompt=user_content,
                system_prompt=system_prompt,
                max_tokens=self.max_tokens,
            )
            draft = llm_res.content
            # Normalize Unicode hyphens (e.g. non-breaking hyphens \u2011, en-dashes \u2013) to standard ASCII hyphens
            draft = draft.replace("\u2011", "-").replace("\u2012", "-").replace("\u2013", "-").replace("\u2014", "-")
            is_fallback = llm_res.is_fallback
            llm_metadata = {
                "provider": llm_res.provider,
                "model": llm_res.model,
                "is_fallback": llm_res.is_fallback,
                "fallback_reason": llm_res.fallback_reason,
                "error": llm_res.error,
            }

        # 5. Extract Citations
        chunk_map = {c.chunk_id: c for c in retrieved}
        extracted_cids = list(dict.fromkeys(re.findall(r"\[([a-zA-Z0-9_\-]+:\d+)\]", draft)))

        citations: List[Citation] = []
        for cid in extracted_cids:
            if cid in chunk_map:
                c = chunk_map[cid]
                citations.append(
                    Citation(
                        chunk_id=c.chunk_id,
                        doc_id=c.doc_id,
                        title=c.title,
                        source_url=c.source_url,
                    )
                )
            else:
                # Fabricated or unretrieved citation detected in draft text
                doc_part = cid.split(":")[0] if ":" in cid else cid
                citations.append(
                    Citation(
                        chunk_id=cid,
                        doc_id=doc_part,
                        title="Unknown (Unverified)",
                        source_url="",
                    )
                )

        if not citations:
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
            "is_fallback": is_fallback,
            "llm_metadata": llm_metadata,
        }
