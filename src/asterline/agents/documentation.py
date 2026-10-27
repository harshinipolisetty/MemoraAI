"""Documentation Research Agent for Asterline Labs.

Answers direct factual questions regarding specifications, pricing, policies,
API limits, and product capabilities. Grounds claims strictly in retrieved corpus chunks
and formats verifiable citations.
"""

from typing import Any, Dict, List, Optional
from asterline.agents.llm import LLMClient
from asterline.graph.state import Citation, RetrievedChunk
from asterline.guardrails.input import wrap_user_prompt
from asterline.rag.retriever import HybridRetriever


class DocumentationAgent:
    """Researches authoritative company documentation for direct factual inquiries."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: Optional[LLMClient] = None,
    ):
        self.retriever = retriever
        self.llm = llm_client or LLMClient()

    def research(
        self,
        query: str,
        user_memories: Optional[List[Dict[str, Any]]] = None,
        feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute focused documentation lookup and synthesize evidence-backed response."""
        # 1. Retrieve candidate evidence chunks
        retrieved = self.retriever.hybrid_search(query, top_k=4)

        if not retrieved:
            return {
                "draft_response": "The available Asterline documentation does not contain information to answer this question.",
                "citations": [],
                "retrieved_chunks": [],
            }

        # 2. Build structured evidence block
        evidence_lines = []
        for c in retrieved:
            evidence_lines.append(
                f"[{c.chunk_id}] {c.title} (Version: {c.version}, Published: {c.published})\n{c.text}\n"
            )
        evidence_block = "\n".join(evidence_lines)

        # 3. Incorporate relevant personal workspace context if applicable (e.g., tier preference)
        context_str = ""
        if user_memories:
            context_notes = [f"- {m['content']}" for m in user_memories if m.get("is_active", True)]
            if context_notes:
                context_str = "\nUser Workspace Context (Use only for personal scoping, not product facts):\n" + "\n".join(context_notes)

        revision_str = ""
        if feedback:
            revision_str = f"\nVERIFIER FEEDBACK FROM PREVIOUS ATTEMPT (Please fix):\n{feedback}\n"

        system_prompt = (
            "You are the Documentation Research Agent for Asterline Labs.\n"
            "Answer the user query strictly and solely using the provided documentation evidence.\n"
            "Rules:\n"
            "1. Ground every factual claim in the retrieved text.\n"
            "2. Cite supporting chunk IDs in square brackets, e.g. [pricing:0].\n"
            "3. If evidence is missing or insufficient, state that explicitly.\n"
            "4. Distinguish current from historical specifications if noted.\n"
            "5. Never invent chunk IDs or factual claims."
        )

        user_content = (
            f"Retrieved Company Evidence:\n{evidence_block}\n"
            f"{context_str}\n"
            f"{revision_str}\n"
            f"User Query:\n{wrap_user_prompt(query)}"
        )

        # 4. Generate draft answer
        # If offline fallback is triggered, generate a deterministic factual summary across retrieved chunks
        if not self.llm.settings.llm_api_key or self.llm.mock_synthesizer is not None:
            draft_lines = []
            query_words = set(query.lower().split())
            for c in retrieved:
                # Check if chunk contains direct answer terms or query words
                if any(qw in c.text.lower() for qw in query_words if len(qw) > 3) or any(
                    kw in c.text.lower() for kw in ("$", "cost", "price", "limit", "sla", "timeout")
                ):
                    draft_lines.append(f"According to {c.title} [{c.chunk_id}], {c.text}")
            if not draft_lines:
                top_chunk = retrieved[0]
                draft_lines.append(f"According to {top_chunk.title} [{top_chunk.chunk_id}], {top_chunk.text}")
            draft = "\n\n".join(draft_lines)
        else:
            draft = self.llm.generate(prompt=user_content, system_prompt=system_prompt)

        # 5. Extract and validate citations
        citations: List[Citation] = []
        for c in retrieved:
            # Include citation if referenced in text or if top relevant evidence used
            if f"[{c.chunk_id}]" in draft or c.chunk_id in draft:
                citations.append(
                    Citation(
                        chunk_id=c.chunk_id,
                        doc_id=c.doc_id,
                        title=c.title,
                        source_url=c.source_url,
                    )
                )

        # If model failed to format bracketed citations, cite the retrieved chunks directly
        if not citations and retrieved:
            citations.append(
                Citation(
                    chunk_id=retrieved[0].chunk_id,
                    doc_id=retrieved[0].doc_id,
                    title=retrieved[0].title,
                    source_url=retrieved[0].source_url,
                )
            )

        return {
            "draft_response": draft,
            "citations": citations,
            "retrieved_chunks": retrieved,
        }
