"""Documentation Research Agent for Asterline Labs.

Answers direct factual questions regarding specifications, pricing, policies,
API limits, and product capabilities. Grounds claims strictly in retrieved corpus chunks
and formats verifiable citations.
"""

import re
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
        existing_chunks: Optional[List[RetrievedChunk]] = None,
    ) -> Dict[str, Any]:
        """Execute focused documentation lookup and synthesize evidence-backed response."""
        # 1. Retrieve candidate evidence chunks or reuse preserved evidence context
        if existing_chunks is not None:
            retrieved = existing_chunks
        else:
            retrieved = self.retriever.hybrid_search(query, top_k=4)

        if not retrieved:
            return {
                "draft_response": "The available Asterline documentation does not contain information to answer this question.",
                "citations": [],
                "retrieved_chunks": [],
                "is_fallback": False,
                "llm_metadata": {"reason": "no_chunks_found"},
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

        # 4. Generate draft answer with explicit fallback tracking
        is_fallback = False
        llm_metadata: Dict[str, Any] = {}

        generic_words = {
            "what", "is", "the", "for", "much", "does", "where", "can", "users", "which", "are", "on", "used",
            "with", "from", "and", "about", "asterline", "how", "many", "when", "that", "this", "tell", "show",
            "give", "explain", "describe", "details", "information", "info", "cost", "costs", "price", "pricing",
            "plan", "plans", "rate", "rates", "fee", "fees", "endpoint", "endpoints", "parameter", "parameters",
            "accepted", "accepts", "method", "methods", "http", "post", "get", "put", "delete", "available",
            "supported", "support", "policy", "documentation", "feature", "features", "third-party", "competitor",
            "competitors", "undocumented", "app", "application"
        }
        distinctive_query_words = {qw.strip("?,.:;!'\"/").lower() for qw in query.split() if len(qw) > 2} - generic_words

        has_relevant_evidence = any(
            any(qw in c.text.lower() or qw in c.title.lower() for qw in distinctive_query_words)
            for c in retrieved
        ) if distinctive_query_words else True

        if not self.llm.settings.llm_api_key and self.llm.mock_synthesizer is None:
            # Deterministic evidence synthesis for offline/test mode without API key
            if not has_relevant_evidence:
                draft = "The available Asterline documentation does not contain information to answer this question."
            else:
                draft_lines = []
                for c in retrieved:
                    if any(qw in c.text.lower() or qw in c.title.lower() for qw in distinctive_query_words):
                        draft_lines.append(f"According to {c.title} [{c.chunk_id}], {c.text}")
                if not draft_lines:
                    top_chunk = retrieved[0]
                    draft_lines.append(f"According to {top_chunk.title} [{top_chunk.chunk_id}], {top_chunk.text}")
                draft = "\n\n".join(draft_lines)

            is_fallback = True
            llm_metadata = {
                "provider": self.llm.settings.llm_provider,
                "model": self.llm.settings.llm_model,
                "is_fallback": True,
                "fallback_reason": "No LLM API key configured (offline deterministic fallback)",
                "error": None,
            }
        else:
            llm_res = self.llm.generate_with_metadata(prompt=user_content, system_prompt=system_prompt)
            draft = llm_res.content
            is_fallback = llm_res.is_fallback
            llm_metadata = {
                "provider": llm_res.provider,
                "model": llm_res.model,
                "is_fallback": llm_res.is_fallback,
                "fallback_reason": llm_res.fallback_reason,
                "error": llm_res.error,
            }

        # 5. Extract and validate citations
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

        # If model failed to format bracketed citations, check if chunk_id appears in text or cite retrieved (only if relevant evidence exists)
        if not citations and has_relevant_evidence:
            for c in retrieved:
                if c.chunk_id in draft:
                    citations.append(
                        Citation(
                            chunk_id=c.chunk_id,
                            doc_id=c.doc_id,
                            title=c.title,
                            source_url=c.source_url,
                        )
                    )
            if not citations and retrieved and "does not contain" not in draft.lower():
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
            "is_fallback": is_fallback,
            "llm_metadata": llm_metadata,
        }
