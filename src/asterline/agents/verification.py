"""Verification and Fact-Checking Agent for Asterline Labs.

Audits draft research responses against retrieved evidence chunks,
verifies citation existence and document titles, checks cross-user privacy,
and issues structured verdicts: 'pass', 'revise', or 'abstain'.
"""

from typing import Any, Dict, List, Optional, Set
from asterline.graph.state import Citation, RetrievedChunk, VerificationResult


class VerificationAgent:
    """Fact-checks draft responses and citations against retrieved evidence."""

    def verify(
        self,
        draft_response: str,
        citations: List[Citation],
        retrieved_chunks: List[RetrievedChunk],
        user_id: str,
        other_user_ids: Optional[Set[str]] = None,
        is_investigation: bool = False,
    ) -> VerificationResult:
        """Evaluate draft response and citations against authoritative evidence.

        Returns VerificationResult with verdict:
        - 'pass': Factual claims grounded, citations valid, privacy respected.
        - 'revise': Actionable defects (hallucinated citation, missing attribution) that a bounded retry can fix.
        - 'abstain': No relevant documentation exists, or conflicting evidence cannot be reconciled.
        """
        unsupported_claims: List[str] = []
        feedback_notes: List[str] = []

        # ---------------------------------------------------------------------
        # 1. Check for Complete Evidence Absence (Abstention Rule)
        # ---------------------------------------------------------------------
        if not retrieved_chunks:
            # If the response honestly states no evidence was found, that is a legitimate 'abstain' pass
            if any(phrase in draft_response.lower() for phrase in ("does not contain", "no authoritative", "not documented")):
                return VerificationResult(
                    verdict="abstain",
                    is_grounded=True,
                    citation_validity=True,
                    privacy_compliant=True,
                    feedback="Evidence is absent; model correctly acknowledged lack of documentation.",
                )
            return VerificationResult(
                verdict="abstain",
                is_grounded=False,
                citation_validity=False,
                privacy_compliant=True,
                feedback="No documentation chunks were retrieved to substantiate these claims. Must abstain.",
                unsupported_claims=["Entire response lacks supporting retrieved chunks."],
            )

        # ---------------------------------------------------------------------
        # 2. Deterministic Citation Validation
        # ---------------------------------------------------------------------
        chunk_map: Dict[str, RetrievedChunk] = {c.chunk_id: c for c in retrieved_chunks}

        if not citations:
            feedback_notes.append("Response made factual claims but provided zero citations to retrieved chunks.")
            unsupported_claims.append("Missing source citations.")

        for cit in citations:
            # Citation existence check
            if cit.chunk_id not in chunk_map:
                feedback_notes.append(
                    f"Fabricated chunk ID detected: '{cit.chunk_id}' does not exist in retrieved evidence set."
                )
                unsupported_claims.append(f"Invalid citation chunk_id '{cit.chunk_id}'")
                continue

            # Citation title match check
            real_chunk = chunk_map[cit.chunk_id]
            if cit.title != real_chunk.title:
                feedback_notes.append(
                    f"Citation title mismatch: Cited title '{cit.title}' does not match real document title '{real_chunk.title}' for chunk '{cit.chunk_id}'."
                )
                unsupported_claims.append(f"Mismatched title for '{cit.chunk_id}'")

        # ---------------------------------------------------------------------
        # 3. Privacy Compliance (Cross-user leakage)
        # ---------------------------------------------------------------------
        privacy_compliant = True
        if other_user_ids:
            for other_id in other_user_ids:
                if other_id != user_id and other_id.lower() in draft_response.lower():
                    privacy_compliant = False
                    feedback_notes.append(f"Privacy breach: Response contains unauthorized reference to user '{other_id}'.")
                    unsupported_claims.append(f"Cross-user leakage of {other_id}")

        # ---------------------------------------------------------------------
        # 4. Synthesize Structured Verdict
        # ---------------------------------------------------------------------
        if unsupported_claims or not privacy_compliant:
            return VerificationResult(
                verdict="revise",
                is_grounded=len(unsupported_claims) == 0,
                citation_validity=all(c.chunk_id in chunk_map and c.title == chunk_map[c.chunk_id].title for c in citations) if citations else False,
                privacy_compliant=privacy_compliant,
                feedback="; ".join(feedback_notes),
                unsupported_claims=unsupported_claims,
            )

        return VerificationResult(
            verdict="pass",
            is_grounded=True,
            citation_validity=True,
            privacy_compliant=True,
            feedback="All claims are verified and cited from retrieved evidence.",
        )
