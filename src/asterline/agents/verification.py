"""Verification and Fact-Checking Agent for Asterline Labs.

Audits draft research responses against retrieved evidence chunks,
verifies citation existence and document titles, checks cross-user privacy,
and issues structured verdicts: 'pass', 'revise', or 'abstain'.
"""

import re
from typing import Any, Dict, List, Optional, Set
from asterline.graph.state import Citation, RetrievedChunk, VerificationResult

ABSTENTION_PHRASES = (
    "does not contain",
    "cannot answer",
    "can't answer",
    "no information",
    "not documented",
    "not mentioned",
    "no authoritative",
    "unable to answer",
    "cannot find any",
    "no records",
    "no details are mentioned",
    "not available in the documentation",
)

# Patterns that suggest affirmative technical facts are being asserted
AFFIRMATIVE_FACT_PATTERNS = [
    re.compile(r"\bcosts?\s+\$\d+", re.IGNORECASE),
    re.compile(r"\b\d+\s+(?:seats|records|days|hours|minutes|seconds)\b", re.IGNORECASE),
    re.compile(r"\b99\.\d+%", re.IGNORECASE),
    re.compile(r"\b(?:shipped on|effective)\s+\d{4}", re.IGNORECASE),
    re.compile(r"\bpatch\s+\d+\.\d+", re.IGNORECASE),
    re.compile(r"\binc-\d{4}-\d{2}\b", re.IGNORECASE),
]


class VerificationAgent:
    """Fact-checks draft responses and citations against retrieved evidence."""

    def _is_genuine_abstention(
        self,
        draft_response: str,
        citations: List[Citation],
        chunk_map: Dict[str, RetrievedChunk],
    ) -> tuple[bool, Optional[str]]:
        """Verify whether draft response is a genuine, evidence-backed abstention.

        Returns (is_genuine, failure_reason).
        """
        lower_resp = draft_response.lower()
        has_abstention_phrase = any(phrase in lower_resp for phrase in ABSTENTION_PHRASES)
        if not has_abstention_phrase:
            return False, "Response does not contain recognized abstention phrasing."

        # If citations were provided with an abstention, every cited chunk MUST be valid
        if citations:
            for cit in citations:
                if cit.chunk_id not in chunk_map:
                    return False, f"Fabricated chunk ID '{cit.chunk_id}' in abstention response."
                if cit.title != chunk_map[cit.chunk_id].title:
                    return False, f"Mismatched title for cited chunk '{cit.chunk_id}'."

        # Must not smuggle uncited affirmative facts
        if not citations:
            for pattern in AFFIRMATIVE_FACT_PATTERNS:
                if pattern.search(draft_response):
                    return False, "Response contains affirmative factual claims without supporting citations."

        return True, None

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
        # 1. Privacy Compliance Check (Cross-user leakage)
        # ---------------------------------------------------------------------
        privacy_compliant = True
        if other_user_ids:
            for other_id in other_user_ids:
                if other_id != user_id and other_id.lower() in draft_response.lower():
                    privacy_compliant = False
                    feedback_notes.append(f"Privacy breach: Response contains unauthorized reference to user '{other_id}'.")
                    unsupported_claims.append(f"Cross-user leakage of {other_id}")

        if not privacy_compliant:
            return VerificationResult(
                verdict="revise",
                is_grounded=False,
                citation_validity=False,
                privacy_compliant=False,
                feedback="; ".join(feedback_notes),
                unsupported_claims=unsupported_claims,
            )

        # ---------------------------------------------------------------------
        # 2. Check for Complete Evidence Absence (Empty retrieved chunks)
        # ---------------------------------------------------------------------
        if not retrieved_chunks:
            if any(phrase in draft_response.lower() for phrase in ABSTENTION_PHRASES):
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

        chunk_map: Dict[str, RetrievedChunk] = {c.chunk_id: c for c in retrieved_chunks}

        # ---------------------------------------------------------------------
        # 3. Check for Genuine Evidence-Based Abstention
        # ---------------------------------------------------------------------
        is_abstention, abstention_issue = self._is_genuine_abstention(draft_response, citations, chunk_map)
        if is_abstention:
            return VerificationResult(
                verdict="abstain",
                is_grounded=True,
                citation_validity=True,
                privacy_compliant=True,
                feedback="Genuine evidence-based abstention: model correctly declined to answer without unsupported claims or fabricated citations.",
            )
        elif abstention_issue and any(p in draft_response.lower() for p in ABSTENTION_PHRASES):
            # Model attempted to abstain but smuggled ungrounded claims or used invalid citations
            feedback_notes.append(f"Invalid abstention: {abstention_issue}")
            unsupported_claims.append(abstention_issue)

        # ---------------------------------------------------------------------
        # 4. Deterministic Citation Validation for Factual Claims
        # ---------------------------------------------------------------------
        if not citations and not is_abstention:
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
        # 5. Synthesize Structured Verdict
        # ---------------------------------------------------------------------
        if unsupported_claims:
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
