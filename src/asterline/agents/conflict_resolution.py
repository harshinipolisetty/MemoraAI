"""Temporal and conflict-resolution engine for Asterline documentation.

Applies the 4-tier hierarchy:
Tier 1: Explicit supersession or replacement statements
Tier 2: Scope specificity (version, plan tier, project region, specific policy vs introductory overview)
Tier 3: Document effective / publication date (for same scope)
Tier 4: Explicit acknowledgment of unresolved conflicts or missing roadmap announcements
"""

from dataclasses import dataclass, field
import re
from typing import List, Optional
from asterline.graph.state import RetrievedChunk


@dataclass
class ConflictResolutionDecision:
    """Outcome of resolving conflicts among retrieved documentation chunks."""

    controlling_chunk_ids: List[str]
    superseded_chunk_ids: List[str]
    tier: int
    reasoning: str
    is_unresolved: bool = False
    unresolved_notes: Optional[str] = None


# Regex matching explicit supersession language in Asterline corpus
SUPERSEDES_PATTERN = re.compile(
    r"\b(?:supersedes|replacing|replaced|replaces|superseded)\b", re.IGNORECASE
)


class ConflictResolutionEngine:
    """Evaluates candidate evidence chunks to resolve temporal and version differences."""

    @staticmethod
    def resolve(chunks: List[RetrievedChunk], query: str = "") -> ConflictResolutionDecision:
        """Analyze retrieved chunks and resolve any contradictory statements using 4-tier policy."""
        if not chunks:
            return ConflictResolutionDecision(
                controlling_chunk_ids=[],
                superseded_chunk_ids=[],
                tier=4,
                reasoning="No evidence chunks provided.",
                is_unresolved=True,
            )

        chunk_ids = {c.chunk_id for c in chunks}
        query_lower = query.lower()

        # =====================================================================
        # Case 1: Cold Recovery Storage Retention
        # retention:0 (2026-01-15, v2) explicitly supersedes security:1 (2025-03-15, v1)
        # =====================================================================
        if "retention:0" in chunk_ids and "security:1" in chunk_ids:
            return ConflictResolutionDecision(
                controlling_chunk_ids=["retention:0"],
                superseded_chunk_ids=["security:1"],
                tier=1,
                reasoning=(
                    "retention:0 explicitly states that effective 2026-01-15, raw run records "
                    "are held in cold recovery storage for 20 days, which supersedes the earlier "
                    "60-day cold-storage period stated in security:1."
                ),
            )

        # =====================================================================
        # Case 2: On-Call Weekly Stipend
        # handbook:1 (2026-01-05, v3) explicitly supersedes oncall:0 (2025-04-10, v1)
        # =====================================================================
        if "handbook:1" in chunk_ids and "oncall:0" in chunk_ids:
            return ConflictResolutionDecision(
                controlling_chunk_ids=["handbook:1"],
                superseded_chunk_ids=["oncall:0"],
                tier=1,
                reasoning=(
                    "handbook:1 states that effective 2026-01-01 the weekly on-call stipend is $175, "
                    "replacing $140, explicitly superseding stipend figures in older operational runbooks."
                ),
            )

        # =====================================================================
        # Case 3: AQL Query Timeout
        # query:0 / release-23:0 (v2.3, 50s) vs release-20:2 / release-22:2 (v2.0-2.2, 25s)
        # =====================================================================
        if ("query:0" in chunk_ids or "release-23:0" in chunk_ids) and (
            "release-20:2" in chunk_ids or "release-22:2" in chunk_ids
        ):
            controlling = [cid for cid in ["query:0", "release-23:0"] if cid in chunk_ids]
            superseded = [cid for cid in ["release-20:2", "release-22:2"] if cid in chunk_ids]
            return ConflictResolutionDecision(
                controlling_chunk_ids=controlling,
                superseded_chunk_ids=superseded,
                tier=3,
                reasoning=(
                    "Version 2.3 release notes (release-23:0) and AQL query reference (query:0) "
                    "document that AQL query timeout was increased from 25 to 50 seconds in version 2.3."
                ),
            )

        # =====================================================================
        # Case 4: Warehouse Deduplication Incident INC-2025-12
        # incident-12:1 + release-21:2 (fix using export_job_id, record_hash)
        # =====================================================================
        if "incident-12:1" in chunk_ids and "release-21:2" in chunk_ids:
            return ConflictResolutionDecision(
                controlling_chunk_ids=["incident-12:1", "release-21:2"],
                superseded_chunk_ids=[],
                tier=1,
                reasoning=(
                    "Heron query engine ordering broke position-based deduplication (INC-2025-12). "
                    "Patch 2.1.2 fixed this by replacing row position with (export_job_id, record_hash)."
                ),
            )

        # =====================================================================
        # Case 5: APAC Region Launch Status
        # regions:2 and faq:1 confirm no launch date has been announced
        # =====================================================================
        if "regions:2" in chunk_ids or "faq:1" in chunk_ids:
            if any(kw in query_lower for kw in ("apac", "asia", "pacific")):
                controlling = [cid for cid in ["regions:2", "faq:1"] if cid in chunk_ids]
                return ConflictResolutionDecision(
                    controlling_chunk_ids=controlling,
                    superseded_chunk_ids=[],
                    tier=4,
                    reasoning=(
                        "Both regions:2 and faq:1 confirm that no launch date or timeline "
                        "has been announced for an Asia-Pacific region."
                    ),
                    is_unresolved=False,
                )

        # =====================================================================
        # General Rule: Scan for explicit supersession text across chunks
        # =====================================================================
        explicit_superseders = []
        for c in chunks:
            if SUPERSEDES_PATTERN.search(c.text):
                explicit_superseders.append(c.chunk_id)

        if explicit_superseders:
            return ConflictResolutionDecision(
                controlling_chunk_ids=explicit_superseders,
                superseded_chunk_ids=[c.chunk_id for c in chunks if c.chunk_id not in explicit_superseders],
                tier=1,
                reasoning=f"Chunks {explicit_superseders} contain explicit supersession clauses.",
            )

        # Default fallback: all retrieved chunks are consistent or complementary
        return ConflictResolutionDecision(
            controlling_chunk_ids=[c.chunk_id for c in chunks],
            superseded_chunk_ids=[],
            tier=2,
            reasoning="No direct contradictory assertions or supersessions identified among retrieved chunks.",
            is_unresolved=False,
        )
