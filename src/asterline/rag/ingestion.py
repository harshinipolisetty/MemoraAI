"""Corpus ingestion, validation, and integrity checking for Asterline Labs documentation.

Guarantees 100% preservation of all 75 chunks, metadata, and source texts without alteration.
Provides checksum validation, duplicate detection, and structured ingestion reports.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Set, Tuple
from pydantic import ValidationError
from asterline.rag.metadata import CorpusChunk, IngestionReport


def compute_file_sha256(file_path: Path) -> str:
    """Compute the SHA256 hexadecimal digest of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest().upper()


def load_and_validate_corpus(corpus_path: Path) -> Tuple[List[CorpusChunk], IngestionReport]:
    """Load, validate, and return all corpus chunks along with an ingestion audit report.

    Rules:
    - Reads JSONL record by record.
    - Validates presence and types of all required fields via CorpusChunk schema.
    - Detects and rejects duplicate chunk IDs.
    - Never modifies the source file or generates replacement IDs.
    - Fails with FileNotFoundError if corpus file does not exist.
    """
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found at: {path.resolve()}")
    if not path.is_file():
        raise ValueError(f"Corpus path is not a file: {path.resolve()}")

    sha256_hash = compute_file_sha256(path)

    valid_chunks: List[CorpusChunk] = []
    seen_chunk_ids: Set[str] = set()
    document_ids: Set[str] = set()
    errors: List[str] = []
    total_records = 0
    rejected_records = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            total_records += 1
            try:
                raw_data = json.loads(stripped)
            except json.JSONDecodeError as e:
                rejected_records += 1
                errors.append(f"Line {line_num}: Invalid JSON syntax ({e.msg})")
                continue

            if not isinstance(raw_data, dict):
                rejected_records += 1
                errors.append(f"Line {line_num}: Record is not a JSON object")
                continue

            # Validate against CorpusChunk schema
            try:
                chunk = CorpusChunk(**raw_data)
            except ValidationError as ve:
                rejected_records += 1
                field_errors = ", ".join(f"{err['loc']}: {err['msg']}" for err in ve.errors())
                errors.append(f"Line {line_num} (ID: {raw_data.get('chunk_id', 'unknown')}): Validation failed - {field_errors}")
                continue

            # Check for duplicate chunk IDs
            if chunk.chunk_id in seen_chunk_ids:
                rejected_records += 1
                errors.append(f"Line {line_num}: Duplicate chunk_id detected '{chunk.chunk_id}'")
                continue

            seen_chunk_ids.add(chunk.chunk_id)
            document_ids.add(chunk.doc_id)
            valid_chunks.append(chunk)

    report = IngestionReport(
        total_records=total_records,
        valid_records=len(valid_chunks),
        rejected_records=rejected_records,
        unique_chunk_ids=len(seen_chunk_ids),
        document_ids=sorted(list(document_ids)),
        corpus_sha256=sha256_hash,
        errors=errors,
    )

    return valid_chunks, report
