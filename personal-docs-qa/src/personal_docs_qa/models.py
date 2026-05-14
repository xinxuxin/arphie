"""Shared data models for the document QA pipeline."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Document:
    """A loaded source document with plain text content."""

    id: str
    path: str
    file_name: str
    file_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A searchable chunk with source metadata."""

    id: str
    document_id: str
    file_name: str
    path: str
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and its relevance score."""

    rank: int
    score: float
    chunk: Chunk


@dataclass(frozen=True)
class Answer:
    """A concise answer with cited sources."""

    question: str
    answer: str
    sources: list[SearchResult]
    warnings: list[str] = field(default_factory=list)
    confidence: str = "unknown"


def make_document_id(path: str | Path) -> str:
    """Create a stable, inspectable document id from a path."""
    normalized = str(Path(path).expanduser().resolve(strict=False))
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"doc-{digest}"


def make_chunk_id(document_id: str, index: int) -> str:
    """Create a stable chunk id within a document."""
    return f"{document_id}-chunk-{index:04d}"
