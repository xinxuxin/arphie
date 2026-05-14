"""Shared data models for the document QA pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceDocument:
    """A loaded source document with plain text content."""

    path: Path
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable chunk with source metadata."""

    id: str
    source_path: Path
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk and its relevance score."""

    chunk: DocumentChunk
    score: float

