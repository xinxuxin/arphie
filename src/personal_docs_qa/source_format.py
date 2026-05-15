"""Small helpers for presenting retrieved sources."""

from __future__ import annotations

import html
from numbers import Real
import re
from pathlib import Path
from typing import Any

from personal_docs_qa.models import Chunk, SearchResult


WORD_RE = re.compile(r"[a-zA-Z0-9']+")
SUPPORTED_FILE_LABELS = {"txt": "TXT", "md": "MD", "pdf": "PDF"}
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "did",
    "do",
    "does",
    "for",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
}


def format_score(score: float | None) -> str:
    """Format optional scores consistently for humans."""
    return f"{score:.3f}" if isinstance(score, Real) else "-"


def compact_text(text: str) -> str:
    """Collapse whitespace in source text."""
    return " ".join((text or "").split())


def make_excerpt(text: str, limit: int = 220, query: str = "") -> str:
    """Create a short excerpt, centered near the first query term when possible."""
    compact = compact_text(text)
    if len(compact) <= limit:
        return compact

    match_start = None
    compact_lower = compact.lower()
    for term in query_terms(query):
        index = compact_lower.find(term.lower())
        if index != -1 and (match_start is None or index < match_start):
            match_start = index

    if match_start is None:
        return compact[: max(0, limit - 3)].rstrip() + "..."

    half_window = max(0, limit // 2)
    start = max(0, match_start - half_window)
    end = min(len(compact), start + limit)
    if end == len(compact):
        start = max(0, end - limit)
    excerpt = compact[start:end].strip()
    if start > 0:
        excerpt = "..." + excerpt.lstrip()
    if end < len(compact):
        excerpt = excerpt.rstrip() + "..."
    return excerpt


def query_terms(query: str) -> list[str]:
    """Extract unique query terms in order, skipping tiny terms."""
    seen: set[str] = set()
    terms: list[str] = []
    for term in WORD_RE.findall(query or ""):
        normalized = term.lower()
        if len(normalized) < 2 or normalized in STOP_WORDS or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(term)
    return terms


def highlight_query_terms_html(text: str, query: str) -> str:
    """Return escaped HTML with query terms wrapped in mark tags."""
    text = text or ""
    terms = query_terms(query)
    if not terms:
        return html.escape(text)

    pattern = re.compile(r"(" + "|".join(re.escape(term) for term in terms) + r")", re.IGNORECASE)
    parts: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        parts.append(html.escape(text[cursor : match.start()]))
        parts.append(f"<mark>{html.escape(match.group(0))}</mark>")
        cursor = match.end()
    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def file_type_label(chunk: Chunk | None) -> str:
    """Return a compact file type label for a chunk."""
    if chunk is None:
        return "FILE"
    extension = str(chunk.metadata.get("extension") or Path(chunk.file_name).suffix).lower().lstrip(".")
    if not extension:
        extension = Path(chunk.path).suffix.lower().lstrip(".")
    return SUPPORTED_FILE_LABELS.get(extension, extension.upper() if extension else "FILE")


def page_number(chunk: Chunk | None) -> int | None:
    """Return the source page number when available."""
    if chunk is None:
        return None
    value = chunk.metadata.get("page_number")
    return value if isinstance(value, int) else None


def source_payload(result: SearchResult, query: str = "", excerpt_limit: int = 220) -> dict[str, Any]:
    """Build a robust serializable source payload for CLI or web display."""
    chunk = result.chunk
    excerpt = make_excerpt(chunk.text, limit=excerpt_limit, query=query)
    should_highlight = result.retrieval_mode_used in {"tfidf", "hybrid"}
    highlighted_excerpt = highlight_query_terms_html(excerpt, query) if should_highlight else html.escape(excerpt)
    page = page_number(chunk)
    return {
        "rank": result.rank,
        "score": result.score,
        "score_display": format_score(result.score),
        "score_tfidf": result.score_tfidf,
        "score_tfidf_display": format_score(result.score_tfidf),
        "score_embedding": result.score_embedding,
        "score_embedding_display": format_score(result.score_embedding),
        "retrieval_mode_used": result.retrieval_mode_used,
        "file_name": chunk.file_name,
        "file_type": file_type_label(chunk),
        "page_number": page,
        "page_label": f"p. {page}" if page is not None else None,
        "chunk_id": chunk.id,
        "excerpt": excerpt,
        "excerpt_html": highlighted_excerpt,
    }
