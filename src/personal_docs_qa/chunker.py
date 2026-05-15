"""Document chunking utilities."""

import re

from personal_docs_qa.models import Chunk, Document, make_chunk_id


SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
DEFAULT_CHUNK_SIZE = 900
DEFAULT_OVERLAP = 150
PDF_CHUNK_SIZE = 500
PDF_OVERLAP = 80


def _split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_END_RE.split(text) if part.strip()]


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start += step

    return [chunk for chunk in chunks if chunk]


def _text_units(text: str, chunk_size: int, overlap: int) -> list[str]:
    units: list[str] = []
    for paragraph in _split_paragraphs(text):
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue

        for sentence in _split_sentences(paragraph):
            if len(sentence) <= chunk_size:
                units.append(sentence)
            else:
                units.extend(_split_long_text(sentence, chunk_size, overlap))

    return units


def _find_span(text: str, chunk_text: str, start_at: int) -> tuple[int, int]:
    start = text.find(chunk_text, start_at)
    if start == -1:
        start = text.find(chunk_text)
    if start == -1:
        start = start_at
    return start, start + len(chunk_text)


def chunk_document(document: Document, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[Chunk]:
    """Split one document into paragraph-aware chunks."""
    text = document.text.strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    overlap = min(overlap, chunk_size - 1)

    units = _text_units(text, chunk_size, overlap)
    if not units:
        return []

    raw_chunks: list[str] = []
    current = ""
    for unit in units:
        separator = "\n\n" if current else ""
        candidate = f"{current}{separator}{unit}"
        if len(candidate) <= chunk_size or not current:
            current = candidate
            continue

        raw_chunks.append(current)
        available_overlap = max(0, chunk_size - len(unit) - 2)
        prefix_size = min(overlap, available_overlap)
        prefix = current[-prefix_size:].strip() if prefix_size else ""
        current = f"{prefix}\n\n{unit}".strip() if prefix else unit

    if current:
        raw_chunks.append(current)

    chunks: list[Chunk] = []
    next_search_start = 0
    for index, chunk_text in enumerate(raw_chunks):
        start_char, end_char = _find_span(text, chunk_text, next_search_start)
        next_search_start = max(start_char + 1, end_char - overlap)
        chunks.append(
            Chunk(
                id=make_chunk_id(document.id, index),
                document_id=document.id,
                file_name=document.file_name,
                path=document.path,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                metadata=dict(document.metadata),
            )
        )

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split multiple documents into searchable chunks."""
    chunks: list[Chunk] = []
    for document in documents:
        document_chunk_size = chunk_size
        document_overlap = overlap
        if document.file_type == "pdf" and chunk_size == DEFAULT_CHUNK_SIZE and overlap == DEFAULT_OVERLAP:
            document_chunk_size = PDF_CHUNK_SIZE
            document_overlap = PDF_OVERLAP
        chunks.extend(chunk_document(document, chunk_size=document_chunk_size, overlap=document_overlap))
    return chunks
