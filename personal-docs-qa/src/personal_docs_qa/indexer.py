"""Local TF-IDF indexing and persistence."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from personal_docs_qa.chunker import chunk_documents
from personal_docs_qa.loaders import load_folder
from personal_docs_qa.models import Chunk


DEFAULT_INDEX_PATH = Path(".docqa/index.joblib")


@dataclass
class LocalIndex:
    """A small persisted TF-IDF index."""

    vectorizer: TfidfVectorizer
    matrix: csr_matrix
    chunks: list[Chunk]
    created_at: str
    document_count: int
    chunk_count: int


def build_index(chunks: list[Chunk]) -> LocalIndex:
    """Build a local searchable TF-IDF index from chunks."""
    usable_chunks = [chunk for chunk in chunks if chunk.text.strip()]
    if not usable_chunks:
        raise ValueError("Cannot build index: no non-empty chunks were provided.")

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunk.text for chunk in usable_chunks)
    document_ids = {chunk.document_id for chunk in usable_chunks}

    return LocalIndex(
        vectorizer=vectorizer,
        matrix=matrix,
        chunks=usable_chunks,
        created_at=datetime.now(timezone.utc).isoformat(),
        document_count=len(document_ids),
        chunk_count=len(usable_chunks),
    )


def save_index(index: LocalIndex, path: str | Path = DEFAULT_INDEX_PATH) -> None:
    """Persist an index to disk."""
    index_path = Path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, index_path)


def load_index(path: str | Path = DEFAULT_INDEX_PATH) -> LocalIndex:
    """Load a persisted index from disk."""
    index_path = Path(path)
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    return joblib.load(index_path)


def index_folder(
    folder_path: str | Path,
    index_path: str | Path = DEFAULT_INDEX_PATH,
) -> LocalIndex:
    """Load, chunk, index, and persist a folder of supported documents."""
    documents, warnings = load_folder(folder_path)
    chunks = chunk_documents(documents)
    if not chunks:
        details = f" Warnings: {'; '.join(warnings)}" if warnings else ""
        raise ValueError(f"Cannot build index: no chunks were produced.{details}")

    index = build_index(chunks)
    save_index(index, index_path)
    return index
