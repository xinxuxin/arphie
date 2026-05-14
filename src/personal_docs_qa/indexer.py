"""Local search indexing and persistence."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from personal_docs_qa.chunker import chunk_documents
from personal_docs_qa.config import (
    get_embedding_dimensions,
    get_embedding_model,
    resolve_retrieval_mode,
)
from personal_docs_qa.loaders import load_folder
from personal_docs_qa.models import Chunk
from personal_docs_qa.openai_embeddings import OpenAIEmbeddingError, embed_texts


DEFAULT_INDEX_PATH = Path(".docqa/index.joblib")


@dataclass
class LocalIndex:
    """A small persisted search index."""

    vectorizer: TfidfVectorizer
    tfidf_matrix: csr_matrix
    chunks: list[Chunk]
    created_at: str
    document_count: int
    chunk_count: int
    chunk_embeddings: np.ndarray | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    embedding_created_at: str | None = None
    retrieval_mode_built: str = "tfidf"
    warnings: list[str] = field(default_factory=list)
    matrix: csr_matrix | None = None
    embedding_matrix: np.ndarray | None = None
    retrieval_mode: str | None = None

    def __post_init__(self) -> None:
        if self.matrix is None:
            self.matrix = self.tfidf_matrix
        if self.embedding_matrix is None:
            self.embedding_matrix = self.chunk_embeddings
        if self.retrieval_mode is None:
            self.retrieval_mode = self.retrieval_mode_built


@dataclass
class IndexingResult:
    """Result of loading, chunking, indexing, and saving documents."""

    index: LocalIndex
    warnings: list[str]


def build_index(chunks: list[Chunk], retrieval_mode: str | None = None) -> LocalIndex:
    """Build a local searchable index from chunks."""
    usable_chunks = [chunk for chunk in chunks if chunk.text.strip()]
    if not usable_chunks:
        raise ValueError("Cannot build index: no non-empty chunks were provided.")

    requested_mode = (retrieval_mode or "auto").strip().lower()
    resolved_mode, warnings = resolve_retrieval_mode(retrieval_mode)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunk.text for chunk in usable_chunks)
    document_ids = {chunk.document_id for chunk in usable_chunks}
    chunk_embeddings = None
    embedding_model = None
    embedding_dimensions = None
    embedding_created_at = None
    retrieval_mode_built = resolved_mode

    if resolved_mode in {"embedding", "hybrid"}:
        embedding_model = get_embedding_model()
        embedding_dimensions = get_embedding_dimensions()
        try:
            embeddings = embed_texts(
                [chunk.text for chunk in usable_chunks],
                model=embedding_model,
                dimensions=embedding_dimensions,
            )
        except OpenAIEmbeddingError as exc:
            if requested_mode == "auto":
                warnings.append(f"Embedding generation failed in auto mode; using tfidf only. {exc}")
                embedding_model = None
                embedding_dimensions = None
                retrieval_mode_built = "tfidf"
            else:
                raise ValueError(str(exc)) from exc
        else:
            chunk_embeddings = np.array(embeddings, dtype=float)
            embedding_created_at = datetime.now(timezone.utc).isoformat()

    if resolved_mode in {"embedding", "hybrid"} and chunk_embeddings is None and requested_mode != "auto":
        raise ValueError("Embedding generation failed; no embeddings were stored.")

    return LocalIndex(
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        chunks=usable_chunks,
        created_at=datetime.now(timezone.utc).isoformat(),
        document_count=len(document_ids),
        chunk_count=len(usable_chunks),
        chunk_embeddings=chunk_embeddings,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        embedding_created_at=embedding_created_at,
        retrieval_mode_built=retrieval_mode_built,
        warnings=warnings,
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
    return normalize_index(joblib.load(index_path))


def normalize_index(index: LocalIndex) -> LocalIndex:
    """Populate compatibility fields for indexes saved by older versions."""
    if not hasattr(index, "tfidf_matrix"):
        index.tfidf_matrix = index.matrix
    if not hasattr(index, "matrix") or index.matrix is None:
        index.matrix = index.tfidf_matrix
    if not hasattr(index, "chunk_embeddings"):
        index.chunk_embeddings = getattr(index, "embedding_matrix", None)
    if not hasattr(index, "embedding_matrix") or index.embedding_matrix is None:
        index.embedding_matrix = index.chunk_embeddings
    if not hasattr(index, "embedding_created_at"):
        index.embedding_created_at = None
    if not hasattr(index, "retrieval_mode_built"):
        index.retrieval_mode_built = getattr(index, "retrieval_mode", "tfidf")
    if not hasattr(index, "retrieval_mode") or index.retrieval_mode is None:
        index.retrieval_mode = index.retrieval_mode_built
    if not hasattr(index, "warnings"):
        index.warnings = []
    return index


def index_folder(
    folder_path: str | Path,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    retrieval_mode: str | None = None,
) -> LocalIndex:
    """Load, chunk, index, and persist a folder of supported documents."""
    return index_folder_with_warnings(
        folder_path,
        index_path=index_path,
        retrieval_mode=retrieval_mode,
    ).index


def index_folder_with_warnings(
    folder_path: str | Path,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    retrieval_mode: str | None = None,
) -> IndexingResult:
    """Load, chunk, index, persist, and return non-fatal loading warnings."""
    documents, warnings = load_folder(folder_path)
    chunks = chunk_documents(documents)
    if not chunks:
        details = f" Warnings: {'; '.join(warnings)}" if warnings else ""
        raise ValueError(f"Cannot build index: no chunks were produced.{details}")

    index = build_index(chunks, retrieval_mode=retrieval_mode)
    save_index(index, index_path)
    return IndexingResult(index=index, warnings=[*warnings, *index.warnings])
