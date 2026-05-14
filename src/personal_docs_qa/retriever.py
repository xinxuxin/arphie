"""Retrieval utilities."""

from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from personal_docs_qa.config import is_openai_available, resolve_retrieval_mode
from personal_docs_qa.indexer import LocalIndex, load_index
from personal_docs_qa.models import SearchResult
from personal_docs_qa.openai_embeddings import OpenAIEmbeddingError, embed_query


MISSING_EMBEDDINGS_MESSAGE = (
    "This index was built without embeddings. Re-run ingest with --retrieval-mode hybrid or embedding."
)


def _ranked_results(
    index: LocalIndex,
    scores: np.ndarray,
    top_k: int,
    retrieval_mode_used: str,
    tfidf_scores: np.ndarray | None = None,
    embedding_scores: np.ndarray | None = None,
) -> list[SearchResult]:
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    results: list[SearchResult] = []
    for rank, chunk_index in enumerate(ranked_indices[:top_k], start=1):
        results.append(
            SearchResult(
                rank=rank,
                score=float(scores[chunk_index]),
                chunk=index.chunks[chunk_index],
                score_tfidf=float(tfidf_scores[chunk_index]) if tfidf_scores is not None else None,
                score_embedding=float(embedding_scores[chunk_index]) if embedding_scores is not None else None,
                retrieval_mode_used=retrieval_mode_used,
            )
        )
    return results


def _get_tfidf_matrix(index: LocalIndex):
    tfidf_matrix = getattr(index, "tfidf_matrix", None)
    if tfidf_matrix is not None:
        return tfidf_matrix
    return getattr(index, "matrix")


def _get_chunk_embeddings(index: LocalIndex):
    chunk_embeddings = getattr(index, "chunk_embeddings", None)
    if chunk_embeddings is not None:
        return chunk_embeddings
    return getattr(index, "embedding_matrix", None)


def _tfidf_scores(index: LocalIndex, query: str) -> np.ndarray:
    query_vector = index.vectorizer.transform([query])
    return cosine_similarity(query_vector, _get_tfidf_matrix(index)).ravel()


def _embedding_scores(index: LocalIndex, query: str) -> np.ndarray:
    chunk_embeddings = _get_chunk_embeddings(index)
    if chunk_embeddings is None:
        raise ValueError(MISSING_EMBEDDINGS_MESSAGE)
    try:
        query_embedding = embed_query(
            query,
            model=index.embedding_model,
            dimensions=index.embedding_dimensions,
        )
    except OpenAIEmbeddingError as exc:
        raise ValueError(str(exc)) from exc
    return cosine_similarity(np.array([query_embedding], dtype=float), chunk_embeddings).ravel()


def _normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    minimum = float(scores.min())
    maximum = float(scores.max())
    if maximum == minimum:
        return np.zeros_like(scores, dtype=float)
    return (scores - minimum) / (maximum - minimum)


def _clean_query(query: str) -> str:
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Search query cannot be empty.")
    return clean_query


def search_tfidf(index: LocalIndex, query: str, top_k: int = 5) -> list[SearchResult]:
    """Search using local TF-IDF only."""
    clean_query = _clean_query(query)
    if top_k <= 0:
        return []
    if not index.chunks:
        return []
    scores = _tfidf_scores(index, clean_query)
    return _ranked_results(index, scores, top_k, retrieval_mode_used="tfidf", tfidf_scores=scores)


def search_embedding(index: LocalIndex, query: str, top_k: int = 5) -> list[SearchResult]:
    """Search using stored chunk embeddings and an OpenAI query embedding."""
    clean_query = _clean_query(query)
    if top_k <= 0:
        return []
    if not index.chunks:
        return []
    scores = _embedding_scores(index, clean_query)
    return _ranked_results(index, scores, top_k, retrieval_mode_used="embedding", embedding_scores=scores)


def search_hybrid(
    index: LocalIndex,
    query: str,
    top_k: int = 5,
    tfidf_weight: float = 0.45,
    embedding_weight: float = 0.55,
) -> list[SearchResult]:
    """Search by combining normalized TF-IDF and embedding scores."""
    clean_query = _clean_query(query)
    if top_k <= 0:
        return []
    if not index.chunks:
        return []
    raw_tfidf_scores = _tfidf_scores(index, clean_query)
    raw_embedding_scores = _embedding_scores(index, clean_query)
    normalized_tfidf = _normalize(raw_tfidf_scores)
    normalized_embedding = _normalize(raw_embedding_scores)
    scores = (tfidf_weight * normalized_tfidf) + (embedding_weight * normalized_embedding)
    return _ranked_results(
        index,
        scores,
        top_k,
        retrieval_mode_used="hybrid",
        tfidf_scores=raw_tfidf_scores,
        embedding_scores=raw_embedding_scores,
    )


def _auto_mode(index: LocalIndex) -> str:
    if _get_chunk_embeddings(index) is not None and is_openai_available():
        return "hybrid"
    return "tfidf"


def search(
    index: LocalIndex,
    query: str,
    top_k: int = 5,
    retrieval_mode: str = "auto",
) -> list[SearchResult]:
    """Search a local index and return ranked chunks."""
    requested_mode = retrieval_mode or "auto"
    mode, _warnings = resolve_retrieval_mode(requested_mode)

    if requested_mode == "auto":
        mode = _auto_mode(index)

    if mode == "embedding":
        return search_embedding(index, query, top_k=top_k)
    if mode == "hybrid":
        try:
            return search_hybrid(index, query, top_k=top_k)
        except ValueError:
            if requested_mode == "auto":
                return search_tfidf(index, query, top_k=top_k)
            raise
    return search_tfidf(index, query, top_k=top_k)


def retrieve(
    question: str,
    index_path: str | Path,
    top_k: int = 5,
    retrieval_mode: str | None = None,
) -> list[SearchResult]:
    """Load an index from disk and retrieve relevant chunks for a question."""
    return search(load_index(index_path), question, top_k=top_k, retrieval_mode=retrieval_mode)
