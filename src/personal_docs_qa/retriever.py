"""Retrieval utilities."""

from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from personal_docs_qa.config import resolve_retrieval_mode
from personal_docs_qa.indexer import LocalIndex, load_index
from personal_docs_qa.models import SearchResult
from personal_docs_qa.openai_embeddings import OpenAIEmbeddingError, embed_query


def _ranked_results(index: LocalIndex, scores: np.ndarray, top_k: int) -> list[SearchResult]:
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    results: list[SearchResult] = []
    for rank, chunk_index in enumerate(ranked_indices[:top_k], start=1):
        results.append(
            SearchResult(
                rank=rank,
                score=float(scores[chunk_index]),
                chunk=index.chunks[chunk_index],
            )
        )
    return results


def _tfidf_scores(index: LocalIndex, query: str) -> np.ndarray:
    query_vector = index.vectorizer.transform([query])
    return cosine_similarity(query_vector, index.matrix).ravel()


def _embedding_scores(index: LocalIndex, query: str) -> np.ndarray:
    if index.embedding_matrix is None:
        raise ValueError("This index does not contain embeddings. Rebuild it with embedding or hybrid mode.")
    try:
        query_embedding = embed_query(
            query,
            model=index.embedding_model,
            dimensions=index.embedding_dimensions,
        )
    except OpenAIEmbeddingError as exc:
        raise ValueError(str(exc)) from exc
    return cosine_similarity(np.array([query_embedding], dtype=float), index.embedding_matrix).ravel()


def _normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    minimum = float(scores.min())
    maximum = float(scores.max())
    if maximum == minimum:
        return np.zeros_like(scores, dtype=float)
    return (scores - minimum) / (maximum - minimum)


def search(
    index: LocalIndex,
    query: str,
    top_k: int = 5,
    retrieval_mode: str | None = None,
) -> list[SearchResult]:
    """Search a local index and return ranked chunks."""
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Search query cannot be empty.")
    if top_k <= 0:
        return []
    if not index.chunks:
        return []

    requested_mode = retrieval_mode or getattr(index, "retrieval_mode", "tfidf")
    mode, _warnings = resolve_retrieval_mode(requested_mode)
    if mode == "embedding":
        scores = _embedding_scores(index, clean_query)
    elif mode == "hybrid":
        tfidf_scores = _normalize(_tfidf_scores(index, clean_query))
        embedding_scores = _normalize(_embedding_scores(index, clean_query))
        scores = (tfidf_scores + embedding_scores) / 2
    else:
        scores = _tfidf_scores(index, clean_query)

    return _ranked_results(index, scores, top_k)


def retrieve(
    question: str,
    index_path: str | Path,
    top_k: int = 5,
    retrieval_mode: str | None = None,
) -> list[SearchResult]:
    """Load an index from disk and retrieve relevant chunks for a question."""
    return search(load_index(index_path), question, top_k=top_k, retrieval_mode=retrieval_mode)
