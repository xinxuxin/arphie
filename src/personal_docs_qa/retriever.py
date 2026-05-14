"""TF-IDF retrieval utilities."""

from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

from personal_docs_qa.indexer import LocalIndex, load_index
from personal_docs_qa.models import SearchResult


def search(index: LocalIndex, query: str, top_k: int = 5) -> list[SearchResult]:
    """Search a local index and return ranked chunks."""
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Search query cannot be empty.")
    if top_k <= 0:
        return []
    if not index.chunks:
        return []

    query_vector = index.vectorizer.transform([clean_query])
    scores = cosine_similarity(query_vector, index.matrix).ravel()
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


def retrieve(question: str, index_path: str | Path, top_k: int = 5) -> list[SearchResult]:
    """Load an index from disk and retrieve relevant chunks for a question."""
    return search(load_index(index_path), question, top_k=top_k)
