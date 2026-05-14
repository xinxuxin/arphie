"""Retrieval placeholders."""

from pathlib import Path

from personal_docs_qa.models import SearchResult


def retrieve(question: str, index_path: Path, top_k: int = 5) -> list[SearchResult]:
    """Retrieve relevant chunks for a question.

    Full retrieval logic will be implemented in a later phase.
    """
    raise NotImplementedError("Retrieval is not implemented yet.")
