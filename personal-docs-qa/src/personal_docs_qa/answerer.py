"""Answer generation placeholders."""

from personal_docs_qa.models import SearchResult


def answer_question(question: str, chunks: list[SearchResult]) -> str:
    """Create a concise answer with citations.

    Full answer generation logic will be implemented in a later phase.
    """
    raise NotImplementedError("Answering is not implemented yet.")
