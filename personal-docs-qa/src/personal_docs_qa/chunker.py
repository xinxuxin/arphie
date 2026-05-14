"""Document chunking placeholders."""

from personal_docs_qa.models import DocumentChunk, SourceDocument


def chunk_documents(documents: list[SourceDocument]) -> list[DocumentChunk]:
    """Split documents into searchable chunks.

    Full chunking logic will be implemented in a later phase.
    """
    raise NotImplementedError("Document chunking is not implemented yet.")

