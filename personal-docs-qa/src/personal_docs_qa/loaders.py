"""Document loading placeholders."""

from pathlib import Path

from personal_docs_qa.models import SourceDocument


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def load_documents(folder: Path) -> list[SourceDocument]:
    """Load supported documents from a folder.

    Full loading logic will be implemented in a later phase.
    """
    raise NotImplementedError("Document loading is not implemented yet.")

