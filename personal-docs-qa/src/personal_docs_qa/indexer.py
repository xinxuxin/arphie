"""Local index persistence placeholders."""

from pathlib import Path

from personal_docs_qa.models import Chunk


def build_index(chunks: list[Chunk], output_path: Path) -> None:
    """Build and persist a local searchable index.

    Full indexing logic will be implemented in a later phase.
    """
    raise NotImplementedError("Indexing is not implemented yet.")
