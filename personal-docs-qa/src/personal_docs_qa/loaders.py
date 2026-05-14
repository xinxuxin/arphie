"""Document loading utilities."""

from pathlib import Path
from typing import Any

from pypdf import PdfReader

from personal_docs_qa.models import Document, make_document_id


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def _base_metadata(path: Path) -> dict[str, Any]:
    return {
        "source_path": str(path),
        "extension": path.suffix.lower(),
    }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _document_from_text(path: Path, text: str) -> Document:
    return Document(
        id=make_document_id(path),
        path=str(path),
        file_name=path.name,
        file_type=path.suffix.lower().lstrip("."),
        text=text,
        metadata=_base_metadata(path),
    )


def load_txt(path: str | Path) -> Document:
    """Load a plain text file."""
    file_path = Path(path)
    return _document_from_text(file_path, _read_text(file_path))


def load_md(path: str | Path) -> Document:
    """Load a Markdown file as plain text."""
    file_path = Path(path)
    return _document_from_text(file_path, _read_text(file_path))


def load_pdf(path: str | Path) -> list[Document]:
    """Load a PDF into one document per page with extractable text."""
    file_path = Path(path)
    reader = PdfReader(str(file_path))
    documents: list[Document] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        metadata = _base_metadata(file_path)
        metadata["page_number"] = page_index
        documents.append(
            Document(
                id=make_document_id(f"{file_path}#page={page_index}"),
                path=str(file_path),
                file_name=file_path.name,
                file_type="pdf",
                text=text,
                metadata=metadata,
            )
        )

    return documents


def load_folder(folder_path: str | Path) -> tuple[list[Document], list[str]]:
    """Recursively load supported documents from a folder.

    Returns loaded documents and non-fatal warnings for skipped or failed files.
    """
    folder = Path(folder_path)
    documents: list[Document] = []
    warnings: list[str] = []

    if not folder.exists():
        return [], [f"Folder not found: {folder}"]
    if not folder.is_dir():
        return [], [f"Not a folder: {folder}"]

    for path in sorted(item for item in folder.rglob("*") if item.is_file()):
        extension = path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue

        try:
            if extension == ".txt":
                document = load_txt(path)
                if document.text:
                    documents.append(document)
                else:
                    warnings.append(f"Skipped empty document: {path}")
            elif extension == ".md":
                document = load_md(path)
                if document.text:
                    documents.append(document)
                else:
                    warnings.append(f"Skipped empty document: {path}")
            elif extension == ".pdf":
                pdf_documents = load_pdf(path)
                if pdf_documents:
                    documents.extend(pdf_documents)
                else:
                    warnings.append(f"Skipped PDF with no extractable text: {path}")
        except Exception as exc:
            warnings.append(f"Failed to load {path}: {exc}")

    return documents, warnings


def load_documents(folder: str | Path) -> list[Document]:
    """Compatibility wrapper returning only documents."""
    documents, _warnings = load_folder(folder)
    return documents
