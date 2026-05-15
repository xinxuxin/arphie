from personal_docs_qa.chunker import chunk_document, chunk_documents
from personal_docs_qa.models import Document


def make_document(text: str, metadata: dict | None = None) -> Document:
    return Document(
        id="doc-test",
        path="/tmp/note.md",
        file_name="note.md",
        file_type="md",
        text=text,
        metadata=metadata or {"source_path": "/tmp/note.md", "extension": ".md"},
    )


def make_pdf_document(text: str) -> Document:
    return Document(
        id="doc-pdf",
        path="/tmp/resume.pdf",
        file_name="resume.pdf",
        file_type="pdf",
        text=text,
        metadata={"source_path": "/tmp/resume.pdf", "extension": ".pdf", "page_number": 1},
    )


def test_long_text_creates_multiple_chunks() -> None:
    text = "\n\n".join(f"Paragraph {index}. " + ("word " * 20) for index in range(10))

    chunks = chunk_document(make_document(text), chunk_size=140, overlap=30)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 140 for chunk in chunks)


def test_overlap_exists_between_chunks() -> None:
    text = "\n\n".join(f"Paragraph {index}. " + ("context " * 10) for index in range(6))

    chunks = chunk_document(make_document(text), chunk_size=130, overlap=25)

    assert len(chunks) > 1
    first_tail = chunks[0].text[-25:].strip()
    assert first_tail
    assert chunks[1].text.startswith(first_tail)


def test_short_text_creates_one_chunk() -> None:
    chunks = chunk_document(make_document("A short note."))

    assert len(chunks) == 1
    assert chunks[0].text == "A short note."


def test_metadata_and_source_fields_are_preserved() -> None:
    metadata = {"source_path": "/tmp/note.md", "extension": ".md", "page_number": 2}
    chunks = chunk_documents([make_document("Some content.", metadata=metadata)])

    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-test"
    assert chunks[0].file_name == "note.md"
    assert chunks[0].path == "/tmp/note.md"
    assert chunks[0].metadata == metadata
    assert chunks[0].metadata is not metadata


def test_empty_text_creates_no_chunks() -> None:
    assert chunk_document(make_document("   \n")) == []


def test_pdf_documents_use_smaller_default_chunks() -> None:
    text = " ".join(f"resume item {index}." for index in range(220))

    chunks = chunk_documents([make_pdf_document(text)])

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 500 for chunk in chunks)
