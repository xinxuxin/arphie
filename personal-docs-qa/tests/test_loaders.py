from pathlib import Path

from personal_docs_qa.loaders import load_folder, load_md, load_txt
from personal_docs_qa.models import make_chunk_id, make_document_id


def test_load_txt(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("hello from a text file\n", encoding="utf-8")

    document = load_txt(path)

    assert document.id == make_document_id(path)
    assert document.path == str(path)
    assert document.file_name == "note.txt"
    assert document.file_type == "txt"
    assert document.text == "hello from a text file"
    assert document.metadata["source_path"] == str(path)
    assert document.metadata["extension"] == ".txt"


def test_load_md(tmp_path: Path) -> None:
    path = tmp_path / "journal.md"
    path.write_text("# Journal\n\nA markdown note.", encoding="utf-8")

    document = load_md(path)

    assert document.file_name == "journal.md"
    assert document.file_type == "md"
    assert document.text == "# Journal\n\nA markdown note."


def test_load_folder_recursively_skips_unsupported_files(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "keep.txt").write_text("keep this", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("not supported", encoding="utf-8")

    documents, warnings = load_folder(tmp_path)

    assert [document.file_name for document in documents] == ["keep.txt"]
    assert warnings == []


def test_empty_files_warn_but_do_not_crash(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("   \n", encoding="utf-8")
    (tmp_path / "real.md").write_text("real content", encoding="utf-8")

    documents, warnings = load_folder(tmp_path)

    assert [document.file_name for document in documents] == ["real.md"]
    assert len(warnings) == 1
    assert "Skipped empty document" in warnings[0]
    assert "empty.txt" in warnings[0]


def test_broken_folder_path_handled_cleanly(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    documents, warnings = load_folder(missing)

    assert documents == []
    assert warnings == [f"Folder not found: {missing}"]


def test_empty_folder_returns_no_documents_or_warnings(tmp_path: Path) -> None:
    documents, warnings = load_folder(tmp_path)

    assert documents == []
    assert warnings == []


def test_duplicate_file_names_in_different_subfolders_keep_distinct_paths(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "note.txt").write_text("first note", encoding="utf-8")
    (second / "note.txt").write_text("second note", encoding="utf-8")

    documents, warnings = load_folder(tmp_path)

    assert warnings == []
    assert len(documents) == 2
    assert [document.file_name for document in documents] == ["note.txt", "note.txt"]
    assert len({document.path for document in documents}) == 2
    assert len({document.id for document in documents}) == 2


def test_malformed_pdf_warns_and_does_not_stop_ingest(tmp_path: Path) -> None:
    (tmp_path / "broken.pdf").write_bytes(b"this is not a real pdf")
    (tmp_path / "note.txt").write_text("valid text survives", encoding="utf-8")

    documents, warnings = load_folder(tmp_path)

    assert [document.file_name for document in documents] == ["note.txt"]
    assert len(warnings) == 1
    assert "Failed to load" in warnings[0]
    assert "broken.pdf" in warnings[0]


def test_make_chunk_id_is_stable_and_readable() -> None:
    assert make_chunk_id("doc-abc", 3) == "doc-abc-chunk-0003"
