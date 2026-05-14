from pathlib import Path

import pytest

from personal_docs_qa.indexer import build_index, index_folder, load_index, save_index
from personal_docs_qa.models import Chunk


def make_chunk(text: str, document_id: str = "doc-1", index: int = 0) -> Chunk:
    return Chunk(
        id=f"{document_id}-chunk-{index}",
        document_id=document_id,
        file_name="note.txt",
        path="/tmp/note.txt",
        text=text,
        start_char=0,
        end_char=len(text),
        metadata={"source_path": "/tmp/note.txt", "extension": ".txt"},
    )


def test_build_index_from_chunks() -> None:
    chunks = [make_chunk("budget groceries rent"), make_chunk("hiking trail map", "doc-2", 1)]

    index = build_index(chunks)

    assert index.chunk_count == 2
    assert index.document_count == 2
    assert index.matrix.shape[0] == 2
    assert index.chunks == chunks
    assert index.created_at
    assert index.retrieval_mode == "tfidf"


def test_save_and_load_index_preserves_chunks(tmp_path: Path) -> None:
    path = tmp_path / ".docqa" / "index.joblib"
    chunks = [make_chunk("tax receipt invoice")]
    index = build_index(chunks)

    save_index(index, path)
    loaded = load_index(path)

    assert loaded.chunk_count == 1
    assert loaded.document_count == 1
    assert loaded.chunks == chunks
    assert loaded.matrix.shape == index.matrix.shape


def test_empty_chunk_list_handled_cleanly() -> None:
    with pytest.raises(ValueError, match="no non-empty chunks"):
        build_index([])


def test_auto_mode_falls_back_to_tfidf_without_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    index = build_index([make_chunk("local search only")], retrieval_mode="auto")

    assert index.retrieval_mode == "tfidf"
    assert index.embedding_matrix is None
    assert index.warnings


def test_embedding_mode_requires_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_index([make_chunk("semantic search")], retrieval_mode="embedding")


def test_empty_folder_cannot_be_indexed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no chunks were produced"):
        index_folder(tmp_path, index_path=tmp_path / ".docqa" / "index.joblib")


def test_missing_index_path_raises_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / ".docqa" / "missing.joblib"

    with pytest.raises(FileNotFoundError, match="Index not found"):
        load_index(missing)


def test_index_folder_saves_default_shape(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.txt").write_text("A simple searchable document.", encoding="utf-8")
    index_path = tmp_path / ".docqa" / "index.joblib"

    index = index_folder(docs, index_path=index_path)
    loaded = load_index(index_path)

    assert index.chunk_count == 1
    assert loaded.chunks[0].text == "A simple searchable document."
