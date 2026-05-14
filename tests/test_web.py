from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from personal_docs_qa.web import AskRequest, IngestPathRequest, create_app


def get_endpoint(path: str, method: str):
    app = create_app()
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"Route not found: {method} {path}")


def test_health() -> None:
    health = get_endpoint("/api/health", "GET")

    assert health() == {"status": "ok"}


def test_config() -> None:
    config = get_endpoint("/api/config", "GET")

    body = config()

    assert body["supported_retrieval_modes"] == ["auto", "tfidf", "embedding", "hybrid"]
    assert body["supported_answer_modes"] == ["auto", "openai", "local"]
    assert body["default_retrieval_mode"] == "auto"
    assert body["default_answer_mode"] == "auto"
    assert body["embedding_model"] == "text-embedding-3-small"
    assert "api" not in body


def test_ask_without_web_index_returns_helpful_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ask = get_endpoint("/api/ask", "POST")

    with pytest.raises(HTTPException) as exc_info:
        ask(AskRequest(question="What is indexed?"))

    assert exc_info.value.status_code == 404
    assert "No web index found" in exc_info.value.detail


def test_ingest_path_and_ask(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "budget.txt").write_text("The grocery budget is 120 dollars per week.", encoding="utf-8")
    ingest_path = get_endpoint("/api/ingest-path", "POST")
    ask = get_endpoint("/api/ask", "POST")

    ingest_response = ingest_path(IngestPathRequest(folder_path=str(docs)))
    ask_response = ask(AskRequest(question="What is the grocery budget?", top_k=3))

    assert ingest_response["document_count"] == 1
    assert ingest_response["chunk_count"] == 1
    assert ingest_response["retrieval_mode_requested"] == "auto"
    assert ingest_response["retrieval_mode_built"] == "tfidf"
    assert ingest_response["embeddings_created"] is False
    assert ingest_response["embedding_model"] is None
    assert "using tfidf" in ingest_response["warnings"][0]
    assert ask_response["question"] == "What is the grocery budget?"
    assert ask_response["retrieval_mode_requested"] == "auto"
    assert ask_response["retrieval_mode_used"] == "tfidf"
    assert ask_response["retrieval_fallback_used"] is True
    assert ask_response["answer_mode_requested"] == "auto"
    assert ask_response["answer_mode_used"] == "local"
    assert "budget.txt" in ask_response["answer"]
    assert ask_response["sources"][0]["file_name"] == "budget.txt"
    assert ask_response["sources"][0]["rank"] == 1
    assert ask_response["sources"][0]["score_tfidf"] is not None
    assert ask_response["sources"][0]["score_embedding"] is None
    assert ask_response["sources"][0]["retrieval_mode_used"] == "tfidf"
    assert ask_response["sources"][0]["chunk_id"]
    assert ask_response["sources"][0]["excerpt"]


@pytest.mark.anyio
async def test_upload_and_index(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    upload_and_index = get_endpoint("/api/upload-and-index", "POST")
    ask = get_endpoint("/api/ask", "POST")
    files = [
        UploadFile(BytesIO(b"Bring a passport for travel."), filename="notes/travel.md"),
        UploadFile(BytesIO(b"unsupported"), filename="skip.csv"),
    ]

    response = await upload_and_index(files=files)

    assert response["document_count"] == 1
    assert response["chunk_count"] == 1
    assert response["retrieval_mode_requested"] == "auto"
    assert response["retrieval_mode_built"] == "tfidf"
    assert response["embeddings_created"] is False
    assert "using tfidf" in response["warnings"][0]

    ask_response = ask(AskRequest(question="What should I bring for travel?"))
    assert ask_response["sources"][0]["file_name"] == "travel.md"
