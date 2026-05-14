"""FastAPI web app for Personal Docs QA."""

from pathlib import Path, PurePosixPath
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from personal_docs_qa.answerer import answer_question_with_metadata
from personal_docs_qa.config import (
    VALID_ANSWER_MODES,
    VALID_RETRIEVAL_MODES,
    get_default_answer_mode,
    get_default_retrieval_mode,
    get_embedding_model,
    is_openai_available,
)
from personal_docs_qa.indexer import index_folder_with_warnings, load_index


STATIC_DIR = Path(__file__).parent / "static"
WEB_INDEX_PATH = Path(".docqa/web_index.joblib")
UPLOAD_ROOT = Path(".docqa/uploads")


class IngestPathRequest(BaseModel):
    folder_path: str
    retrieval_mode: str | None = None


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    retrieval_mode: str = "auto"
    answer_mode: str = "auto"


def _excerpt(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _safe_upload_path(file_name: str) -> Path:
    normalized = file_name.replace("\\", "/")
    parts = [
        part
        for part in PurePosixPath(normalized).parts
        if part not in {"", ".", "..", "/"} and ":" not in part
    ]
    return Path(*parts) if parts else Path("upload.txt")


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="Personal Docs QA")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config")
    def config() -> dict:
        index_present = WEB_INDEX_PATH.exists()
        index_built_with_embeddings = False
        index_retrieval_mode_built = None
        if index_present:
            try:
                current_index = load_index(WEB_INDEX_PATH)
                index_built_with_embeddings = current_index.chunk_embeddings is not None
                index_retrieval_mode_built = current_index.retrieval_mode_built
            except Exception:
                index_present = False
        return {
            "openai_available": is_openai_available(),
            "supported_retrieval_modes": ["auto", "tfidf", "embedding", "hybrid"],
            "supported_answer_modes": ["auto", "openai", "local"],
            "default_retrieval_mode": get_default_retrieval_mode(),
            "default_answer_mode": get_default_answer_mode(),
            "embedding_model": get_embedding_model(),
            "index_present": index_present,
            "index_built_with_embeddings": index_built_with_embeddings,
            "index_retrieval_mode_built": index_retrieval_mode_built,
        }

    @app.post("/api/ingest-path")
    def ingest_path(request: IngestPathRequest) -> dict:
        retrieval_mode_requested = request.retrieval_mode or get_default_retrieval_mode()
        if retrieval_mode_requested not in VALID_RETRIEVAL_MODES:
            raise HTTPException(status_code=400, detail=f"Invalid retrieval mode: {retrieval_mode_requested}")
        try:
            result = index_folder_with_warnings(
                request.folder_path,
                WEB_INDEX_PATH,
                retrieval_mode=retrieval_mode_requested,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "document_count": result.index.document_count,
            "chunk_count": result.index.chunk_count,
            "retrieval_mode_requested": retrieval_mode_requested,
            "retrieval_mode_built": result.index.retrieval_mode_built,
            "embeddings_created": result.index.chunk_embeddings is not None,
            "embedding_model": result.index.embedding_model,
            "warnings": result.warnings,
        }

    @app.post("/api/upload-and-index")
    async def upload_and_index(
        files: list[UploadFile] = File(...),
        retrieval_mode: Annotated[str, Form()] = "auto",
    ) -> dict:
        if not files:
            raise HTTPException(status_code=400, detail="No files were uploaded.")
        if retrieval_mode not in VALID_RETRIEVAL_MODES:
            raise HTTPException(status_code=400, detail=f"Invalid retrieval mode: {retrieval_mode}")

        session_dir = UPLOAD_ROOT / uuid4().hex
        session_dir.mkdir(parents=True, exist_ok=True)

        for upload in files:
            target_path = session_dir / _safe_upload_path(upload.filename or "upload.txt")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(await upload.read())

        try:
            result = index_folder_with_warnings(session_dir, WEB_INDEX_PATH, retrieval_mode=retrieval_mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "document_count": result.index.document_count,
            "chunk_count": result.index.chunk_count,
            "retrieval_mode_requested": retrieval_mode,
            "retrieval_mode_built": result.index.retrieval_mode_built,
            "embeddings_created": result.index.chunk_embeddings is not None,
            "embedding_model": result.index.embedding_model,
            "warnings": result.warnings,
        }

    @app.post("/api/ask")
    def ask(request: AskRequest) -> dict:
        try:
            index = load_index(WEB_INDEX_PATH)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="No web index found. Ingest a folder or upload files first.",
            ) from exc

        if request.retrieval_mode not in VALID_RETRIEVAL_MODES:
            raise HTTPException(status_code=400, detail=f"Invalid retrieval mode: {request.retrieval_mode}")
        if request.answer_mode not in VALID_ANSWER_MODES:
            raise HTTPException(status_code=400, detail=f"Invalid answer mode: {request.answer_mode}")

        result = answer_question_with_metadata(
            index,
            request.question,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            answer_mode=request.answer_mode,
        )
        answer = result.answer
        return {
            "question": request.question,
            "answer": answer.answer,
            "confidence": answer.confidence,
            "retrieval_mode_requested": result.retrieval_mode_requested,
            "retrieval_mode_used": result.retrieval_mode_used,
            "retrieval_fallback_used": result.retrieval_fallback_used,
            "answer_mode_requested": result.answer_mode_requested,
            "answer_mode_used": result.answer_mode_used,
            "answer_fallback_used": result.answer_fallback_used,
            "warnings": answer.warnings,
            "sources": [
                {
                    "rank": source.rank,
                    "score": source.score,
                    "score_tfidf": source.score_tfidf,
                    "score_embedding": source.score_embedding,
                    "retrieval_mode_used": source.retrieval_mode_used,
                    "file_name": source.chunk.file_name,
                    "chunk_id": source.chunk.id,
                    "excerpt": _excerpt(source.chunk.text),
                }
                for source in answer.sources
            ],
        }

    return app


app = create_app()
