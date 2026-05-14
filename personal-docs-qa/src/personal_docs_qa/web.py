"""FastAPI web app for Personal Docs QA."""

from pathlib import Path, PurePosixPath
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from personal_docs_qa.answerer import answer_question
from personal_docs_qa.indexer import index_folder_with_warnings, load_index


STATIC_DIR = Path(__file__).parent / "static"
WEB_INDEX_PATH = Path(".docqa/web_index.joblib")
UPLOAD_ROOT = Path(".docqa/uploads")


class IngestPathRequest(BaseModel):
    folder_path: str


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


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

    @app.post("/api/ingest-path")
    def ingest_path(request: IngestPathRequest) -> dict:
        try:
            result = index_folder_with_warnings(request.folder_path, WEB_INDEX_PATH)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "document_count": result.index.document_count,
            "chunk_count": result.index.chunk_count,
            "warnings": result.warnings,
        }

    @app.post("/api/upload-and-index")
    async def upload_and_index(files: list[UploadFile] = File(...)) -> dict:
        if not files:
            raise HTTPException(status_code=400, detail="No files were uploaded.")

        session_dir = UPLOAD_ROOT / uuid4().hex
        session_dir.mkdir(parents=True, exist_ok=True)

        for upload in files:
            target_path = session_dir / _safe_upload_path(upload.filename or "upload.txt")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(await upload.read())

        try:
            result = index_folder_with_warnings(session_dir, WEB_INDEX_PATH)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "document_count": result.index.document_count,
            "chunk_count": result.index.chunk_count,
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

        answer = answer_question(index, request.question, top_k=request.top_k)
        return {
            "answer": answer.answer,
            "confidence": answer.confidence,
            "warnings": answer.warnings,
            "sources": [
                {
                    "rank": source.rank,
                    "score": source.score,
                    "file_name": source.chunk.file_name,
                    "excerpt": _excerpt(source.chunk.text),
                }
                for source in answer.sources
            ],
        }

    return app


app = create_app()
