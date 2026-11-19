import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse

from app import documents
from app.config import settings
from app.generator import REFUSAL, LLMError, filter_relevant, generate
from app.ingest import embed, process_document
from app.retriever import search
from app.schemas import AskRequest, AskResponse, DocumentOut, Source

SNIPPET_CHARS = 200


@asynccontextmanager
async def lifespan(app: FastAPI):
    embed(["warm up"])  # load the embedding model now, not during the first user request
    yield


app = FastAPI(title="tech-docs-rag", lifespan=lifespan)


@app.exception_handler(psycopg.OperationalError)
async def database_unavailable(request, exc):
    return JSONResponse(status_code=503, content={"detail": "Database unavailable"})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentOut, status_code=202)
def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    """Accept a PDF and ingest it in the background. Poll GET /documents for its status."""
    filename = Path(file.filename or "").name
    max_bytes = settings.max_upload_mb * 1024 * 1024
    content = file.file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB")
    if not filename.lower().endswith(".pdf") or not content.startswith(b"%PDF-"):
        raise HTTPException(400, "Only PDF files are accepted")
    if documents.filename_exists(filename):
        raise HTTPException(409, f"A document named '{filename}' already exists")

    # The upload is closed when this request ends, so keep a copy for the background job.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
    document = documents.create_document(filename)
    background_tasks.add_task(process_document, document["id"], tmp.name)
    return document


@app.get("/documents", response_model=list[DocumentOut])
def list_documents():
    return documents.list_documents()


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: int):
    if not documents.delete_document(document_id):
        raise HTTPException(404, "Document not found")
    return Response(status_code=204)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    start = time.perf_counter()
    if not documents.has_ready_documents():
        raise HTTPException(404, "No documents available yet. Upload a PDF first.")

    chunks = search(request.question)
    try:
        answer = generate(request.question, chunks)
    except LLMError as exc:
        raise HTTPException(503, f"The language model is unavailable: {exc}") from exc

    sources = []
    if answer != REFUSAL:
        sources = [
            Source(
                filename=c["filename"],
                page=c["page"],
                score=round(c["score"], 3),
                snippet=c["content"][:SNIPPET_CHARS],
            )
            for c in filter_relevant(chunks)
        ]
    return AskResponse(
        answer=answer, sources=sources, latency_ms=int((time.perf_counter() - start) * 1000)
    )
