"""API tests. The database, the embedding model and the LLM are all replaced by fakes,
so these run offline and without PostgreSQL.

TestClient is used without `with`, so the startup hook (which loads the embedding model) is skipped.
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app import main
from app.generator import REFUSAL, LLMError

client = TestClient(main.app)

CHUNKS = [
    {"id": 1, "content": "AWS reserves five IP addresses in each subnet.", "page": 41,
     "filename": "vpc.pdf", "score": 0.78},
    {"id": 2, "content": "Unrelated text.", "page": 3, "filename": "vpc.pdf", "score": 0.10},
]
PDF_BYTES = b"%PDF-1.4 fake pdf body"


@pytest.fixture
def docs_available(monkeypatch):
    monkeypatch.setattr(main.documents, "has_ready_documents", lambda: True)


def fake_document(filename="a.pdf"):
    return {"id": 7, "filename": filename, "status": "processing", "error": None,
            "uploaded_at": datetime(2026, 11, 23), "chunk_count": 0}


# /health

def test_health():
    assert client.get("/health").json() == {"status": "ok"}


# /ask

def test_ask_returns_answer_and_only_relevant_sources(monkeypatch, docs_available):
    monkeypatch.setattr(main, "search", lambda q: CHUNKS)
    monkeypatch.setattr(main, "generate", lambda q, c: "Five [vpc.pdf, page 41].")
    response = client.post("/ask", json={"question": "How many IPs are reserved?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Five [vpc.pdf, page 41]."
    assert [s["page"] for s in body["sources"]] == [41]  # the 0.10 chunk is below min_score
    assert body["latency_ms"] >= 0


def test_ask_refusal_has_no_sources(monkeypatch, docs_available):
    monkeypatch.setattr(main, "search", lambda q: CHUNKS)
    monkeypatch.setattr(main, "generate", lambda q, c: REFUSAL)
    body = client.post("/ask", json={"question": "Capital of France?"}).json()
    assert body["answer"] == REFUSAL
    assert body["sources"] == []


@pytest.mark.parametrize("question", ["", "   "])
def test_ask_empty_question_is_422(question):
    assert client.post("/ask", json={"question": question}).status_code == 422


def test_ask_missing_question_is_422():
    assert client.post("/ask", json={}).status_code == 422


def test_ask_without_documents_is_404(monkeypatch):
    monkeypatch.setattr(main.documents, "has_ready_documents", lambda: False)
    assert client.post("/ask", json={"question": "anything"}).status_code == 404


def test_ask_llm_failure_is_503(monkeypatch, docs_available):
    def boom(question, chunks):
        raise LLMError("ServerError: 503 UNAVAILABLE")

    monkeypatch.setattr(main, "search", lambda q: CHUNKS)
    monkeypatch.setattr(main, "generate", boom)
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


# POST /documents

def upload(name, content, monkeypatch=None):
    return client.post("/documents", files={"file": (name, content, "application/pdf")})


def test_upload_accepts_pdf_and_starts_background_job(monkeypatch):
    started = []
    monkeypatch.setattr(main.documents, "filename_exists", lambda n: False)
    monkeypatch.setattr(main.documents, "create_document", lambda n: fake_document(n))
    monkeypatch.setattr(main, "process_document", lambda doc_id, path: started.append(doc_id))
    response = upload("manual.pdf", PDF_BYTES)
    assert response.status_code == 202
    assert response.json()["status"] == "processing"
    assert started == [7]


def test_upload_non_pdf_extension_is_400():
    assert upload("notes.txt", PDF_BYTES).status_code == 400


def test_upload_fake_pdf_content_is_400():
    assert upload("fake.pdf", b"this is not a pdf").status_code == 400


def test_upload_too_large_is_413(monkeypatch):
    monkeypatch.setattr(main.settings, "max_upload_mb", 0)
    assert upload("big.pdf", PDF_BYTES).status_code == 413


def test_upload_duplicate_filename_is_409(monkeypatch):
    monkeypatch.setattr(main.documents, "filename_exists", lambda n: True)
    assert upload("manual.pdf", PDF_BYTES).status_code == 409


# GET / DELETE /documents

def test_list_documents(monkeypatch):
    monkeypatch.setattr(main.documents, "list_documents", lambda: [fake_document("a.pdf")])
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json()[0]["filename"] == "a.pdf"


def test_delete_existing_document_is_204(monkeypatch):
    monkeypatch.setattr(main.documents, "delete_document", lambda i: True)
    assert client.delete("/documents/7").status_code == 204


def test_delete_unknown_document_is_404(monkeypatch):
    monkeypatch.setattr(main.documents, "delete_document", lambda i: False)
    assert client.delete("/documents/999").status_code == 404


def test_delete_non_numeric_id_is_422():
    assert client.delete("/documents/abc").status_code == 422
