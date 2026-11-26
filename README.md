# tech-docs-rag

Upload technical manuals (PDF) and ask questions in natural language. Answers come with page-level citations, and the system refuses to answer when the documents do not contain the information.

Built with FastAPI, PostgreSQL + pgvector, sentence-transformers, Gemini, Docker and AWS EC2.

> **Status:** MVP (v0.1) is deployed. Hybrid search, reranking and a retrieval evaluation are implemented and compared below. Authentication and rate limiting are planned.

**Live demo:** <http://16.178.77.124/docs> (Swagger UI)

> The demo runs on a small AWS EC2 instance that may be stopped between demos to save cost. If the link does not respond, it is probably offline: the code runs the same locally (see [Run it](#run-it)).

## Try the demo

The demo has two public AWS documents loaded: the *Amazon VPC User Guide* and the *AWS Well-Architected Framework*.

1. Open the [Swagger UI](http://16.178.77.124/docs), expand `POST /ask` and click **Try it out**.
2. Use a body like:

```json
{"question": "How many IP addresses does AWS reserve in each VPC subnet?"}
```

Real response (shortened):

```json
{
  "answer": "AWS reserves five IP addresses in each VPC subnet [aws_vpc_user_guide.pdf, page 41].",
  "sources": [{"filename": "aws_vpc_user_guide.pdf", "page": 41, "score": 0.783, "snippet": "..."}],
  "latency_ms": 4869
}
```

More questions to try:

- `What is the difference between a public and a private subnet?`
- `What is the capital of France?` (not in the documents: the system should refuse instead of guessing)

Uploading and deleting documents (`/documents`) is password-protected on the public demo, and `/ask` is rate limited, because the API has no user accounts yet. To use your own PDFs, run it locally.

## How it works

```
PDF ──► read pages ──► clean text ──► chunk (500 chars, 50 overlap) ──► embed ──► PostgreSQL + pgvector

question ──► embed ──► top-k nearest chunks (cosine, HNSW index) ──► Gemini (context only, cites sources) ──► answer + sources
```

- **Ingest** runs in the background; `GET /documents` shows `processing`, `ready` or `failed`.
- **Retrieval** uses pgvector cosine distance with an HNSW index by default. Hybrid search (plus PostgreSQL full-text search, fused with Reciprocal Rank Fusion) and cross-encoder reranking can be switched on; see [Experiments](#experiments).
- **Generation** sends only the retrieved chunks to Gemini with instructions to answer from them, cite `[file, page N]`, and refuse otherwise. Chunks below a similarity threshold never reach the model.

## Run it

Requirements: Docker, and a Gemini API key from [Google AI Studio](https://aistudio.google.com/).

```bash
cp .env.example .env        # then set LLM_API_KEY (and LLM_MODEL if you like)
docker compose up --build
```

Open <http://localhost:8000/docs>. The first build downloads PyTorch (CPU) and the embedding model, so it takes a few minutes.

### Without Docker

```bash
docker compose up -d db
python -m venv .venv && .venv/Scripts/activate      # Windows; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Command-line tools:

```bash
python -m app.ingest data/samples/                  # ingest a folder of PDFs
python -m app.ask "your question"                   # retrieve + answer from the terminal
python -m pytest -q                                 # tests (no database or API key needed)
```

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/documents` | Upload a PDF. Returns `202`; ingestion continues in the background |
| `GET` | `/documents` | List documents with `status` and `chunk_count` |
| `DELETE` | `/documents/{id}` | Delete a document and its chunks |
| `POST` | `/ask` | `{"question": "..."}` → `{answer, sources, latency_ms}` |
| `GET` | `/health` | Liveness check |

Errors: `400` not a PDF, `404` unknown document or nothing uploaded yet, `409` duplicate filename, `413` file too large, `422` invalid input, `503` language model or database unavailable.

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How many IP addresses does AWS reserve in each VPC subnet?"}'
```

## Configuration

Set in `.env` (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `LLM_API_KEY` | – | API key of your LLM provider (the provider is currently Gemini; the call is isolated in `app/generator.py`, so another provider only needs changes there) |
| `LLM_MODEL` | see `.env.example` | Model name for that provider |
| `TOP_K` | 5 | Chunks retrieved per question |
| `RETRIEVAL_MODE` | `vector` | `vector`, `keyword` or `hybrid` |
| `KEYWORD_MATCH` | `any` | Keyword search needs `any` or `all` of the question's words in a chunk |
| `RERANK` | `false` | Rerank candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `CANDIDATE_K` | 20 | Candidates per list before fusion and reranking |
| `MIN_SCORE` | 0.25 | Cosine similarity below which a chunk is ignored |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 50 | Chunking in characters |
| `MAX_UPLOAD_MB` | 20 | Upload size limit |

## Experiments

Evaluation set: 30 questions on the two sample documents, 25 answerable (10 using the document's exact terms, 8 paraphrased, 7 needing several facts) and 5 unanswerable. A hit means a chunk from a labelled page is in the top k. Latency is retrieval only (no LLM), measured on a laptop CPU.

```bash
python eval/run_eval.py --mode hybrid --rerank
```

**Chunking and embedding model** (vector search):

| Chunk / overlap | Embedding model | Recall@5 | Recall@8 | MRR |
|---|---|---|---|---|
| 300 / 30 | all-MiniLM-L6-v2 | 0.72 | 0.88 | 0.57 |
| **500 / 50** | **all-MiniLM-L6-v2** | **0.88** | **0.92** | **0.68** |
| 1000 / 100 | all-MiniLM-L6-v2 | 0.80 | 0.80 | 0.57 |
| 500 / 50 | bge-small-en-v1.5 (with query instruction) | 0.80 | 0.88 | 0.71 |

**Retrieval method** (500 / 50, all-MiniLM-L6-v2):

| Method | Recall@5 | Recall@8 | MRR | Recall@5 exact terms | Recall@5 paraphrased | p95 latency |
|---|---|---|---|---|---|---|
| **Vector (default)** | 0.88 | 0.92 | 0.68 | 0.90 | 0.88 | 81 ms |
| Keyword | 0.68 | 0.80 | 0.52 | 1.00 | 0.38 | 112 ms |
| Hybrid (RRF) | 0.84 | 0.92 | 0.74 | 0.80 | 0.75 | 175 ms |
| Hybrid + rerank | 0.84 | 0.88 | 0.65 | 0.90 | 0.62 | 767 ms |

- Keyword search found every exact-term question in its top 5 but only 3 of 8 paraphrased ones.
- Hybrid search ranked correct pages higher on average (MRR 0.68 → 0.74) but did not raise Recall@5.
- Reranking did not help overall and made retrieval several times slower.
- Requiring all words of the question (`plainto_tsquery` as is) gave keyword search a Recall@5 of only 0.20, so the words are combined with OR. Hybrid with the all-words form scored 0.92 / MRR 0.70.
- With 25 questions, one question is 0.04, so most of these differences are too small to be conclusive. Vector search stays the default because nothing clearly beat it.
- The page labels were extended after looking at vector search results and are still incomplete, which can favour vector search over the other methods.

## Deployment

Runs on a single AWS EC2 instance with Docker Compose, behind an Nginx reverse proxy. The database port is never exposed publicly.

A database created before full-text search was added needs one upgrade before using `keyword` or `hybrid` (safe to run again):

```bash
docker compose exec -T db psql -U postgres < sql/init.sql
```

## Known limitations

- English text only (the embedding model is English); scanned PDFs without a text layer are rejected (no OCR).
- Tables are read as plain text, and an answer spread over several chunks can come back incomplete.
- Repeated page headers add noise to similarity scores.
- Duplicate detection is by filename, not content.
- No authentication or rate limiting yet: a public instance needs protection at the proxy level (e.g. Basic Auth on `/documents` and a rate limit on `/ask`).
- Ingestion runs inside the API process; a restart during ingestion leaves the document in `processing`.

## Sample documents

`data/samples/*.pdf` is not committed. The development set is public AWS documentation:
[VPC User Guide](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-ug.pdf) and
[Well-Architected Framework](https://docs.aws.amazon.com/wellarchitected/latest/framework/wellarchitected-framework.pdf).
