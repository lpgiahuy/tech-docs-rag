import logging
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.db import get_connection

logging.getLogger("pypdf").setLevel(logging.ERROR)
logger = logging.getLogger("app.ingest")

BATCH_SIZE = 64


def clean_text(text: str) -> str:
    """Normalize extracted PDF text before chunking.

    - NFKC turns ligatures such as 'ﬃ' into 'ffi', so keyword search for 'traffic' matches.
    - Runs of 4+ dots (table-of-contents leaders) are dropped; they carry no meaning.
    - Whitespace is collapsed to single spaces.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?:\s?\.){4,}", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_pdf(path: str | Path) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for every page that has text. Pages are 1-based."""
    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append((number, text))
    return pages


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into chunks of `size` characters; consecutive chunks share `overlap` characters."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    step = size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts into L2-normalized vectors (shape: len(texts) x 384)."""
    return _model().encode(
        texts, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False
    )


def build_rows(path: str | Path) -> list[tuple[int, int, str]]:
    """Read a PDF and return [(page, chunk_index, content)] for all its chunks."""
    rows = []
    for page, text in load_pdf(path):
        for index, chunk in enumerate(chunk_text(text, settings.chunk_size, settings.chunk_overlap)):
            rows.append((page, index, chunk))
    return rows


def _store_chunks(conn, document_id: int, rows: list[tuple[int, int, str]]) -> None:
    """Embed the rows in batches and insert them into `chunks` using `conn`."""
    with conn.cursor() as cur:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            vectors = embed([content for _, _, content in batch])
            cur.executemany(
                "INSERT INTO chunks (document_id, page, chunk_index, content, embedding)"
                " VALUES (%s, %s, %s, %s, %s)",
                [
                    (document_id, page, index, content, vec)
                    for (page, index, content), vec in zip(batch, vectors)
                ],
            )


def ingest_file(path: str | Path) -> int:
    """Ingest one PDF from the command line. Returns the number of chunks stored."""
    path = Path(path)
    rows = build_rows(path)
    if not rows:
        print(f"{path.name}: no extractable text (scanned PDF?), skipped")
        return 0

    with get_connection() as conn:
        document_id = conn.execute(
            "INSERT INTO documents (filename) VALUES (%s) RETURNING id", (path.name,)
        ).fetchone()[0]
        _store_chunks(conn, document_id, rows)
    print(f"{path.name}: {len(rows)} chunks stored")
    return len(rows)


def process_document(document_id: int, path: str | Path) -> None:
    """Background job for the API: fill an existing 'processing' document, then mark it.

    Chunks and the 'ready' status are committed together, so a search never sees a
    half-ingested document. On any error the document is marked 'failed' with the reason.
    The uploaded temp file is always removed.
    """
    path = Path(path)
    try:
        rows = build_rows(path)
        if not rows:
            raise ValueError("no extractable text (scanned PDF? OCR is not supported)")
        with get_connection() as conn:
            _store_chunks(conn, document_id, rows)
            conn.execute(
                "UPDATE documents SET status = 'ready', error = NULL WHERE id = %s", (document_id,)
            )
        logger.info("document %s ingested: %d chunks", document_id, len(rows))
    except Exception as exc:  # noqa: BLE001 - any failure must be recorded, not lost
        logger.exception("ingest of document %s failed", document_id)
        with get_connection() as conn:
            conn.execute(
                "UPDATE documents SET status = 'failed', error = %s WHERE id = %s",
                (str(exc)[:500], document_id),
            )
    finally:
        path.unlink(missing_ok=True)


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        sys.exit("usage: python -m app.ingest <pdf file or folder>")
    target = Path(argv[1])
    files = sorted(target.glob("*.pdf")) if target.is_dir() else [target]
    if not files:
        sys.exit(f"no PDF files found in {target}")
    for file in files:
        ingest_file(file)


if __name__ == "__main__":
    main(sys.argv)
