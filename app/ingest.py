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


def ingest_file(path: str | Path) -> int:
    """Ingest one PDF. Returns the number of chunks stored."""
    path = Path(path)
    rows = []  # (page, chunk_index, content)
    for page, text in load_pdf(path):
        for index, chunk in enumerate(chunk_text(text, settings.chunk_size, settings.chunk_overlap)):
            rows.append((page, index, chunk))
    if not rows:
        print(f"{path.name}: no extractable text (scanned PDF?), skipped")
        return 0

    with get_connection() as conn:
        document_id = conn.execute(
            "INSERT INTO documents (filename) VALUES (%s) RETURNING id", (path.name,)
        ).fetchone()[0]
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
    print(f"{path.name}: {len(rows)} chunks stored")
    return len(rows)


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
