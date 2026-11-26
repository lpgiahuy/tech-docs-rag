"""Retrieval: vector search, keyword search, their fusion, and optional reranking.

Every result is {id, content, page, filename, score}, best first. `score` is always the
cosine similarity to the question, whatever the ranking method, so `min_score` keeps one meaning.
"""

import numpy as np
from psycopg.rows import dict_row

from app.config import settings
from app.db import get_connection
from app.ingest import embed
from app.reranker import rerank

RRF_K = 60

VECTOR_SQL = """
SELECT c.id, c.content, c.page, d.filename,
       1 - (c.embedding <=> %(q)s) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id
ORDER BY c.embedding <=> %(q)s
LIMIT %(k)s
"""

# plainto_tsquery requires every word of the question to be in the chunk ("all").
# For "any", the same stemmed words are joined with OR and ts_rank sorts the matches.
TSQUERY = {
    "all": "plainto_tsquery('english', %(text)s)",
    "any": "replace(plainto_tsquery('english', %(text)s)::text, ' & ', ' | ')::tsquery",
}

KEYWORD_SQL = """
SELECT c.id, c.content, c.page, d.filename,
       1 - (c.embedding <=> %(q)s) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id
CROSS JOIN (SELECT {tsquery} AS query) AS t
WHERE c.content_tsv @@ t.query
ORDER BY ts_rank(c.content_tsv, t.query) DESC, c.id
LIMIT %(k)s
"""


def _fetch(sql: str, params: dict) -> list[dict]:
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def vector_search(query_vector: np.ndarray, k: int) -> list[dict]:
    """The k chunks nearest to the question embedding."""
    return _fetch(VECTOR_SQL, {"q": query_vector, "k": k})


def keyword_search(question: str, query_vector: np.ndarray, k: int) -> list[dict]:
    """The k chunks ranking highest in PostgreSQL full-text search (may return fewer)."""
    sql = KEYWORD_SQL.format(tsquery=TSQUERY[settings.keyword_match])
    return _fetch(sql, {"text": question, "q": query_vector, "k": k})


def rrf(rankings: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion: each chunk scores sum(1 / (k + rank)) over the lists it is in.

    Only ranks are used, because cosine similarity and ts_rank are on different scales.
    """
    fused: dict[int, float] = {}
    chunks: dict[int, dict] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            fused[chunk["id"]] = fused.get(chunk["id"], 0.0) + 1 / (k + rank)
            chunks[chunk["id"]] = chunk
    return [chunks[i] for i in sorted(fused, key=fused.get, reverse=True)]


def search(question: str, k: int | None = None) -> list[dict]:
    """Return the k best chunks for the question using the configured retrieval method."""
    k = k or settings.top_k
    depth = max(k, settings.candidate_k) if settings.rerank else k
    query_vector = embed([question])[0]

    if settings.retrieval_mode == "vector":
        chunks = vector_search(query_vector, depth)
    elif settings.retrieval_mode == "keyword":
        chunks = keyword_search(question, query_vector, depth)
    else:
        per_list = max(depth, settings.candidate_k)
        chunks = rrf(
            [
                vector_search(query_vector, per_list),
                keyword_search(question, query_vector, per_list),
            ]
        )[:depth]

    if settings.rerank:
        chunks = rerank(question, chunks)
    return chunks[:k]
