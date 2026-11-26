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

# The chunk just before and just after each hit, in reading order across page boundaries.
NEIGHBOR_SQL = """
SELECT h.id AS hit_id, n.id, n.content, n.page, d.filename,
       1 - (n.embedding <=> %(q)s) AS score, n.offset_from_hit
FROM chunks h
JOIN documents d ON d.id = h.document_id
CROSS JOIN LATERAL (
    (SELECT c.id, c.content, c.page, c.embedding, -1 AS offset_from_hit FROM chunks c
     WHERE c.document_id = h.document_id AND (c.page, c.chunk_index) < (h.page, h.chunk_index)
     ORDER BY c.page DESC, c.chunk_index DESC LIMIT 1)
    UNION ALL
    (SELECT c.id, c.content, c.page, c.embedding, 1 FROM chunks c
     WHERE c.document_id = h.document_id AND (c.page, c.chunk_index) > (h.page, h.chunk_index)
     ORDER BY c.page, c.chunk_index LIMIT 1)
) AS n
WHERE h.id = ANY(%(ids)s)
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


def add_neighbors(question: str, chunks: list[dict]) -> list[dict]:
    """Surround each relevant hit with the chunks next to it, if `context_neighbors` is on.

    Hits keep their ranking; each comes out as before, hit, after. Added chunks are marked
    `neighbor`, so `min_score` does not drop them: they are there for their hit's sake.
    Hits below `min_score` bring no neighbours. A chunk appears once.
    """
    hits = [c for c in chunks if c["score"] >= settings.min_score]
    if not settings.context_neighbors or not hits:
        return chunks

    rows = _fetch(NEIGHBOR_SQL, {"q": embed([question])[0], "ids": [c["id"] for c in hits]})
    around = {(row.pop("hit_id"), row.pop("offset_from_hit")): row for row in rows}
    hit_ids = {c["id"] for c in chunks}
    result, seen = [], set()
    for chunk in chunks:
        group = [chunk]
        before, after = around.get((chunk["id"], -1)), around.get((chunk["id"], 1))
        if before:
            group.insert(0, before)
        if after:
            group.append(after)
        for item in group:
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            if item["id"] in hit_ids:  # a neighbour that was also found keeps its hit entry
                result.append(next(c for c in chunks if c["id"] == item["id"]))
            else:
                result.append({**item, "neighbor": True})
    return result


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
