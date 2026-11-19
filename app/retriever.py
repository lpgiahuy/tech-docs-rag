from psycopg.rows import dict_row

from app.config import settings
from app.db import get_connection
from app.ingest import embed

SEARCH_SQL = """
SELECT c.id, c.content, c.page, d.filename,
       1 - (c.embedding <=> %(q)s) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id
ORDER BY c.embedding <=> %(q)s
LIMIT %(k)s
"""

def search(question: str, k: int | None = None) -> list[dict]:
    """Return the k chunks most similar to the question, best first.

    Each item: {id, content, page, filename, score}.
    """
    query_vector = embed([question])[0]
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(SEARCH_SQL, {"q": query_vector, "k": k or settings.top_k})
        return cur.fetchall()
