from psycopg.rows import dict_row

from app.db import get_connection

COLUMNS = "id, filename, status, error, uploaded_at"


def create_document(filename: str) -> dict:
    """Insert a document in 'processing' state and return it."""
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"INSERT INTO documents (filename, status) VALUES (%s, 'processing')"
            f" RETURNING {COLUMNS}, 0 AS chunk_count",
            (filename,),
        )
        return cur.fetchone()


def list_documents() -> list[dict]:
    """All documents, newest first, with their chunk counts."""
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT d.id, d.filename, d.status, d.error, d.uploaded_at, COUNT(c.id) AS chunk_count"
            " FROM documents d LEFT JOIN chunks c ON c.document_id = d.id"
            " GROUP BY d.id ORDER BY d.uploaded_at DESC, d.id DESC"
        )
        return cur.fetchall()


def delete_document(document_id: int) -> bool:
    """Delete a document (its chunks go with it). Returns False if it does not exist."""
    with get_connection() as conn:
        row = conn.execute(
            "DELETE FROM documents WHERE id = %s RETURNING id", (document_id,)
        ).fetchone()
    return row is not None


def filename_exists(filename: str) -> bool:
    with get_connection() as conn:
        return conn.execute(
            "SELECT EXISTS (SELECT 1 FROM documents WHERE filename = %s)", (filename,)
        ).fetchone()[0]


def has_ready_documents() -> bool:
    with get_connection() as conn:
        return conn.execute(
            "SELECT EXISTS (SELECT 1 FROM documents WHERE status = 'ready')"
        ).fetchone()[0]


def set_status(document_id: int, status: str, error: str | None = None) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE documents SET status = %s, error = %s WHERE id = %s",
            (status, error, document_id),
        )
