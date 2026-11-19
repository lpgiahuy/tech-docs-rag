import psycopg
from pgvector.psycopg import register_vector

from app.config import settings


def get_connection() -> psycopg.Connection:
    """Open a PostgreSQL connection with pgvector types registered."""
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn
