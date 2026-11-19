CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    filename    TEXT NOT NULL,
    uploaded_at TIMESTAMPTZ DEFAULT now()
);

-- all-MiniLM-L6-v2 produces 384-dimensional vectors
CREATE TABLE IF NOT EXISTS chunks (
    id           SERIAL PRIMARY KEY,
    document_id  INT REFERENCES documents(id) ON DELETE CASCADE,
    page         INT,
    chunk_index  INT,
    content      TEXT NOT NULL,
    embedding    vector(384)
);
