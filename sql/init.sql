CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    filename    TEXT NOT NULL,
    uploaded_at TIMESTAMPTZ DEFAULT now(),
    -- 'processing' while the background ingest runs, then 'ready' or 'failed'
    status      TEXT NOT NULL DEFAULT 'ready',
    error       TEXT
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

-- Approximate nearest-neighbour index for cosine distance (<=>)
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Full-text search for hybrid retrieval. IF NOT EXISTS so that running this file again
-- also upgrades a database created before the column existed.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS chunks_content_tsv ON chunks USING gin (content_tsv);
