"""Cross-encoder reranking: score each (question, chunk) pair together.

Slower than comparing two precomputed embeddings, so it only reorders a short candidate list.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import settings


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    return CrossEncoder(settings.rerank_model)


def rerank(question: str, chunks: list[dict]) -> list[dict]:
    """Return the chunks ordered by cross-encoder relevance, best first. `score` is kept."""
    if not chunks:
        return []
    scores = _model().predict([(question, c["content"]) for c in chunks], show_progress_bar=False)
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in order]
