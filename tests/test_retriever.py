"""Fusion, mode selection and reranking, with the database and the models replaced by fakes."""

import pytest

from app import reranker, retriever
from app.config import settings


def chunk(i):
    return {"id": i, "content": f"chunk {i}", "page": i, "filename": "a.pdf", "score": 0.5}


def ids(chunks):
    return [c["id"] for c in chunks]


def test_rrf_rewards_chunks_found_by_both_lists():
    vector = [chunk(1), chunk(2), chunk(3)]
    keyword = [chunk(3), chunk(4)]
    # 3: 1/63 + 1/61; 1: 1/61; 2 and 4: 1/62 (a tie keeps first-seen order)
    assert ids(retriever.rrf([vector, keyword])) == [3, 1, 2, 4]


def test_rrf_with_one_empty_list_keeps_the_other_order():
    assert ids(retriever.rrf([[chunk(5), chunk(2)], []])) == [5, 2]


@pytest.fixture
def depths(monkeypatch):
    """Replace both searches and the reranker; return the depth each search was asked for."""
    calls = {}

    def vector(query_vector, k):
        calls["vector"] = k
        return [chunk(1), chunk(2)]

    def keyword(question, query_vector, k):
        calls["keyword"] = k
        return [chunk(2), chunk(3)]

    monkeypatch.setattr(retriever, "embed", lambda texts: [None])
    monkeypatch.setattr(retriever, "vector_search", vector)
    monkeypatch.setattr(retriever, "keyword_search", keyword)
    monkeypatch.setattr(retriever, "rerank", lambda q, chunks: chunks[::-1])
    monkeypatch.setattr(settings, "candidate_k", 20)
    return calls


@pytest.mark.parametrize(
    "mode, rerank, expected, asked",
    [
        ("vector", False, [1, 2], {"vector": 2}),
        ("keyword", False, [2, 3], {"keyword": 2}),
        ("hybrid", False, [2, 1], {"vector": 20, "keyword": 20}),
        ("hybrid", True, [3, 1], {"vector": 20, "keyword": 20}),
        ("vector", True, [2, 1], {"vector": 20}),
    ],
)
def test_search_follows_settings(monkeypatch, depths, mode, rerank, expected, asked):
    monkeypatch.setattr(settings, "retrieval_mode", mode)
    monkeypatch.setattr(settings, "rerank", rerank)
    assert ids(retriever.search("q", k=2)) == expected
    assert depths == asked


def test_rerank_orders_by_cross_encoder_score(monkeypatch):
    class FakeModel:
        def predict(self, pairs, **kwargs):
            return [0.1, 2.0, -1.0]

    monkeypatch.setattr(reranker, "_model", lambda: FakeModel())
    assert ids(reranker.rerank("q", [chunk(1), chunk(2), chunk(3)])) == [2, 1, 3]
    assert reranker.rerank("q", []) == []
