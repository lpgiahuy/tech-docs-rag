"""Fallback between LLM models, with a fake client (no network)."""

from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

from app import generator
from app.config import settings

CHUNKS = [{"filename": "a.pdf", "page": 1, "content": "text", "score": 0.9}]


def api_error(code: int) -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"code": code, "message": "boom"}})


@pytest.fixture
def fake_llm(monkeypatch):
    """Set the outcome per model; returns the list of models that were called."""
    monkeypatch.setattr(settings, "llm_model", "primary")
    monkeypatch.setattr(settings, "llm_fallback_models", "second, primary ,third")
    outcomes, calls = {}, []

    def generate_content(model, **kwargs):
        calls.append(model)
        outcome = outcomes.get(model, f"answer from {model}")
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(generator, "_client", lambda: client)
    return outcomes, calls


def test_primary_success_calls_no_fallback(fake_llm):
    _, calls = fake_llm
    assert generator.generate("q", CHUNKS) == "answer from primary"
    assert calls == ["primary"]


@pytest.mark.parametrize("error", [api_error(429), api_error(503), httpx.ReadTimeout("slow")])
def test_falls_back_on_quota_overload_and_timeout(fake_llm, error):
    outcomes, calls = fake_llm
    outcomes["primary"] = error
    assert generator.generate("q", CHUNKS) == "answer from second"
    assert calls == ["primary", "second"]


def test_bad_request_does_not_fall_back(fake_llm):
    outcomes, calls = fake_llm
    outcomes["primary"] = api_error(400)
    with pytest.raises(generator.LLMError):
        generator.generate("q", CHUNKS)
    assert calls == ["primary"]


def test_all_models_failing_raises_llm_error(fake_llm):
    outcomes, calls = fake_llm
    outcomes.update({name: api_error(429) for name in ("primary", "second", "third")})
    with pytest.raises(generator.LLMError):
        generator.generate("q", CHUNKS)
    assert calls == ["primary", "second", "third"]
