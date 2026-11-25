import logging
from functools import lru_cache

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.config import settings

logger = logging.getLogger("app.generator")

# Another model may succeed: not found, quota exhausted, or the service is failing.
FALLBACK_STATUS = {404, 429, 500, 502, 503, 504}


class LLMError(Exception):
    """The LLM could not produce an answer (quota, overload, timeout, missing key...)."""


REFUSAL = "I could not find this information in the uploaded documents."

SYSTEM_PROMPT = f"""You answer questions about technical documents using ONLY the context provided.

Rules:
1. Use only facts stated in the context. Do not use outside knowledge, even if you know the answer.
2. If the context does not contain the answer, reply exactly: "{REFUSAL}"
3. Cite every fact with its source in the form [filename, page N], using the labels given in the context.
4. Be concise. Do not invent page numbers, values or filenames.
5. Answer in the same language as the question."""


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    if not settings.llm_api_key:
        raise LLMError("LLM_API_KEY is not set (put it in .env)")
    return genai.Client(
        api_key=settings.llm_api_key,
        http_options=types.HttpOptions(api_version="v1beta", timeout=60_000),
    )


def build_context(chunks: list[dict]) -> str:
    """Format chunks as labelled blocks so the model can cite [filename, page N]."""
    return "\n\n".join(
        f"[{c['filename']}, page {c['page']}]\n{c['content']}" for c in chunks
    )


def filter_relevant(chunks: list[dict]) -> list[dict]:
    """Keep only chunks whose score reaches `min_score`."""
    return [c for c in chunks if c["score"] >= settings.min_score]


def model_chain() -> list[str]:
    """The primary model followed by the fallbacks, without blanks or repeats."""
    names = [settings.llm_model, *settings.llm_fallback_models.split(",")]
    return list(dict.fromkeys(name.strip() for name in names if name.strip()))


def _call_llm(prompt: str):
    """Call the first model of the chain that works. Raises LLMError when none does.

    Errors another model cannot fix (bad request, bad key) are raised at once.
    """
    models = model_chain()
    for position, model in enumerate(models, start=1):
        try:
            return _client().models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0),
            )
        except (genai_errors.APIError, httpx.HTTPError) as exc:
            retryable = isinstance(exc, httpx.HTTPError) or exc.code in FALLBACK_STATUS
            if not retryable or position == len(models):
                raise LLMError(f"{type(exc).__name__}: {exc}") from exc
            logger.warning("model %s failed (%s), trying %s", model, type(exc).__name__, models[position])


def generate(question: str, chunks: list[dict]) -> str:
    """Answer the question from the retrieved chunks, with citations.

    Refuses without calling the LLM when no chunk is relevant enough.
    Raises LLMError when the LLM call fails.
    """
    relevant = filter_relevant(chunks)
    if not relevant:
        return REFUSAL

    prompt = f"Context:\n{build_context(relevant)}\n\nQuestion: {question}"
    response = _call_llm(prompt)
    return (response.text or "").strip() or REFUSAL
