from functools import lru_cache

from google import genai
from google.genai import types

from app.config import settings

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
        raise RuntimeError("LLM_API_KEY is not set (put it in .env)")
    return genai.Client(
        api_key=settings.llm_api_key,
        http_options=types.HttpOptions(api_version="v1beta", timeout=60_000),
    )


def build_context(chunks: list[dict]) -> str:
    """Format chunks as labelled blocks so the model can cite [filename, page N]."""
    return "\n\n".join(
        f"[{c['filename']}, page {c['page']}]\n{c['content']}" for c in chunks
    )


def generate(question: str, chunks: list[dict]) -> str:
    """Answer the question from the retrieved chunks, with citations.

    Refuses without calling the LLM when no chunk is relevant enough.
    """
    relevant = [c for c in chunks if c["score"] >= settings.min_score]
    if not relevant:
        return REFUSAL

    prompt = f"Context:\n{build_context(relevant)}\n\nQuestion: {question}"
    response = _client().models.generate_content(
        model=settings.llm_model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0),
    )
    return (response.text or "").strip() or REFUSAL
