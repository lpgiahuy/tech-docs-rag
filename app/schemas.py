from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, StringConstraints

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class AskRequest(BaseModel):
    question: Question


class Source(BaseModel):
    filename: str
    page: int
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: int


class DocumentOut(BaseModel):
    id: int
    filename: str
    status: str  # processing | ready | failed
    error: str | None = None
    uploaded_at: datetime
    chunk_count: int
