from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class AskRequest(BaseModel):
    # The example is what Swagger UI pre-fills in "Try it out".
    question: Question = Field(
        examples=["How many IP addresses does AWS reserve in each VPC subnet?"]
    )


class Source(BaseModel):
    filename: str
    page: int
    score: float
    snippet: str
    neighbor: bool = False  # added next to a hit, not found by the search itself


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
