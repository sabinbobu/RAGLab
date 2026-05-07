from typing import Protocol

from pydantic import BaseModel, ConfigDict


class RetrievedChunk(BaseModel):
    """Represents a single retrieved chunk with its metadata."""

    model_config = ConfigDict(frozen=True)

    text: str
    source: str
    page: int
    score: float


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]: ...
