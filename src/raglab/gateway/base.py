from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    model: str
    provider: str


class LLMProvider(Protocol):
    def generate(self, messages: list[dict[str, Any]], model: str) -> LLMResponse: ...
