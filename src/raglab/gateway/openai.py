import time
from typing import Any

import logfire
from openai import OpenAI

from raglab.config import MODEL_PRICING
from raglab.gateway.base import LLMResponse


class OpenAIProvider:
    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key)

    def generate(self, messages: list[dict[str, Any]], model: str) -> LLMResponse:
        start = time.perf_counter()

        with logfire.span("openai.chat.completions", model=model):
            response = self.client.chat.completions.create(
                model=model,
                # OpenAI accepts messages list natively — system + user roles work as-is
                messages=messages,  # type: ignore[arg-type]
            )

        latency_ms = (time.perf_counter() - start) * 1000

        assert response.usage is not None
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost_usd = (
            input_tokens * pricing["input"] + output_tokens * pricing["output"]
        ) / 1_000_000

        return LLMResponse(
            text=response.choices[0].message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=round(latency_ms, 2),
            model=model,
            provider="openai",
        )
