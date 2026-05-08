import time

from anthropic import Anthropic

from raglab.config import MODEL_PRICING
from raglab.gateway.base import LLMResponse


class AnthropicProvider:
    def __init__(self, api_key: str) -> None:
        self.client = Anthropic(api_key=api_key)

    def generate(self, messages: list[dict], model: str) -> LLMResponse:
        start = time.perf_counter()

        # Anthropic requires system prompt as a separate parameter
        # extract it from the messages list before calling the SDK
        system = next(
            (m["content"] for m in messages if m["role"] == "system"),
            "You are a helpful assistant.",  # fallback if no system message
        )

        # filter out system messages — Anthropic only accepts user/assistant roles
        # in the messages list
        user_messages = [m for m in messages if m["role"] != "system"]

        response = self.client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=user_messages,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost_usd = (
            input_tokens * pricing["input"] + output_tokens * pricing["output"]
        ) / 1_000_000

        return LLMResponse(
            text=response.content[0].text or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=round(latency_ms, 2),
            model=model,
            provider="anthropic",
        )
