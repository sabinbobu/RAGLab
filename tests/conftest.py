import pytest

from raglab.gateway.base import LLMResponse


@pytest.fixture
def fake_llm_response() -> LLMResponse:
    return LLMResponse(
        text="This is a test answer.",
        input_tokens=10,
        output_tokens=20,
        cost_usd=0.000005,
        latency_ms=123.45,
        model="gpt-4o-mini",
        provider="openai",
    )
