from unittest.mock import MagicMock, patch

from anthropic.types import TextBlock

from raglab.gateway.anthropic import AnthropicProvider
from raglab.gateway.openai import OpenAIProvider


def test_openai_provider_returns_normalized_response():
    # build a fake OpenAI response object
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello from OpenAI"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.model = "gpt-4o-mini"

    with patch("raglab.gateway.openai.OpenAI") as mock_client_class:
        mock_client_class.return_value.chat.completions.create.return_value = (
            mock_response
        )

        provider = OpenAIProvider(api_key="test-key")
        result = provider.generate(
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-4o-mini",
        )

    assert result.text == "Hello from OpenAI"
    assert result.provider == "openai"
    assert result.input_tokens == 10
    assert result.output_tokens == 20
    assert result.cost_usd >= 0
    assert result.latency_ms > 0


def test_anthropic_provider_returns_normalized_response():
    mock_response = MagicMock()
    mock_response.content = [TextBlock(type="text", text="Hello from Anthropic")]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 20

    with patch("raglab.gateway.anthropic.Anthropic") as mock_client_class:
        mock_client_class.return_value.messages.create.return_value = mock_response

        provider = AnthropicProvider(api_key="test-key")
        result = provider.generate(
            messages=[
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "hello"},
            ],
            model="claude-haiku-4-5-20251001",
        )

    assert result.text == "Hello from Anthropic"
    assert result.provider == "anthropic"
    assert result.input_tokens == 10
    assert result.output_tokens == 20


def test_gateway_normalizes_both_providers_to_same_shape():
    # both providers must return LLMResponse with identical fields
    from raglab.gateway.base import LLMResponse

    mock_openai = MagicMock()
    mock_openai.choices[0].message.content = "answer"
    mock_openai.usage.prompt_tokens = 5
    mock_openai.usage.completion_tokens = 5

    mock_anthropic = MagicMock()
    mock_anthropic.content = [TextBlock(type="text", text="answer")]
    mock_anthropic.usage.input_tokens = 5
    mock_anthropic.usage.output_tokens = 5

    with patch("raglab.gateway.openai.OpenAI") as oai:
        oai.return_value.chat.completions.create.return_value = mock_openai
        openai_result = OpenAIProvider("test").generate(
            [{"role": "user", "content": "hi"}], "gpt-4o-mini"
        )

    with patch("raglab.gateway.anthropic.Anthropic") as ant:
        ant.return_value.messages.create.return_value = mock_anthropic
        anthropic_result = AnthropicProvider("test").generate(
            [{"role": "user", "content": "hi"}], "claude-haiku-4-5-20251001"
        )

    # both results must be LLMResponse instances with same fields
    assert isinstance(openai_result, LLMResponse)
    assert isinstance(anthropic_result, LLMResponse)
    assert set(openai_result.__class__.model_fields) == set(
        anthropic_result.__class__.model_fields
    )
