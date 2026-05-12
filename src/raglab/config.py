from pydantic_settings import BaseSettings, SettingsConfigDict

# Price per 1M tokens — update when pricing changes
# Source: https://openai.com/api/pricing/ and https://docs.anthropic.com/en/docs/about-claude/models
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    openai_api_key: str
    anthropic_api_key: str
    default_model: str = "gpt-4o-mini"
    log_level: str = "INFO"
    logfire_token: str = ""


# Singleton — import this everywhere, don't create new instances
settings = Settings()  # type: ignore[call-arg]
