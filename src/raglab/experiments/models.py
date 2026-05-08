from datetime import datetime

from sqlmodel import Field, SQLModel


class RunResult(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    experiment_id: str
    model: str
    prompt_version: str
    question: str
    contexts: str = Field(
        default="[]"
    )  # stored as JSON string — list[str] serialized with json.dumps
    answer: str
    cost_usd: float
    latency_ms: float
    created_at: datetime
