from pydantic import BaseModel
from sqlmodel import SQLModel, create_engine

from raglab.experiments.models import RunResult

DATABASE_URL = "sqlite:///raglab.db"
engine = create_engine(DATABASE_URL)


def init_db() -> None:
    """Create all tables if they don't exist."""
    SQLModel.metadata.create_all(engine)


class ExperimentConfig(BaseModel):
    models: list[str]
    prompt_versions: list[str]
    questions: list[str]
    top_k: int = 5
    provider: str = "openai"


def run_experiment(config: ExperimentConfig) -> list[RunResult]:
    """
    Run a matrix of models x prompt_versions x questions.
    Each combination is one run, stored as a RunResult in SQLite.

    Args:
        config: experiment configuration

    Returns:
        list of RunResult — one per combination
    """
    raise NotImplementedError
