import itertools
import json
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel
from sqlmodel import Session, SQLModel, create_engine

from raglab.experiments.models import RunResult
from raglab.gateway.factory import get_provider
from raglab.prompts import load_prompt
from raglab.retrieval.factory import get_retriever

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
    retriever: str = "chroma"


def run_experiment(config: ExperimentConfig) -> list[RunResult]:
    """
    Run a matrix of models x prompt_versions x questions.
    Each combination is one run, stored as a RunResult in SQLite.

    Args:
        config: experiment configuration

    Returns:
        list of RunResult — one per combination
    """
    init_db()

    # unique ID groups all runs from this experiment together
    # so you can query "show me all runs from experiment X"
    experiment_id = str(uuid4())

    retriever = get_retriever(config.retriever, collection="raglab")
    results = []

    # itertools.product expands the matrix
    # e.g. [gpt-4o-mini, claude] x [v1, v2] x [Q1, Q2]
    # → (gpt-4o-mini, v1, Q1), (gpt-4o-mini, v1, Q2) ...
    combinations = list(
        itertools.product(config.models, config.prompt_versions, config.questions)
    )

    print(f"Running {len(combinations)} combinations...")

    for model, prompt_version, question in combinations:
        print(f"  → {model} | {prompt_version} | {question[:50]}")

        try:
            # retrieve relevant chunks for this question
            chunks = retriever.retrieve(question, top_k=config.top_k)
            contexts = [chunk.text for chunk in chunks]

            if not chunks:
                print("    ⚠ No chunks found, skipping.")
                continue

            # load and fill prompt template
            prompt_template = load_prompt(prompt_version)
            context = "\n\n".join(
                f"[{chunk.source}, page {chunk.page}]\n{chunk.text}" for chunk in chunks
            )
            user_content = prompt_template["user"].format(
                context=context,
                question=question,
            )

            messages = [
                {"role": "system", "content": prompt_template["system"]},
                {"role": "user", "content": user_content},
            ]

            # call LLM via gateway
            provider = get_provider(config.provider)
            llm_response = provider.generate(messages, model)

            run = RunResult(
                experiment_id=experiment_id,
                model=model,
                prompt_version=prompt_version,
                question=question,
                answer=llm_response.text,
                cost_usd=llm_response.cost_usd,
                latency_ms=llm_response.latency_ms,
                contexts=json.dumps(contexts),
                # timezone-aware UTC timestamp
                created_at=datetime.now(timezone.utc),
            )

            # persist to SQLite immediately — don't wait for all runs to finish
            # if the experiment crashes halfway, you keep the completed runs
            with Session(engine) as session:
                session.add(run)
                session.commit()
                session.refresh(run)

            results.append(run)

        except Exception as e:
            # log and continue — one failed run shouldn't kill the experiment
            print(f"    ✗ Failed: {e}")
            continue

    print(f"Done. {len(results)}/{len(combinations)} runs completed.")
    return results
