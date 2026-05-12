import json
import os
from itertools import groupby
from typing import Union

from pydantic import BaseModel
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Faithfulness
from sqlmodel import Session, select

from raglab.config import settings
from raglab.experiments.models import RunResult
from raglab.experiments.runner import engine


class Scorecard(BaseModel):
    model: str
    prompt_version: str
    faithfulness: float
    avg_cost_usd: float
    avg_latency_ms: float
    run_count: int


# Ragas creates its own OpenAI client internally — it reads from env directly
os.environ["OPENAI_API_KEY"] = settings.openai_api_key


def evaluate_experiment(experiment_id: str) -> list[Scorecard]:
    """
    Load all runs for an experiment from SQLite,
    score them with Ragas,
    return one Scorecard per (model, prompt_version) combination.
    """
    # load all runs for this experiment from SQLite
    with Session(engine) as session:
        runs = session.exec(
            select(RunResult).where(RunResult.experiment_id == experiment_id)
        ).all()

    if not runs:
        raise ValueError(f"No runs found for experiment_id: {experiment_id}")

    # group runs by (model, prompt_version) - one scorecard per combination
    def group_key(r: RunResult) -> tuple[str, str]:
        return (r.model, r.prompt_version)

    sorted_runs = sorted(runs, key=group_key)
    scorecards = []

    for (model, prompt_version), group in groupby(sorted_runs, key=group_key):
        group_runs = list(group)

        # build Ragas dataset from this group
        samples: list[Union[SingleTurnSample, object]] = []
        for run in group_runs:
            samples.append(
                SingleTurnSample(
                    user_input=run.question,
                    response=run.answer,
                    # deserialize JSON string back to list of strings
                    retrieved_contexts=json.loads(run.contexts),
                )
            )

        dataset = EvaluationDataset(samples=samples)  # type: ignore[arg-type]

        # run Ragas evaluation — this makes LLM calls internally
        results = evaluate(
            dataset=dataset,
            metrics=[Faithfulness()],
        )

        # results.to_pandas() gives a DataFrame — extract mean per metric
        df = results.to_pandas()  # type: ignore[union-attr]

        scorecards.append(
            Scorecard(
                model=model,
                prompt_version=prompt_version,
                faithfulness=round(df["faithfulness"].mean(), 4),
                avg_cost_usd=round(
                    sum(r.cost_usd for r in group_runs) / len(group_runs), 6
                ),
                avg_latency_ms=round(
                    sum(r.latency_ms for r in group_runs) / len(group_runs), 2
                ),
                run_count=len(group_runs),
            )
        )

    return scorecards
