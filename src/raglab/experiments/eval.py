class Scorecard:
    # fields: model, prompt_version, faithfulness, answer_relevance,
    #         context_precision, avg_cost_usd, avg_latency_ms, run_count
    pass


def evaluate_experiment(experiment_id: str) -> list[Scorecard]:
    """
    Load all runs for an experiment from SQLite,
    score them with Ragas,
    return one Scorecard per (model, prompt_version) combination.
    """
    raise NotImplementedError
