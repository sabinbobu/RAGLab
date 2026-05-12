from raglab.experiments.runner import ExperimentConfig


def test_experiment_config_defaults():
    config = ExperimentConfig(
        models=["gpt-4o-mini"],
        prompt_versions=["v1"],
        questions=["What is RAG?"],
    )
    assert config.top_k == 5
    assert config.provider == "openai"
    assert config.retriever == "chroma"


def test_experiment_config_matrix_size():
    config = ExperimentConfig(
        models=["gpt-4o-mini", "claude-haiku-4-5-20251001"],
        prompt_versions=["v1", "v2"],
        questions=["Q1", "Q2", "Q3"],
    )
    import itertools

    combinations = list(
        itertools.product(config.models, config.prompt_versions, config.questions)
    )
    # 2 models x 2 prompts x 3 questions = 12
    assert len(combinations) == 12
