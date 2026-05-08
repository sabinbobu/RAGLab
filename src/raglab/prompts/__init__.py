from pathlib import Path

import yaml


def load_prompt(version: str) -> dict:
    """
    Load and validate a prompt template by version string.

    Args:
        version: prompt version string e.g. "v1"

    Returns:
        dict with keys: version, system, user
    """
    prompts_dir = Path(__file__).parent
    prompt_file = prompts_dir / f"rag_{version}.yaml"

    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt version '{version}' not found at {prompt_file}"
        )

    with open(prompt_file, encoding="utf-8") as f:
        prompt = yaml.safe_load(f)

    _validate_prompt(prompt, version)

    return prompt


def _validate_prompt(prompt: dict, version: str) -> None:
    """
    Validate that a prompt template has all required keys.

    Args:
        prompt: loaded prompt dict
        version: version string for error messages

    Raises:
        ValueError if required keys are missing
    """
    required_keys = {"version", "system", "user"}
    missing = required_keys - prompt.keys()

    if missing:
        raise ValueError(f"Prompt '{version}' is missing required keys: {missing}")

    # ensure placeholders exist in user template
    # both {context} and {question} must be present for RAG to work
    for placeholder in ["{context}", "{question}"]:
        if placeholder not in prompt["user"]:
            raise ValueError(
                f"Prompt '{version}' user template missing placeholder: {placeholder}"
            )
