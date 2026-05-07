from pathlib import Path

import yaml


def load_prompt(version: str) -> dict:
    """
    Load a prompt template by version string from the prompts directory.

    Args:
        version: prompt version string e.g. "v1"

    Returns:
        dict with keys: version, system, user
    """
    # prompts live next to this __init__.py file
    prompts_dir = Path(__file__).parent
    prompt_file = prompts_dir / f"rag_{version}.yaml"

    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt version '{version}' not found at {prompt_file}"
        )

    with open(prompt_file, encoding="utf-8") as f:
        return yaml.safe_load(f)
