import json
from pathlib import Path
from typing import Any


def load_golden_dataset(file_path: str) -> list[dict[str, Any]]:
    """
    Load and validate a golden evaluation dataset.

    Expected structure:
    {
        "name": "dataset_name",
        "golden_samples": [
            {
                "input": "question",
                "expected_output": "expected answer",
                "retrieval_context": ["expected relevant context"]
            }
        ]
    }
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in dataset: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Dataset must be a JSON object.")

    samples = data.get("golden_samples", [])

    if not isinstance(samples, list) or not samples:
        raise ValueError("Dataset must contain a non-empty golden_samples list.")

    for index, sample in enumerate(samples, start=1):

        if not isinstance(sample, dict):
            raise ValueError(f"Sample {index} must be a JSON object.")

        if not sample.get("input"):
            raise ValueError(f"Sample {index} is missing input.")

        if not sample.get("expected_output"):
            raise ValueError(f"Sample {index} is missing expected_output.")

        context = sample.get("retrieval_context")

        if not isinstance(context, list) or not context:
            raise ValueError(
                f"Sample {index} must contain a non-empty " "retrieval_context list."
            )

        if not all(isinstance(chunk, str) and chunk.strip() for chunk in context):
            raise ValueError(f"Sample {index} contains an invalid context chunk.")

    print(
        f"Loaded {len(samples)} golden samples " f"from '{data.get('name', 'unnamed')}'"
    )

    return samples
