"""
Run retriever evaluation with default configuration.

Usage:
    python -m evals.scripts.run_eval

Custom top-k:
    python -m evals.scripts.run_eval --k 5
"""

import argparse

from pathlib import Path

from evals.scripts.evaluate_retriever import evaluate_retriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET = PROJECT_ROOT / "evals" / "datasets" / "golden_dataset.json"


def main():

    parser = argparse.ArgumentParser(description="Run RAG retriever evaluation.")

    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help="Path to golden dataset.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of documents to retrieve.",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():

        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    print("\nStarting retriever evaluation...")

    evaluate_retriever(
        dataset_path=str(dataset_path),
        k=args.k,
    )


if __name__ == "__main__":
    main()
