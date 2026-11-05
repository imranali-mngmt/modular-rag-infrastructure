"""
Retriever-only evaluation using DeepEval and Groq.

Evaluates:
1. Contextual Precision
2. Contextual Recall
3. Contextual Relevancy

The existing application retriever is used for retrieval.
Groq is used only as the evaluation LLM (judge).
"""

import os
import sys
import json

from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

from deepeval.models import OpenAIModel

from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)

from deepeval.test_case import LLMTestCase

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

from src.app.api.dependencies.providers import get_retriever

from evals.scripts.dataset_loader import load_golden_dataset

# --------------------------------------------------
# 2. Groq configuration
# --------------------------------------------------


def get_evaluation_model():
    """
    Configure Groq as the DeepEval evaluation judge.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is missing. " "Please configure it in your project .env file."
        )

    model_name = os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile",
    )

    base_url = os.getenv(
        "GROQ_BASE_URL",
        "https://api.groq.com/openai/v1",
    )

    print("\nGroq API Configuration")
    print(f"Model: {model_name}")
    print(f"Base URL: {base_url}")
    print("API Key: Loaded successfully")

    return OpenAIModel(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


# --------------------------------------------------
# 3. Retriever
# --------------------------------------------------


def retrieve_context(retriever, query: str, k: int = 4) -> list[str]:
    """
    Retrieve actual documents using the existing application retriever.
    """

    results = retriever.retrieve(query, k=k)

    contexts = [
        result.content for result in results if getattr(result, "content", None)
    ]

    dropped = len(results) - len(contexts)

    if dropped:
        print(
            f"WARNING: Dropped {dropped} retrieved item(s) "
            "missing a 'content' attribute."
        )

    return contexts


# --------------------------------------------------
# 4. DeepEval metrics
# --------------------------------------------------


def create_metrics(judge_model):

    return [
        ContextualPrecisionMetric(
            threshold=0.5,
            model=judge_model,
            include_reason=True,
        ),
        ContextualRecallMetric(
            threshold=0.5,
            model=judge_model,
            include_reason=True,
        ),
        ContextualRelevancyMetric(
            threshold=0.5,
            model=judge_model,
            include_reason=True,
        ),
    ]


# --------------------------------------------------
# 5. Save evaluation results
# --------------------------------------------------


def save_results(results: dict, dataset_path: str):

    results_dir = PROJECT_ROOT / "evals" / "results"

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_name = Path(dataset_path).stem

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    results_file = results_dir / (f"{dataset_name}_{timestamp}_results.json")

    with results_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nResults saved to: {results_file}")

    return results_file


# --------------------------------------------------
# 6. Main evaluation
# --------------------------------------------------


def evaluate_retriever(
    dataset_path: str,
    k: int = 4,
):
    """
    Run retriever evaluation against a golden dataset.
    """

    if k < 1:
        raise ValueError("k must be greater than zero.")

    print("\n" + "=" * 65)
    print("       RETRIEVER EVALUATION - DEEPEVAL (Groq judge)")
    print("=" * 65)

    # Load dataset
    print("\n[1/5] Loading golden dataset...")

    golden_samples = load_golden_dataset(dataset_path)

    # Initialize evaluation model
    print("\n[2/5] Initializing Groq evaluation model...")

    judge_model = get_evaluation_model()

    # Initialize retriever only once
    print("\n[3/5] Initializing application retriever...")

    retriever = get_retriever()

    print("Retriever initialized successfully.")

    # Create metrics once and reuse them for every sample
    metrics = create_metrics(judge_model)

    # Run retrieval and evaluation
    print("\n[4/5] Running retrieval and evaluation...")

    evaluation_results = []

    metric_totals = {}

    skipped_samples = 0

    for index, golden in enumerate(golden_samples, start=1):

        query = golden["input"]

        print("\n" + "-" * 65)
        print(f"Query [{index}/{len(golden_samples)}]")
        print(query)

        # Retrieve actual context
        actual_context = retrieve_context(
            retriever=retriever,
            query=query,
            k=k,
        )

        # Gracefully skip samples where the retriever returned nothing
        if not actual_context:

            print("WARNING: Retriever returned no documents. Skipping metrics.")

            skipped_samples += 1

            evaluation_results.append(
                {
                    "input": query,
                    "expected_output": golden["expected_output"],
                    "expected_context": golden["retrieval_context"],
                    "retrieved_context": actual_context,
                    "retrieved_count": 0,
                    "metrics": {"error": "No documents retrieved - metrics skipped."},
                }
            )

            continue

        # Construct DeepEval test case
        test_case = LLMTestCase(
            input=query,
            # Placeholder: this is a retriever-only evaluation.
            # We are not evaluating answer generation here.
            actual_output="Retriever-only evaluation.",
            expected_output=golden["expected_output"],
            # Actual documents returned by your retriever
            retrieval_context=actual_context,
            # Ground-truth context from the golden dataset.
            # Note: the three RAG metrics above do NOT read 'context'
            # (they score retrieval_context against expected_output).
            # Kept here for future generation-side metrics (e.g. faithfulness).
            context=golden["retrieval_context"],
        )

        sample_result = {
            "input": query,
            "expected_output": golden["expected_output"],
            "expected_context": golden["retrieval_context"],
            "retrieved_context": actual_context,
            "retrieved_count": len(actual_context),
            "metrics": {},
        }

        # Measure each metric
        for metric in metrics:

            metric_name = metric.__class__.__name__

            print(f"\nEvaluating {metric_name}...")

            # One flaky judge response must not kill the whole run
            try:

                metric.measure(test_case)

            except Exception as exc:

                print(f"ERROR: {metric_name} failed: {exc}")

                sample_result["metrics"][metric_name] = {
                    "score": None,
                    "reason": f"Evaluation failed: {exc}",
                    "threshold": metric.threshold,
                    "passed": False,
                    "error": str(exc),
                }

                continue

            score = metric.score
            reason = metric.reason

            sample_result["metrics"][metric_name] = {
                "score": score,
                "reason": reason,
                "threshold": metric.threshold,
                "passed": (score is not None and score >= metric.threshold),
            }

            if score is not None:
                metric_totals.setdefault(metric_name, []).append(score)

            print(f"Score: {score}")
            print(f"Reason: {reason}")

        evaluation_results.append(sample_result)

    # Calculate averages
    print("\n[5/5] Generating evaluation report...")

    summary = {}

    for metric_name, scores in metric_totals.items():

        average_score = sum(scores) / len(scores) if scores else 0

        summary[metric_name] = {
            "average_score": round(average_score, 4),
            "evaluated_samples": len(scores),
        }

    final_results = {
        "dataset": Path(dataset_path).stem,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_provider": "Groq",
        "evaluation_model": os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile",
        ),
        "retrieval_k": k,
        "total_samples": len(golden_samples),
        "evaluated_samples": len(golden_samples) - skipped_samples,
        "skipped_samples": skipped_samples,
        "summary": summary,
        "samples": evaluation_results,
    }

    save_results(
        results=final_results,
        dataset_path=dataset_path,
    )

    # Print summary
    print("\n" + "=" * 65)
    print("             EVALUATION SUMMARY")
    print("=" * 65)

    if skipped_samples:
        print(f"\nSkipped samples (no documents retrieved): {skipped_samples}")

    for metric_name, values in summary.items():

        print(f"\n{metric_name}")
        print(f"Average Score: {values['average_score']:.4f}")
        print(f"Evaluated Samples: {values['evaluated_samples']}")

    print("\n" + "=" * 65)
    print("Evaluation completed successfully.")
    print("=" * 65)

    return final_results


# --------------------------------------------------
# 7. CLI entry point
# --------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate retriever using DeepEval and Groq."
    )

    parser.add_argument(
        "dataset",
        help="Path to golden dataset JSON.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of documents to retrieve.",
    )

    args = parser.parse_args()

    evaluate_retriever(
        dataset_path=args.dataset,
        k=args.k,
    )
