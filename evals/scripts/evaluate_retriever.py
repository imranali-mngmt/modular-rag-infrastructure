"""
script:
sifr is script sy run hota hai
    python -m evals.scripts.evaluate_retriever
    )


    python -m evals.scripts.evaluate_hallucination --top-k 6 --threshold 0.4
python -m evals.scripts.evaluate_prompt_injection            # built-in attacks chalti hain agar dataset file nahi hai
python -m evals.scripts.evaluate_latency --runs 5 --max-p95 20.0
"""

import argparse
import json
from dataclasses import dataclass, field

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple

from deepeval import evaluate
from deepeval.metrics import (
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase


import os
from openai import OpenAI
from deepeval.models import DeepEvalBaseLLM

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALS_DIR = PROJECT_ROOT / "evals"
DATASET_DIR = EVALS_DIR / "datasets"
RESULTS_DIR = EVALS_DIR / "results"

DEFAULT_DATASET_PATH = DATASET_DIR / "golden_dataset.json"

# ============================================================
# CONFIGURATION
# ============================================================


class Mercury3Judge(DeepEvalBaseLLM):

    def __init__(self):
        api_key = os.getenv("INCEPTION_API_KEY")

        if not api_key:
            raise ValueError("INCEPTION_API_KEY not found in environment.")

        self.client = OpenAI(
            api_key=api_key, base_url="https://api.inceptionlabs.ai/v1/"
        )

    def load_model(self):
        return self.client

    def get_model_name(self):
        return "mercury-3"

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="mercury-3",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model="mercury-3",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content or ""


@dataclass
class RetrieverEvalConfig:
    """All knobs for a retriever evaluation run."""

    judge_model: DeepEvalBaseLLM = field(default_factory=Mercury3Judge)
    dataset_path: Path = DEFAULT_DATASET_PATH
    results_dir: Path = RESULTS_DIR
    top_k: int = 4
    metric_threshold: float = 0.7
    verbose: bool = True


class Retriever(Protocol):
    """Minimal interface the evaluation expects from a retrieval service."""

    def retrieve(self, query: str, k: int) -> Any: ...


ChunkExtractor = Callable[[Any], List[str]]


def default_chunk_extractor(response: Any) -> List[str]:
    """Extract non-empty chunk texts from a retrieval service response."""
    return [
        result.content
        for result in response.results
        if getattr(result, "content", None)
    ]


def _log(config: RetrieverEvalConfig, message: str) -> None:
    if config.verbose:
        print(message)


# ============================================================
# DATASET LOADING
# ============================================================


def load_golden_dataset(dataset_path: Path) -> Tuple[str, List[dict]]:
    """Load and validate the golden dataset. Returns (name, samples)."""

    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    golden_samples = dataset.get("golden_samples", [])

    if not golden_samples:
        raise ValueError(f"Golden dataset contains no samples: {dataset_path}")

    dataset_name = dataset.get("name", dataset_path.stem)

    return dataset_name, golden_samples


# ============================================================
# METRICS
# ============================================================


def build_default_metrics(config: RetrieverEvalConfig) -> List[BaseMetric]:
    """Default metric suite: contextual recall + contextual precision."""
    return [
        ContextualRecallMetric(
            threshold=config.metric_threshold, model=config.judge_model
        ),
        ContextualPrecisionMetric(
            threshold=config.metric_threshold, model=config.judge_model
        ),
    ]


# ============================================================
# TEST CASE BUILDING
# ============================================================


def build_test_cases(
    golden_samples: Sequence[dict],
    retriever: Retriever,
    config: RetrieverEvalConfig,
    chunk_extractor: ChunkExtractor = default_chunk_extractor,
) -> Tuple[List[LLMTestCase], List[dict], List[dict]]:
    """
    Retrieve documents for every golden sample.

    Returns:
        test_cases: DeepEval test cases ready for evaluation.
        detailed_results: Per-question retrieval details (metrics filled in later).
        failures: Questions that could not be retrieved.
    """

    test_cases: List[LLMTestCase] = []
    detailed_results: List[dict] = []
    failures: List[dict] = []

    total = len(golden_samples)

    for idx, sample in enumerate(golden_samples, start=1):

        question = sample.get("input")
        golden_context = sample.get("retrieval_context", [])

        if not question:
            failures.append(
                {"question": None, "reason": "Missing input question in dataset."}
            )
            continue

        if not golden_context:
            failures.append(
                {
                    "question": question,
                    "reason": "Missing retrieval_context in dataset.",
                }
            )
            continue

        _log(config, f"[{idx}/{total}] Retrieving: {question[:70]}...")

        try:

            response = retriever.retrieve(question, k=config.top_k)
            retrieved_chunks = chunk_extractor(response)

            if not retrieved_chunks:
                failures.append(
                    {
                        "question": question,
                        "reason": "Retriever returned no usable document chunks.",
                    }
                )
                _log(config, "   WARNING: No documents retrieved.")
                continue

            test_cases.append(
                LLMTestCase(
                    input=question,
                    actual_output=" ".join(retrieved_chunks),
                    expected_output=" ".join(golden_context),
                    retrieval_context=retrieved_chunks,
                )
            )

            detailed_results.append(
                {
                    "question": question,
                    "golden_context": golden_context,
                    "actual_retrieved_context": retrieved_chunks,
                    "retrieved_chunks_count": len(retrieved_chunks),
                    "status": "PENDING",
                }
            )

            _log(config, f"   Retrieved {len(retrieved_chunks)} chunks.")

        except Exception as error:
            _log(config, f"   Retrieval failed: {error}")
            failures.append(
                {"question": question, "reason": f"Retrieval Error: {error}"}
            )

    return test_cases, detailed_results, failures


# ============================================================
# RESULT PROCESSING
# ============================================================


def process_results(
    results: Any,
    detailed_results: List[dict],
    failures: List[dict],
) -> List[dict]:
    """Merge DeepEval metric scores back into the per-question details."""

    evaluated_results: List[dict] = []

    for i, test_result in enumerate(results.test_results):

        detail = detailed_results[i]
        metrics_data: dict = {}
        case_failures: List[dict] = []

        for metric_result in test_result.metrics_data:

            score = metric_result.score

            metrics_data[metric_result.name] = {
                "score": round(score, 4) if score is not None else None,
                "passed": metric_result.success,
                "reason": metric_result.reason,
            }

            if not metric_result.success:
                case_failures.append(
                    {
                        "metric": metric_result.name,
                        "score": score,
                        "reason": metric_result.reason,
                    }
                )

        detail["metrics"] = metrics_data
        detail["overall_status"] = "PASS" if test_result.success else "FAIL"

        evaluated_results.append(detail)

        if case_failures:
            failures.append({"question": detail["question"], "reasons": case_failures})

    return evaluated_results


def count_retrieval_failures(failures: Sequence[dict]) -> int:
    """Count failures caused by the retrieval step (not metric failures)."""
    return len(
        [
            failure
            for failure in failures
            if "Retrieval Error" in failure.get("reason", "")
            or "No usable document" in failure.get("reason", "")
        ]
    )


def build_summary(
    dataset_name: str,
    config: RetrieverEvalConfig,
    total_golden_samples: int,
    total_cases: int,
    evaluated_results: List[dict],
    failures: List[dict],
    pass_rate: float,
) -> dict:

    return {
        "evaluation_type": "RETRIEVER",
        "dataset_name": dataset_name,
        "dataset_path": str(config.dataset_path),
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "top_k": config.top_k,
            "metric_threshold": config.metric_threshold,
            "judge_model": "mercury-3",
        },
        "total_golden_samples": total_golden_samples,
        "total_cases": total_cases,
        "failed_retrievals": count_retrieval_failures(failures),
        "pass_rate": f"{pass_rate * 100:.1f}%",
        "detailed_results": evaluated_results,
        "failures": failures,
    }


# ============================================================
# OUTPUT
# ============================================================


def save_results(summary: dict, results_dir: Path) -> Path:

    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"retriever_eval_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    return output_file


def print_report(summary: dict, output_file: Path) -> None:

    print("\n" + "=" * 60)
    print("RETRIEVER EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Dataset:            {summary['dataset_name']}")
    print(f"Evaluated cases:    {summary['total_cases']}")
    print(f"Retrieval failures: {summary['failed_retrievals']}")
    print(f"Pass rate:          {summary['pass_rate']}")
    print(f"\nResults saved to:\n{output_file}")
    print("=" * 60 + "\n")


# ============================================================
# ORCHESTRATION
# ============================================================


def run_retriever_evaluation(
    retriever: Optional[Retriever] = None,
    config: Optional[RetrieverEvalConfig] = None,
    metrics: Optional[Sequence[BaseMetric]] = None,
    chunk_extractor: ChunkExtractor = default_chunk_extractor,
) -> dict:
    """
    Run a full retriever evaluation.

    Args:
        retriever: Retrieval service to evaluate. Defaults to the app's
            retrieval service (imported lazily so this module can be reused
            without the app package).
        config: Evaluation settings (paths, top_k, thresholds).
        metrics: DeepEval metrics to run. Defaults to contextual recall and
            contextual precision built from the config.
        chunk_extractor: Extracts chunk texts from a retriever response,
            allowing reuse with different retrieval service implementations.

    Returns:
        The evaluation summary dict (also written to disk as JSON).
    """

    config = config or RetrieverEvalConfig()

    _log(config, "\nStarting standalone retriever evaluation...\n")

    # 1. Load golden dataset.
    _log(config, "Loading golden dataset...")
    dataset_name, golden_samples = load_golden_dataset(config.dataset_path)
    _log(config, f"   Dataset: {dataset_name}")
    _log(config, f"   Total samples: {len(golden_samples)}")
    _log(config, f"   Dataset path: {config.dataset_path}\n")

    # 2. Resolve retriever.
    if retriever is None:
        _log(config, "Initializing default retrieval service...")
        from app.api.dependencies.providers import get_retrieval_service

        retriever = get_retrieval_service()
        _log(config, "   Retrieval service initialized successfully.\n")

    # 3. Resolve metrics.
    if metrics is None:
        metrics = build_default_metrics(config)

    # 4. Retrieve documents and build test cases.
    _log(config, "Starting document retrieval...\n")

    test_cases, detailed_results, failures = build_test_cases(
        golden_samples=golden_samples,
        retriever=retriever,
        config=config,
        chunk_extractor=chunk_extractor,
    )

    if not test_cases:
        raise RuntimeError(
            "Evaluation stopped because no test cases were created. "
            "Check your retriever and golden dataset."
        )

    _log(config, "\n" + "=" * 60)
    _log(config, "RETRIEVAL SUMMARY")
    _log(config, "=" * 60)
    _log(config, f"Total golden samples: {len(golden_samples)}")
    _log(config, f"Successful retrievals: {len(test_cases)}")
    _log(config, f"Failed retrievals: {len(failures)}")

    # 5. Run DeepEval metrics.
    _log(config, "\nRunning DeepEval retriever metrics...\n")

    results = evaluate(test_cases=test_cases, metrics=list(metrics))

    # 6. Merge metric scores into per-question details.
    _log(config, "\nProcessing evaluation results...")

    evaluated_results = process_results(results, detailed_results, failures)

    # 7. Build summary.
    summary = build_summary(
        dataset_name=dataset_name,
        config=config,
        total_golden_samples=len(golden_samples),
        total_cases=len(test_cases),
        evaluated_results=evaluated_results,
        failures=failures,
        pass_rate=results.pass_rate,
    )

    # 8. Save report.
    output_file = save_results(summary, config.results_dir)

    # 9. Final report.
    if config.verbose:
        print_report(summary, output_file)

    return summary


# ============================================================
# CLI ENTRY POINT
# ============================================================


def parse_args() -> argparse.Namespace:

    base = RetrieverEvalConfig()

    parser = argparse.ArgumentParser(description="Run standalone retriever evaluation.")

    parser.add_argument(
        "--dataset",
        type=Path,
        default=base.dataset_path,
        help="Path to the golden dataset JSON file.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=base.results_dir,
        help="Directory where reports are written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=base.top_k,
        help="Number of chunks to retrieve per question.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=base.metric_threshold,
        help="Minimum score for a metric to pass.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=base.judge_model,
        help="Model used by DeepEval to grade the metrics.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output."
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    run_retriever_evaluation(
        config=RetrieverEvalConfig(
            dataset_path=args.dataset,
            results_dir=args.results_dir,
            top_k=args.top_k,
            metric_threshold=args.threshold,
            verbose=not args.quiet,
        )
    )
