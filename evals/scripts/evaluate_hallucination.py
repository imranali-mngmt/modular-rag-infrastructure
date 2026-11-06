"""
Standalone hallucination evaluation harness built on DeepEval.

Retrieves context, generates answers for every golden question, then scores
each answer with DeepEval's HallucinationMetric, which detects claims in the
generated answer that contradict the retrieved context.

Run as a script:
    python -m evals.scripts.evaluate_hallucination
    python -m evals.scripts.evaluate_hallucination --top-k 8 --threshold 0.3

Switch the judge model provider with --judge-model:
    python -m evals.scripts.evaluate_hallucination --judge-model gpt-4o-mini

NOTE on score semantics (INVERTED vs. most DeepEval metrics):
    score 0.0 = no contradictions with the retrieved context (best)
    score 1.0 = every claim contradicts the retrieved context (worst)
    A test case PASSES when score <= threshold.

Programmatic use:
    from evals.scripts.evaluate_hallucination import (
        HallucinationEvalConfig,
        run_hallucination_evaluation,
    )

    summary = run_hallucination_evaluation(config=HallucinationEvalConfig(top_k=8))
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple

from openai import AsyncOpenAI, OpenAI
from deepeval import evaluate
from deepeval.metrics import BaseMetric, HallucinationMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALS_DIR = PROJECT_ROOT / "evals"
DATASET_DIR = EVALS_DIR / "datasets"
RESULTS_DIR = EVALS_DIR / "results"

DEFAULT_DATASET_PATH = DATASET_DIR / "golden_dataset.json"

# ============================================================
# JUDGE MODEL
# ============================================================


class Mercury3Judge(DeepEvalBaseLLM):

    def __init__(self):
        api_key = os.getenv("INCEPTION_API_KEY")

        if not api_key:
            raise ValueError("INCEPTION_API_KEY not found in environment.")

        self.client = OpenAI(
            api_key=api_key, base_url="https://api.inceptionlabs.ai/v1/"
        )
        self.async_client = AsyncOpenAI(
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
        response = await self.async_client.chat.completions.create(
            model="mercury-3",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


def resolve_judge_model(model_name: str):
    """Resolve the custom Mercury judge or a LiteLLM model string."""
    if model_name.lower() == "mercury-3":
        return Mercury3Judge()
    return model_name


def _judge_model_name(judge: Any) -> str:
    """JSON-safe name for the judge (objects are not serializable)."""
    if isinstance(judge, str):
        return judge
    get_model_name = getattr(judge, "get_model_name", None)
    if callable(get_model_name):
        try:
            return str(get_model_name())
        except Exception:
            pass
    return type(judge).__name__


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class HallucinationEvalConfig:
    """All knobs for a hallucination evaluation run."""

    judge_model: Any = field(default_factory=Mercury3Judge)
    dataset_path: Path = DEFAULT_DATASET_PATH
    results_dir: Path = RESULTS_DIR
    top_k: int = 4
    # A case passes when hallucination score <= this (0.0 = no contradictions).
    max_hallucination_rate: float = 0.5
    verbose: bool = True


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> Any: ...


class Generator(Protocol):
    def generate_answer(self, query: str, results: Any) -> str: ...


ChunkExtractor = Callable[[Any], List[str]]


def default_chunk_extractor(response: Any) -> List[str]:
    return [
        result.content
        for result in response.results
        if getattr(result, "content", None)
    ]


def _log(config: HallucinationEvalConfig, message: str) -> None:
    if config.verbose:
        print(message)


# ============================================================
# DATASET LOADING
# ============================================================


def load_golden_dataset(dataset_path: Path) -> Tuple[str, List[dict]]:
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


def build_default_metrics(config: HallucinationEvalConfig) -> List[BaseMetric]:
    """
    HallucinationMetric flags claims in the answer that CONTRADICT the
    retrieved context. It is complementary to FaithfulnessMetric:
    faithfulness checks whether claims are grounded; hallucination checks
    whether claims are contradicted.
    """
    return [
        HallucinationMetric(
            threshold=config.max_hallucination_rate,
            model=config.judge_model,
            include_reason=True,
            async_mode=False,
        )
    ]


# ============================================================
# TEST CASE BUILDING
# ============================================================


def build_test_cases(
    golden_samples: Sequence[dict],
    retriever: Retriever,
    generator: Generator,
    config: HallucinationEvalConfig,
    chunk_extractor: ChunkExtractor = default_chunk_extractor,
) -> Tuple[List[LLMTestCase], List[dict], List[dict]]:

    test_cases: List[LLMTestCase] = []
    detailed_results: List[dict] = []
    failures: List[dict] = []

    total = len(golden_samples)

    for idx, sample in enumerate(golden_samples, start=1):

        question = sample.get("input")
        expected_output = sample.get("expected_output")

        if not question:
            failures.append(
                {"question": None, "reason": "Missing input question in dataset."}
            )
            continue

        _log(config, f"[{idx}/{total}] Generating: '{question[:50]}...'")

        try:
            retrieval_response = retriever.retrieve(question, k=config.top_k)
            retrieved_chunks = chunk_extractor(retrieval_response)

            if not retrieved_chunks:
                failures.append(
                    {
                        "question": question,
                        "reason": "Retriever returned no usable document chunks.",
                    }
                )
                _log(config, "   WARNING: No documents retrieved.")
                continue

            actual_output = generator.generate_answer(
                question, retrieval_response.results
            )

            # HallucinationMetric uses `context`, NOT `retrieval_context`.
            case_kwargs = {
                "input": question,
                "actual_output": actual_output,
                "context": retrieved_chunks,
            }
            if expected_output:
                case_kwargs["expected_output"] = expected_output

            test_cases.append(LLMTestCase(**case_kwargs))

            detailed_results.append(
                {
                    "question": question,
                    "expected_output": expected_output,
                    "actual_output": actual_output,
                    "retrieved_context": retrieved_chunks,
                    "status": "PENDING",
                }
            )

        except Exception as error:
            _log(config, f"   Generation failed: {error}")
            failures.append(
                {"question": question, "reason": f"Generation Error: {error}"}
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

    evaluated_results: List[dict] = []

    for i, test_result in enumerate(results.test_results):

        detail = detailed_results[i]
        metrics_data: dict = {}
        case_failures: List[str] = []

        for metric_result in test_result.metrics_data:

            score = metric_result.score

            metrics_data[metric_result.name] = {
                "score": round(score, 4) if score is not None else None,
                "passed": metric_result.success,
                "reason": metric_result.reason,
            }

            if not metric_result.success:
                score_str = f"{score:.2f}" if score is not None else "N/A"
                case_failures.append(
                    f"{metric_result.name} (Score: {score_str}): {metric_result.reason}"
                )

        hallucination_score = metrics_data.get("Hallucination", {}).get("score")
        detail["metrics"] = metrics_data
        detail["hallucination_rate"] = hallucination_score
        detail["overall_status"] = "PASS" if test_result.success else "FAIL"

        evaluated_results.append(detail)

        if case_failures:
            failures.append(
                {
                    "question": detail["question"],
                    "reasons": case_failures,
                    "actual_output": detail["actual_output"],
                }
            )

    return evaluated_results


def count_generation_failures(failures: Sequence[dict]) -> int:
    return len([f for f in failures if "Generation Error" in f.get("reason", "")])


def average_hallucination_rate(evaluated_results: List[dict]) -> Optional[float]:
    scores = [
        detail["hallucination_rate"]
        for detail in evaluated_results
        if detail.get("hallucination_rate") is not None
    ]
    return round(sum(scores) / len(scores), 4) if scores else None


def build_summary(
    dataset_name: str,
    dataset_path: Path,
    config: HallucinationEvalConfig,
    total_golden_samples: int,
    total_cases: int,
    evaluated_results: List[dict],
    failures: List[dict],
    pass_rate: float,
) -> dict:

    return {
        "evaluation_type": "HALLUCINATION",
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "top_k": config.top_k,
            "max_hallucination_rate": config.max_hallucination_rate,
            "judge_model": _judge_model_name(config.judge_model),
        },
        "total_golden_samples": total_golden_samples,
        "total_cases": total_cases,
        "failed_generations": count_generation_failures(failures),
        "average_hallucination_rate": average_hallucination_rate(evaluated_results),
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
    output_file = results_dir / f"hallucination_eval_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return output_file


def print_report(summary: dict, output_file: Path) -> None:
    avg = summary.get("average_hallucination_rate")
    avg_str = f"{avg * 100:.1f}%" if avg is not None else "N/A"

    print("\n" + "=" * 60)
    print("HALLUCINATION EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Dataset:                  {summary['dataset_name']}")
    print(f"Evaluated cases:          {summary['total_cases']}")
    print(f"Generation failures:      {summary['failed_generations']}")
    print(f"Avg hallucination rate:   {avg_str}  (lower is better)")
    print(f"Pass rate:                {summary['pass_rate']}")
    print(f"Judge model:              {summary['configuration']['judge_model']}")
    print(f"\nResults saved to:\n{output_file}")
    print("=" * 60 + "\n")


# ============================================================
# ORCHESTRATION
# ============================================================


def run_hallucination_evaluation(
    retriever: Optional[Retriever] = None,
    generator: Optional[Generator] = None,
    config: Optional[HallucinationEvalConfig] = None,
    metrics: Optional[Sequence[BaseMetric]] = None,
    chunk_extractor: ChunkExtractor = default_chunk_extractor,
) -> dict:

    config = config or HallucinationEvalConfig()

    _log(config, "\nStarting standalone hallucination evaluation...\n")

    # 1. Load golden dataset.
    dataset_name, golden_samples = load_golden_dataset(config.dataset_path)
    _log(config, f"   Dataset: {dataset_name}")
    _log(config, f"   Total samples: {len(golden_samples)}\n")

    # 2. Resolve services.
    if retriever is None or generator is None:
        _log(config, "Initializing default retrieval and generator services...")
        # NOTE: adjust this import path to your project layout.
        from src.app.api.dependencies.providers import (
            get_generator_service,
            get_retrieval_service,
        )

        retriever = retriever or get_retrieval_service()
        generator = generator or get_generator_service()
        _log(config, "   Services initialized successfully.\n")

    # 3. Resolve metrics.
    if metrics is None:
        metrics = build_default_metrics(config)

    # 4. Retrieve, generate, and build test cases.
    test_cases, detailed_results, failures = build_test_cases(
        golden_samples=golden_samples,
        retriever=retriever,
        generator=generator,
        config=config,
        chunk_extractor=chunk_extractor,
    )

    if not test_cases:
        raise RuntimeError(
            "Evaluation stopped because no test cases were created. "
            "Check your services and golden dataset."
        )

    _log(config, f"\nSuccessful generations: {len(test_cases)}")
    _log(config, f"Failed generations:     {count_generation_failures(failures)}")

    # 5. Run DeepEval metrics.
    _log(config, "\nRunning HallucinationMetric...\n")
    results = evaluate(test_cases=test_cases, metrics=list(metrics))

    # 6. Merge metric scores into per-question details.
    evaluated_results = process_results(results, detailed_results, failures)

    # 7. Build summary.
    summary = build_summary(
        dataset_name=dataset_name,
        dataset_path=config.dataset_path,
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
    base = HallucinationEvalConfig()

    parser = argparse.ArgumentParser(
        description="Run standalone hallucination evaluation."
    )
    parser.add_argument("--dataset", type=Path, default=base.dataset_path)
    parser.add_argument("--results-dir", type=Path, default=base.results_dir)
    parser.add_argument("--top-k", type=int, default=base.top_k)
    parser.add_argument(
        "--threshold",
        type=float,
        default=base.max_hallucination_rate,
        help="Max acceptable hallucination rate (case passes when score <= threshold).",
    )
    parser.add_argument("--judge-model", type=str, default="mercury-3")
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_hallucination_evaluation(
        config=HallucinationEvalConfig(
            dataset_path=args.dataset,
            results_dir=args.results_dir,
            top_k=args.top_k,
            max_hallucination_rate=args.threshold,
            judge_model=resolve_judge_model(args.judge_model),
            verbose=not args.quiet,
        )
    )
