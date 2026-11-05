"""
Standalone generator evaluation harness built on DeepEval.

Retrieves context, generates answers for every golden question, scores the
outputs with DeepEval metrics (faithfulness, answer relevancy), and writes
a JSON report.

Run as a script:
    python evals/generator_eval.py
    python evals/generator_eval.py --top-k 8 --threshold 0.75 --quiet

Switch the judge model provider with --judge-model (see JUDGE PROVIDERS below):
    python evals/generator_eval.py --judge-model openrouter/openai/gpt-4o-mini

Or call programmatically:
    from evals.generator_eval import GeneratorEvalConfig, run_generator_evaluation

    summary = run_generator_evaluation(
        generator=my_custom_generator,
        config=GeneratorEvalConfig(top_k=8),
    )
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase


import os
from openai import OpenAI
from deepeval.models import DeepEvalBaseLLM


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


# Make the project root importable so we can import from 'src'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = SCRIPT_DIR / "golden_dataset.json"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

# ============================================================
# JUDGE PROVIDERS
# ============================================================
# DeepEval runs its judge model through LiteLLM. The provider is selected
# by the model string prefix; each provider needs its API key env var:
#
#   OpenAI      "gpt-4o-mini"                        OPENAI_API_KEY
#   OpenRouter  "openrouter/openai/gpt-4o-mini"      OPENROUTER_API_KEY
#   Gemini      "gemini/gemini-1.5-flash"            GEMINI_API_KEY
#   Anthropic   "anthropic/claude-3-5-sonnet-latest" ANTHROPIC_API_KEY
#   Azure       "azure/<deployment-name>"            AZURE_API_KEY + AZURE_API_BASE + AZURE_API_VERSION
#   Groq        "groq/llama-3.3-70b-versatile"       GROQ_API_KEY
#   Ollama      "ollama/llama3"                      OLLAMA_API_BASE (local, no key)

# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class GeneratorEvalConfig:
    """All knobs for a generator evaluation run."""

    judge_model: DeepEvalBaseLLM = field(default_factory=Mercury3Judge)

    dataset_path: Path = DEFAULT_DATASET_PATH
    results_dir: Path = DEFAULT_RESULTS_DIR
    top_k: int = 4
    metric_threshold: float = 0.7
    verbose: bool = True


class Retriever(Protocol):
    """Minimal interface the evaluation expects from a retrieval service."""

    def retrieve(self, query: str, k: int) -> Any: ...


class Generator(Protocol):
    """Minimal interface the evaluation expects from a generation service."""

    def generate_answer(self, query: str, results: Any) -> str: ...


ChunkExtractor = Callable[[Any], List[str]]


def default_chunk_extractor(response: Any) -> List[str]:
    """Extract chunk texts from a retrieval service response."""
    return [result.content for result in response.results]


def _log(config: GeneratorEvalConfig, message: str) -> None:
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


def build_default_metrics(config: GeneratorEvalConfig) -> List[BaseMetric]:
    """Default metric suite: faithfulness + answer relevancy."""
    return [
        FaithfulnessMetric(threshold=config.metric_threshold, model=config.judge_model),
        AnswerRelevancyMetric(
            threshold=config.metric_threshold, model=config.judge_model
        ),
    ]


# ============================================================
# TEST CASE BUILDING
# ============================================================


def build_test_cases(
    golden_samples: Sequence[dict],
    retriever: Retriever,
    generator: Generator,
    config: GeneratorEvalConfig,
    chunk_extractor: ChunkExtractor = default_chunk_extractor,
) -> Tuple[List[LLMTestCase], List[dict], List[dict]]:
    """
    Retrieve context and generate an answer for every golden sample.

    Returns:
        test_cases: DeepEval test cases ready for evaluation.
        detailed_results: Per-question details (metrics filled in later).
        failures: Questions that could not be processed.
    """

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

        if expected_output is None:
            failures.append(
                {"question": question, "reason": "Missing expected_output in dataset."}
            )
            continue

        _log(config, f"[{idx}/{total}] Generating for: '{question[:50]}...'")

        try:

            # A. Retrieve context from the vector store
            retrieval_response = retriever.retrieve(question, k=config.top_k)
            retrieved_chunks = chunk_extractor(retrieval_response)

            # B. Generate the answer
            actual_output = generator.generate_answer(
                question, retrieval_response.results
            )

            # C. Build the DeepEval test case
            test_cases.append(
                LLMTestCase(
                    input=question,
                    actual_output=actual_output,
                    expected_output=expected_output,
                    retrieval_context=retrieved_chunks,
                )
            )

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
    """Merge DeepEval metric scores back into the per-question details."""

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

        detail["metrics"] = metrics_data
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
    """Count failures caused by the retrieve/generate step (not metrics)."""
    return len(
        [
            failure
            for failure in failures
            if "Generation Error" in failure.get("reason", "")
        ]
    )


def build_summary(
    dataset_name: str,
    dataset_path: Path,
    config: GeneratorEvalConfig,
    total_golden_samples: int,
    total_cases: int,
    evaluated_results: List[dict],
    failures: List[dict],
    pass_rate: float,
) -> dict:

    return {
        "evaluation_type": "GENERATOR",
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "top_k": config.top_k,
            "metric_threshold": config.metric_threshold,
            "judge_model": config.judge_model,
        },
        "total_golden_samples": total_golden_samples,
        "total_cases": total_cases,
        "failed_generations": count_generation_failures(failures),
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
    output_file = results_dir / f"generator_eval_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return output_file


def print_report(summary: dict, output_file: Path) -> None:

    print("\n" + "=" * 60)
    print("GENERATOR EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Dataset:              {summary['dataset_name']}")
    print(f"Evaluated cases:      {summary['total_cases']}")
    print(f"Generation failures:  {summary['failed_generations']}")
    print(f"Pass rate:            {summary['pass_rate']}")
    print(f"Judge model:          {summary['configuration']['judge_model']}")
    print(f"\nResults saved to:\n{output_file}")
    print("=" * 60 + "\n")


# ============================================================
# ORCHESTRATION
# ============================================================


def run_generator_evaluation(
    retriever: Optional[Retriever] = None,
    generator: Optional[Generator] = None,
    config: Optional[GeneratorEvalConfig] = None,
    metrics: Optional[Sequence[BaseMetric]] = None,
    chunk_extractor: ChunkExtractor = default_chunk_extractor,
) -> dict:
    """
    Run a full generator evaluation (retrieve -> generate -> score).

    Args:
        retriever: Retrieval service. Defaults to the app's retrieval service
            (imported lazily so this module can be reused without the app).
        generator: Generation service. Defaults to the app's generator service.
        config: Evaluation settings (paths, top_k, threshold, judge model).
        metrics: DeepEval metrics to run. Defaults to faithfulness +
            answer relevancy built from the config.
        chunk_extractor: Extracts chunk texts from a retriever response.

    Returns:
        The evaluation summary dict (also written to disk as JSON).
    """

    config = config or GeneratorEvalConfig()

    _log(config, "\nStarting standalone generator evaluation...\n")

    # 1. Load golden dataset.
    _log(config, "Loading golden dataset...")
    dataset_name, golden_samples = load_golden_dataset(config.dataset_path)
    _log(config, f"   Dataset: {dataset_name}")
    _log(config, f"   Total samples: {len(golden_samples)}")
    _log(config, f"   Dataset path: {config.dataset_path}\n")

    # 2. Resolve services.
    if retriever is None or generator is None:
        _log(config, "Initializing default retrieval and generator services...")
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

    _log(config, "\n" + "=" * 60)
    _log(config, "GENERATION SUMMARY")
    _log(config, "=" * 60)
    _log(config, f"Total golden samples: {len(golden_samples)}")
    _log(config, f"Successful generations: {len(test_cases)}")
    _log(config, f"Failed generations: {count_generation_failures(failures)}")

    # 5. Run DeepEval metrics.
    _log(config, "\nRunning DeepEval generator metrics (Faithfulness, Relevancy)...\n")

    results = evaluate(test_cases=test_cases, metrics=list(metrics))

    # 6. Merge metric scores into per-question details.
    _log(config, "\nProcessing evaluation results...")

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

    base = GeneratorEvalConfig()

    parser = argparse.ArgumentParser(description="Run standalone generator evaluation.")

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
        help=(
            "Judge model in LiteLLM format. Examples: 'gpt-4o-mini', "
            "'openrouter/openai/gpt-4o-mini', 'gemini/gemini-1.5-flash', "
            "'anthropic/claude-3-5-sonnet-latest', 'ollama/llama3'. "
            "Set the matching API key env var for the provider."
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output."
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    run_generator_evaluation(
        config=GeneratorEvalConfig(
            dataset_path=args.dataset,
            results_dir=args.results_dir,
            top_k=args.top_k,
            metric_threshold=args.threshold,
            judge_model=args.judge_model,
            verbose=not args.quiet,
        )
    )
