"""
Standalone latency evaluation harness (no judge model needed).

Runs every golden question through the full RAG pipeline N times, measures:
    - retrieval latency (vector store + embedding round trip)
    - generation latency (LLM round trip)
    - end-to-end latency
and reports mean / median / stdev / min / max / p50 / p90 / p95 / p99.

Optionally fails the run if average or p95 end-to-end latency exceeds
configurable limits.

Run as a script:
    python -m evals.scripts.evaluate_latency
    python -m evals.scripts.evaluate_latency --runs 5 --warmup 1 --max-p95 20.0

Programmatic use:
    from evals.scripts.evaluate_latency import LatencyEvalConfig, run_latency_evaluation

    summary = run_latency_evaluation(config=LatencyEvalConfig(runs_per_question=5))

Tips:
    - Use --runs >= 3 for stable percentiles.
    - Warmup runs absorb cold-start costs (model loading, connections).
    - Run in an environment comparable to production.
"""

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Protocol, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALS_DIR = PROJECT_ROOT / "evals"
DATASET_DIR = EVALS_DIR / "datasets"
RESULTS_DIR = EVALS_DIR / "results"

DEFAULT_DATASET_PATH = DATASET_DIR / "golden_dataset.json"

# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class LatencyEvalConfig:
    """All knobs for a latency evaluation run."""

    dataset_path: Path = DEFAULT_DATASET_PATH
    results_dir: Path = RESULTS_DIR
    top_k: int = 4
    runs_per_question: int = 3
    warmup_runs: int = 1
    # Thresholds (None = check disabled). End-to-end only for simplicity.
    max_avg_end_to_end_seconds: Optional[float] = 15.0
    max_p95_end_to_end_seconds: Optional[float] = 30.0
    verbose: bool = True


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> Any: ...


class Generator(Protocol):
    def generate_answer(self, query: str, results: Any) -> str: ...


def _log(config: LatencyEvalConfig, message: str) -> None:
    if config.verbose:
        print(message)


# ============================================================
# DATASET LOADING
# ============================================================


def load_questions(
    dataset_path: Path,
) -> Tuple[str, List[str], int]:
    """Load golden questions. Returns (name, questions, total_samples)."""

    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    golden_samples = dataset.get("golden_samples", [])

    questions = [s["input"] for s in golden_samples if s.get("input")]

    if not questions:
        raise ValueError(f"Golden dataset contains no questions: {dataset_path}")

    dataset_name = dataset.get("name", dataset_path.stem)
    return dataset_name, questions, len(golden_samples)


# ============================================================
# MEASUREMENT
# ============================================================


def _percentile(values: List[float], percentile: float) -> float:
    """Linear-interpolation percentile (numpy default behaviour), no numpy needed."""
    if not values:
        return 0.0

    ordered = sorted(values)
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)

    if lower == upper:
        return ordered[int(rank)]

    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def compute_stage_stats(values: List[float]) -> dict:
    if not values:
        return {"samples": 0}

    return {
        "samples": len(values),
        "mean_s": round(statistics.mean(values), 4),
        "median_s": round(statistics.median(values), 4),
        "stdev_s": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "min_s": round(min(values), 4),
        "max_s": round(max(values), 4),
        "p50_s": round(_percentile(values, 50), 4),
        "p90_s": round(_percentile(values, 90), 4),
        "p95_s": round(_percentile(values, 95), 4),
        "p99_s": round(_percentile(values, 99), 4),
    }


def _measure_single_run(
    retriever: Retriever,
    generator: Generator,
    question: str,
    config: LatencyEvalConfig,
) -> dict:
    """One retrieve -> generate pass with stage timings."""

    start = time.perf_counter()
    response = retriever.retrieve(question, k=config.top_k)
    after_retrieval = time.perf_counter()

    generator.generate_answer(question, response.results)
    end = time.perf_counter()

    return {
        "retrieval_seconds": round(after_retrieval - start, 4),
        "generation_seconds": round(end - after_retrieval, 4),
        "end_to_end_seconds": round(end - start, 4),
    }


def build_latency_results(
    questions: Sequence[str],
    retriever: Retriever,
    generator: Generator,
    config: LatencyEvalConfig,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Run warmup + timed runs per question.

    Returns:
        per_question_results: per-question run records + stats.
        failures: failed runs / questions.
        pooled_runs: all stage timings pooled across every run.
    """

    per_question_results: List[dict] = []
    failures: List[dict] = []

    pooled: dict = {"retrieval": [], "generation": [], "end_to_end": []}

    # Warmup (cold start: model loading, connection pools, caches).
    if config.warmup_runs > 0:
        warmup_question = questions[0]
        _log(config, f"Warming up with {config.warmup_runs} run(s) (not measured)...")
        for _ in range(config.warmup_runs):
            try:
                _measure_single_run(retriever, generator, warmup_question, config)
            except Exception as error:
                _log(config, f"   Warmup run failed (ignored): {error}")

    for idx, question in enumerate(questions, start=1):
        _log(config, f"[{idx}/{len(questions)}] Measuring: {question[:60]}...")

        run_records: List[dict] = []

        for run_idx in range(1, config.runs_per_question + 1):
            try:
                timings = _measure_single_run(retriever, generator, question, config)
            except Exception as error:
                _log(config, f"   Run {run_idx} failed: {error}")
                failures.append(
                    {
                        "question": question,
                        "run": run_idx,
                        "reason": f"Run Error: {error}",
                    }
                )
                continue

            run_records.append(timings)
            pooled["retrieval"].append(timings["retrieval_seconds"])
            pooled["generation"].append(timings["generation_seconds"])
            pooled["end_to_end"].append(timings["end_to_end_seconds"])

            _log(
                config,
                f"   Run {run_idx}: retrieval {timings['retrieval_seconds']:.3f}s | "
                f"generation {timings['generation_seconds']:.3f}s | "
                f"e2e {timings['end_to_end_seconds']:.3f}s",
            )

        if run_records:
            per_question_results.append(
                {
                    "question": question,
                    "successful_runs": len(run_records),
                    "runs": run_records,
                    "stats": {
                        "retrieval": compute_stage_stats(
                            [r["retrieval_seconds"] for r in run_records]
                        ),
                        "generation": compute_stage_stats(
                            [r["generation_seconds"] for r in run_records]
                        ),
                        "end_to_end": compute_stage_stats(
                            [r["end_to_end_seconds"] for r in run_records]
                        ),
                    },
                }
            )
        else:
            failures.append(
                {"question": question, "reason": "All runs failed for this question."}
            )

    return per_question_results, failures, pooled


# ============================================================
# THRESHOLD CHECK
# ============================================================


def check_thresholds(pooled: dict, config: LatencyEvalConfig) -> Tuple[str, List[dict]]:
    """Check aggregate end-to-end latency against configured limits."""

    breaches: List[dict] = []
    e2e = pooled.get("end_to_end", [])

    if config.max_avg_end_to_end_seconds is not None and e2e:
        avg = statistics.mean(e2e)
        if avg > config.max_avg_end_to_end_seconds:
            breaches.append(
                {
                    "check": "average_end_to_end",
                    "observed_s": round(avg, 4),
                    "limit_s": config.max_avg_end_to_end_seconds,
                }
            )

    if config.max_p95_end_to_end_seconds is not None and e2e:
        p95 = _percentile(e2e, 95)
        if p95 > config.max_p95_end_to_end_seconds:
            breaches.append(
                {
                    "check": "p95_end_to_end",
                    "observed_s": round(p95, 4),
                    "limit_s": config.max_p95_end_to_end_seconds,
                }
            )

    return ("PASS" if not breaches else "FAIL"), breaches


# ============================================================
# SUMMARY / OUTPUT
# ============================================================


def build_summary(
    dataset_name: str,
    dataset_path: Path,
    config: LatencyEvalConfig,
    total_questions: int,
    per_question_results: List[dict],
    pooled: dict,
    overall_status: str,
    threshold_breaches: List[dict],
    failures: List[dict],
) -> dict:

    return {
        "evaluation_type": "LATENCY",
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "top_k": config.top_k,
            "runs_per_question": config.runs_per_question,
            "warmup_runs": config.warmup_runs,
            "max_avg_end_to_end_seconds": config.max_avg_end_to_end_seconds,
            "max_p95_end_to_end_seconds": config.max_p95_end_to_end_seconds,
        },
        "total_questions": total_questions,
        "measured_questions": len(per_question_results),
        "failed_runs": len(failures),
        "overall_status": overall_status,
        "threshold_breaches": threshold_breaches,
        "aggregate_latency_stats": {
            "retrieval": compute_stage_stats(pooled.get("retrieval", [])),
            "generation": compute_stage_stats(pooled.get("generation", [])),
            "end_to_end": compute_stage_stats(pooled.get("end_to_end", [])),
        },
        "per_question_results": per_question_results,
        "failures": failures,
    }


def save_results(summary: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"latency_eval_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return output_file


def _format_stage(stats: dict) -> str:
    if not stats.get("samples"):
        return "no samples"
    return (
        f"avg {stats['mean_s']:.3f}s | p50 {stats['p50_s']:.3f}s | "
        f"p95 {stats['p95_s']:.3f}s | max {stats['max_s']:.3f}s"
    )


def print_report(summary: dict, output_file: Path) -> None:

    aggregate = summary["aggregate_latency_stats"]

    print("\n" + "=" * 60)
    print("LATENCY EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Dataset:            {summary['dataset_name']}")
    print(f"Measured questions: {summary['measured_questions']}")
    print(f"Failed runs:        {summary['failed_runs']}")
    print(f"Retrieval:          {_format_stage(aggregate['retrieval'])}")
    print(f"Generation:         {_format_stage(aggregate['generation'])}")
    print(f"End-to-end:         {_format_stage(aggregate['end_to_end'])}")
    print(f"Overall status:     {summary['overall_status']}")

    for breach in summary["threshold_breaches"]:
        print(
            f"  BREACH: {breach['check']} observed {breach['observed_s']}s > limit {breach['limit_s']}s"
        )

    print(f"\nResults saved to:\n{output_file}")
    print("=" * 60 + "\n")


# ============================================================
# ORCHESTRATION
# ============================================================


def run_latency_evaluation(
    retriever: Optional[Retriever] = None,
    generator: Optional[Generator] = None,
    config: Optional[LatencyEvalConfig] = None,
) -> dict:

    config = config or LatencyEvalConfig()

    _log(config, "\nStarting standalone latency evaluation...\n")

    # 1. Load golden questions.
    dataset_name, questions, total_samples = load_questions(config.dataset_path)
    _log(config, f"   Dataset: {dataset_name}")
    _log(config, f"   Total questions: {len(questions)}\n")

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

    # 3. Measure.
    per_question_results, failures, pooled = build_latency_results(
        questions=questions,
        retriever=retriever,
        generator=generator,
        config=config,
    )

    if not per_question_results:
        raise RuntimeError(
            "Evaluation stopped because no runs succeeded. "
            "Check your services and golden dataset."
        )

    # 4. Threshold check.
    overall_status, threshold_breaches = check_thresholds(pooled, config)

    # 5. Build summary.
    summary = build_summary(
        dataset_name=dataset_name,
        dataset_path=config.dataset_path,
        config=config,
        total_questions=total_samples,
        per_question_results=per_question_results,
        pooled=pooled,
        overall_status=overall_status,
        threshold_breaches=threshold_breaches,
        failures=failures,
    )

    # 6. Save report.
    output_file = save_results(summary, config.results_dir)

    # 7. Final report.
    if config.verbose:
        print_report(summary, output_file)

    return summary


# ============================================================
# CLI ENTRY POINT
# ============================================================


def parse_args() -> argparse.Namespace:
    base = LatencyEvalConfig()

    parser = argparse.ArgumentParser(description="Run standalone latency evaluation.")
    parser.add_argument("--dataset", type=Path, default=base.dataset_path)
    parser.add_argument("--results-dir", type=Path, default=base.results_dir)
    parser.add_argument("--top-k", type=int, default=base.top_k)
    parser.add_argument(
        "--runs",
        type=int,
        default=base.runs_per_question,
        help="Timed runs per question (>= 3 recommended for stable percentiles).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=base.warmup_runs,
        help="Unmeasured warmup runs to absorb cold-start costs.",
    )
    parser.add_argument(
        "--max-avg",
        type=float,
        default=base.max_avg_end_to_end_seconds,
        help="Fail if average end-to-end latency exceeds this (seconds).",
    )
    parser.add_argument(
        "--max-p95",
        type=float,
        default=base.max_p95_end_to_end_seconds,
        help="Fail if p95 end-to-end latency exceeds this (seconds).",
    )
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_latency_evaluation(
        config=LatencyEvalConfig(
            dataset_path=args.dataset,
            results_dir=args.results_dir,
            top_k=args.top_k,
            runs_per_question=args.runs,
            warmup_runs=args.warmup,
            max_avg_end_to_end_seconds=args.max_avg,
            max_p95_end_to_end_seconds=args.max_p95,
            verbose=not args.quiet,
        )
    )
