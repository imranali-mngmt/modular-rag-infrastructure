import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
)

from app.application.services.retrieval_service import RetrievalService
from app.application.services.generator_service import GeneratorService


class EvaluationService:
    """Comprehensive RAG evaluation using DeepEval."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        generator_service: GeneratorService,
    ):
        self.retrieval_service = retrieval_service
        self.generator_service = generator_service
        self.results_dir = Path("evals/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize DeepEval metrics with a threshold of 0.7 (70%)
        self.faithfulness_metric = FaithfulnessMetric(threshold=0.7)
        self.relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
        self.recall_metric = ContextualRecallMetric(threshold=0.7)

    def run_evaluation(
        self, test_cases: List[Dict[str, Any]], save_to_json: bool = True
    ) -> Dict[str, Any]:
        """
        Runs DeepEval on a list of test cases and saves detailed results.
        """
        print(f"\n🚀 Starting DeepEval on {len(test_cases)} test cases...")

        deepeval_test_cases = []
        test_case_mapping = []  # To map back to original questions

        # 1. Prepare Test Cases
        for idx, tc in enumerate(test_cases):
            question = tc["question"]
            expected_output = tc.get("expected_output", "")
            retrieval_context = tc.get("retrieval_context", [])  # Ground truth context

            print(f"  [{idx+1}/{len(test_cases)}] Processing: '{question[:50]}...'")

            try:
                # A. Retrieve actual context from your Qdrant store
                retrieval_response = self.retrieval_service.retrieve(question, k=4)
                actual_retrieval_context = [
                    r.content for r in retrieval_response.results
                ]

                # B. Generate actual answer from your Generator
                actual_output = self.generator_service.generate_answer(
                    question, retrieval_response.results
                )

                # C. Create DeepEval LLMTestCase
                test_case = LLMTestCase(
                    input=question,
                    actual_output=actual_output,
                    expected_output=expected_output,
                    retrieval_context=actual_retrieval_context,
                )
                deepeval_test_cases.append(test_case)
                test_case_mapping.append(
                    {
                        "question": question,
                        "expected_output": expected_output,
                        "actual_output": actual_output,
                        "retrieved_context": actual_retrieval_context,
                    }
                )

            except Exception as e:
                print(f"  ⚠️ Error processing case {idx+1}: {e}")

        # 2. Run DeepEval
        print("\n⚖️  Running DeepEval metrics (Faithfulness, Relevancy, Recall)...")
        results = evaluate(
            test_cases=deepeval_test_cases,
            metrics=[
                self.faithfulness_metric,
                self.relevancy_metric,
                self.recall_metric,
            ],
        )

        # 3. Parse Results into a Clean JSON Structure
        evaluation_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_cases": len(test_case_mapping),
            "overall_pass_rate": f"{results.pass_rate * 100:.1f}%",
            "detailed_results": [],
            "failures": [],
        }

        for i, test_result in enumerate(results.test_results):
            original_data = test_case_mapping[i]

            # Extract metrics data
            metrics_data = {}
            failures = []

            for metric_result in test_result.metrics_data:
                metric_name = metric_result.name
                score = metric_result.score
                reason = metric_result.reason  # <-- This is the "why it failed" reason!

                metrics_data[metric_name] = {
                    "score": round(score, 4),
                    "passed": metric_result.success,
                    "reason": reason,
                }

                if not metric_result.success:
                    failures.append(
                        f"{metric_name} failed (Score: {score:.2f}): {reason}"
                    )

            # Build detailed result entry
            result_entry = {
                "question": original_data["question"],
                "expected_output": original_data["expected_output"],
                "actual_output": original_data["actual_output"],
                "metrics": metrics_data,
                "overall_status": "PASS" if test_result.success else "FAIL",
            }
            evaluation_summary["detailed_results"].append(result_entry)

            # Track failures separately for easy review
            if not test_result.success:
                evaluation_summary["failures"].append(
                    {
                        "question": original_data["question"],
                        "reasons": failures,
                        "actual_output": original_data["actual_output"],
                    }
                )

        # 4. Save to JSON
        if save_to_json:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"deepeval_results_{timestamp}.json"
            filepath = self.results_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(evaluation_summary, f, indent=2, ensure_ascii=False)

            print(f"\n💾 Evaluation results saved to: {filepath}")

        # 5. Print Summary
        print("\n" + "=" * 60)
        print(" 📊 DEEPEVAL SUMMARY")
        print("=" * 60)
        print(f" Total Cases:      {evaluation_summary['total_cases']}")
        print(f" Pass Rate:        {evaluation_summary['overall_pass_rate']}")
        print(f" Total Failures:   {len(evaluation_summary['failures'])}")
        print("=" * 60)

        return evaluation_summary
