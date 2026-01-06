"""
Evaluation module for MiRAGE - Metrics and dataset evaluation.
"""

from mirage.evaluation.metrics import (
    evaluate_qa_dataset,
    compute_faithfulness,
    compute_relevancy,
)

from mirage.evaluation.metrics_optimized import (
    OptimizedEvaluator,
    evaluate_subset,
    generate_evaluation_report,
)

__all__ = [
    # Basic metrics
    "evaluate_qa_dataset",
    "compute_faithfulness",
    "compute_relevancy",
    # Optimized evaluation
    "OptimizedEvaluator",
    "evaluate_subset",
    "generate_evaluation_report",
]
