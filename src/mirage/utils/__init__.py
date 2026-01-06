"""
Utilities module for MiRAGE - Preflight checks, statistics, and ablation studies.
"""

from mirage.utils.preflight import (
    run_preflight_checks,
    check_gpu_availability,
    check_api_connectivity,
    PreflightChecker,
)

from mirage.utils.stats import (
    compute_dataset_stats,
    print_dataset_stats,
    compute_qa_category_stats,
    print_qa_category_stats,
)

from mirage.utils.ablation import (
    run_ablation_study,
    AblationConfig,
)

__all__ = [
    # Preflight
    "run_preflight_checks",
    "check_gpu_availability",
    "check_api_connectivity",
    "PreflightChecker",
    # Statistics
    "compute_dataset_stats",
    "print_dataset_stats",
    "compute_qa_category_stats",
    "print_qa_category_stats",
    # Ablation
    "run_ablation_study",
    "AblationConfig",
]
