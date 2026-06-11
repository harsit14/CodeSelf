"""Dataset schemas and registry helpers."""

from codeself.datasets.loaders import (
    DatasetLoadError,
    apply_test_sidecar,
    load_humaneval_like,
    load_mbpp,
    load_task_specs,
    load_test_sidecar,
)
from codeself.datasets.contamination import (
    ContaminationFinding,
    DatasetQualityReport,
    HiddenLeakFinding,
    check_dataset_quality,
    detect_hidden_test_leaks,
    detect_train_eval_contamination,
    normalize_text,
)
from codeself.datasets.registry import TaskRegistry
from codeself.datasets.schemas import ResourceLimits, Split, TaskSpec, TestSpec
from codeself.datasets.splits import (
    SplitFractions,
    assign_splits,
    dataset_fingerprint,
    split_counts,
    write_split_manifest,
)

__all__ = [
    "DatasetLoadError",
    "ContaminationFinding",
    "DatasetQualityReport",
    "HiddenLeakFinding",
    "ResourceLimits",
    "Split",
    "SplitFractions",
    "TaskRegistry",
    "TaskSpec",
    "TestSpec",
    "apply_test_sidecar",
    "assign_splits",
    "check_dataset_quality",
    "dataset_fingerprint",
    "detect_hidden_test_leaks",
    "detect_train_eval_contamination",
    "load_humaneval_like",
    "load_mbpp",
    "load_task_specs",
    "load_test_sidecar",
    "normalize_text",
    "split_counts",
    "write_split_manifest",
]
