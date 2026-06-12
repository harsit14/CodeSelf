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
    normalize_code,
    normalize_text,
)
from codeself.datasets.dataset_config import (
    DatasetBuildResult,
    DatasetConfig,
    DatasetConfigError,
    DatasetContaminationError,
    DatasetSourceConfig,
    auto_split_visible_hidden,
    build_dataset,
    load_dataset_config,
)
from codeself.datasets.registry import TaskRegistry
from codeself.datasets.schemas import ResourceLimits, Split, TaskSpec, TestSpec, TestVisibility
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
    "DatasetBuildResult",
    "DatasetConfig",
    "DatasetConfigError",
    "DatasetContaminationError",
    "DatasetQualityReport",
    "DatasetSourceConfig",
    "HiddenLeakFinding",
    "ResourceLimits",
    "Split",
    "SplitFractions",
    "TaskRegistry",
    "TaskSpec",
    "TestSpec",
    "TestVisibility",
    "apply_test_sidecar",
    "assign_splits",
    "auto_split_visible_hidden",
    "build_dataset",
    "check_dataset_quality",
    "load_dataset_config",
    "dataset_fingerprint",
    "detect_hidden_test_leaks",
    "detect_train_eval_contamination",
    "load_humaneval_like",
    "load_mbpp",
    "load_task_specs",
    "load_test_sidecar",
    "normalize_code",
    "normalize_text",
    "split_counts",
    "write_split_manifest",
]
