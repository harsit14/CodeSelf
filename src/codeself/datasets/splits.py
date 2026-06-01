"""Deterministic split helpers."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path

from codeself.datasets.schemas import Split, TaskSpec


@dataclass(frozen=True)
class SplitFractions:
    """Fractions for train/dev/public/private splits."""

    train: float = 0.80
    dev: float = 0.10
    test_public: float = 0.10
    test_private: float = 0.0

    def __post_init__(self) -> None:
        total = self.train + self.dev + self.test_public + self.test_private
        if total <= 0:
            raise ValueError("split fractions must sum to a positive value")
        for value in (self.train, self.dev, self.test_public, self.test_private):
            if value < 0:
                raise ValueError("split fractions must be non-negative")


def assign_splits(
    tasks: list[TaskSpec],
    *,
    fractions: SplitFractions,
    seed: int,
) -> list[TaskSpec]:
    """Assign deterministic splits by task ID."""

    ordered = sorted(tasks, key=lambda task: task.task_id)
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)

    counts = _counts(len(shuffled), fractions)
    boundaries = {
        Split.TRAIN: counts[Split.TRAIN],
        Split.DEV: counts[Split.TRAIN] + counts[Split.DEV],
        Split.TEST_PUBLIC: counts[Split.TRAIN] + counts[Split.DEV] + counts[Split.TEST_PUBLIC],
    }

    assigned: list[TaskSpec] = []
    for index, task in enumerate(shuffled):
        if index < boundaries[Split.TRAIN]:
            split = Split.TRAIN
        elif index < boundaries[Split.DEV]:
            split = Split.DEV
        elif index < boundaries[Split.TEST_PUBLIC]:
            split = Split.TEST_PUBLIC
        else:
            split = Split.TEST_PRIVATE
        assigned.append(replace(task, split=split))
    return sorted(assigned, key=lambda task: task.task_id)


def split_counts(tasks: list[TaskSpec]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.split.value] = counts.get(task.split.value, 0) + 1
    return dict(sorted(counts.items()))


def dataset_fingerprint(tasks: list[TaskSpec]) -> str:
    """Stable SHA-256 fingerprint over canonical task records."""

    payload = "\n".join(json.dumps(task.to_dict(), sort_keys=True) for task in tasks)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_split_manifest(
    tasks: list[TaskSpec],
    path: str | Path,
    *,
    dataset_name: str,
    version: str,
    seed: int,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": {
            "name": dataset_name,
            "version": version,
            "seed": seed,
            "task_count": len(tasks),
            "fingerprint": dataset_fingerprint(tasks),
        },
        "splits": split_counts(tasks),
    }
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _counts(total: int, fractions: SplitFractions) -> dict[Split, int]:
    fraction_total = fractions.train + fractions.dev + fractions.test_public + fractions.test_private
    train = int(total * fractions.train / fraction_total)
    dev = int(total * fractions.dev / fraction_total)
    test_public = int(total * fractions.test_public / fraction_total)
    used = train + dev + test_public
    test_private = total - used
    return {
        Split.TRAIN: train,
        Split.DEV: dev,
        Split.TEST_PUBLIC: test_public,
        Split.TEST_PRIVATE: test_private,
    }
