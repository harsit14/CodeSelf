"""Small in-memory registry for task specs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from codeself.datasets.schemas import Split, TaskSpec


class TaskRegistry:
    """Holds task specs and protects against duplicate task IDs."""

    def __init__(self, tasks: Iterable[TaskSpec] | None = None) -> None:
        self._tasks: dict[str, TaskSpec] = {}
        for task in tasks or ():
            self.add(task)

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[TaskSpec]:
        return iter(self._tasks.values())

    def add(self, task: TaskSpec) -> None:
        if task.task_id in self._tasks:
            raise ValueError(f"duplicate task_id: {task.task_id}")
        self._tasks[task.task_id] = task

    def get(self, task_id: str) -> TaskSpec:
        return self._tasks[task_id]

    def by_split(self, split: Split | str) -> list[TaskSpec]:
        split_value = Split(split)
        return [task for task in self._tasks.values() if task.split == split_value]

    def to_jsonl(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for task in self._tasks.values():
                handle.write(json.dumps(task.to_dict(), sort_keys=True))
                handle.write("\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "TaskRegistry":
        input_path = Path(path)
        tasks: list[TaskSpec] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    tasks.append(TaskSpec.from_dict(json.loads(stripped)))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid task JSONL at {input_path}:{line_number}") from exc
        return cls(tasks)
