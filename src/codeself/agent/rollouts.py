"""Rollout records and generation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeself.agent.generation import CodeGenerator, GenerationRequest
from codeself.agent.parser import ParseStatus, ParsedCompletion, extract_code
from codeself.agent.prompts import PromptTemplate
from codeself.datasets import TaskSpec
from codeself.execution import SandboxedTestRunner
from codeself.rewards import CompositeRewardScorer


@dataclass(frozen=True)
class RolloutRecord:
    """One generated, parsed, executed, and scored sample."""

    task_id: str
    sample_index: int
    prompt_template: str
    prompt: str
    raw_completion: str
    parsed: ParsedCompletion
    backend: str
    model_name: str
    generation_metadata: dict[str, str | int | float | bool]
    execution: dict[str, Any]
    reward: dict[str, Any]
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "sample_index": self.sample_index,
            "prompt_template": self.prompt_template,
            "prompt": self.prompt,
            "raw_completion": self.raw_completion,
            "parsed": {
                "code": self.parsed.code,
                "status": self.parsed.status.value,
                "parser": self.parsed.parser,
                "error": self.parsed.error,
            },
            "backend": self.backend,
            "model_name": self.model_name,
            "generation_metadata": dict(self.generation_metadata),
            "execution": self.execution,
            "reward": self.reward,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RolloutRecord":
        parsed = ParsedCompletion(
            raw_text=str(value.get("raw_completion", "")),
            code=str(value["parsed"]["code"]),
            status=ParseStatus(str(value["parsed"]["status"])),
            parser=str(value["parsed"]["parser"]),
            error=str(value["parsed"].get("error", "")),
        )
        return cls(
            task_id=str(value["task_id"]),
            sample_index=int(value["sample_index"]),
            prompt_template=str(value["prompt_template"]),
            prompt=str(value["prompt"]),
            raw_completion=str(value["raw_completion"]),
            parsed=parsed,
            backend=str(value["backend"]),
            model_name=str(value["model_name"]),
            generation_metadata=dict(value.get("generation_metadata", {})),
            execution=dict(value.get("execution", {})),
            reward=dict(value.get("reward", {})),
            metadata=dict(value.get("metadata", {})),
        )


def generate_rollouts(
    tasks: list[TaskSpec],
    *,
    generator: CodeGenerator,
    prompt_template: PromptTemplate,
    samples_per_task: int,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    include_hidden: bool,
) -> list[RolloutRecord]:
    runner = SandboxedTestRunner()
    scorer = CompositeRewardScorer()
    records: list[RolloutRecord] = []
    for task in tasks:
        prompt = prompt_template.render(task)
        for sample_index in range(samples_per_task):
            request = GenerationRequest(
                task_id=task.task_id,
                prompt=prompt,
                sample_index=sample_index,
                seed=seed,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                entry_point=task.entry_point,
            )
            generation = generator.generate(request)
            parsed = extract_code(generation.text)
            execution_result = runner.run(task, parsed.code, include_hidden=include_hidden)
            reward = scorer.score(execution_result, solution_code=parsed.code)
            records.append(
                RolloutRecord(
                    task_id=task.task_id,
                    sample_index=sample_index,
                    prompt_template=prompt_template.name,
                    prompt=prompt,
                    raw_completion=generation.text,
                    parsed=parsed,
                    backend=generation.backend,
                    model_name=generation.model_name,
                    generation_metadata=generation.metadata,
                    execution=execution_result.to_dict(),
                    reward=reward.to_dict(),
                    metadata={
                        "seed": seed,
                        "include_hidden": include_hidden,
                    },
                )
            )
    return records


def write_rollouts_jsonl(records: list[RolloutRecord], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")


def read_rollouts_jsonl(path: str | Path) -> list[RolloutRecord]:
    input_path = Path(path)
    records: list[RolloutRecord] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(RolloutRecord.from_dict(json.loads(stripped)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid rollout JSONL at {input_path}:{line_number}") from exc
    return records
