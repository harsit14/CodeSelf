"""Prompt templates for coding tasks."""

from __future__ import annotations

from dataclasses import dataclass

from codeself.datasets import TaskSpec


@dataclass(frozen=True)
class PromptTemplate:
    """A versioned prompt template."""

    name: str
    include_public_tests: bool = False
    require_code_fence: bool = True

    def render(self, task: TaskSpec) -> str:
        parts = [
            "You are a careful Python coding assistant.",
            "Solve the task by writing correct, executable Python code.",
            "Return only the final code.",
        ]
        if self.require_code_fence:
            parts.append("Wrap the code in a single ```python fenced block.")
        parts.append("")
        parts.append("Task:")
        parts.append(task.prompt.strip())
        if task.entry_point:
            parts.append("")
            parts.append(f"Required entry point: `{task.entry_point}`")
        if task.starter_code.strip():
            parts.append("")
            parts.append("Starter code:")
            parts.append("```python")
            parts.append(task.starter_code.strip())
            parts.append("```")
        if self.include_public_tests and task.public_tests:
            parts.append("")
            parts.append("Public tests:")
            parts.append("```python")
            parts.extend(test.code.strip() for test in task.public_tests)
            parts.append("```")
        return "\n".join(parts).strip() + "\n"


@dataclass(frozen=True)
class RevisionPromptTemplate:
    """A versioned prompt template for self-debug revisions."""

    name: str
    include_public_tests: bool = True
    require_code_fence: bool = True

    def render(
        self,
        task: TaskSpec,
        *,
        current_code: str,
        observation: str,
        revision_index: int,
    ) -> str:
        parts = [
            "You are revising a Python solution after public-test feedback.",
            "Return a complete corrected solution, not a patch.",
            "Do not mention the debugging process.",
        ]
        if self.require_code_fence:
            parts.append("Wrap the revised code in a single ```python fenced block.")
        parts.extend(
            [
                "",
                f"Revision attempt: {revision_index}",
                "",
                "Task:",
                task.prompt.strip(),
            ]
        )
        if task.entry_point:
            parts.extend(["", f"Required entry point: `{task.entry_point}`"])
        if task.starter_code.strip():
            parts.extend(["", "Starter code:", "```python", task.starter_code.strip(), "```"])
        if self.include_public_tests and task.public_tests:
            parts.extend(["", "Public tests:", "```python"])
            parts.extend(test.code.strip() for test in task.public_tests)
            parts.append("```")
        parts.extend(
            [
                "",
                "Current code:",
                "```python",
                current_code.strip(),
                "```",
                "",
                "Public-test feedback:",
                observation.strip() or "public tests failed",
            ]
        )
        return "\n".join(parts).strip() + "\n"


DIRECT_SOLUTION_TEMPLATE = PromptTemplate(
    name="direct_solution_v1",
    include_public_tests=False,
    require_code_fence=True,
)

DIRECT_WITH_PUBLIC_TESTS_TEMPLATE = PromptTemplate(
    name="direct_solution_with_public_tests_v1",
    include_public_tests=True,
    require_code_fence=True,
)

SELF_DEBUG_REVISION_TEMPLATE = RevisionPromptTemplate(
    name="self_debug_revision_v1",
    include_public_tests=True,
    require_code_fence=True,
)


def get_prompt_template(name: str) -> PromptTemplate:
    templates = {
        DIRECT_SOLUTION_TEMPLATE.name: DIRECT_SOLUTION_TEMPLATE,
        DIRECT_WITH_PUBLIC_TESTS_TEMPLATE.name: DIRECT_WITH_PUBLIC_TESTS_TEMPLATE,
    }
    try:
        return templates[name]
    except KeyError as exc:
        available = ", ".join(sorted(templates))
        raise ValueError(f"unknown prompt template {name!r}; available: {available}") from exc


def get_revision_prompt_template(name: str) -> RevisionPromptTemplate:
    templates = {
        SELF_DEBUG_REVISION_TEMPLATE.name: SELF_DEBUG_REVISION_TEMPLATE,
    }
    try:
        return templates[name]
    except KeyError as exc:
        available = ", ".join(sorted(templates))
        raise ValueError(
            f"unknown revision prompt template {name!r}; available: {available}"
        ) from exc
