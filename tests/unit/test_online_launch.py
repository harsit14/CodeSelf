from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.datasets import Split, TaskRegistry, TaskSpec, TestSpec  # noqa: E402
from codeself.training import (  # noqa: E402
    CausalLMWithValueHead,
    GenerationRequest,
    build_generated_sequence,
    run_online_training_from_config,
)
import codeself.training.online_launch as online_launch_module  # noqa: E402
from codeself.training import torch_training_available  # noqa: E402


@unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
class OnlineTrainingLaunchTests(unittest.TestCase):
    def test_runs_grpo_online_training_from_single_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            output_dir = root / "artifacts"
            TaskRegistry([_task()]).to_jsonl(tasks_path)

            launch = run_online_training_from_config(
                _base_config(
                    "grpo",
                    tasks_path=tasks_path,
                    output_dir=output_dir,
                    training={
                        "max_batches": 1,
                        "optimizer": {"learning_rate": 0.05},
                        "loss": {"kl_beta": 0.0},
                    },
                )
            )
            summary = json.loads(
                (output_dir / "launch_summary.json").read_text(encoding="utf-8")
            )
            rollouts_exists = (output_dir / "cycle_0001" / "rollouts.jsonl").exists()
            checkpoint_exists = (output_dir / "cycle_0001" / "checkpoint.json").exists()

        self.assertEqual(launch.algorithm, "grpo")
        self.assertEqual(launch.model_backend, "toy")
        self.assertEqual(launch.task_count, 1)
        self.assertEqual(launch.cycle_count, 1)
        self.assertEqual(launch.total_rollouts, 2)
        self.assertEqual(launch.records_used, 2)
        self.assertEqual(summary["algorithm"], "grpo")
        self.assertEqual(summary["total_rollouts"], 2)
        self.assertTrue(rollouts_exists)
        self.assertTrue(checkpoint_exists)

    def test_run_online_training_script_runs_ppo_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            output_dir = root / "artifacts"
            config_path = root / "ppo_online.json"
            TaskRegistry([_task()]).to_jsonl(tasks_path)
            config = _base_config(
                "ppo",
                tasks_path=tasks_path,
                output_dir=output_dir,
                training={
                    "max_batches": 1,
                    "optimizer": {"learning_rate": 0.05},
                    "loss": {
                        "kl_beta": 0.0,
                        "normalize_advantages": False,
                        "value_clip_epsilon": None,
                    },
                },
            )
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_online_training.py"),
                    "--config",
                    str(config_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(
                (output_dir / "launch_summary.json").read_text(encoding="utf-8")
            )
            metrics_exists = (output_dir / "cycle_0001" / "metrics.jsonl").exists()
            checkpoint_exists = (output_dir / "cycle_0001" / "checkpoint.json").exists()

        self.assertIn("algorithm: ppo", result.stdout)
        self.assertIn("model_backend: toy", result.stdout)
        self.assertEqual(summary["algorithm"], "ppo")
        self.assertEqual(summary["total_rollouts"], 2)
        self.assertTrue(metrics_exists)
        self.assertTrue(checkpoint_exists)

    def test_runs_ppo_transformers_with_integrated_value_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            TaskRegistry([_task()]).to_jsonl(tasks_path)
            config = _base_config(
                "ppo",
                tasks_path=tasks_path,
                output_dir=root / "artifacts",
                model={
                    "backend": "transformers",
                    "name": "local-model",
                    "use_lora": True,
                    "full_finetune": False,
                    "lora_rank": 4,
                    "lora_alpha": 8,
                    "lora_dropout": 0.1,
                    "lora_target_modules": ["q_proj", "v_proj"],
                    "value_head": {"hidden_size": 4},
                    "old_policy": {"enabled": True},
                    "old_value": {"enabled": True},
                },
                training={
                    "max_batches": 1,
                    "optimizer": {"learning_rate": 0.05},
                    "loss": {
                        "kl_beta": 0.0,
                        "normalize_advantages": False,
                        "value_clip_epsilon": None,
                    },
                },
            )

            _FakeTransformersEngine.seen_model_names = []
            with patch.object(
                online_launch_module,
                "TransformersModelEngine",
                _FakeTransformersEngine,
            ):
                launch = run_online_training_from_config(config)

        self.assertEqual(launch.algorithm, "ppo")
        self.assertEqual(launch.model_backend, "transformers")
        self.assertEqual(launch.generation_backend, "policy_engine")
        model_runtime = online_launch_module._build_model_runtime_config(config)
        self.assertTrue(model_runtime.use_lora)
        self.assertFalse(model_runtime.full_finetune)
        self.assertEqual(model_runtime.lora_rank, 4)
        self.assertEqual(model_runtime.lora_alpha, 8)
        self.assertAlmostEqual(model_runtime.lora_dropout, 0.1)
        self.assertEqual(model_runtime.lora_target_modules, ("q_proj", "v_proj"))
        self.assertEqual(launch.total_rollouts, 2)
        self.assertEqual(launch.records_used, 2)
        self.assertEqual(
            launch.result.steps[0].result.training.policy_model_class,
            "CausalLMWithValueHead",
        )
        self.assertTrue(launch.result.steps[0].old_policy_synced)
        self.assertTrue(launch.result.steps[0].old_value_synced)
        self.assertEqual(
            _FakeTransformersEngine.seen_model_names,
            ["local-model", "local-model"],
        )

    def test_runs_ppo_transformers_with_separate_value_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            TaskRegistry([_task()]).to_jsonl(tasks_path)
            config = _base_config(
                "ppo",
                tasks_path=tasks_path,
                output_dir=root / "artifacts",
                model={
                    "backend": "transformers",
                    "name": "policy-model",
                    "use_lora": True,
                    "full_finetune": False,
                    "lora_rank": 4,
                    "lora_alpha": 8,
                    "lora_target_modules": ["q_proj", "v_proj"],
                    "value_model": {
                        "enabled": True,
                        "name": "critic-model",
                        "use_lora": False,
                        "full_finetune": True,
                        "value_head": {"hidden_size": 4},
                    },
                    "old_policy": {"enabled": True},
                    "old_value": {"enabled": True},
                },
                training={
                    "max_batches": 1,
                    "optimizer": {"learning_rate": 0.05},
                    "loss": {
                        "kl_beta": 0.0,
                        "normalize_advantages": False,
                        "value_clip_epsilon": None,
                    },
                },
            )

            _FakeTransformersEngine.seen_model_names = []
            with patch.object(
                online_launch_module,
                "TransformersModelEngine",
                _FakeTransformersEngine,
            ):
                launch = run_online_training_from_config(config)

        policy_runtime = online_launch_module._build_model_runtime_config(config)
        value_engine_config = online_launch_module._build_value_model_engine_config(
            config,
            policy_runtime,
        )
        training = launch.result.steps[0].result.training

        self.assertEqual(launch.algorithm, "ppo")
        self.assertEqual(training.policy_model_class, "_FakeTransformerCausalLM")
        self.assertEqual(training.value_model_class, "CausalLMWithValueHead")
        self.assertEqual(training.old_policy_model_class, "_FakeTransformerCausalLM")
        self.assertEqual(training.old_value_model_class, "CausalLMWithValueHead")
        self.assertTrue(launch.result.steps[0].old_policy_synced)
        self.assertTrue(launch.result.steps[0].old_value_synced)
        self.assertEqual(value_engine_config.model.name, "critic-model")
        self.assertFalse(value_engine_config.model.use_lora)
        self.assertTrue(value_engine_config.model.full_finetune)
        self.assertEqual(
            _FakeTransformersEngine.seen_model_names,
            ["policy-model", "critic-model", "policy-model", "critic-model"],
        )

    def test_value_head_wrapper_returns_logits_and_values(self) -> None:
        import torch

        model = CausalLMWithValueHead(_FakeTransformerCausalLM(vocab_size=8, hidden_size=4))
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

        self.assertEqual(output.logits.shape, (1, 3, 8))
        self.assertEqual(output.values.shape, (1, 3))
        self.assertTrue(any(parameter.requires_grad for parameter in model.parameters()))

    def test_value_head_wrapper_loads_saved_state_paths(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = CausalLMWithValueHead(
                _FakeTransformerCausalLM(vocab_size=8, hidden_size=4)
            )
            target = CausalLMWithValueHead(
                _FakeTransformerCausalLM(vocab_size=8, hidden_size=4)
            )
            with torch.no_grad():
                source.value_head.weight.fill_(0.25)
                source.value_head.bias.fill_(0.5)
            state_path = root / "value_model.pt"
            value_head_path = root / "value_head.pt"
            torch.save(source.state_dict(), state_path)
            torch.save(source.value_head.state_dict(), value_head_path)

            target.load_state_path(state_path)
            from_full_state = target.value_head.weight.detach().clone()
            with torch.no_grad():
                target.value_head.weight.zero_()
                target.value_head.bias.zero_()
            target.load_value_head_path(value_head_path)

        self.assertTrue(torch.equal(from_full_state, source.value_head.weight))
        self.assertTrue(torch.equal(target.value_head.weight, source.value_head.weight))
        self.assertTrue(torch.equal(target.value_head.bias, source.value_head.bias))


def _base_config(
    algorithm: str,
    *,
    tasks_path: Path,
    output_dir: Path,
    training: dict[str, object] | None = None,
    model: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "algorithm": algorithm,
        "data": {
            "tasks": str(tasks_path),
            "split": "train",
            "limit": 1,
            "skip_quality_checks": True,
        },
        "output": {"dir": str(output_dir)},
        "model": model or {"backend": "toy", "vocab_size": 512},
        "prompt": {"template": "direct_solution_v1"},
        "online": {"cycles": 1},
        "rollout": {
            "mode": "direct",
            "group_size": 2,
            "samples_per_task": 2,
            "max_prompt_tokens": 64,
            "max_response_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "batch": {"include_failed_parses": True},
        "training": training or {"max_batches": 1},
    }


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="agent/add-one",
        source="unit-test",
        prompt="Write add_one(x).",
        split=Split.TRAIN,
        entry_point="add_one",
        public_tests=(TestSpec(name="public", code="assert add_one(1) == 2"),),
        hidden_tests=(),
    )


class _FakeTransformersEngine:
    seen_model_names: list[str] = []

    def __init__(self, config: object) -> None:
        self.config = config
        self.seen_model_names.append(str(config.model.name))
        self.tokenizer = _SequentialTokenizer()
        self.model = _FakeTransformerCausalLM(vocab_size=512, hidden_size=4)

    def generate(self, request: GenerationRequest) -> object:
        return build_generated_sequence(
            self.tokenizer,
            prompt=request.prompt,
            response="```python\ndef add_one(x):\n    return x + 1\n```",
            max_response_tokens=request.max_new_tokens,
            finish_reason="stop",
            metadata={"engine": "fake_transformers"},
        )


class _FakeTransformerCausalLM:
    def __init__(self, *, vocab_size: int, hidden_size: int) -> None:
        import torch

        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.logit_table = torch.nn.Parameter(torch.zeros((vocab_size, vocab_size)))
        self.hidden_table = torch.nn.Parameter(torch.zeros((vocab_size, hidden_size)))

    def parameters(self) -> tuple[object, object]:
        return (self.logit_table, self.hidden_table)

    def train(self) -> None:
        return None

    def eval(self) -> None:
        return None

    def state_dict(self) -> dict[str, object]:
        return {
            "logit_table": self.logit_table.detach().clone(),
            "hidden_table": self.hidden_table.detach().clone(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        import torch

        with torch.no_grad():
            self.logit_table.copy_(state["logit_table"])
            self.hidden_table.copy_(state["hidden_table"])

    def __call__(
        self,
        *,
        input_ids: object,
        attention_mask: object,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> object:
        logits = self.logit_table[input_ids]
        hidden = self.hidden_table[input_ids]
        if return_dict:
            return SimpleNamespace(logits=logits, hidden_states=(hidden,))
        return (logits, (hidden,))


class _SequentialTokenizer:
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    @property
    def eos_token_id(self) -> int | None:
        return 0

    def encode(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
        truncation_side: str = "right",
    ) -> tuple[int, ...]:
        token_ids = tuple(self._id_for_token(token) for token in text.split())
        if max_tokens is None or len(token_ids) <= max_tokens:
            return token_ids
        if truncation_side == "left":
            return token_ids[-max_tokens:]
        return token_ids[:max_tokens]

    def decode(self, token_ids: Any) -> str:
        return " ".join(self._id_to_token[int(token_id)] for token_id in token_ids)

    def _id_for_token(self, token: str) -> int:
        if token not in self._token_to_id:
            token_id = len(self._token_to_id) + 1
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return self._token_to_id[token]


if __name__ == "__main__":
    unittest.main()
