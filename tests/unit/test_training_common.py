from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.training import (  # noqa: E402
    ModelRuntimeConfig,
    OptimizerConfig,
    RolloutRuntimeConfig,
    SequenceTrainingSample,
    TokenLogprobs,
    TrainingBatch,
    backend_availability,
    build_prompt_response_mask,
    build_training_core_config,
    get_backend_spec,
    list_backend_specs,
    mask_mean,
    mask_sum,
    summarize_logprobs,
)


class TrainingCommonTests(unittest.TestCase):
    def test_backend_registry_has_smoke_and_trainable_backends(self) -> None:
        specs = {spec.name: spec for spec in list_backend_specs()}

        self.assertIn("smoke", specs)
        self.assertIn("from_scratch", specs)
        self.assertFalse(specs["smoke"].updates_model_weights)
        self.assertTrue(specs["from_scratch"].updates_model_weights)
        self.assertEqual(get_backend_spec("trl").name, "trl")

    def test_backend_availability_reports_missing_packages_without_import_side_effects(self) -> None:
        availability = backend_availability("smoke")

        self.assertTrue(availability.available)
        self.assertEqual(availability.missing_packages, ())

    def test_model_runtime_config_validates_lora_vs_full_finetune(self) -> None:
        config = ModelRuntimeConfig(name="Qwen/Qwen2.5-Coder-0.5B", tokenizer=None)

        self.assertEqual(config.tokenizer_name, "Qwen/Qwen2.5-Coder-0.5B")
        with self.assertRaises(ValueError):
            ModelRuntimeConfig(name="m", use_lora=True, full_finetune=True)

    def test_training_core_config_builds_from_nested_mapping(self) -> None:
        config = build_training_core_config(
            {
                "experiment": {"seed": 7},
                "model": {
                    "name": "Qwen/Qwen2.5-Coder-0.5B",
                    "tokenizer": "Qwen/Qwen2.5-Coder-0.5B",
                    "lora_rank": 8,
                    "dtype": "bf16",
                },
                "rollout": {"group_size": 3, "max_response_tokens": 128},
                "training": {
                    "algorithm": "grpo",
                    "backend": "smoke",
                    "max_steps": 5,
                    "learning_rate": 2e-6,
                    "kl_beta": 0.02,
                },
            }
        )

        self.assertEqual(config.algorithm, "grpo")
        self.assertEqual(config.backend, "smoke")
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.model.lora_rank, 8)
        self.assertEqual(config.optimizer.learning_rate, 2e-6)
        self.assertEqual(config.rollout.group_size, 3)
        self.assertEqual(config.rollout.max_response_tokens, 128)

    def test_optimizer_and_rollout_configs_validate_ranges(self) -> None:
        self.assertEqual(OptimizerConfig().gradient_accumulation_steps, 1)
        self.assertEqual(RolloutRuntimeConfig().top_p, 0.95)
        with self.assertRaises(ValueError):
            OptimizerConfig(learning_rate=0.0)
        with self.assertRaises(ValueError):
            RolloutRuntimeConfig(top_p=1.5)

    def test_prompt_response_masks_are_non_overlapping(self) -> None:
        mask = build_prompt_response_mask(
            token_count=7,
            prompt_token_count=3,
            response_token_count=2,
            pad_token_count=2,
        )

        self.assertEqual(mask.attention_mask, (1, 1, 1, 1, 1, 0, 0))
        self.assertEqual(mask.prompt_mask, (1, 1, 1, 0, 0, 0, 0))
        self.assertEqual(mask.response_mask, (0, 0, 0, 1, 1, 0, 0))
        self.assertEqual(mask.response_tokens, 2)

    def test_mask_sum_and_mean_use_included_positions(self) -> None:
        self.assertEqual(mask_sum([1.0, 2.0, 3.0], [1, 0, 1]), 4.0)
        self.assertEqual(mask_mean([1.0, 2.0, 3.0], [1, 0, 1]), 2.0)
        self.assertEqual(mask_mean([1.0, 2.0, 3.0], [0, 0, 0]), 0.0)
        with self.assertRaises(ValueError):
            mask_sum([1.0], [1, 0])

    def test_logprob_summary_uses_response_tokens_only(self) -> None:
        mask = build_prompt_response_mask(token_count=4, prompt_token_count=2)
        logprobs = TokenLogprobs(
            token_ids=(10, 11, 12, 13),
            policy_logprobs=(-9.0, -9.0, -0.2, -0.4),
            reference_logprobs=(-9.0, -9.0, -0.3, -0.8),
            old_policy_logprobs=(-9.0, -9.0, -0.1, -0.6),
            entropy=(0.0, 0.0, 0.7, 0.9),
        )

        summary = summarize_logprobs(logprobs, mask)

        self.assertEqual(summary.response_tokens, 2)
        self.assertAlmostEqual(summary.mean_policy_logprob, -0.3)
        self.assertAlmostEqual(summary.mean_reference_logprob, -0.55)
        self.assertAlmostEqual(summary.mean_kl_to_reference, 0.25)
        self.assertGreater(summary.mean_importance_ratio, 0.0)
        self.assertAlmostEqual(summary.mean_entropy, 0.8)

    def test_training_batch_groups_samples_and_summarizes(self) -> None:
        mask = build_prompt_response_mask(token_count=4, prompt_token_count=2)
        first = SequenceTrainingSample(
            task_id="task/a",
            sample_index=1,
            input_ids=(1, 2, 3, 4),
            masks=mask,
            reward=0.5,
        )
        second = SequenceTrainingSample(
            task_id="task/a",
            sample_index=0,
            input_ids=(1, 2, 5, 6),
            masks=mask,
            reward=1.0,
        )
        batch = TrainingBatch((first, second))

        self.assertEqual(batch.size, 2)
        self.assertEqual(batch.total_response_tokens, 4)
        self.assertAlmostEqual(batch.mean_reward, 0.75)
        self.assertEqual([sample.sample_index for sample in batch.by_task()["task/a"]], [0, 1])

    def test_inspect_training_core_script_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "training_core.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "inspect_training_core.py"),
                    "--config",
                    str(ROOT / "configs" / "experiments" / "training_core_debug.example.json"),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(payload["training_core"]["algorithm"], "grpo")
        self.assertEqual(payload["training_core"]["backend"], "smoke")
        self.assertTrue(payload["backend_availability"]["available"])


if __name__ == "__main__":
    unittest.main()
