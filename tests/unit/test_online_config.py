from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.config import load_config_file  # noqa: E402
from codeself.training import (  # noqa: E402
    GRPOOnlineTrainingConfig,
    PPOOnlineTrainingConfig,
    build_grpo_online_training_config,
    build_online_training_config,
    build_ppo_online_training_config,
    resolve_online_algorithm,
)


class OnlineTrainingConfigBuilderTests(unittest.TestCase):
    def test_builds_grpo_self_debug_config_from_mapping(self) -> None:
        config = {
            "algorithm": "grpo",
            "experiment": {"seed": 123},
            "online": {
                "cycles": 3,
                "seed_stride": 7,
                "sync_old_policy_before_cycle": False,
            },
            "rollout": {
                "mode": "self_debug",
                "group_size": 2,
                "samples_per_task": 4,
                "max_prompt_tokens": 128,
                "max_response_tokens": 64,
                "temperature": 0.7,
                "top_p": 0.9,
            },
            "self_debug": {
                "max_revisions": 2,
                "revision_strategy": "model",
                "revision_prompt_template": "self_debug_revision_v2",
                "revision_reward_discount": 0.75,
                "revision_reward_discount_mode": "all",
                "revision_reward_step_penalty": 0.05,
                "revision_max_new_tokens": 96,
                "revision_temperature": 0.25,
                "revision_top_p": 0.85,
            },
            "execution": {"include_hidden": False},
            "batch": {
                "response_source": "parsed_code",
                "include_failed_parses": False,
                "min_reward_std": 0.01,
            },
            "training": {
                "max_batches": 5,
                "step_scheduler": False,
                "dtype": "bf16",
                "optimizer": {
                    "learning_rate": 2e-6,
                    "weight_decay": 0.01,
                    "warmup_steps": 3,
                    "gradient_accumulation_steps": 2,
                    "max_grad_norm": 0.5,
                },
                "loss": {
                    "clip_epsilon": 0.15,
                    "kl_beta": 0.02,
                    "max_log_ratio": 12.0,
                },
            },
            "evaluation": {
                "enabled": True,
                "samples_per_task": 2,
                "max_new_tokens": 32,
                "temperature": 0.1,
                "top_p": 0.8,
                "include_hidden": False,
                "ks": [1, 2],
            },
        }

        built = build_grpo_online_training_config(config)

        self.assertIsInstance(built, GRPOOnlineTrainingConfig)
        self.assertEqual(resolve_online_algorithm(config), "grpo")
        self.assertEqual(built.cycles, 3)
        self.assertEqual(built.seed_stride, 7)
        self.assertFalse(built.sync_old_policy_before_cycle)
        self.assertEqual(built.cycle_config_for(2).seed, 130)
        self.assertEqual(built.cycle.rollout_mode, "self_debug")
        self.assertEqual(built.cycle.rollout.samples_per_task, 4)
        self.assertEqual(built.cycle.self_debug.revision_strategy, "model")
        self.assertEqual(
            built.cycle.self_debug.revision_prompt_template,
            "self_debug_revision_v2",
        )
        self.assertAlmostEqual(built.cycle.self_debug.revision_reward_discount, 0.75)
        self.assertEqual(built.cycle.self_debug.revision_reward_discount_mode, "all")
        self.assertAlmostEqual(built.cycle.self_debug.revision_reward_step_penalty, 0.05)
        self.assertEqual(built.cycle.self_debug.revision_max_new_tokens, 96)
        self.assertAlmostEqual(built.cycle.self_debug.revision_temperature, 0.25)
        self.assertAlmostEqual(built.cycle.self_debug.revision_top_p, 0.85)
        self.assertFalse(built.cycle.include_hidden)
        self.assertEqual(built.cycle.batch.response_source, "parsed_code")
        self.assertFalse(built.cycle.batch.include_failed_parses)
        self.assertAlmostEqual(built.cycle.batch.min_reward_std, 0.01)
        self.assertEqual(built.cycle.training.max_batches, 5)
        self.assertFalse(built.cycle.training.step_scheduler)
        self.assertEqual(built.cycle.training.dtype, "bf16")
        self.assertAlmostEqual(built.cycle.training.optimizer.learning_rate, 2e-6)
        self.assertAlmostEqual(built.cycle.training.optimizer.weight_decay, 0.01)
        self.assertEqual(built.cycle.training.optimizer.gradient_accumulation_steps, 2)
        self.assertAlmostEqual(built.cycle.training.optimizer.max_grad_norm, 0.5)
        self.assertAlmostEqual(built.cycle.training.loss.clip_epsilon, 0.15)
        self.assertAlmostEqual(built.cycle.training.loss.kl_beta, 0.02)
        self.assertIsNotNone(built.evaluation)
        assert built.evaluation is not None
        self.assertEqual(built.evaluation.ks, (1, 2))
        self.assertFalse(built.evaluation.include_hidden)

    def test_builds_ppo_self_debug_config_from_mapping(self) -> None:
        config = {
            "experiment": {"seed": 50},
            "online": {
                "cycles": 2,
                "seed_stride": 5,
                "sync_old_value_before_cycle": False,
            },
            "generation": {
                "group_size": 3,
                "samples_per_task": 3,
                "max_new_tokens": 96,
                "temperature": 0.6,
                "top_p": 0.88,
            },
            "rollout": {"mode": "self_debug", "max_prompt_tokens": 256},
            "agent": {"max_revisions": 0, "use_rule_based_repair": False},
            "training": {
                "algorithm": "ppo",
                "learning_rate": 3e-6,
                "gradient_accumulation_steps": 3,
                "value_clip_epsilon": None,
                "value_loss_coef": 0.7,
                "entropy_coef": 0.01,
                "normalize_advantages": False,
                "gamma": 0.99,
                "gae_lambda": 0.95,
            },
            "evaluation": {"enabled": False},
        }

        built = build_ppo_online_training_config(config)

        self.assertIsInstance(built, PPOOnlineTrainingConfig)
        self.assertEqual(resolve_online_algorithm(config), "ppo")
        self.assertEqual(built.cycles, 2)
        self.assertEqual(built.cycle_config_for(2).seed, 55)
        self.assertFalse(built.sync_old_value_before_cycle)
        self.assertEqual(built.cycle.rollout.group_size, 3)
        self.assertEqual(built.cycle.rollout.samples_per_task, 3)
        self.assertEqual(built.cycle.rollout.max_response_tokens, 96)
        self.assertEqual(built.cycle.self_debug.revision_strategy, "none")
        self.assertEqual(built.cycle.self_debug.max_revisions, 0)
        self.assertAlmostEqual(built.cycle.training.optimizer.learning_rate, 3e-6)
        self.assertEqual(built.cycle.training.optimizer.gradient_accumulation_steps, 3)
        self.assertIsNone(built.cycle.training.loss.value_clip_epsilon)
        self.assertAlmostEqual(built.cycle.training.loss.value_loss_coef, 0.7)
        self.assertAlmostEqual(built.cycle.training.loss.entropy_coef, 0.01)
        self.assertFalse(built.cycle.training.loss.normalize_advantages)
        self.assertAlmostEqual(built.cycle.training.loss.gamma, 0.99)
        self.assertAlmostEqual(built.cycle.training.loss.gae_lambda, 0.95)
        self.assertIsNone(built.evaluation)

    def test_example_configs_load_into_typed_online_configs(self) -> None:
        cases = (
            (
                "grpo",
                "self_debug",
                True,
                ROOT / "configs/experiments/grpo_online_self_debug.example.yaml",
            ),
            (
                "ppo",
                "self_debug",
                True,
                ROOT / "configs/experiments/ppo_online_self_debug.example.yaml",
            ),
            (
                "grpo",
                "self_debug",
                True,
                ROOT / "configs/experiments/grpo_online_transformers_launch.example.yaml",
            ),
            (
                "ppo",
                "direct",
                False,
                ROOT / "configs/experiments/ppo_online_toy_launch.example.yaml",
            ),
            (
                "ppo",
                "self_debug",
                True,
                ROOT / "configs/experiments/ppo_online_transformers_launch.example.yaml",
            ),
        )

        for algorithm, rollout_mode, evaluation_enabled, path in cases:
            with self.subTest(path=path):
                raw_config = load_config_file(path)
                built = build_online_training_config(raw_config)

                self.assertEqual(resolve_online_algorithm(raw_config), algorithm)
                self.assertEqual(built.cycle.rollout_mode, rollout_mode)
                self.assertEqual(built.evaluation is not None, evaluation_enabled)

    def test_inspect_online_training_config_script_writes_normalized_json(self) -> None:
        config = {
            "algorithm": "grpo",
            "experiment": {"seed": 9},
            "online": {"cycles": 2},
            "rollout": {"mode": "self_debug", "samples_per_task": 2},
            "self_debug": {"max_revisions": 1},
            "evaluation": {"enabled": True, "ks": [1]},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "online.json"
            output_path = tmp_path / "normalized.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/inspect_online_training_config.py"),
                    "--config",
                    str(config_path),
                    "--output",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            normalized = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("algorithm: grpo", result.stdout)
        self.assertIn("rollout_mode: self_debug", result.stdout)
        self.assertIn("evaluation_enabled: true", result.stdout)
        self.assertEqual(normalized["algorithm"], "grpo")
        self.assertEqual(normalized["online_training"]["cycles"], 2)
        self.assertEqual(normalized["online_training"]["cycle"]["seed"], 9)


if __name__ == "__main__":
    unittest.main()
