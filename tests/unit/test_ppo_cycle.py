from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.agent import DIRECT_SOLUTION_TEMPLATE, MockGenerator  # noqa: E402
from codeself.datasets import Split, TaskSpec, TestSpec  # noqa: E402
from codeself.training import (  # noqa: E402
    OptimizerConfig,
    PPOLossConfig,
    PPOModelTrainingConfig,
    PPORolloutTrainingCycleConfig,
    RolloutRuntimeConfig,
    run_ppo_rollout_training_cycle,
    torch_training_available,
    truncate_token_ids,
)


class PPORolloutTrainingCycleTests(unittest.TestCase):
    def test_config_resolves_rollout_token_limits_into_batch_config(self) -> None:
        config = PPORolloutTrainingCycleConfig(
            rollout=RolloutRuntimeConfig(
                group_size=2,
                samples_per_task=2,
                max_prompt_tokens=5,
                max_response_tokens=7,
            )
        )

        batch_config = config.resolved_batch_config()

        self.assertEqual(batch_config.max_prompt_tokens, 5)
        self.assertEqual(batch_config.max_response_tokens, 7)
        self.assertEqual(config.to_dict()["training"]["loss"]["kl_beta"], 0.0)
        with self.assertRaises(ValueError):
            PPORolloutTrainingCycleConfig(seed=-1)

    def test_cycle_rejects_empty_task_list_before_torch_work(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one task"):
            run_ppo_rollout_training_cycle(
                (),
                generator=MockGenerator(),
                prompt_template=DIRECT_SOLUTION_TEMPLATE,
                tokenizer=_SequentialTokenizerEngine(),
                policy_model=object(),
            )

    @unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
    def test_cycle_collects_rollouts_trains_and_writes_artifacts(self) -> None:
        import torch

        tokenizer = _SequentialTokenizerEngine()
        policy_model = _ToyPPOCausalLM(vocab_size=512)
        before_logits = policy_model.logit_table.detach().clone()
        before_values = policy_model.value_table.detach().clone()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            rollouts_path = tmp_path / "rollouts.jsonl"
            metrics_path = tmp_path / "metrics.jsonl"
            checkpoint_path = tmp_path / "checkpoint.json"
            state_dir = tmp_path / "state"

            result = run_ppo_rollout_training_cycle(
                [_task()],
                generator=MockGenerator(),
                prompt_template=DIRECT_SOLUTION_TEMPLATE,
                tokenizer=tokenizer,
                policy_model=policy_model,
                config=PPORolloutTrainingCycleConfig(
                    seed=4,
                    rollout=RolloutRuntimeConfig(
                        group_size=2,
                        samples_per_task=2,
                        max_prompt_tokens=64,
                        max_response_tokens=64,
                        temperature=0.8,
                        top_p=0.95,
                    ),
                    training=PPOModelTrainingConfig(
                        optimizer=OptimizerConfig(learning_rate=0.05),
                        loss=PPOLossConfig(
                            kl_beta=0.0,
                            normalize_advantages=False,
                            value_clip_epsilon=None,
                        ),
                    ),
                ),
                rollouts_path=rollouts_path,
                metrics_path=metrics_path,
                checkpoint_path=checkpoint_path,
                state_dir=state_dir,
            )
            rollout_lines = rollouts_path.read_text(encoding="utf-8").splitlines()
            metric_lines = metrics_path.read_text(encoding="utf-8").splitlines()
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        self.assertEqual(result.rollout_count, 2)
        self.assertEqual(result.rollout_batch.records_used, 2)
        self.assertEqual([record.reward["reward"] for record in result.rollouts], [0.4, 1.0])
        self.assertIsNone(result.rollout_batch.batch.samples[0].advantage)
        self.assertEqual(result.training.loop.microbatch_count, 1)
        self.assertEqual(result.training.loop.optimizer_step_count, 1)
        self.assertTrue(policy_model.train_called)
        self.assertFalse(torch.equal(before_logits, policy_model.logit_table.detach()))
        self.assertFalse(torch.equal(before_values, policy_model.value_table.detach()))
        self.assertEqual(len(rollout_lines), 2)
        self.assertEqual(len(metric_lines), 1)
        self.assertEqual(json.loads(metric_lines[0])["phase"], "ppo_model_training")
        self.assertTrue(checkpoint["has_state_artifacts"])
        self.assertEqual(checkpoint["kind"], "ppo_model_training_checkpoint")
        self.assertEqual(result.to_dict()["artifacts"]["rollouts_path"], str(rollouts_path))


class _SequentialTokenizerEngine:
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
        return truncate_token_ids(
            token_ids,
            max_tokens=max_tokens,
            truncation_side=truncation_side,  # type: ignore[arg-type]
        )

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(self._id_to_token[int(token_id)] for token_id in token_ids)

    def _id_for_token(self, token: str) -> int:
        if token not in self._token_to_id:
            token_id = len(self._token_to_id) + 1
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return self._token_to_id[token]


class _ToyPPOCausalLM:
    def __init__(self, *, vocab_size: int) -> None:
        import torch

        self.logit_table = torch.nn.Parameter(torch.zeros((vocab_size, vocab_size)))
        self.value_table = torch.nn.Parameter(torch.zeros(vocab_size))
        self.train_called = False

    def parameters(self) -> tuple[object, object]:
        return (self.logit_table, self.value_table)

    def train(self) -> None:
        self.train_called = True

    def state_dict(self) -> dict[str, object]:
        return {
            "logit_table": self.logit_table.detach().clone(),
            "value_table": self.value_table.detach().clone(),
        }

    def __call__(self, *, input_ids: object, attention_mask: object) -> object:
        return SimpleNamespace(
            logits=self.logit_table[input_ids],
            values=self.value_table[input_ids],
        )


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


if __name__ == "__main__":
    unittest.main()
