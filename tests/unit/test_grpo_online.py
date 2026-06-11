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
    GRPOLossConfig,
    GRPOModelTrainingConfig,
    GRPOOnlineEvaluationConfig,
    GRPOOnlineTrainingConfig,
    GRPORolloutTrainingCycleConfig,
    OptimizerConfig,
    RolloutRuntimeConfig,
    run_grpo_online_training,
    torch_training_available,
    truncate_token_ids,
)


class GRPOOnlineTrainingTests(unittest.TestCase):
    def test_config_builds_per_cycle_seeds(self) -> None:
        config = GRPOOnlineTrainingConfig(
            cycles=3,
            cycle=GRPORolloutTrainingCycleConfig(seed=10),
            seed_stride=5,
        )

        self.assertEqual(config.cycle_config_for(1).seed, 10)
        self.assertEqual(config.cycle_config_for(2).seed, 15)
        self.assertEqual(config.cycle_config_for(3).seed, 20)
        self.assertEqual(config.to_dict()["cycles"], 3)
        self.assertTrue(config.to_dict()["sync_old_policy_before_cycle"])
        with self.assertRaises(ValueError):
            GRPOOnlineTrainingConfig(cycles=0)
        with self.assertRaises(ValueError):
            GRPOOnlineTrainingConfig(seed_stride=0)
        with self.assertRaises(ValueError):
            config.cycle_config_for(0)

    def test_online_training_rejects_empty_tasks_before_torch_work(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one task"):
            run_grpo_online_training(
                (),
                generator=MockGenerator(),
                prompt_template=DIRECT_SOLUTION_TEMPLATE,
                tokenizer=_SequentialTokenizerEngine(),
                policy_model=object(),
            )

    @unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
    def test_online_training_runs_multiple_cycles_and_writes_artifacts(self) -> None:
        import torch

        tokenizer = _SequentialTokenizerEngine()
        policy_model = _ToyCausalLM(vocab_size=512)
        old_policy_model = _ToyCausalLM(vocab_size=512)
        before = policy_model.logit_table.detach().clone()
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "online"
            result = run_grpo_online_training(
                [_task()],
                generator=MockGenerator(),
                prompt_template=DIRECT_SOLUTION_TEMPLATE,
                tokenizer=tokenizer,
                policy_model=policy_model,
                old_policy_model=old_policy_model,
                eval_tasks=[_task()],
                config=GRPOOnlineTrainingConfig(
                    cycles=2,
                    evaluation=GRPOOnlineEvaluationConfig(
                        samples_per_task=1,
                        max_new_tokens=64,
                        ks=(1,),
                    ),
                    cycle=GRPORolloutTrainingCycleConfig(
                        seed=4,
                        rollout=RolloutRuntimeConfig(
                            group_size=2,
                            samples_per_task=2,
                            max_prompt_tokens=64,
                            max_response_tokens=64,
                        ),
                        training=GRPOModelTrainingConfig(
                            optimizer=OptimizerConfig(learning_rate=0.05),
                            loss=GRPOLossConfig(kl_beta=0.0),
                        ),
                    ),
                ),
                artifact_dir=artifact_dir,
            )

            first_metrics = json.loads(
                (artifact_dir / "cycle_0001" / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            second_checkpoint = json.loads(
                (artifact_dir / "cycle_0002" / "checkpoint.json").read_text(
                    encoding="utf-8"
                )
            )
            first_rollouts_exists = (artifact_dir / "cycle_0001" / "rollouts.jsonl").exists()
            second_policy_state_exists = (
                artifact_dir / "cycle_0002" / "state" / "policy_model.pt"
            ).exists()
            first_eval = json.loads(
                (artifact_dir / "cycle_0001" / "evaluation.json").read_text(
                    encoding="utf-8"
                )
            )
            first_eval_rollouts_exists = (
                artifact_dir / "cycle_0001" / "eval_rollouts.jsonl"
            ).exists()

        self.assertEqual(result.cycle_count, 2)
        self.assertEqual(result.total_rollouts, 4)
        self.assertEqual(result.records_used, 4)
        self.assertEqual(result.total_optimizer_steps, 2)
        self.assertEqual([step.seed for step in result.steps], [4, 5])
        self.assertTrue(all(step.old_policy_synced for step in result.steps))
        self.assertTrue(result.steps[0].to_dict()["old_policy_synced"])
        self.assertEqual(old_policy_model.load_count, 2)
        self.assertEqual(result.steps[0].result.rollouts[0].metadata["online_cycle"], 1)
        self.assertEqual(result.steps[1].result.rollouts[0].metadata["online_cycle"], 2)
        self.assertEqual(first_metrics["phase"], "grpo_model_training")
        self.assertTrue(second_checkpoint["has_state_artifacts"])
        self.assertIsNotNone(result.steps[0].evaluation)
        self.assertEqual(result.steps[0].evaluation.summary.rollout_count, 1)
        self.assertEqual(first_eval["rollout_count"], 1)
        self.assertTrue(first_eval_rollouts_exists)
        self.assertTrue(first_rollouts_exists)
        self.assertTrue(second_policy_state_exists)
        self.assertGreater(result.mean_reward, 0.0)
        self.assertFalse(torch.equal(before, policy_model.logit_table.detach()))
        self.assertEqual(result.to_dict()["artifact_dir"], str(artifact_dir))


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


class _ToyCausalLM:
    def __init__(self, *, vocab_size: int) -> None:
        import torch

        self.logit_table = torch.nn.Parameter(torch.zeros((vocab_size, vocab_size)))
        self.load_count = 0

    def parameters(self) -> tuple[object]:
        return (self.logit_table,)

    def train(self) -> None:
        pass

    def state_dict(self) -> dict[str, object]:
        return {"logit_table": self.logit_table.detach().clone()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        import torch

        self.load_count += 1
        with torch.no_grad():
            self.logit_table.copy_(state["logit_table"])

    def __call__(self, *, input_ids: object, attention_mask: object) -> object:
        return SimpleNamespace(logits=self.logit_table[input_ids])


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
