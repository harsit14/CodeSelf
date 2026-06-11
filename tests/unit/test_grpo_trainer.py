from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.training import (  # noqa: E402
    GRPOLossConfig,
    GRPOModelTrainingConfig,
    OptimizerConfig,
    SequenceTrainingSample,
    TokenLogprobs,
    TrainingBatch,
    assign_group_relative_advantages,
    build_prompt_response_mask,
    run_grpo_model_training,
    torch_training_available,
)


@unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
class GRPOModelTrainerTests(unittest.TestCase):
    def test_model_training_runs_updates_and_writes_outputs(self) -> None:
        import torch

        policy_model = _ToyCausalLM(vocab_size=8)
        batches = (
            _training_batch(input_ids_a=(1, 2, 3), input_ids_b=(1, 4, 5)),
            _training_batch(input_ids_a=(1, 2, 6), input_ids_b=(1, 4, 7)),
        )
        before = policy_model.logit_table.detach().clone()
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.jsonl"
            checkpoint_path = Path(tmpdir) / "checkpoint_manifest.json"
            state_dir = Path(tmpdir) / "state"
            result = run_grpo_model_training(
                batches,
                policy_model=policy_model,
                config=GRPOModelTrainingConfig(
                    optimizer=OptimizerConfig(
                        learning_rate=0.05,
                        gradient_accumulation_steps=2,
                        max_grad_norm=1.0,
                    ),
                    loss=GRPOLossConfig(kl_beta=0.0),
                ),
                metrics_path=metrics_path,
                checkpoint_path=checkpoint_path,
                state_dir=state_dir,
            )
            metrics_lines = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
            ]
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            artifacts = {artifact["kind"]: artifact for artifact in checkpoint["artifacts"]}
            loaded_policy_state = torch.load(
                artifacts["policy_model_state"]["path"],
                weights_only=True,
            )

        self.assertEqual(result.loop.microbatch_count, 2)
        self.assertEqual(result.loop.optimizer_step_count, 1)
        self.assertEqual(result.policy_model_class, "_ToyCausalLM")
        self.assertEqual(result.optimizer_class, "AdamW")
        self.assertTrue(policy_model.train_called)
        self.assertFalse(torch.equal(before, policy_model.logit_table.detach()))
        self.assertEqual(len(metrics_lines), 2)
        self.assertEqual(metrics_lines[0]["phase"], "grpo_model_training")
        self.assertEqual(checkpoint["kind"], "grpo_model_training_checkpoint")
        self.assertTrue(checkpoint["has_real_model_weights"])
        self.assertTrue(checkpoint["has_state_artifacts"])
        self.assertEqual(set(artifacts), {"policy_model_state", "optimizer_state"})
        self.assertEqual(len(artifacts["policy_model_state"]["sha256"]), 64)
        self.assertGreater(artifacts["optimizer_state"]["bytes"], 0)
        self.assertIn("logit_table", loaded_policy_state)

    def test_model_training_uses_reference_model_for_kl(self) -> None:
        policy_model = _ToyCausalLM(vocab_size=8)
        reference_model = _ToyCausalLM(vocab_size=8)

        result = run_grpo_model_training(
            (_training_batch(input_ids_a=(1, 2, 3), input_ids_b=(1, 4, 5)),),
            policy_model=policy_model,
            reference_model=reference_model,
            config=GRPOModelTrainingConfig(
                optimizer=OptimizerConfig(learning_rate=0.05),
                loss=GRPOLossConfig(kl_beta=0.1),
            ),
        )

        self.assertTrue(reference_model.eval_called)
        self.assertEqual(result.reference_model_class, "_ToyCausalLM")
        self.assertGreaterEqual(result.loop.mean_kl_loss, 0.0)

    def test_model_training_writes_save_pretrained_artifacts(self) -> None:
        policy_model = _PretrainedToyCausalLM(vocab_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / "state"
            result = run_grpo_model_training(
                (_training_batch(input_ids_a=(1, 2, 3), input_ids_b=(1, 4, 5)),),
                policy_model=policy_model,
                config=GRPOModelTrainingConfig(loss=GRPOLossConfig(kl_beta=0.0)),
                state_dir=state_dir,
            )
            artifacts = {artifact.kind: artifact for artifact in result.checkpoint_artifacts}

        self.assertIn("policy_model_state", artifacts)
        self.assertIn("policy_pretrained:adapter_config.json", artifacts)
        self.assertIn("policy_pretrained:adapter_model.bin", artifacts)
        self.assertEqual(len(artifacts["policy_pretrained:adapter_config.json"].sha256), 64)
        self.assertGreater(artifacts["policy_pretrained:adapter_model.bin"].bytes, 0)

    def test_model_training_rejects_model_without_trainable_parameters(self) -> None:
        with self.assertRaises(ValueError):
            run_grpo_model_training(
                (_training_batch(input_ids_a=(1, 2, 3), input_ids_b=(1, 4, 5)),),
                policy_model=_NoParameterModel(),
                config=GRPOModelTrainingConfig(loss=GRPOLossConfig(kl_beta=0.0)),
            )

    def test_model_training_config_validates_ranges(self) -> None:
        with self.assertRaises(ValueError):
            GRPOModelTrainingConfig(max_batches=0)
        with self.assertRaises(ValueError):
            GRPOModelTrainingConfig(pad_token_id=-1)
        with self.assertRaises(ValueError):
            GRPOModelTrainingConfig(dtype="int8")


class _ToyCausalLM:
    def __init__(self, *, vocab_size: int) -> None:
        import torch

        self.logit_table = torch.nn.Parameter(torch.zeros((vocab_size, vocab_size)))
        self.train_called = False
        self.eval_called = False

    def parameters(self) -> tuple[object]:
        return (self.logit_table,)

    def train(self) -> None:
        self.train_called = True

    def eval(self) -> None:
        self.eval_called = True

    def state_dict(self) -> dict[str, object]:
        return {"logit_table": self.logit_table.detach().clone()}

    def __call__(self, *, input_ids: object, attention_mask: object) -> object:
        return SimpleNamespace(logits=self.logit_table[input_ids])


class _NoParameterModel:
    def parameters(self) -> tuple[object, ...]:
        return ()


class _PretrainedToyCausalLM(_ToyCausalLM):
    def save_pretrained(self, path: object) -> None:
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter_config.json").write_text(
            json.dumps({"model_type": "toy_adapter"}),
            encoding="utf-8",
        )
        (output_dir / "adapter_model.bin").write_bytes(b"toy-adapter-state")


def _training_batch(
    *,
    input_ids_a: tuple[int, ...],
    input_ids_b: tuple[int, ...],
) -> TrainingBatch:
    return assign_group_relative_advantages(
        TrainingBatch(
            (
                _sample(sample_index=0, reward=0.0, input_ids=input_ids_a),
                _sample(sample_index=1, reward=1.0, input_ids=input_ids_b),
            )
        )
    ).batch


def _sample(
    *,
    sample_index: int,
    reward: float,
    input_ids: tuple[int, ...],
) -> SequenceTrainingSample:
    masks = build_prompt_response_mask(
        token_count=len(input_ids),
        prompt_token_count=1,
    )
    old_policy_logprobs = (0.0,) + tuple(-math.log(8.0) for _ in input_ids[1:])
    return SequenceTrainingSample(
        task_id="task/a",
        sample_index=sample_index,
        input_ids=input_ids,
        masks=masks,
        reward=reward,
        logprobs=TokenLogprobs(
            token_ids=input_ids,
            policy_logprobs=old_policy_logprobs,
            old_policy_logprobs=old_policy_logprobs,
        ),
    )


if __name__ == "__main__":
    unittest.main()
