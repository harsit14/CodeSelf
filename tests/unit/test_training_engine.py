from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.config import load_config_file  # noqa: E402
from codeself.agent import GenerationRequest as AgentGenerationRequest  # noqa: E402
from codeself.training import (  # noqa: E402
    GeneratedSequence,
    GenerationRequest,
    ModelEngineCodeGenerator,
    ModelRuntimeConfig,
    TransformersEngineConfig,
    WhitespaceTokenizerEngine,
    build_generated_sequence,
    build_training_core_config,
    encode_prompt_response,
    truncate_token_ids,
)


class TrainingEngineTests(unittest.TestCase):
    def test_truncate_token_ids_supports_left_and_right_sides(self) -> None:
        token_ids = (1, 2, 3, 4)

        self.assertEqual(
            truncate_token_ids(token_ids, max_tokens=2, truncation_side="right"),
            (1, 2),
        )
        self.assertEqual(
            truncate_token_ids(token_ids, max_tokens=2, truncation_side="left"),
            (3, 4),
        )
        self.assertEqual(
            truncate_token_ids(token_ids, max_tokens=None, truncation_side="left"),
            token_ids,
        )
        with self.assertRaises(ValueError):
            truncate_token_ids(token_ids, max_tokens=2, truncation_side="middle")

    def test_whitespace_tokenizer_is_deterministic_and_decodable(self) -> None:
        tokenizer = WhitespaceTokenizerEngine(eos_token_id=0)

        first = tokenizer.encode("return x plus y")
        second = tokenizer.encode("return x plus y")
        truncated = tokenizer.encode("return x plus y", max_tokens=2, truncation_side="left")

        self.assertEqual(first, second)
        self.assertEqual(truncated, first[-2:])
        self.assertEqual(tokenizer.decode(first), "return x plus y")
        self.assertEqual(tokenizer.eos_token_id, 0)

    def test_encode_prompt_response_builds_loss_masks_after_truncation(self) -> None:
        tokenizer = WhitespaceTokenizerEngine()
        packed = encode_prompt_response(
            tokenizer,
            prompt="old context useful instruction",
            response="answer token tail",
            max_prompt_tokens=2,
            max_response_tokens=2,
        )

        self.assertEqual(packed.prompt_token_count, 2)
        self.assertEqual(packed.response_token_count, 2)
        self.assertTrue(packed.prompt.truncated)
        self.assertTrue(packed.response.truncated)
        self.assertEqual(tokenizer.decode(packed.prompt.token_ids), "useful instruction")
        self.assertEqual(tokenizer.decode(packed.response.token_ids), "answer token")
        self.assertEqual(packed.masks.prompt_mask, (1, 1, 0, 0))
        self.assertEqual(packed.masks.response_mask, (0, 0, 1, 1))

    def test_build_generated_sequence_can_become_training_sample(self) -> None:
        tokenizer = WhitespaceTokenizerEngine()
        sequence = build_generated_sequence(
            tokenizer,
            prompt="write add function",
            response="def add return",
            max_prompt_tokens=3,
            max_response_tokens=2,
            finish_reason="length",
            metadata={"backend": "fake"},
        )
        sample = sequence.to_training_sample(
            task_id="task/add",
            sample_index=0,
            reward=0.75,
        )

        self.assertIsInstance(sequence, GeneratedSequence)
        self.assertEqual(sequence.response_token_count, 2)
        self.assertEqual(sequence.finish_reason, "length")
        self.assertEqual(sample.task_id, "task/add")
        self.assertEqual(sample.reward, 0.75)
        self.assertEqual(sample.metadata["finish_reason"], "length")
        self.assertTrue(sample.metadata["response_truncated"])

    def test_generation_request_and_generated_sequence_validate_ranges(self) -> None:
        with self.assertRaises(ValueError):
            GenerationRequest(prompt="p", max_new_tokens=0)
        with self.assertRaises(ValueError):
            GenerationRequest(prompt="p", max_new_tokens=1, top_p=1.5)

        tokenizer = WhitespaceTokenizerEngine()
        packed = encode_prompt_response(tokenizer, prompt="p", response="r")
        with self.assertRaises(ValueError):
            GeneratedSequence(
                prompt="p",
                response="r",
                input_ids=packed.input_ids,
                masks=packed.masks,
                finish_reason="quota",
            )

    def test_transformers_engine_config_is_lazy_and_serializable(self) -> None:
        config = TransformersEngineConfig(
            model=ModelRuntimeConfig(
                name="Qwen/Qwen2.5-Coder-0.5B",
                tokenizer="Qwen/Qwen2.5-Coder-0.5B",
                dtype="bf16",
            ),
            local_files_only=True,
        )

        payload = config.to_dict()

        self.assertTrue(payload["local_files_only"])
        self.assertEqual(payload["model"]["name"], "Qwen/Qwen2.5-Coder-0.5B")
        self.assertIsNone(payload["device_map"])

    def test_model_engine_generator_adapts_training_engine_to_rollouts(self) -> None:
        engine = _FakeModelEngine()
        generator = ModelEngineCodeGenerator(
            engine,
            model_name="shared-policy",
            backend_name="policy_engine",
        )

        result = generator.generate(
            AgentGenerationRequest(
                task_id="task/a",
                prompt="write add",
                sample_index=0,
                seed=11,
                max_new_tokens=4,
                temperature=0.0,
                top_p=1.0,
            )
        )

        self.assertEqual(result.text, "def add")
        self.assertEqual(result.backend, "policy_engine")
        self.assertEqual(result.model_name, "shared-policy")
        self.assertEqual(result.metadata["finish_reason"], "stop")
        self.assertEqual(result.metadata["response_tokens"], 2)

    def test_model_engine_debug_config_parses_training_core_fields(self) -> None:
        raw_config = load_config_file(
            ROOT / "configs" / "experiments" / "model_engine_debug.example.json"
        )
        core_config = build_training_core_config(raw_config)

        self.assertEqual(core_config.backend, "from_scratch")
        self.assertEqual(core_config.model.tokenizer_name, "Qwen/Qwen2.5-Coder-0.5B")
        self.assertEqual(core_config.rollout.max_response_tokens, 128)


class _FakeModelEngine:
    def __init__(self) -> None:
        self.tokenizer = WhitespaceTokenizerEngine()

    def generate(self, request: GenerationRequest) -> GeneratedSequence:
        return build_generated_sequence(
            self.tokenizer,
            prompt=request.prompt,
            response="def add",
            max_response_tokens=request.max_new_tokens,
            finish_reason="stop",
            metadata={"engine": "fake"},
        )


if __name__ == "__main__":
    unittest.main()
