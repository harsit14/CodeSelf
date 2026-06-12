"""Value-head adapters for PPO model training."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from codeself.training.grpo_torch import require_torch


class CausalLMWithValueHead:
    """Attach a token-level value head to a causal language model."""

    def __init__(self, base_model: Any, *, hidden_size: int | None = None) -> None:
        torch = require_torch()
        self.base_model = base_model
        self.value_head = torch.nn.Linear(_hidden_size(base_model, hidden_size), 1)
        self._move_value_head_to_base_model()

    def parameters(self) -> Any:
        return tuple(self.base_model.parameters()) + tuple(self.value_head.parameters())

    def train(self) -> None:
        if hasattr(self.base_model, "train"):
            self.base_model.train()
        self.value_head.train()

    def eval(self) -> None:
        if hasattr(self.base_model, "eval"):
            self.base_model.eval()
        self.value_head.eval()

    def to(self, *args: Any, **kwargs: Any) -> "CausalLMWithValueHead":
        if hasattr(self.base_model, "to"):
            self.base_model.to(*args, **kwargs)
        self.value_head.to(*args, **kwargs)
        return self

    def state_dict(self) -> dict[str, object]:
        return {
            "base_model": self.base_model.state_dict(),
            "value_head": self.value_head.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if "base_model" not in state or "value_head" not in state:
            raise ValueError("value-head checkpoint requires base_model and value_head")
        self.base_model.load_state_dict(state["base_model"])
        self.value_head.load_state_dict(state["value_head"])

    def load_state_path(self, path: str | Path) -> None:
        """Load a full value-head wrapper state dict from disk."""

        torch = require_torch()
        self.load_state_dict(torch.load(Path(path), map_location="cpu"))

    def load_value_head_path(self, path: str | Path) -> None:
        """Load only the token-value head weights from disk."""

        torch = require_torch()
        self.value_head.load_state_dict(torch.load(Path(path), map_location="cpu"))

    def save_pretrained(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.base_model, "save_pretrained"):
            self.base_model.save_pretrained(path)
        torch = require_torch()
        torch.save(self.value_head.state_dict(), path / "value_head.pt")
        (path / "value_head_config.json").write_text(
            json.dumps({"hidden_size": self.value_head.in_features}, indent=2) + "\n",
            encoding="utf-8",
        )

    def __call__(self, *, input_ids: Any, attention_mask: Any) -> Any:
        output = self._forward_base(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = _hidden_states(output)
        values = self.value_head(hidden_states[-1]).squeeze(-1)
        return SimpleNamespace(logits=_logits(output), values=values)

    def _forward_base(self, *, input_ids: Any, attention_mask: Any) -> Any:
        try:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        except TypeError:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

    def _move_value_head_to_base_model(self) -> None:
        try:
            parameter = next(iter(self.base_model.parameters()))
        except StopIteration:
            return
        dtype = (
            parameter.dtype
            if getattr(parameter, "is_floating_point", lambda: False)()
            else None
        )
        if dtype is None:
            self.value_head.to(device=parameter.device)
        else:
            self.value_head.to(device=parameter.device, dtype=dtype)


def _hidden_size(base_model: Any, explicit_hidden_size: int | None) -> int:
    if explicit_hidden_size is not None:
        if explicit_hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        return explicit_hidden_size
    config = getattr(base_model, "config", None)
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    embeddings = (
        base_model.get_input_embeddings()
        if hasattr(base_model, "get_input_embeddings")
        else None
    )
    embedding_dim = getattr(embeddings, "embedding_dim", None)
    if embedding_dim is not None:
        return int(embedding_dim)
    raise ValueError("could not infer hidden size for PPO value head")


def _logits(output: Any) -> Any:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if isinstance(output, tuple):
        return output[0]
    raise ValueError("causal LM output must include logits")


def _hidden_states(output: Any) -> tuple[Any, ...]:
    if hasattr(output, "hidden_states") and output.hidden_states is not None:
        return tuple(output.hidden_states)
    if isinstance(output, dict) and output.get("hidden_states") is not None:
        return tuple(output["hidden_states"])
    if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
        return tuple(output[1])
    raise ValueError("causal LM output must include hidden states for PPO values")
