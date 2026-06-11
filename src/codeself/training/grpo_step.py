"""Optional Torch optimizer step for GRPO tensor batches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from codeself.training.grpo_loss import GRPOLossConfig
from codeself.training.grpo_torch import (
    GRPOTensorBatch,
    compute_grpo_tensor_loss,
    require_torch,
)


@dataclass(frozen=True)
class GRPOOptimizerStepConfig:
    """Settings for one GRPO optimizer step."""

    loss: GRPOLossConfig = field(default_factory=GRPOLossConfig)
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = 1.0
    zero_grad: bool = True
    step_optimizer: bool = True

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when set")

    def to_dict(self) -> dict[str, object]:
        return {
            "loss": self.loss.to_dict(),
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "zero_grad": self.zero_grad,
            "step_optimizer": self.step_optimizer,
        }


@dataclass(frozen=True)
class GRPOOptimizerStepResult:
    """Detached diagnostics from one GRPO optimizer step."""

    loss: float
    backward_loss: float
    policy_loss: float
    kl_loss: float
    grad_norm: float | None
    optimizer_step: bool
    zero_grad: bool
    gradient_accumulation_steps: int
    total_response_tokens: int
    mean_ratio: float
    mean_clipped_ratio: float
    mean_approx_kl: float
    clipped_token_fraction: float

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "loss": self.loss,
            "backward_loss": self.backward_loss,
            "policy_loss": self.policy_loss,
            "kl_loss": self.kl_loss,
            "grad_norm": self.grad_norm,
            "optimizer_step": self.optimizer_step,
            "zero_grad": self.zero_grad,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "total_response_tokens": self.total_response_tokens,
            "mean_ratio": self.mean_ratio,
            "mean_clipped_ratio": self.mean_clipped_ratio,
            "mean_approx_kl": self.mean_approx_kl,
            "clipped_token_fraction": self.clipped_token_fraction,
        }


def run_grpo_optimizer_step(
    batch: GRPOTensorBatch,
    *,
    optimizer: Any,
    config: GRPOOptimizerStepConfig | None = None,
) -> GRPOOptimizerStepResult:
    """Backpropagate GRPO tensor loss and optionally step an optimizer."""

    torch = require_torch()
    step_config = config or GRPOOptimizerStepConfig()
    _validate_optimizer(optimizer)
    if step_config.zero_grad:
        _zero_grad(optimizer)

    loss_result = compute_grpo_tensor_loss(batch, config=step_config.loss)
    backward_loss = loss_result.loss / step_config.gradient_accumulation_steps
    backward_loss.backward()

    grad_norm = None
    if step_config.max_grad_norm is not None:
        parameters = _optimizer_parameters(optimizer)
        if parameters:
            grad_norm_value = torch.nn.utils.clip_grad_norm_(
                parameters,
                step_config.max_grad_norm,
            )
            grad_norm = _detach_optional_float(grad_norm_value)

    if step_config.step_optimizer:
        optimizer.step()

    return GRPOOptimizerStepResult(
        loss=_detach_float(loss_result.loss),
        backward_loss=_detach_float(backward_loss),
        policy_loss=_detach_float(loss_result.policy_loss),
        kl_loss=_detach_float(loss_result.kl_loss),
        grad_norm=grad_norm,
        optimizer_step=step_config.step_optimizer,
        zero_grad=step_config.zero_grad,
        gradient_accumulation_steps=step_config.gradient_accumulation_steps,
        total_response_tokens=loss_result.total_response_tokens,
        mean_ratio=loss_result.mean_ratio,
        mean_clipped_ratio=loss_result.mean_clipped_ratio,
        mean_approx_kl=loss_result.mean_approx_kl,
        clipped_token_fraction=loss_result.clipped_token_fraction,
    )


def _validate_optimizer(optimizer: Any) -> None:
    if not hasattr(optimizer, "step"):
        raise TypeError("optimizer must define step()")
    if not hasattr(optimizer, "zero_grad"):
        raise TypeError("optimizer must define zero_grad()")


def _zero_grad(optimizer: Any) -> None:
    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:
        optimizer.zero_grad()


def _optimizer_parameters(optimizer: Any) -> list[Any]:
    parameters: list[Any] = []
    for group in getattr(optimizer, "param_groups", ()):
        for parameter in group.get("params", ()):
            if getattr(parameter, "grad", None) is not None:
                parameters.append(parameter)
    return parameters


def _detach_float(value: Any) -> float:
    return float(value.detach().cpu().item())


def _detach_optional_float(value: Any) -> float:
    if hasattr(value, "detach"):
        return _detach_float(value)
    return float(value)
