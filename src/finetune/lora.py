from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    base_model_id: str = "local"
    base_model_revision: str | None = None

    def __post_init__(self) -> None:
        if self.r < 1:
            raise ValueError("LoRA rank r must be positive.")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive.")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("LoRA dropout must be in [0, 1).")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty.")

    @property
    def scaling(self) -> float:
        return self.alpha / self.r


@dataclass(frozen=True)
class QLoRAConfig:
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    double_quant: bool = True
    gradient_checkpointing: bool = True
    max_seq_len: int = 1024
    effective_batch_size: int = 32

    def __post_init__(self) -> None:
        if self.quant_type not in {"nf4", "fp4", "int8"}:
            raise ValueError("quant_type must be nf4, fp4, or int8.")
        if self.compute_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("compute_dtype must be bfloat16, float16, or float32.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if self.effective_batch_size <= 0:
            raise ValueError("effective_batch_size must be positive.")


@dataclass(frozen=True)
class TrainableParameterSummary:
    trainable: int
    total: int

    @property
    def ratio(self) -> float:
        return self.trainable / self.total if self.total else 0.0


class LoRALinear(nn.Module):
    """A frozen Linear layer plus trainable low-rank LoRA adapters."""

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.config = config
        self.base = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None)
        self.base.load_state_dict(base.state_dict())
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.lora_A = nn.Linear(base.in_features, config.r, bias=False)
        self.lora_B = nn.Linear(config.r, base.out_features, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        adapter_out = self.lora_B(self.lora_A(self.dropout(x))) * self.config.scaling
        return base_out + adapter_out

    def merged_weight(self) -> Tensor:
        delta = self.lora_B.weight @ self.lora_A.weight
        return self.base.weight.detach() + delta.detach() * self.config.scaling


def get_parent_module(root: nn.Module, module_path: str) -> tuple[nn.Module, str]:
    parts = module_path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora_adapters(model: nn.Module, config: LoRAConfig) -> list[str]:
    matched: list[str] = []
    named_modules = list(model.named_modules())
    for name, module in named_modules:
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(target) for target in config.target_modules):
            continue
        parent, child_name = get_parent_module(model, name)
        setattr(parent, child_name, LoRALinear(module, config))
        matched.append(name)

    if not matched:
        raise ValueError(f"no target modules matched {config.target_modules}")
    return matched


def discover_lora_target_modules(model: nn.Module, suffixes: tuple[str, ...]) -> list[str]:
    if not suffixes:
        raise ValueError("suffixes must not be empty.")
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(name.endswith(suffix) for suffix in suffixes)
    ]


def assert_lora_training_setup(model: nn.Module) -> None:
    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names:
        raise ValueError("LoRA setup has no trainable parameters.")
    if not any(".lora_A." in name or ".lora_B." in name for name in trainable_names):
        raise ValueError("LoRA setup has no trainable adapter parameters.")
    frozen_base_violations = [
        name
        for name, parameter in model.named_parameters()
        if ".base." in name and parameter.requires_grad
    ]
    if frozen_base_violations:
        raise ValueError(f"LoRA base parameters must be frozen: {frozen_base_violations}")


def trainable_parameter_summary(model: nn.Module) -> TrainableParameterSummary:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return TrainableParameterSummary(trainable=trainable, total=total)


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }


def save_lora_adapter(
    path: str | Path,
    model: nn.Module,
    config: LoRAConfig,
    extra: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "config": asdict(config),
        "state_dict": lora_state_dict(model),
        "extra": extra or {},
    }
    torch.save(checkpoint, Path(path))


def load_lora_adapter(path: str | Path, model: nn.Module) -> tuple[LoRAConfig, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    config = LoRAConfig(**checkpoint["config"])
    inject_lora_adapters(model, config)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    unexpected_lora = [name for name in unexpected if "lora_" in name]
    missing_lora = [name for name in missing if "lora_" in name]
    if unexpected_lora or missing_lora:
        raise ValueError(
            f"adapter load mismatch: missing={missing_lora}, unexpected={unexpected_lora}",
        )
    return config, checkpoint.get("extra", {})


def estimate_lora_parameter_count(in_features: int, out_features: int, rank: int) -> int:
    if rank < 1:
        raise ValueError("rank must be positive.")
    return rank * (in_features + out_features)


def estimate_linear_parameter_count(in_features: int, out_features: int, bias: bool = True) -> int:
    return out_features * in_features + (out_features if bias else 0)


def qlora_memory_note(config: QLoRAConfig) -> str:
    return (
        f"load_in_4bit={config.load_in_4bit}, quant_type={config.quant_type}, "
        f"compute_dtype={config.compute_dtype}, double_quant={config.double_quant}, "
        f"gradient_checkpointing={config.gradient_checkpointing}, "
        f"max_seq_len={config.max_seq_len}, "
        f"effective_batch_size={config.effective_batch_size}"
    )


def write_lora_manifest(
    path: str | Path,
    config: LoRAConfig,
    summary: TrainableParameterSummary,
) -> None:
    payload = {
        "config": asdict(config),
        "trainable_parameters": summary.trainable,
        "total_parameters": summary.total,
        "trainable_ratio": summary.ratio,
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))
