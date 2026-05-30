from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

IGNORE_INDEX = -100


class SavePretrained(Protocol):
    def save_pretrained(self, output_dir: str | Path) -> None: ...


@dataclass(frozen=True)
class HFLoadPlan:
    model_id: str
    revision: str | None = None
    torch_dtype: str | None = None
    device_map: str | None = None
    trust_remote_code: bool = False

    def tokenizer_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.revision is not None:
            kwargs["revision"] = self.revision
        kwargs["trust_remote_code"] = self.trust_remote_code
        return kwargs

    def model_kwargs(self) -> dict[str, Any]:
        kwargs = self.tokenizer_kwargs()
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        return kwargs


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 32
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive when provided.")


def validate_tokenizer_batch(batch: dict[str, Tensor]) -> None:
    if "input_ids" not in batch or "attention_mask" not in batch:
        raise ValueError("tokenizer output must contain input_ids and attention_mask.")
    if batch["input_ids"].shape != batch["attention_mask"].shape:
        raise ValueError("input_ids and attention_mask must have the same shape.")
    if batch["input_ids"].dtype != torch.long:
        raise TypeError("input_ids must use dtype torch.long.")


def causal_lm_labels(input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have the same shape.")
    labels = input_ids.clone()
    labels = labels.masked_fill(attention_mask == 0, IGNORE_INDEX)
    return labels


def causal_lm_data_collator(
    examples: list[dict[str, list[int]]],
    pad_token_id: int,
) -> dict[str, Tensor]:
    if not examples:
        raise ValueError("examples must not be empty.")
    max_length = max(len(example["input_ids"]) for example in examples)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []

    for example in examples:
        ids = list(example["input_ids"])
        pad_count = max_length - len(ids)
        input_ids.append([*ids, *([pad_token_id] * pad_count)])
        attention_mask.append([1] * len(ids) + [0] * pad_count)

    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }
    batch["labels"] = causal_lm_labels(batch["input_ids"], batch["attention_mask"])
    return batch


def assert_causal_lm_logits_shape(
    logits: Tensor,
    input_ids: Tensor,
    expected_vocab_size: int,
) -> None:
    expected_shape = (*input_ids.shape, expected_vocab_size)
    if tuple(logits.shape) != expected_shape:
        raise ValueError(f"expected logits shape {expected_shape}, got {tuple(logits.shape)}.")


def greedy_next_token(logits: Tensor) -> Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocab_size).")
    return logits.argmax(dim=-1)


def deterministic_split(
    items: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in (0, 1).")
    indices = list(range(len(items)))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(round(len(items) * val_ratio))) if items else 0
    val_indices = set(indices[:val_size])
    train = [item for index, item in enumerate(items) if index not in val_indices]
    val = [item for index, item in enumerate(items) if index in val_indices]
    return train, val


def format_messages_fallback(
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    pieces: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported role: {role}")
        if content is None:
            raise ValueError("message content must not be None.")
        pieces.append(f"<|{role}|>\n{content}\n")
    if add_generation_prompt:
        pieces.append("<|assistant|>\n")
    return "".join(pieces)


def save_pretrained_artifacts(
    output_dir: str | Path,
    model: SavePretrained,
    tokenizer: SavePretrained,
    manifest: dict[str, Any],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    (output_path / "workflow_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
    )


def verify_pretrained_artifacts(output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    required = ["config.json", "tokenizer_config.json", "workflow_manifest.json"]
    missing = [name for name in required if not (output_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing pretrained artifacts: {missing}")
    return json.loads((output_path / "workflow_manifest.json").read_text())


def write_generation_settings(path: str | Path, settings: GenerationSettings) -> None:
    Path(path).write_text(json.dumps(asdict(settings), indent=2, sort_keys=True))
