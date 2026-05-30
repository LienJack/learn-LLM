from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.tokenizer.simple_tokenizer import CharacterTokenizer

IGNORE_INDEX = -100
VALID_ROLES = {"system", "user", "assistant"}


class LanguageModelingTextDataset(Dataset[tuple[Tensor, Tensor]]):
    """Slices a 1D token stream into right-shifted causal LM examples."""

    def __init__(self, token_ids: Tensor, block_size: int) -> None:
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a 1D tensor.")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use dtype torch.long.")
        if block_size < 1:
            raise ValueError("block_size must be positive.")
        if len(token_ids) <= block_size:
            raise ValueError("token_ids must contain at least block_size + 1 tokens.")

        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        input_ids = self.token_ids[index : index + self.block_size]
        labels = self.token_ids[index + 1 : index + self.block_size + 1]
        return input_ids, labels


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_ROLES)}, got {self.role}.")


@dataclass(frozen=True)
class SFTFeatures:
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor


def _role_prefix(role: str) -> str:
    return f"<|{role}|>\n"


def build_sft_features(
    messages: list[ChatMessage],
    tokenizer: CharacterTokenizer,
    max_length: int,
) -> SFTFeatures:
    """Builds one SFT example where only assistant content contributes to loss."""

    if not messages:
        raise ValueError("messages must not be empty.")
    if max_length < 1:
        raise ValueError("max_length must be positive.")

    input_ids: list[int] = [tokenizer.bos_token_id]
    labels: list[int] = [IGNORE_INDEX]

    for message in messages:
        prefix_ids = tokenizer.encode(_role_prefix(message.role), add_special_tokens=False)
        content_ids = tokenizer.encode(message.content, add_special_tokens=False)
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)

        input_ids.extend(prefix_ids)
        labels.extend([IGNORE_INDEX] * len(prefix_ids))

        input_ids.extend(content_ids)
        if message.role == "assistant":
            labels.extend(content_ids)
        else:
            labels.extend([IGNORE_INDEX] * len(content_ids))

        input_ids.extend(newline_ids)
        labels.extend([IGNORE_INDEX] * len(newline_ids))

    input_ids.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id if messages[-1].role == "assistant" else IGNORE_INDEX)

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    attention_mask = [1] * len(input_ids)

    pad_count = max_length - len(input_ids)
    if pad_count > 0:
        input_ids.extend([tokenizer.pad_token_id] * pad_count)
        labels.extend([IGNORE_INDEX] * pad_count)
        attention_mask.extend([0] * pad_count)

    return SFTFeatures(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
    )
