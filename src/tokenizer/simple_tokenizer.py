from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)


@dataclass(frozen=True)
class TokenBatch:
    input_ids: Tensor
    attention_mask: Tensor


class CharacterTokenizer:
    """A tiny character-level tokenizer with fixed special token ids."""

    def __init__(self, token_to_id: dict[str, int]) -> None:
        expected_special_ids = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
            BOS_TOKEN: 2,
            EOS_TOKEN: 3,
        }
        for token, expected_id in expected_special_ids.items():
            actual_id = token_to_id.get(token)
            if actual_id != expected_id:
                raise ValueError(f"{token} must have id {expected_id}, got {actual_id}.")

        self.token_to_id = dict(token_to_id)
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        if len(self.id_to_token) != len(self.token_to_id):
            raise ValueError("token ids must be unique.")

    @classmethod
    def from_texts(cls, texts: list[str]) -> CharacterTokenizer:
        token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        for text in texts:
            for char in text:
                if char not in token_to_id:
                    token_to_id[char] = len(token_to_id)
        return cls(token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self.token_to_id.get(char, self.unk_token_id) for char in text]
        if add_special_tokens:
            return [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def decode(self, ids: list[int] | Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, Tensor):
            ids = ids.tolist()

        pieces: list[str] = []
        for index in ids:
            token = self.id_to_token.get(int(index), UNK_TOKEN)
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            pieces.append(token)
        return "".join(pieces)

    def batch_encode(
        self,
        texts: list[str],
        max_length: int,
        padding: bool = True,
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> TokenBatch:
        if max_length < 1:
            raise ValueError("max_length must be positive.")

        encoded = [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]
        if not truncation and any(len(ids) > max_length for ids in encoded):
            raise ValueError("encoded sequence exceeds max_length and truncation=False.")

        if truncation:
            encoded = [ids[:max_length] for ids in encoded]

        target_length = max_length if padding else max(len(ids) for ids in encoded)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for ids in encoded:
            if len(ids) > target_length:
                raise ValueError("encoded sequence exceeds target length.")
            pad_count = target_length - len(ids)
            input_ids.append([*ids, *([self.pad_token_id] * pad_count)])
            attention_mask.append([1] * len(ids) + [0] * pad_count)

        return TokenBatch(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        )
