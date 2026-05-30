from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from torch import Tensor

from src.data.text_datasets import IGNORE_INDEX, ChatMessage, SFTFeatures, build_sft_features
from src.finetune.hf_workflow import format_messages_fallback
from src.tokenizer.simple_tokenizer import CharacterTokenizer


@dataclass(frozen=True)
class SFTExample:
    id: str
    messages: list[ChatMessage]
    source: str
    source_group: str
    risk_tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("SFTExample.id must not be empty.")
        if not self.messages:
            raise ValueError("SFTExample.messages must not be empty.")
        if not any(message.role == "assistant" for message in self.messages):
            raise ValueError("SFTExample must contain at least one assistant message.")
        if not self.source:
            raise ValueError("SFTExample.source must not be empty.")
        if not self.source_group:
            raise ValueError("SFTExample.source_group must not be empty.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SFTExample:
        required = {"id", "messages", "source", "source_group"}
        missing = sorted(required - data.keys())
        if missing:
            raise ValueError(f"missing required fields: {missing}")
        messages = [
            ChatMessage(role=message["role"], content=message["content"])
            for message in data["messages"]
        ]
        return cls(
            id=data["id"],
            messages=messages,
            source=data["source"],
            source_group=data["source_group"],
            risk_tags=list(data.get("risk_tags", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["messages"] = [asdict(message) for message in self.messages]
        return data


@dataclass(frozen=True)
class SFTBatchItem:
    id: str
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor


@dataclass(frozen=True)
class BehaviorComparison:
    prompt: str
    before: str
    after: str
    changed: bool


def load_sft_jsonl(path: str | Path) -> list[SFTExample]:
    examples: list[SFTExample] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            examples.append(SFTExample.from_dict(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"invalid SFT example at line {line_number}: {exc}") from exc
    ensure_unique_ids(examples)
    return examples


def dump_sft_jsonl(path: str | Path, examples: list[SFTExample]) -> None:
    ensure_unique_ids(examples)
    lines = [
        json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True)
        for example in examples
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def ensure_unique_ids(examples: list[SFTExample]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for example in examples:
        if example.id in seen:
            duplicates.add(example.id)
        seen.add(example.id)
    if duplicates:
        raise ValueError(f"duplicate SFT ids: {sorted(duplicates)}")


def render_training_text(example: SFTExample, add_generation_prompt: bool = False) -> str:
    return format_messages_fallback(
        [asdict(message) for message in example.messages],
        add_generation_prompt=add_generation_prompt,
    )


def build_sft_batch_item(
    example: SFTExample,
    tokenizer: CharacterTokenizer,
    max_length: int,
) -> SFTBatchItem:
    features = build_sft_features(example.messages, tokenizer=tokenizer, max_length=max_length)
    return SFTBatchItem(
        id=example.id,
        input_ids=features.input_ids,
        attention_mask=features.attention_mask,
        labels=features.labels,
    )


def build_sft_feature_list(
    examples: list[SFTExample],
    tokenizer: CharacterTokenizer,
    max_length: int,
) -> list[SFTBatchItem]:
    return [build_sft_batch_item(example, tokenizer, max_length) for example in examples]


def supervised_label_text(
    features: SFTFeatures | SFTBatchItem,
    tokenizer: CharacterTokenizer,
) -> str:
    supervised = features.labels[features.labels != IGNORE_INDEX]
    return tokenizer.decode(supervised)


def split_by_source_group(
    examples: list[SFTExample],
    val_ratio: float,
    seed: int,
) -> tuple[list[SFTExample], list[SFTExample]]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in (0, 1).")

    groups = sorted({example.source_group for example in examples})
    random.Random(seed).shuffle(groups)
    val_size = max(1, int(round(len(groups) * val_ratio))) if groups else 0
    val_groups = set(groups[:val_size])
    train = [example for example in examples if example.source_group not in val_groups]
    val = [example for example in examples if example.source_group in val_groups]
    assert_no_split_leakage(train, val)
    return train, val


def assert_no_split_leakage(train: list[SFTExample], val: list[SFTExample]) -> None:
    train_ids = {example.id for example in train}
    val_ids = {example.id for example in val}
    repeated_ids = train_ids & val_ids
    if repeated_ids:
        raise ValueError(f"ids appear in both splits: {sorted(repeated_ids)}")

    train_groups = {example.source_group for example in train}
    val_groups = {example.source_group for example in val}
    repeated_groups = train_groups & val_groups
    if repeated_groups:
        raise ValueError(f"source groups appear in both splits: {sorted(repeated_groups)}")


def compare_behaviors(
    prompts: list[str],
    before_outputs: list[str],
    after_outputs: list[str],
) -> list[BehaviorComparison]:
    if not (len(prompts) == len(before_outputs) == len(after_outputs)):
        raise ValueError("prompts, before_outputs, and after_outputs must have the same length.")
    return [
        BehaviorComparison(
            prompt=prompt,
            before=before,
            after=after,
            changed=before != after,
        )
        for prompt, before, after in zip(prompts, before_outputs, after_outputs, strict=True)
    ]


def write_behavior_report(path: str | Path, comparisons: list[BehaviorComparison]) -> None:
    lines = ["# SFT 行为对比报告", ""]
    for item in comparisons:
        lines.extend(
            [
                f"## Prompt: {item.prompt}",
                "",
                f"- changed: `{item.changed}`",
                f"- before: {item.before}",
                f"- after: {item.after}",
                "",
            ],
        )
    Path(path).write_text("\n".join(lines))
