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

ANSWERABILITY_VALUES = {"answerable", "unanswerable", "partial", "red_flag"}
SUPPORT_LEVELS = {"full", "partial", "none", "contradicted"}


@dataclass(frozen=True)
class EvidenceReference:
    source_id: str
    span_id: str
    support_level: str

    def __post_init__(self) -> None:
        if not self.source_id or not self.span_id:
            raise ValueError("EvidenceReference source_id and span_id must not be empty.")
        if self.support_level not in SUPPORT_LEVELS:
            raise ValueError(f"support_level must be one of {sorted(SUPPORT_LEVELS)}.")


@dataclass(frozen=True)
class SFTExample:
    id: str
    messages: list[ChatMessage]
    source: str
    source_group: str
    task_type: str = "general"
    answerability: str = "answerable"
    template_version: str = "template_v1"
    evidence_ids: list[EvidenceReference] = field(default_factory=list)
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
        if not self.task_type:
            raise ValueError("SFTExample.task_type must not be empty.")
        if self.answerability not in ANSWERABILITY_VALUES:
            raise ValueError(f"answerability must be one of {sorted(ANSWERABILITY_VALUES)}.")
        if not self.template_version:
            raise ValueError("SFTExample.template_version must not be empty.")

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
        evidence_ids = [
            EvidenceReference(
                source_id=item["source_id"],
                span_id=item["span_id"],
                support_level=item["support_level"],
            )
            for item in data.get("evidence_ids", [])
        ]
        return cls(
            id=data["id"],
            messages=messages,
            source=data["source"],
            source_group=data["source_group"],
            task_type=data.get("task_type", "general"),
            answerability=data.get("answerability", "answerable"),
            template_version=data.get("template_version", "template_v1"),
            evidence_ids=evidence_ids,
            risk_tags=list(data.get("risk_tags", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["messages"] = [asdict(message) for message in self.messages]
        data["sample_id"] = self.id
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


@dataclass(frozen=True)
class LabelDebugRow:
    token: str
    role: str
    label: int
    contributes_to_loss: bool


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


def debug_labels(
    example: SFTExample,
    tokenizer: CharacterTokenizer,
    max_length: int,
) -> list[LabelDebugRow]:
    """Returns token-level roles and label visibility for assistant-only loss checks."""

    features = build_sft_batch_item(example, tokenizer, max_length)
    roles = _token_roles(example, tokenizer, max_length)
    rows: list[LabelDebugRow] = []
    for token_id, role, label in zip(
        features.input_ids.tolist(),
        roles,
        features.labels.tolist(),
        strict=True,
    ):
        token = tokenizer.decode([token_id], skip_special_tokens=False)
        rows.append(
            LabelDebugRow(
                token=token,
                role=role,
                label=label,
                contributes_to_loss=label != IGNORE_INDEX,
            ),
        )
    return rows


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


def _token_roles(
    example: SFTExample,
    tokenizer: CharacterTokenizer,
    max_length: int,
) -> list[str]:
    roles: list[str] = ["special"]
    for message in example.messages:
        prefix_ids = tokenizer.encode(f"<|{message.role}|>\n", add_special_tokens=False)
        content_ids = tokenizer.encode(message.content, add_special_tokens=False)
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        roles.extend(["template"] * len(prefix_ids))
        roles.extend([message.role] * len(content_ids))
        roles.extend(["template"] * len(newline_ids))
    roles.append("assistant" if example.messages[-1].role == "assistant" else "special")
    roles = roles[:max_length]
    roles.extend(["padding"] * (max_length - len(roles)))
    return roles
