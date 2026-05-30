from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.data.text_datasets import VALID_ROLES, ChatMessage


@dataclass(frozen=True)
class DomainExample:
    id: str
    source_id: str
    source_type: str
    created_by: str
    license: str
    contains_personal_data: bool
    risk_tags: list[str]
    messages: list[ChatMessage]

    def __post_init__(self) -> None:
        for field_name in ["id", "source_id", "source_type", "created_by", "license"]:
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty.")
        if not self.messages:
            raise ValueError("messages must not be empty.")
        if not self.risk_tags:
            raise ValueError("risk_tags must not be empty.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainExample:
        required = {
            "id",
            "source_id",
            "source_type",
            "created_by",
            "license",
            "contains_personal_data",
            "risk_tags",
            "messages",
        }
        missing = sorted(required - data.keys())
        if missing:
            raise ValueError(f"missing required fields: {missing}")
        messages = [
            ChatMessage(role=message["role"], content=message["content"])
            for message in data["messages"]
        ]
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            source_type=data["source_type"],
            created_by=data["created_by"],
            license=data["license"],
            contains_personal_data=bool(data["contains_personal_data"]),
            risk_tags=list(data["risk_tags"]),
            messages=messages,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["messages"] = [asdict(message) for message in self.messages]
        return data

    @property
    def text(self) -> str:
        return "\n".join(message.content for message in self.messages)


@dataclass(frozen=True)
class RedactionResult:
    text: str
    hits: dict[str, int]


@dataclass(frozen=True)
class QualityReport:
    sample_count: int
    source_type_counts: dict[str, int]
    risk_tag_counts: dict[str, int]
    min_length: int
    max_length: int
    avg_length: float
    duplicate_count: int
    near_duplicate_pairs: list[tuple[str, str]]
    redaction_hits: dict[str, int]
    schema_error_count: int
    split_rule: str

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "# 数据质量报告",
                "",
                f"- sample_count: {self.sample_count}",
                f"- source_type_counts: {self.source_type_counts}",
                f"- risk_tag_counts: {self.risk_tag_counts}",
                (
                    f"- length: min={self.min_length}, "
                    f"max={self.max_length}, avg={self.avg_length:.2f}"
                ),
                f"- duplicate_count: {self.duplicate_count}",
                f"- near_duplicate_pairs: {self.near_duplicate_pairs}",
                f"- redaction_hits: {self.redaction_hits}",
                f"- schema_error_count: {self.schema_error_count}",
                f"- split_rule: {self.split_rule}",
                "",
            ],
        )


PHONE_RE = re.compile(r"(?<!\d)(?:1[3-9]\d{9}|\d{3,4}-\d{7,8})(?!\d)")
ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
ADDRESS_RE = re.compile(r"[\u4e00-\u9fa5]{2,}(?:省|市|区|县|路|街|号)")
AMOUNT_RE = re.compile(r"(?:人民币)?\d+(?:\.\d+)?(?:元|万元)")
DATE_RE = re.compile(r"\d{4}年\d{1,2}月\d{1,2}日")


def redact_sensitive_text(text: str) -> RedactionResult:
    replacements = [
        ("phone", PHONE_RE, "<PHONE>"),
        ("id_card", ID_CARD_RE, "<ID_CARD>"),
        ("address", ADDRESS_RE, "<ADDRESS>"),
        ("amount", AMOUNT_RE, "<AMOUNT>"),
        ("date", DATE_RE, "<DATE>"),
    ]
    hits: dict[str, int] = {}
    redacted = text
    for name, pattern, placeholder in replacements:
        redacted, count = pattern.subn(placeholder, redacted)
        if count:
            hits[name] = count
    return RedactionResult(text=redacted, hits=hits)


def validate_domain_examples(examples: list[DomainExample]) -> None:
    ids: set[str] = set()
    source_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for example in examples:
        if example.id in ids:
            duplicate_ids.add(example.id)
        ids.add(example.id)
        source_ids.add(example.source_id)
        for message in example.messages:
            if message.role not in VALID_ROLES:
                raise ValueError(f"invalid role: {message.role}")
    if duplicate_ids:
        raise ValueError(f"duplicate ids: {sorted(duplicate_ids)}")
    if "" in source_ids:
        raise ValueError("source_id must not be empty.")


def normalize_for_duplicate(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def duplicate_count(examples: list[DomainExample]) -> int:
    normalized = [normalize_for_duplicate(example.text) for example in examples]
    counts = Counter(normalized)
    return sum(count - 1 for count in counts.values() if count > 1)


def char_ngrams(text: str, n: int = 3) -> set[str]:
    normalized = normalize_for_duplicate(text)
    if len(normalized) < n:
        return {normalized} if normalized else set()
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def find_near_duplicate_pairs(
    examples: list[DomainExample],
    threshold: float = 0.8,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    ngrams = {example.id: char_ngrams(example.text) for example in examples}
    for left_index, left in enumerate(examples):
        for right in examples[left_index + 1 :]:
            if jaccard_similarity(ngrams[left.id], ngrams[right.id]) >= threshold:
                pairs.append((left.id, right.id))
    return pairs


def assert_no_domain_split_leakage(
    train: list[DomainExample],
    val: list[DomainExample],
    test: list[DomainExample],
) -> None:
    splits = {"train": train, "val": val, "test": test}
    seen_ids: dict[str, str] = {}
    seen_sources: dict[str, str] = {}
    for split_name, examples in splits.items():
        for example in examples:
            if example.id in seen_ids:
                raise ValueError(
                    f"id {example.id} appears in {seen_ids[example.id]} and {split_name}",
                )
            if example.source_id in seen_sources:
                raise ValueError(
                    f"source_id {example.source_id} appears in "
                    f"{seen_sources[example.source_id]} and {split_name}",
                )
            seen_ids[example.id] = split_name
            seen_sources[example.source_id] = split_name


def build_quality_report(
    examples: list[DomainExample],
    split_rule: str,
    schema_error_count: int = 0,
) -> QualityReport:
    validate_domain_examples(examples)
    lengths = [len(example.text) for example in examples]
    redaction_total: Counter[str] = Counter()
    for example in examples:
        redaction_total.update(redact_sensitive_text(example.text).hits)
    source_type_counts = Counter(example.source_type for example in examples)
    risk_tag_counts = Counter(tag for example in examples for tag in example.risk_tags)
    return QualityReport(
        sample_count=len(examples),
        source_type_counts=dict(source_type_counts),
        risk_tag_counts=dict(risk_tag_counts),
        min_length=min(lengths) if lengths else 0,
        max_length=max(lengths) if lengths else 0,
        avg_length=sum(lengths) / len(lengths) if lengths else 0.0,
        duplicate_count=duplicate_count(examples),
        near_duplicate_pairs=find_near_duplicate_pairs(examples),
        redaction_hits=dict(redaction_total),
        schema_error_count=schema_error_count,
        split_rule=split_rule,
    )


def write_quality_report(path: str | Path, report: QualityReport) -> None:
    Path(path).write_text(report.to_markdown())
