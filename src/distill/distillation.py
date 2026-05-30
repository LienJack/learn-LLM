from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.data.text_datasets import ChatMessage
from src.finetune.sft import SFTExample

APPROVED_STATUS = "approved"
ALLOWED_FILTER_STATUSES = {"approved", "rejected", "needs_review"}
FORMAT_MARKERS = ("[answer]", "[citation]")
RISK_TERMS = ("ignore evidence", "unsupported claim", "medical dosage", "legal guarantee")


@dataclass(frozen=True)
class DistillationExample:
    id: str
    prompt: str
    teacher_response: str
    teacher_model: str
    teacher_prompt_version: str
    generation_config: dict[str, Any]
    filter_status: str
    citations: list[str]
    source_group: str
    source: str = "teacher"
    risk_tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("DistillationExample.id must not be empty.")
        if not self.prompt:
            raise ValueError("DistillationExample.prompt must not be empty.")
        if not self.teacher_model:
            raise ValueError("DistillationExample.teacher_model must not be empty.")
        if not self.teacher_prompt_version:
            raise ValueError("DistillationExample.teacher_prompt_version must not be empty.")
        if self.filter_status not in ALLOWED_FILTER_STATUSES:
            raise ValueError(
                "DistillationExample.filter_status must be approved, rejected, or needs_review.",
            )
        if not self.source_group:
            raise ValueError("DistillationExample.source_group must not be empty.")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistillationFilterResult:
    example_id: str
    approved: bool
    reasons: list[str]


@dataclass(frozen=True)
class DistillationSplit:
    train: list[DistillationExample]
    val: list[DistillationExample]
    test: list[DistillationExample]


@dataclass(frozen=True)
class DistillationComparison:
    prompt: str
    base_student: str
    teacher: str
    student: str
    base_score: float
    teacher_score: float
    student_score: float

    @property
    def student_changed_from_base(self) -> bool:
        return self.base_student != self.student

    @property
    def student_delta(self) -> float:
        return self.student_score - self.base_score


def filter_distillation_example(
    example: DistillationExample,
    min_response_chars: int = 8,
    required_markers: tuple[str, ...] = FORMAT_MARKERS,
) -> DistillationFilterResult:
    reasons: list[str] = []
    response = example.teacher_response.strip()
    lower_response = response.lower()

    if example.filter_status != APPROVED_STATUS:
        reasons.append("not_approved")
    if len(response) < min_response_chars:
        reasons.append("empty_or_too_short_answer")
    if not example.citations:
        reasons.append("missing_citation")
    if any(marker not in lower_response for marker in required_markers):
        reasons.append("format_error")
    if any(term in lower_response for term in RISK_TERMS):
        reasons.append("unsafe_or_unsupported_content")

    return DistillationFilterResult(
        example_id=example.id,
        approved=not reasons,
        reasons=reasons,
    )


def filter_distillation_dataset(
    examples: list[DistillationExample],
    min_response_chars: int = 8,
) -> tuple[list[DistillationExample], list[DistillationFilterResult], dict[str, int]]:
    approved: list[DistillationExample] = []
    rejected: list[DistillationFilterResult] = []
    reason_counts: dict[str, int] = {}

    for example in examples:
        result = filter_distillation_example(
            example,
            min_response_chars=min_response_chars,
        )
        if result.approved:
            approved.append(example)
            continue
        rejected.append(result)
        for reason in result.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return approved, rejected, reason_counts


def split_distillation_by_source_group(
    examples: list[DistillationExample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> DistillationSplit:
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1).")
    if not 0 <= test_ratio < 1:
        raise ValueError("test_ratio must be in [0, 1).")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1.")

    groups = sorted({example.source_group for example in examples})
    random.Random(seed).shuffle(groups)

    val_size = _ratio_count(len(groups), val_ratio)
    remaining_after_val = len(groups) - val_size
    test_size = min(_ratio_count(len(groups), test_ratio), remaining_after_val)

    val_groups = set(groups[:val_size])
    test_groups = set(groups[val_size : val_size + test_size])
    train_groups = set(groups[val_size + test_size :])

    split = DistillationSplit(
        train=[example for example in examples if example.source_group in train_groups],
        val=[example for example in examples if example.source_group in val_groups],
        test=[example for example in examples if example.source_group in test_groups],
    )
    assert_no_distillation_leakage(split.train, split.val, split.test)
    return split


def assert_no_distillation_leakage(
    train: list[DistillationExample],
    val: list[DistillationExample],
    test: list[DistillationExample],
) -> None:
    split_examples = {"train": train, "val": val, "test": test}
    seen_ids: dict[str, str] = {}
    seen_groups: dict[str, str] = {}

    for split_name, examples in split_examples.items():
        for example in examples:
            if example.id in seen_ids:
                raise ValueError(
                    f"id {example.id!r} appears in both {seen_ids[example.id]} and {split_name}.",
                )
            if (
                example.source_group in seen_groups
                and seen_groups[example.source_group] != split_name
            ):
                raise ValueError(
                    "source group "
                    f"{example.source_group!r} appears in both "
                    f"{seen_groups[example.source_group]} and {split_name}.",
                )
            seen_ids[example.id] = split_name
            seen_groups[example.source_group] = split_name


def to_sft_example(
    example: DistillationExample,
    system_prompt: str = "你是一个基于证据回答问题的领域助教。",
) -> SFTExample:
    filter_result = filter_distillation_example(example)
    if not filter_result.approved:
        raise ValueError(
            f"cannot convert rejected distillation example {example.id}: "
            f"{filter_result.reasons}",
        )

    return SFTExample(
        id=example.id,
        source=f"distill:{example.teacher_model}:{example.teacher_prompt_version}",
        source_group=example.source_group,
        risk_tags=example.risk_tags,
        messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=example.prompt),
            ChatMessage(role="assistant", content=example.teacher_response),
        ],
    )


def compare_base_teacher_student(
    prompts: list[str],
    base_student_outputs: list[str],
    teacher_outputs: list[str],
    student_outputs: list[str],
    base_scores: list[float],
    teacher_scores: list[float],
    student_scores: list[float],
) -> list[DistillationComparison]:
    lengths = {
        len(prompts),
        len(base_student_outputs),
        len(teacher_outputs),
        len(student_outputs),
        len(base_scores),
        len(teacher_scores),
        len(student_scores),
    }
    if len(lengths) != 1:
        raise ValueError("all comparison inputs must have the same length.")

    return [
        DistillationComparison(
            prompt=prompt,
            base_student=base_student,
            teacher=teacher,
            student=student,
            base_score=base_score,
            teacher_score=teacher_score,
            student_score=student_score,
        )
        for prompt, base_student, teacher, student, base_score, teacher_score, student_score in zip(
            prompts,
            base_student_outputs,
            teacher_outputs,
            student_outputs,
            base_scores,
            teacher_scores,
            student_scores,
            strict=True,
        )
    ]


def write_distillation_eval_report(
    path: str | Path,
    comparisons: list[DistillationComparison],
) -> None:
    lines = [
        "# 蒸馏评测报告",
        "",
        (
            "| prompt | base_student | teacher | student | base_score | teacher_score | "
            "student_score | student_delta |"
        ),
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in comparisons:
        lines.append(
            "| "
            f"{_escape_table(item.prompt)} | "
            f"{_escape_table(item.base_student)} | "
            f"{_escape_table(item.teacher)} | "
            f"{_escape_table(item.student)} | "
            f"{item.base_score:.2f} | "
            f"{item.teacher_score:.2f} | "
            f"{item.student_score:.2f} | "
            f"{item.student_delta:.2f} |",
        )
    Path(path).write_text("\n".join(lines) + "\n")


def _ratio_count(total: int, ratio: float) -> int:
    if total == 0 or ratio == 0:
        return 0
    return max(1, int(round(total * ratio)))


def _escape_table(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")
