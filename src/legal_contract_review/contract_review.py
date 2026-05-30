from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.data.text_datasets import ChatMessage
from src.rag.baseline import Citation, VectorStore
from src.safety.governance import ModelCard, validate_model_card

RISK_LEVELS = {"low", "medium", "high", "unknown"}
SENSITIVE_PATTERNS = (
    re.compile(r"1[3-9]\d{9}"),
    re.compile(r"\d{17}[\dXx]"),
    re.compile(r"[\w.-]+@[\w.-]+"),
    re.compile(r"北京市|上海市|广州市|深圳市"),
)
LEGAL_BOUNDARY_TERMS = ("不替代律师", "不提供最终法律意见", "人工复核", "用途限制")


@dataclass(frozen=True)
class ContractSFTExample:
    id: str
    source_id: str
    clause: str
    risk_tags: list[str]
    messages: list[ChatMessage]
    deidentified: bool = True

    def __post_init__(self) -> None:
        validate_contract_sft_example(self)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["messages"] = [asdict(message) for message in self.messages]
        return data


@dataclass(frozen=True)
class LegalRiskPoint:
    issue: str
    why_it_matters: str
    evidence: list[str]

    def __post_init__(self) -> None:
        if not self.issue:
            raise ValueError("LegalRiskPoint.issue must not be empty.")
        if not self.why_it_matters:
            raise ValueError("LegalRiskPoint.why_it_matters must not be empty.")
        if not self.evidence:
            raise ValueError("LegalRiskPoint.evidence must not be empty.")


@dataclass(frozen=True)
class ContractReviewOutput:
    risk_level: str
    risk_points: list[LegalRiskPoint]
    suggested_revision: str
    uncertainty: str
    needs_human_review: bool
    citations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        validate_contract_review_output(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContractReviewOutput:
        risk_points = [
            LegalRiskPoint(
                issue=item["issue"],
                why_it_matters=item["why_it_matters"],
                evidence=list(item["evidence"]),
            )
            for item in data.get("risk_points", [])
        ]
        return cls(
            risk_level=data["risk_level"],
            risk_points=risk_points,
            suggested_revision=data["suggested_revision"],
            uncertainty=data["uncertainty"],
            needs_human_review=bool(data["needs_human_review"]),
            citations=list(data.get("citations", [])),
        )


def deidentify_contract_clause(clause: str) -> str:
    replacements = [
        (r"甲方", "PARTY_A"),
        (r"乙方", "PARTY_B"),
        (r"人民币\s*\d+(?:\.\d+)?\s*万?元", "AMOUNT_1"),
        (r"\d{4}年\d{1,2}月\d{1,2}日", "DATE_1"),
        (r"1[3-9]\d{9}", "PHONE_1"),
        (r"\d{17}[\dXx]", "ID_1"),
        (r"北京市[^，。；;]*", "ADDRESS_1"),
    ]
    result = clause
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result)
    return result


def build_contract_sft_example(
    example_id: str,
    source_id: str,
    clause: str,
    assistant_output: ContractReviewOutput,
    risk_tags: list[str],
) -> ContractSFTExample:
    deidentified_clause = deidentify_contract_clause(clause)
    return ContractSFTExample(
        id=example_id,
        source_id=source_id,
        clause=deidentified_clause,
        risk_tags=risk_tags,
        messages=[
            ChatMessage(
                role="system",
                content="你是谨慎的合同风险分析助手，不提供最终法律意见。",
            ),
            ChatMessage(role="user", content=f"分析以下条款：{deidentified_clause}"),
            ChatMessage(role="assistant", content=assistant_output.to_json()),
        ],
        deidentified=True,
    )


def validate_contract_sft_example(example: ContractSFTExample) -> None:
    if not example.id:
        raise ValueError("ContractSFTExample.id must not be empty.")
    if not example.source_id:
        raise ValueError("ContractSFTExample.source_id must not be empty.")
    if not example.risk_tags:
        raise ValueError("ContractSFTExample.risk_tags must not be empty.")
    if not example.messages:
        raise ValueError("ContractSFTExample.messages must not be empty.")
    if not any(message.role == "assistant" for message in example.messages):
        raise ValueError("ContractSFTExample must include an assistant message.")
    if not example.deidentified or contains_sensitive_data(example.clause):
        raise ValueError("contract SFT clause must be deidentified.")


def validate_contract_review_output(output: ContractReviewOutput) -> None:
    if output.risk_level not in RISK_LEVELS:
        raise ValueError(f"risk_level must be one of {sorted(RISK_LEVELS)}.")
    if output.risk_level == "unknown":
        if output.risk_points:
            raise ValueError("unknown risk output must not include risk points.")
        if not output.needs_human_review:
            raise ValueError("unknown risk output must require human review.")
    else:
        if not output.risk_points:
            raise ValueError("non-unknown risk output must include risk points.")
        if not output.citations:
            raise ValueError("non-unknown risk output must include citations.")
    if not output.suggested_revision:
        raise ValueError("suggested_revision must not be empty.")
    if not output.uncertainty:
        raise ValueError("uncertainty must not be empty.")


def validate_legal_citations(output: ContractReviewOutput, citations: list[Citation]) -> None:
    known_chunks = {
        citation.chunk_id
        for citation in citations
    }
    referenced = set(output.citations)
    for risk_point in output.risk_points:
        referenced.update(risk_point.evidence)
    missing = sorted(referenced - known_chunks)
    if missing:
        raise ValueError(f"legal review cites missing chunks: {missing}")


def review_contract_clause(
    clause: str,
    store: VectorStore,
    top_k: int = 2,
    min_score: float = 0.1,
) -> ContractReviewOutput:
    results = store.search(clause, top_k=top_k, min_score=0.0)
    useful_results = [result for result in results if result.score >= min_score]
    if not useful_results:
        return ContractReviewOutput(
            risk_level="unknown",
            risk_points=[],
            suggested_revision="资料不足，无法给出具体修改建议。",
            uncertainty="当前知识库没有足够依据，需要律师人工复核。",
            needs_human_review=True,
            citations=[],
        )

    best = useful_results[0].chunk
    citation_id = best.chunk_id
    if _looks_like_broad_liability(clause):
        risk_level = "high"
        issue = "违约责任范围过宽"
        why_it_matters = "条款包含一切损失、间接损失或律师费，可能超出常见责任边界。"
        suggested_revision = "建议限定为直接损失，并结合交易背景设置责任上限。"
    else:
        risk_level = "medium"
        issue = "条款需要结合上下文复核"
        why_it_matters = "检索到相关审查规范，但单条条款不足以形成最终法律意见。"
        suggested_revision = "建议补充适用条件、例外情形和责任上限。"

    return ContractReviewOutput(
        risk_level=risk_level,
        risk_points=[
            LegalRiskPoint(
                issue=issue,
                why_it_matters=why_it_matters,
                evidence=[citation_id],
            ),
        ],
        suggested_revision=suggested_revision,
        uncertainty="该输出仅为风险提示，不替代律师最终法律意见。",
        needs_human_review=True,
        citations=[citation_id],
    )


def validate_legal_model_card(model_card: ModelCard) -> None:
    validate_model_card(model_card)
    combined = " ".join(
        model_card.out_of_scope_use
        + model_card.limitations
        + model_card.safety
    )
    if not any(term in combined for term in LEGAL_BOUNDARY_TERMS):
        raise ValueError("legal model card must state legal-use limits and human review.")


def contains_sensitive_data(text: str) -> bool:
    return any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)


def _looks_like_broad_liability(clause: str) -> bool:
    return any(term in clause for term in ("一切损失", "间接损失", "可得利益", "律师费"))
