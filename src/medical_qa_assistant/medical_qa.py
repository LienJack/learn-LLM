from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.data.text_datasets import ChatMessage
from src.rag.baseline import Citation, VectorStore
from src.safety.governance import ModelCard, validate_model_card

MEDICAL_BOUNDARY_TERMS = ("不替代医生", "不替代医生诊断", "不是诊断", "就医", "急救")
DOSAGE_PATTERNS = (
    re.compile(r"\d+\s*(mg|毫克|片|粒|ml|毫升)", re.IGNORECASE),
    re.compile(r"吃多少|用多少|剂量|加量|停药|联合用药"),
)
UNSAFE_MEDICATION_ADVICE_PATTERNS = (
    re.compile(r"\d+\s*(mg|毫克|片|粒|ml|毫升)", re.IGNORECASE),
    re.compile(r"可以.*(加量|停药|联合用药)"),
)
RED_FLAG_RULES = {
    "chest_pain": ("胸痛", "胸口痛", "胸闷"),
    "shortness_of_breath": ("呼吸困难", "喘不上气", "气短"),
    "consciousness_change": ("意识异常", "昏迷", "晕厥"),
    "self_harm": ("自伤", "轻生", "不想活"),
}


@dataclass(frozen=True)
class MedicalQAOutput:
    plain_explanation: str
    possible_causes: list[str]
    when_to_seek_care: list[str]
    red_flags: list[str]
    self_care_general: list[str]
    uncertainty: str
    not_medical_advice: bool
    citations: list[str] = field(default_factory=list)
    refused: bool = False

    def __post_init__(self) -> None:
        validate_medical_output(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MedicalQAOutput:
        return cls(
            plain_explanation=data["plain_explanation"],
            possible_causes=list(data.get("possible_causes", [])),
            when_to_seek_care=list(data.get("when_to_seek_care", [])),
            red_flags=list(data.get("red_flags", [])),
            self_care_general=list(data.get("self_care_general", [])),
            uncertainty=data["uncertainty"],
            not_medical_advice=bool(data["not_medical_advice"]),
            citations=list(data.get("citations", [])),
            refused=bool(data.get("refused", False)),
        )


@dataclass(frozen=True)
class MedicalSFTExample:
    id: str
    source_id: str
    question: str
    risk_tags: list[str]
    messages: list[ChatMessage]
    not_medical_advice: bool = True

    def __post_init__(self) -> None:
        validate_medical_sft_example(self)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["messages"] = [asdict(message) for message in self.messages]
        return data


def build_medical_sft_example(
    example_id: str,
    source_id: str,
    question: str,
    assistant_output: MedicalQAOutput,
    risk_tags: list[str],
) -> MedicalSFTExample:
    return MedicalSFTExample(
        id=example_id,
        source_id=source_id,
        question=question,
        risk_tags=risk_tags,
        messages=[
            ChatMessage(
                role="system",
                content="你是谨慎的医学科普助手，不替代医生诊断。",
            ),
            ChatMessage(role="user", content=question),
            ChatMessage(role="assistant", content=assistant_output.to_json()),
        ],
        not_medical_advice=assistant_output.not_medical_advice,
    )


def review_medical_question(
    question: str,
    store: VectorStore,
    top_k: int = 2,
    min_score: float = 0.1,
) -> MedicalQAOutput:
    red_flags = detect_red_flags(question)
    if contains_medication_dosage_request(question):
        return MedicalQAOutput(
            plain_explanation="不能根据聊天内容提供具体用药、剂量、停药或联合用药建议。",
            possible_causes=[],
            when_to_seek_care=["请咨询医生或药师，若症状严重应及时就医。"],
            red_flags=red_flags,
            self_care_general=[],
            uncertainty="无法通过当前信息判断具体病因或用药方案。",
            not_medical_advice=True,
            citations=[],
            refused=True,
        )

    results = store.search(question, top_k=top_k, min_score=0.0)
    useful_results = [result for result in results if result.score >= min_score]
    if not useful_results:
        return MedicalQAOutput(
            plain_explanation="资料不足，无法基于当前知识库做出可靠科普解释。",
            possible_causes=[],
            when_to_seek_care=["如症状持续或加重，请及时就医。"],
            red_flags=red_flags,
            self_care_general=[],
            uncertainty="当前资料不足，不能诊断。",
            not_medical_advice=True,
            citations=[],
            refused=True,
        )

    citation_id = useful_results[0].chunk.chunk_id
    seek_care = ["如出现危险信号，应及时就医或急救。"]
    if red_flags:
        seek_care.insert(0, "当前问题包含危险信号，建议尽快就医或急救。")

    return MedicalQAOutput(
        plain_explanation="根据资料，相关症状需要谨慎对待，并结合持续时间、严重程度和伴随表现判断。",
        possible_causes=["疲劳、感染、压力等都可能相关，但不能据此诊断。"],
        when_to_seek_care=seek_care,
        red_flags=red_flags,
        self_care_general=["记录症状出现时间、诱因和伴随表现，便于就医沟通。"],
        uncertainty="无法根据当前信息诊断，不能替代医生面对面评估。",
        not_medical_advice=True,
        citations=[citation_id],
        refused=False,
    )


def validate_medical_output(output: MedicalQAOutput) -> None:
    if not output.plain_explanation:
        raise ValueError("plain_explanation must not be empty.")
    if not output.uncertainty:
        raise ValueError("uncertainty must not be empty.")
    if not output.not_medical_advice:
        raise ValueError("medical output must set not_medical_advice=True.")
    if output.red_flags and not output.when_to_seek_care:
        raise ValueError("red flag output must include when_to_seek_care.")
    if not output.refused and not output.citations:
        raise ValueError("non-refusal medical output must include citations.")
    if contains_unsafe_medication_advice(output.plain_explanation):
        raise ValueError("medical output must not provide medication dosage instructions.")


def validate_medical_sft_example(example: MedicalSFTExample) -> None:
    if not example.id:
        raise ValueError("MedicalSFTExample.id must not be empty.")
    if not example.source_id:
        raise ValueError("MedicalSFTExample.source_id must not be empty.")
    if not example.risk_tags:
        raise ValueError("MedicalSFTExample.risk_tags must not be empty.")
    if not example.not_medical_advice:
        raise ValueError("medical SFT example must include not_medical_advice.")
    if not any(message.role == "assistant" for message in example.messages):
        raise ValueError("MedicalSFTExample must include an assistant message.")


def validate_medical_citations(output: MedicalQAOutput, citations: list[Citation]) -> None:
    known_chunks = {citation.chunk_id for citation in citations}
    missing = sorted(set(output.citations) - known_chunks)
    if missing:
        raise ValueError(f"medical output cites missing chunks: {missing}")


def validate_medical_model_card(model_card: ModelCard) -> None:
    validate_model_card(model_card)
    combined = " ".join(
        model_card.out_of_scope_use
        + model_card.limitations
        + model_card.safety
    )
    if not any(term in combined for term in MEDICAL_BOUNDARY_TERMS):
        raise ValueError("medical model card must state diagnosis limits and seek-care boundary.")


def detect_red_flags(text: str) -> list[str]:
    flags: list[str] = []
    for flag, terms in RED_FLAG_RULES.items():
        if any(term in text for term in terms):
            flags.append(flag)
    return flags


def contains_medication_dosage_request(text: str) -> bool:
    return any(pattern.search(text) for pattern in DOSAGE_PATTERNS)


def contains_unsafe_medication_advice(text: str) -> bool:
    return any(pattern.search(text) for pattern in UNSAFE_MEDICATION_ADVICE_PATTERNS)
