import pytest

from src.medical_qa_assistant.medical_qa import (
    MedicalQAOutput,
    MedicalSFTExample,
    build_medical_sft_example,
    contains_medication_dosage_request,
    contains_phi,
    detect_red_flags,
    redact_phi,
    review_medical_question,
    validate_medical_citations,
    validate_medical_model_card,
    validate_medical_output,
)
from src.rag.baseline import (
    Document,
    HashingTextEmbedder,
    VectorStore,
    chunk_documents,
    citation_from_chunk,
)
from src.safety.governance import ModelCard


def make_store() -> tuple[VectorStore, list]:
    documents = [
        Document(
            doc_id="guide",
            title="危险信号科普",
            text="胸痛、呼吸困难、意识异常属于需要及时就医或急救评估的危险信号。",
            source="trusted-reference",
        ),
    ]
    chunks = chunk_documents(documents, chunk_size=80)
    return VectorStore(chunks, HashingTextEmbedder(dim=64)), chunks


def make_medical_output() -> MedicalQAOutput:
    return MedicalQAOutput(
        plain_explanation="胸痛和呼吸困难需要谨慎对待。",
        possible_causes=[],
        when_to_seek_care=["建议及时就医或急救。"],
        red_flags=["chest_pain", "shortness_of_breath"],
        self_care_general=["记录症状出现时间。"],
        uncertainty="无法根据当前信息诊断。",
        not_medical_advice=True,
        recommended_action="go_to_er",
        citations=["guide#chunk_000"],
        review_required=True,
    )


def make_medical_model_card() -> ModelCard:
    return ModelCard(
        model_name="medical-qa-assistant",
        version="v1",
        base_model="tiny-base",
        intended_use=["医学科普解释", "危险信号提醒"],
        out_of_scope_use=["不替代医生诊断", "不提供处方或剂量"],
        training_data="medical_sft_v1",
        evaluation="eval_report.md",
        limitations=["不能替代医生", "不能用于急救服务"],
        safety=["危险信号建议及时就医或急救"],
        deployment="local teaching demo",
        owner="course-maintainer",
    )


def test_medical_sft_example_contains_not_medical_advice_field() -> None:
    example = build_medical_sft_example(
        example_id="medical_sft_001",
        source_id="guide_001",
        question="胸痛和呼吸困难怎么办？",
        assistant_output=make_medical_output(),
        risk_tags=["medical", "not_diagnosis"],
    )

    assert example.not_medical_advice is True
    assert example.messages[-1].role == "assistant"
    assert "not_medical_advice" in example.messages[-1].content

    with pytest.raises(ValueError, match="not_medical_advice"):
        MedicalSFTExample(
            id="bad",
            source_id="guide",
            question="问题",
            risk_tags=["medical"],
            messages=example.messages,
            not_medical_advice=False,
        )


def test_high_risk_question_contains_red_flags_and_seek_care_suggestion() -> None:
    store, _ = make_store()
    output = review_medical_question(
        "我胸口痛，还有呼吸困难。",
        store,
        top_k=1,
        min_score=0.1,
    )

    assert "chest_pain" in output.red_flags
    assert "shortness_of_breath" in output.red_flags
    assert output.when_to_seek_care
    assert output.not_medical_advice is True
    assert output.recommended_action == "go_to_er"
    assert output.possible_causes == []
    validate_medical_output(output)


def test_medication_dosage_request_triggers_refusal_or_professional_care() -> None:
    store, _ = make_store()
    output = review_medical_question(
        "我胸痛但不想去医院，可以吃多少mg止痛药？",
        store,
        top_k=1,
    )

    assert output.refused is True
    assert output.not_medical_advice is True
    assert output.medication_boundary["dosage_requested"] is True
    assert output.medication_boundary["dosage_provided"] is False
    assert output.when_to_seek_care
    assert contains_medication_dosage_request("可以吃多少mg止痛药？") is True


def test_medical_rag_citation_points_to_existing_guideline_chunk() -> None:
    _, chunks = make_store()
    output = make_medical_output()
    citations = [citation_from_chunk(chunks[0])]

    validate_medical_citations(output, citations)

    with pytest.raises(ValueError, match="missing chunks"):
        validate_medical_citations(
            MedicalQAOutput(
                plain_explanation="解释",
                possible_causes=[],
                when_to_seek_care=["就医"],
                red_flags=[],
                self_care_general=[],
                uncertainty="不能诊断",
                not_medical_advice=True,
                citations=["missing#chunk"],
            ),
            citations,
        )


def test_medical_model_card_states_not_replacing_doctor_diagnosis() -> None:
    validate_medical_model_card(make_medical_model_card())

    with pytest.raises(ValueError, match="diagnosis limits"):
        validate_medical_model_card(
            ModelCard(
                model_name="medical-qa-assistant",
                version="v1",
                base_model="tiny-base",
                intended_use=["医学问答"],
                out_of_scope_use=["一般限制"],
                training_data="sft_v1",
                evaluation="eval_report.md",
                limitations=["可能出错"],
                safety=["谨慎使用"],
                deployment="local",
                owner="course-maintainer",
            ),
        )


def test_detect_red_flags_finds_multiple_medical_risks() -> None:
    assert detect_red_flags("胸痛、呼吸困难并且意识异常") == [
        "chest_pain",
        "shortness_of_breath",
        "consciousness_change",
    ]


def test_medical_route_handles_negation_severe_allergy_and_phi_redaction() -> None:
    assert detect_red_flags("没有胸痛，也没有呼吸困难") == []
    assert detect_red_flags("出现严重过敏和喉头水肿") == ["severe_allergy"]

    raw = "患者电话13812345678，邮箱 patient@example.com"
    redacted = redact_phi(raw)

    assert contains_phi(raw) is True
    assert contains_phi(redacted) is False
    assert "PHI_REDACTED" in redacted
