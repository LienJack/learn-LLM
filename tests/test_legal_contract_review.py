import pytest

from src.legal_contract_review.contract_review import (
    ContractReviewOutput,
    ContractSFTExample,
    LegalEvidence,
    LegalReviewInput,
    LegalRiskPoint,
    build_contract_sft_example,
    review_contract_clause,
    scan_forbidden_legal_claims,
    validate_contract_review_output,
    validate_legal_citations,
    validate_legal_model_card,
)
from src.rag.baseline import (
    Document,
    HashingTextEmbedder,
    VectorStore,
    chunk_documents,
    citation_from_chunk,
)
from src.safety.governance import ModelCard


def make_review_output() -> ContractReviewOutput:
    return ContractReviewOutput(
        risk_level="high",
        risk_points=[
            LegalRiskPoint(
                issue="违约责任范围过宽",
                why_it_matters="条款要求承担间接损失和律师费，需复核责任边界。",
                evidence=["guideline#chunk_000"],
            ),
        ],
        suggested_revision="建议限定为直接损失，并设置责任上限。",
        uncertainty="该输出仅为风险提示，不替代律师最终法律意见。",
        needs_human_review=True,
        citations=["guideline#chunk_000"],
    )


def make_store() -> tuple[VectorStore, list]:
    documents = [
        Document(
            doc_id="guideline",
            title="合同审查规范",
            text="违约责任条款应关注责任范围、间接损失、可得利益损失、律师费和责任上限。",
        ),
    ]
    chunks = chunk_documents(documents, chunk_size=80)
    return VectorStore(chunks, HashingTextEmbedder(dim=64)), chunks


def make_legal_model_card() -> ModelCard:
    return ModelCard(
        model_name="legal-contract-review",
        version="v1",
        base_model="tiny-base",
        intended_use=["合同条款风险提示"],
        out_of_scope_use=["不提供最终法律意见", "不替代律师"],
        training_data="deidentified_contract_sft_v1",
        evaluation="eval_report.md",
        limitations=["单条条款不足以形成完整法律判断"],
        safety=["高风险条款必须人工复核"],
        deployment="local teaching demo",
        owner="course-maintainer",
    )


def test_contract_sft_example_schema_is_valid_and_deidentified() -> None:
    raw_clause = "甲方在2026年5月28日向乙方支付人民币 120 万元，联系人电话13812345678。"
    output = make_review_output()

    example = build_contract_sft_example(
        example_id="contract_sft_001",
        source_id="contract_doc_001",
        clause=raw_clause,
        assistant_output=output,
        risk_tags=["liability", "needs_human_review"],
    )

    assert "PARTY_A" in example.clause
    assert "AMOUNT_1" in example.clause
    assert "PHONE_1" in example.clause
    assert example.messages[-1].role == "assistant"

    with pytest.raises(ValueError, match="deidentified"):
        ContractSFTExample(
            id="bad",
            source_id="doc",
            clause="联系人电话13812345678",
            risk_tags=["privacy_sensitive"],
            messages=example.messages,
            deidentified=False,
        )


def test_contract_review_output_json_contains_required_fields() -> None:
    output = ContractReviewOutput.from_dict(make_review_output().to_dict())
    data = output.to_dict()

    assert data["risk_level"] == "high"
    assert data["risk_points"][0]["evidence"] == ["guideline#chunk_000"]
    assert data["needs_human_review"] is True
    assert data["review_required"] is True
    assert data["review_type"] == "lawyer"
    assert "suggested_revision" in data

    with pytest.raises(ValueError, match="risk_level"):
        ContractReviewOutput(
            risk_level="certain",
            risk_points=[],
            suggested_revision="x",
            uncertainty="x",
            needs_human_review=True,
        )


def test_legal_input_evidence_and_forbidden_claim_scan_are_strict() -> None:
    review_input = LegalReviewInput(
        clause_text="违约责任条款",
        contract_type="service_agreement",
        party_role="buyer",
        jurisdiction="unknown",
        source_confidentiality="internal",
        redaction_status="redacted",
    )
    evidence = LegalEvidence(
        source_id="guideline",
        source_type="statute",
        jurisdiction="CN",
        effective_date="2026-01-01",
        authority_level="guideline",
        span_id="span_001",
        support_level="partial",
    )

    assert review_input.jurisdiction == "unknown"
    assert evidence.support_level == "partial"
    assert scan_forbidden_legal_claims({"answer": ["该条款一定无效"]}) == ["一定无效"]

    with pytest.raises(ValueError, match="forbidden"):
        ContractReviewOutput(
            risk_level="high",
            risk_points=[
                LegalRiskPoint(issue="风险", why_it_matters="该条款一定无效", evidence=["x"]),
            ],
            suggested_revision="建议",
            uncertainty="需复核",
            needs_human_review=True,
            citations=["x"],
        )


def test_rag_citation_points_to_existing_contract_or_guideline_chunk() -> None:
    _, chunks = make_store()
    output = make_review_output()
    citations = [citation_from_chunk(chunks[0])]

    validate_legal_citations(output, citations)

    with pytest.raises(ValueError, match="missing chunks"):
        validate_legal_citations(
            ContractReviewOutput(
                risk_level="high",
                risk_points=[
                    LegalRiskPoint(
                        issue="风险",
                        why_it_matters="原因",
                        evidence=["missing#chunk"],
                    ),
                ],
                suggested_revision="建议",
                uncertainty="需复核",
                needs_human_review=True,
                citations=["missing#chunk"],
            ),
            citations,
        )


def test_no_evidence_clause_triggers_unknown_and_human_review() -> None:
    store, _ = make_store()
    output = review_contract_clause(
        "量子芯片制造步骤由PARTY_A负责。",
        store,
        top_k=1,
        min_score=0.95,
    )

    assert output.risk_level == "unknown"
    assert output.needs_human_review is True
    assert output.citations == []
    validate_contract_review_output(output)


def test_legal_model_card_states_use_limits_and_human_review() -> None:
    validate_legal_model_card(make_legal_model_card())

    with pytest.raises(ValueError, match="legal-use limits"):
        validate_legal_model_card(
            ModelCard(
                model_name="legal-contract-review",
                version="v1",
                base_model="tiny-base",
                intended_use=["合同条款风险提示"],
                out_of_scope_use=["合同分析"],
                training_data="sft_v1",
                evaluation="eval_report.md",
                limitations=["可能出错"],
                safety=["需要谨慎使用"],
                deployment="local",
                owner="course-maintainer",
            ),
        )
