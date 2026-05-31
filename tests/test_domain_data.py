import pytest

from src.data.domain_data import (
    CleaningLogEntry,
    DataManifestEntry,
    DomainExample,
    assert_no_domain_split_leakage,
    build_quality_report,
    duplicate_count,
    find_near_duplicate_pairs,
    redact_sensitive_text,
    validate_domain_examples,
    write_quality_report,
)
from src.data.text_datasets import ChatMessage


def make_domain_example(
    example_id: str,
    source_id: str,
    content: str = "分析合同条款风险",
    risk_tags: list[str] | None = None,
) -> DomainExample:
    return DomainExample(
        id=example_id,
        source_id=source_id,
        source_type="contract_clause",
        created_by="manual",
        license="internal_review_only",
        contains_personal_data=False,
        risk_tags=risk_tags or ["contract", "needs_human_review"],
        messages=[
            ChatMessage(role="user", content=content),
            ChatMessage(role="assistant", content="需要人工复核"),
        ],
    )


def test_domain_example_requires_unique_id_and_source_id() -> None:
    examples = [
        make_domain_example("a", "source_a"),
        make_domain_example("a", "source_b"),
    ]

    with pytest.raises(ValueError, match="duplicate ids"):
        validate_domain_examples(examples)


def test_domain_example_from_dict_rejects_invalid_role() -> None:
    with pytest.raises(ValueError, match="role"):
        DomainExample.from_dict(
            {
                "id": "bad",
                "source_id": "source",
                "source_type": "contract_clause",
                "created_by": "manual",
                "license": "internal",
                "contains_personal_data": False,
                "risk_tags": ["contract"],
                "messages": [{"role": "bad", "content": "x"}],
            },
        )


def test_redact_sensitive_text_replaces_phone_id_address_amount_and_date() -> None:
    text = (
        "张三电话13812345678，身份证110101199001011234，"
        "住北京市朝阳区。金额人民币20万元，日期2026年5月28日。"
    )

    result = redact_sensitive_text(text)

    assert "<PHONE>" in result.text
    assert "<ID_CARD>" in result.text
    assert "<ADDRESS>" in result.text
    assert "<AMOUNT>" in result.text
    assert "<DATE>" in result.text
    assert result.hits["phone"] == 1


def test_duplicate_and_near_duplicate_detection() -> None:
    examples = [
        make_domain_example("a", "source_a", "合同条款要求甲方承担全部责任"),
        make_domain_example("b", "source_b", "合同条款要求甲方承担全部责任"),
        make_domain_example("c", "source_c", "合同条款要求甲方承担全部责任。"),
    ]

    assert duplicate_count(examples) == 1
    assert ("a", "b") in find_near_duplicate_pairs(examples)


def test_domain_split_leakage_rejects_repeated_id_or_source() -> None:
    with pytest.raises(ValueError, match="id"):
        assert_no_domain_split_leakage(
            [make_domain_example("same", "source_a")],
            [make_domain_example("same", "source_b")],
            [],
        )

    with pytest.raises(ValueError, match="source_id"):
        assert_no_domain_split_leakage(
            [make_domain_example("a", "same_source")],
            [],
            [make_domain_example("b", "same_source")],
        )


def test_quality_report_contains_required_sections(tmp_path) -> None:
    examples = [
        make_domain_example("a", "source_a", "合同电话13812345678", ["contract"]),
        make_domain_example(
            "b",
            "source_b",
            "医学危险信号需要就医",
            ["medical", "needs_human_review"],
        ),
    ]
    report = build_quality_report(examples, split_rule="split by source_id", schema_error_count=0)
    path = tmp_path / "data_quality_report.md"

    write_quality_report(path, report)
    text = path.read_text()

    assert report.sample_count == 2
    assert report.min_length > 0
    assert "contract" in report.risk_tag_counts
    assert "phone" in report.redaction_hits
    assert "duplicate_count" in text
    assert "risk_tag_counts" in text


def test_data_manifest_entry_enforces_license_and_publishable_flags() -> None:
    entry = DataManifestEntry(
        sample_id="s1",
        source_id="doc1",
        source_group="group1",
        parent_doc_id="parent1",
        content_hash="abc",
        license="internal_review_only",
        allowed_uses=["rag", "eval"],
        commercial_use_allowed=False,
        redistribution_allowed=False,
        version="v1",
        effective_date="2026-05-31",
        purpose="rag",
        pii_status="redacted",
        review_status="human_reviewed",
    )

    assert entry.publishable is False
    assert entry.to_dict()["source_group"] == "group1"

    with pytest.raises(ValueError, match="demo_only"):
        DataManifestEntry(
            sample_id="bad",
            source_id="doc",
            source_group="group",
            parent_doc_id="parent",
            content_hash="hash",
            license="unknown",
            allowed_uses=["sft"],
            commercial_use_allowed=False,
            redistribution_allowed=False,
            version="v1",
            effective_date="2026-05-31",
            purpose="sft",
            pii_status="unknown",
            review_status="auto",
        )


def test_cleaning_log_entry_validates_action() -> None:
    entry = CleaningLogEntry(
        sample_id="s1",
        action="drop",
        reason="license_unknown",
        rule_version="clean_v1",
        timestamp="2026-05-31T00:00:00Z",
    )

    assert entry.action == "drop"

    with pytest.raises(ValueError, match="action"):
        CleaningLogEntry(
            sample_id="s1",
            action="delete",
            reason="duplicate",
            rule_version="clean_v1",
            timestamp="2026-05-31T00:00:00Z",
        )
