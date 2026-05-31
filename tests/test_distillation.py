import pytest

from src.distill.distillation import (
    ClaimSupport,
    DistillationExample,
    assert_no_distillation_leakage,
    compare_base_teacher_student,
    count_unique_rejected_examples,
    filter_distillation_dataset,
    filter_distillation_example,
    split_distillation_by_source_group,
    to_sft_example,
    write_distillation_eval_report,
)


def make_example(
    example_id: str = "distill_001",
    source_group: str = "contract_a",
    response: str = "[answer] 违约金条款需要结合合同上下文判断。[citation] contract_a:12",
    citations: list[str] | None = None,
    filter_status: str = "approved",
    claims: list[ClaimSupport] | None = None,
) -> DistillationExample:
    default_claims = [
        ClaimSupport(
            claim_id="claim_001",
            claim_text="违约金条款需要结合合同上下文判断。",
            span_id="contract_a:12",
            support_label="full",
            review_method="manual",
            reviewer="domain_reviewer",
            confidence=0.9,
        ),
    ]
    return DistillationExample(
        id=example_id,
        prompt="违约金条款是否可以直接执行？",
        teacher_response=response,
        teacher_model="teacher-pro-2026-05",
        teacher_prompt_version="rag_prompt_v3",
        generation_config={"temperature": 0.2, "top_p": 0.9},
        filter_status=filter_status,
        citations=["contract_a:12"] if citations is None else citations,
        source_group=source_group,
        claims=default_claims if claims is None else claims,
        normalized_question_hash="q_hash",
        source_span_hash="span_hash",
        teacher_prompt_hash="prompt_hash",
        risk_tags=["legal"],
    )


def test_distillation_example_requires_teacher_metadata() -> None:
    example = make_example()

    assert example.teacher_model == "teacher-pro-2026-05"
    assert example.teacher_prompt_version == "rag_prompt_v3"
    assert example.filter_status == "approved"

    with pytest.raises(ValueError, match="teacher_model"):
        DistillationExample(
            id="bad",
            prompt="问题",
            teacher_response="[answer] a [citation] c",
            teacher_model="",
            teacher_prompt_version="rag_prompt_v3",
            generation_config={},
            filter_status="approved",
            citations=["c"],
            source_group="doc",
        )


def test_filter_rejects_missing_citation_empty_answer_and_format_error() -> None:
    no_citation = make_example("no_cite", citations=[], claims=[])
    empty = make_example("empty", response="")
    bad_format = make_example("bad_format", response="答案里没有固定标记", citations=["doc:1"])

    assert filter_distillation_example(no_citation).reasons == ["missing_citation"]
    assert "empty_or_too_short_answer" in filter_distillation_example(empty).reasons
    assert "format_error" in filter_distillation_example(bad_format).reasons


def test_filter_dataset_reports_pass_rate_and_reasons() -> None:
    approved, rejected, reason_counts = filter_distillation_dataset(
        [
            make_example("ok"),
            make_example("no_cite", citations=[]),
            make_example("not_approved", filter_status="needs_review"),
        ],
    )

    assert [example.id for example in approved] == ["ok"]
    assert {result.example_id for result in rejected} == {"no_cite", "not_approved"}
    assert reason_counts["missing_citation"] == 1
    assert reason_counts["not_approved"] == 1
    assert count_unique_rejected_examples(rejected) == 2


def test_filter_rejects_cited_answer_without_claim_support() -> None:
    result = filter_distillation_example(make_example("no_claims", claims=[]))

    assert "missing_claim_support" in result.reasons


def test_split_by_source_group_prevents_train_val_test_leakage() -> None:
    examples = [
        make_example("a1", "doc_a"),
        make_example("a2", "doc_a"),
        make_example("b1", "doc_b"),
        make_example("c1", "doc_c"),
        make_example("d1", "doc_d"),
    ]

    split = split_distillation_by_source_group(
        examples,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=1,
    )
    train_groups = {example.source_group for example in split.train}
    val_groups = {example.source_group for example in split.val}
    test_groups = {example.source_group for example in split.test}

    assert split.train
    assert split.val
    assert split.test
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_assert_no_distillation_leakage_rejects_reused_id_or_source_group() -> None:
    with pytest.raises(ValueError, match="id"):
        assert_no_distillation_leakage(
            [make_example("same", "a")],
            [make_example("same", "b")],
            [],
        )

    with pytest.raises(ValueError, match="source group"):
        assert_no_distillation_leakage(
            [make_example("a", "same")],
            [],
            [make_example("b", "same")],
        )


def test_approved_distillation_example_converts_to_sft_example() -> None:
    sft = to_sft_example(make_example())

    assert sft.source == "distill:teacher-pro-2026-05:rag_prompt_v3"
    assert sft.source_group == "contract_a"
    assert sft.messages[-1].role == "assistant"
    assert "[citation]" in sft.messages[-1].content


def test_distillation_comparison_observes_student_change_and_writes_three_columns(
    tmp_path,
) -> None:
    comparisons = compare_base_teacher_student(
        prompts=["违约金条款是否可以直接执行？"],
        base_student_outputs=["可以直接执行。"],
        teacher_outputs=["需要结合合同上下文和证据判断。"],
        student_outputs=["需要结合合同上下文和证据判断。"],
        base_scores=[0.2],
        teacher_scores=[0.9],
        student_scores=[0.8],
    )
    report_path = tmp_path / "distill_eval.md"

    write_distillation_eval_report(report_path, comparisons)
    report = report_path.read_text()

    assert comparisons[0].student_changed_from_base is True
    assert comparisons[0].student_delta == pytest.approx(0.6)
    assert "| base_student | teacher | student |" in report
