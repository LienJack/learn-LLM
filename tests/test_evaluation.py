import json

import pytest

from src.evaluation.metrics import check_citations, check_json_output
from src.evaluation.runner import (
    EvalExample,
    compare_metric_reports,
    ensure_unique_eval_ids,
    run_eval,
    write_regression_report,
)


def make_example(
    example_id: str = "eval_001",
    required_citations: list[str] | None = None,
    expected_refusal: bool = False,
) -> EvalExample:
    return EvalExample(
        id=example_id,
        input="分析合同条款风险",
        expected_behavior="输出 JSON，引用依据；资料不足时拒答",
        gold_facts=["liability cap matters"],
        required_citations=["contract_a#chunk_1"]
        if required_citations is None
        else required_citations,
        risk_tags=["contract", "needs_human_review"],
        rubric="legal_contract_v1",
        required_json_fields=["answer", "citations", "needs_human_review"],
        expected_refusal=expected_refusal,
        source_group="contract_a",
    )


def test_eval_example_schema_requires_risk_tags_and_unique_ids() -> None:
    with pytest.raises(ValueError, match="risk_tags"):
        EvalExample(
            id="bad",
            input="问题",
            expected_behavior="应拒答",
            risk_tags=[],
        )

    with pytest.raises(ValueError, match="duplicate"):
        ensure_unique_eval_ids([make_example("same"), make_example("same")])


def test_json_metric_accepts_valid_json_and_rejects_invalid_or_missing_fields() -> None:
    valid = check_json_output(
        '{"answer": "ok", "citations": [], "needs_human_review": true}',
        ["answer", "citations", "needs_human_review"],
    )
    missing = check_json_output('{"answer": "ok"}', ["answer", "citations"])
    invalid = check_json_output("not json", ["answer"])

    assert valid.score == 1.0
    assert missing.score == 0.0
    assert missing.missing_fields == ["citations"]
    assert invalid.valid is False


def test_citation_metric_finds_missing_and_unsupported_citations() -> None:
    knowledge_base = {
        "contract_a#chunk_1": "liability cap matters",
        "contract_b#chunk_2": "payment term",
    }

    supported = check_citations(
        output_citations=["contract_a#chunk_1"],
        required_citations=["contract_a#chunk_1"],
        knowledge_base=knowledge_base,
        output_text="liability cap matters in this clause",
    )
    missing = check_citations(
        output_citations=[],
        required_citations=["contract_a#chunk_1"],
        knowledge_base=knowledge_base,
    )
    unsupported = check_citations(
        output_citations=["contract_z#chunk_9"],
        required_citations=[],
        knowledge_base=knowledge_base,
    )

    assert supported.score == 1.0
    assert missing.missing_citations == ["contract_a#chunk_1"]
    assert unsupported.unsupported_citations == ["contract_z#chunk_9"]


def test_eval_runner_writes_predictions_metrics_report_and_failure_cases(tmp_path) -> None:
    examples = [
        make_example("ok"),
        make_example("bad", required_citations=["missing#chunk"]),
    ]
    knowledge_base = {"contract_a#chunk_1": "liability cap matters"}

    def predict(example: EvalExample) -> str:
        if example.id == "ok":
            return json.dumps(
                {
                    "answer": "liability cap matters",
                    "citations": ["contract_a#chunk_1"],
                    "needs_human_review": True,
                },
                ensure_ascii=False,
            )
        return "not json"

    result = run_eval(
        examples=examples,
        predict_fn=predict,
        output_dir=tmp_path,
        model_id="student-v1",
        knowledge_base=knowledge_base,
        generation_config={"temperature": 0.2},
    )

    assert result.prediction_path.exists()
    assert result.metrics_path.exists()
    assert result.report_path.exists()
    assert result.failure_cases_path.exists()
    assert len(result.predictions) == 2
    assert result.metrics["json_valid"] == pytest.approx(0.5)
    assert "模型评测报告" in result.report_path.read_text()
    assert "bad" in result.failure_cases_path.read_text()


def test_refusal_metric_counts_expected_refusal_in_eval_runner(tmp_path) -> None:
    example = make_example(
        "refusal",
        required_citations=[],
        expected_refusal=True,
    )

    result = run_eval(
        examples=[example],
        predict_fn=lambda _: json.dumps(
            {
                "answer": "资料不足，无法判断，需要人工复核。",
                "citations": [],
                "needs_human_review": True,
            },
            ensure_ascii=False,
        ),
        output_dir=tmp_path,
        model_id="student-v1",
        knowledge_base={},
    )

    assert result.metrics["refusal_correct"] == 1.0
    assert result.metrics["failure_rate"] == 0.0


def test_regression_report_compares_two_metric_versions(tmp_path) -> None:
    deltas = compare_metric_reports(
        {"json_valid": 0.5, "refusal_correct": 1.0},
        {"json_valid": 1.0, "refusal_correct": 0.75},
    )
    path = tmp_path / "regression.md"

    write_regression_report(path, "base", "student", deltas)
    report = path.read_text()

    assert {delta.metric for delta in deltas} == {"json_valid", "refusal_correct"}
    assert "| json_valid | 0.500 | 1.000 | +0.500 |" in report
    assert "student" in report
