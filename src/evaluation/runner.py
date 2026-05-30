from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.evaluation.metrics import (
    CitationMetricResult,
    JsonMetricResult,
    RefusalMetricResult,
    check_citations,
    check_json_output,
    check_refusal,
)

PredictFn = Callable[["EvalExample"], str]


@dataclass(frozen=True)
class EvalExample:
    id: str
    input: str
    expected_behavior: str
    gold_facts: list[str] = field(default_factory=list)
    required_citations: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)
    rubric: str = "general_v1"
    required_json_fields: list[str] = field(default_factory=list)
    expected_refusal: bool = False
    source_group: str = "default"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("EvalExample.id must not be empty.")
        if not self.input:
            raise ValueError("EvalExample.input must not be empty.")
        if not self.expected_behavior:
            raise ValueError("EvalExample.expected_behavior must not be empty.")
        if not self.risk_tags:
            raise ValueError("EvalExample.risk_tags must not be empty.")
        if not self.rubric:
            raise ValueError("EvalExample.rubric must not be empty.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalExample:
        required = {"id", "input", "expected_behavior", "risk_tags"}
        missing = sorted(required - data.keys())
        if missing:
            raise ValueError(f"missing required eval fields: {missing}")
        return cls(
            id=data["id"],
            input=data["input"],
            expected_behavior=data["expected_behavior"],
            gold_facts=list(data.get("gold_facts", [])),
            required_citations=list(data.get("required_citations", [])),
            risk_tags=list(data["risk_tags"]),
            rubric=data.get("rubric", "general_v1"),
            required_json_fields=list(data.get("required_json_fields", [])),
            expected_refusal=bool(data.get("expected_refusal", False)),
            source_group=data.get("source_group", "default"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalPrediction:
    eval_id: str
    model_id: str
    input: str
    raw_output: str
    parsed_output: dict[str, Any] | None
    metrics: dict[str, float]
    metric_failures: list[str]
    risk_tags: list[str]
    generation_config: dict[str, Any]
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalRunResult:
    predictions: list[EvalPrediction]
    metrics: dict[str, float]
    prediction_path: Path
    metrics_path: Path
    report_path: Path
    failure_cases_path: Path


@dataclass(frozen=True)
class RegressionDelta:
    metric: str
    old: float
    new: float
    delta: float


def ensure_unique_eval_ids(examples: list[EvalExample]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for example in examples:
        if example.id in seen:
            duplicates.add(example.id)
        seen.add(example.id)
    if duplicates:
        raise ValueError(f"duplicate eval ids: {sorted(duplicates)}")


def run_eval(
    examples: list[EvalExample],
    predict_fn: PredictFn,
    output_dir: str | Path,
    model_id: str,
    knowledge_base: dict[str, str],
    generation_config: dict[str, Any] | None = None,
) -> EvalRunResult:
    ensure_unique_eval_ids(examples)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generation_config = generation_config or {}

    predictions = [
        _score_prediction(
            example=example,
            raw_output=predict_fn(example),
            model_id=model_id,
            knowledge_base=knowledge_base,
            generation_config=generation_config,
        )
        for example in examples
    ]
    metrics = aggregate_metric_scores(predictions)

    prediction_path = output_path / "predictions.jsonl"
    metrics_path = output_path / "metrics.json"
    report_path = output_path / "eval_report.md"
    failure_cases_path = output_path / "failure_cases.csv"

    _write_predictions(prediction_path, predictions)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    _write_eval_report(report_path, model_id, metrics, predictions)
    _write_failure_cases(failure_cases_path, predictions, examples)

    return EvalRunResult(
        predictions=predictions,
        metrics=metrics,
        prediction_path=prediction_path,
        metrics_path=metrics_path,
        report_path=report_path,
        failure_cases_path=failure_cases_path,
    )


def aggregate_metric_scores(predictions: list[EvalPrediction]) -> dict[str, float]:
    if not predictions:
        return {}

    metric_names = sorted(
        {metric_name for prediction in predictions for metric_name in prediction.metrics},
    )
    metrics = {
        metric_name: sum(prediction.metrics[metric_name] for prediction in predictions)
        / len(predictions)
        for metric_name in metric_names
    }
    metrics["failure_rate"] = sum(
        1.0 for prediction in predictions if prediction.metric_failures
    ) / len(predictions)
    return metrics


def compare_metric_reports(
    old_metrics: dict[str, float],
    new_metrics: dict[str, float],
) -> list[RegressionDelta]:
    metric_names = sorted(set(old_metrics) | set(new_metrics))
    return [
        RegressionDelta(
            metric=metric_name,
            old=old_metrics.get(metric_name, 0.0),
            new=new_metrics.get(metric_name, 0.0),
            delta=new_metrics.get(metric_name, 0.0) - old_metrics.get(metric_name, 0.0),
        )
        for metric_name in metric_names
    ]


def write_regression_report(
    path: str | Path,
    old_model_id: str,
    new_model_id: str,
    deltas: list[RegressionDelta],
) -> None:
    lines = [
        "# 回归评测报告",
        "",
        f"- old_model: `{old_model_id}`",
        f"- new_model: `{new_model_id}`",
        "",
        "| metric | old | new | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for delta in deltas:
        lines.append(
            f"| {delta.metric} | {delta.old:.3f} | {delta.new:.3f} | "
            f"{delta.delta:+.3f} |",
        )
    Path(path).write_text("\n".join(lines) + "\n")


def _score_prediction(
    example: EvalExample,
    raw_output: str,
    model_id: str,
    knowledge_base: dict[str, str],
    generation_config: dict[str, Any],
) -> EvalPrediction:
    json_result = check_json_output(raw_output, example.required_json_fields)
    output_citations = _extract_citations(json_result.parsed, raw_output)
    output_text = _extract_answer_text(json_result.parsed, raw_output)
    citation_result = check_citations(
        output_citations=output_citations,
        required_citations=example.required_citations,
        knowledge_base=knowledge_base,
        output_text=output_text,
    )
    refusal_result = check_refusal(raw_output, example.expected_refusal)

    metrics = _metric_dict(json_result, citation_result, refusal_result)
    failures = _metric_failures(json_result, citation_result, refusal_result)
    return EvalPrediction(
        eval_id=example.id,
        model_id=model_id,
        input=example.input,
        raw_output=raw_output,
        parsed_output=json_result.parsed,
        metrics=metrics,
        metric_failures=failures,
        risk_tags=example.risk_tags,
        generation_config=generation_config,
    )


def _metric_dict(
    json_result: JsonMetricResult,
    citation_result: CitationMetricResult,
    refusal_result: RefusalMetricResult,
) -> dict[str, float]:
    return {
        "json_valid": json_result.score,
        "citation_supported": citation_result.score,
        "refusal_correct": refusal_result.score,
    }


def _metric_failures(
    json_result: JsonMetricResult,
    citation_result: CitationMetricResult,
    refusal_result: RefusalMetricResult,
) -> list[str]:
    failures: list[str] = []
    if json_result.score == 0.0:
        failures.append("json_invalid_or_missing_fields")
    if citation_result.score == 0.0:
        failures.append("citation_missing_or_unsupported")
    if not refusal_result.correct:
        failures.append("refusal_mismatch")
    return failures


def _extract_citations(parsed: dict[str, Any] | None, raw_output: str) -> list[str]:
    if parsed:
        citations = parsed.get("citations", [])
        if isinstance(citations, list):
            return [str(citation) for citation in citations]
        if isinstance(citations, str):
            return [citations]

    citations: list[str] = []
    for token in raw_output.replace("[", " ").replace("]", " ").split():
        if "#" in token:
            citations.append(token.strip(".,;:，。；："))
    return citations


def _extract_answer_text(parsed: dict[str, Any] | None, raw_output: str) -> str:
    if not parsed:
        return raw_output
    for key in ("answer", "plain_explanation", "suggested_revision"):
        value = parsed.get(key)
        if isinstance(value, str):
            return value
    return raw_output


def _write_predictions(path: Path, predictions: list[EvalPrediction]) -> None:
    lines = [
        json.dumps(prediction.to_dict(), ensure_ascii=False, sort_keys=True)
        for prediction in predictions
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _write_eval_report(
    path: Path,
    model_id: str,
    metrics: dict[str, float],
    predictions: list[EvalPrediction],
) -> None:
    lines = ["# 模型评测报告", "", f"- model_id: `{model_id}`", ""]
    lines.extend(["## Metrics", ""])
    for metric_name, value in sorted(metrics.items()):
        lines.append(f"- {metric_name}: `{value:.3f}`")
    lines.extend(["", "## Failure Cases", ""])
    failures = [prediction for prediction in predictions if prediction.metric_failures]
    if not failures:
        lines.append("No failure cases.")
    for prediction in failures:
        lines.extend(
            [
                f"### {prediction.eval_id}",
                "",
                f"- failures: {', '.join(prediction.metric_failures)}",
                f"- risk_tags: {', '.join(prediction.risk_tags)}",
                f"- raw_output: {prediction.raw_output}",
                "",
            ],
        )
    path.write_text("\n".join(lines) + "\n")


def _write_failure_cases(
    path: Path,
    predictions: list[EvalPrediction],
    examples: list[EvalExample],
) -> None:
    examples_by_id = {example.id: example for example in examples}
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "eval_id",
                "input",
                "expected_behavior",
                "model_output",
                "metric_failures",
                "risk_tags",
                "suspected_root_cause",
                "next_action",
            ],
        )
        writer.writeheader()
        for prediction in predictions:
            if not prediction.metric_failures:
                continue
            example = examples_by_id[prediction.eval_id]
            writer.writerow(
                {
                    "eval_id": prediction.eval_id,
                    "input": example.input,
                    "expected_behavior": example.expected_behavior,
                    "model_output": prediction.raw_output,
                    "metric_failures": ";".join(prediction.metric_failures),
                    "risk_tags": ";".join(prediction.risk_tags),
                    "suspected_root_cause": "needs_review",
                    "next_action": "inspect_prediction_and_update_eval_or_data",
                },
            )
