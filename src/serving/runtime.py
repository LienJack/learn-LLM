from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path

from src.data.text_datasets import ChatMessage

GenerateFn = Callable[[list[ChatMessage], "GenerationConfig"], "GeneratedOutput"]


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    temperature: float = 0.2
    top_p: float = 1.0
    config_id: str = "generation_default"

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("GenerationConfig.max_new_tokens must be positive.")
        if self.temperature < 0:
            raise ValueError("GenerationConfig.temperature must be non-negative.")
        if not 0 < self.top_p <= 1:
            raise ValueError("GenerationConfig.top_p must be in (0, 1].")
        if not self.config_id:
            raise ValueError("GenerationConfig.config_id must not be empty.")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class ServingRequest:
    request_id: str
    messages: list[ChatMessage]
    generation_config: GenerationConfig

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("ServingRequest.request_id must not be empty.")
        if not self.messages:
            raise ValueError("ServingRequest.messages must not be empty.")


@dataclass(frozen=True)
class GeneratedOutput:
    answer: str
    citations: list[str] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    needs_human_review: bool = False
    finish_reason: str = "stop"
    parse_status: str = "not_required"
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.answer:
            raise ValueError("GeneratedOutput.answer must not be empty.")
        if self.finish_reason not in {"stop", "length", "timeout", "error"}:
            raise ValueError("GeneratedOutput.finish_reason is invalid.")
        if self.parse_status not in {"valid_json", "invalid_json", "not_required"}:
            raise ValueError("GeneratedOutput.parse_status is invalid.")


@dataclass(frozen=True)
class ServingResponse:
    request_id: str
    answer: str
    citations: list[str]
    safety_flags: list[str]
    needs_human_review: bool
    model_version: str
    adapter_version: str
    rag_index_version: str
    prompt_template_version: str
    safety_policy_version: str
    quantization: str
    generation_config_id: str
    finish_reason: str
    parse_status: str
    latency_ms: int
    token_usage: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ServingError:
    code: str
    message: str

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("ServingError code and message must not be empty.")


@dataclass(frozen=True)
class ServingErrorResponse:
    request_id: str
    error: ServingError
    model_version: str
    retryable: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DeploymentConfig:
    model_version: str
    rollback_target: str
    rag_index_version: str
    quantization: str
    owner: str
    adapter_version: str
    prompt_template_version: str
    safety_policy_version: str
    tokenizer_version: str
    max_concurrency: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RollbackConfig:
    current_model_version: str
    rollback_model_version: str
    current_adapter_version: str
    rollback_adapter_version: str
    current_rag_index_version: str
    rollback_rag_index_version: str
    prompt_template_version: str
    safety_policy_version: str
    generation_config_id: str
    tokenizer_version: str

    def __post_init__(self) -> None:
        missing = [
            name
            for name, value in asdict(self).items()
            if isinstance(value, str) and not value
        ]
        if missing:
            raise ValueError(f"rollback config missing fields: {sorted(missing)}")
        if self.current_model_version == self.rollback_model_version:
            raise ValueError("rollback_model_version must differ from current_model_version.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkResult:
    request_count: int
    error_count: int
    p50_latency_ms: float
    p95_latency_ms: float
    tokens_per_second: float
    p99_latency_ms: float = 0.0
    requests_per_second: float = 0.0
    peak_memory_mb: int = 0
    input_length_distribution: dict[str, float] = field(default_factory=dict)
    output_length_distribution: dict[str, float] = field(default_factory=dict)
    warmup_requests: int = 0
    concurrency: int = 1

    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["error_rate"] = self.error_rate
        return data


@dataclass(frozen=True)
class QuantizationRun:
    version: str
    precision: str
    memory_mb: int
    benchmark: BenchmarkResult
    eval_metrics: dict[str, float]
    eval_set_id: str

    def __post_init__(self) -> None:
        if self.memory_mb <= 0:
            raise ValueError("QuantizationRun.memory_mb must be positive.")
        if not self.eval_set_id:
            raise ValueError("QuantizationRun.eval_set_id must not be empty.")


@dataclass(frozen=True)
class DeploymentManifest:
    model_version: str
    adapter_version: str
    rag_index_version: str
    prompt_template_version: str
    safety_policy_version: str
    tokenizer_version: str
    quantization: str
    generation_config_id: str
    eval_report: str
    risk_report: str
    model_card: str
    benchmark_report: str
    rollback_target: str

    def __post_init__(self) -> None:
        missing = [
            name
            for name, value in asdict(self).items()
            if isinstance(value, str) and not value
        ]
        if missing:
            raise ValueError(f"deployment manifest missing fields: {sorted(missing)}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReleaseGateConfig:
    min_json_valid_rate: float = 0.98
    min_citation_support_rate: float = 0.9
    max_high_risk_unsafe_answer_rate: float = 0.0
    max_p95_latency_ms: float = 2_000
    max_error_rate: float = 0.01


@dataclass(frozen=True)
class ReleaseCandidate:
    manifest: DeploymentManifest
    eval_metrics: dict[str, float]
    benchmark: BenchmarkResult
    required_files: dict[str, bool]


@dataclass(frozen=True)
class ReleaseGateResult:
    passed: bool
    failed_reasons: list[str]


class LocalServingEngine:
    def __init__(self, deployment: DeploymentConfig, generate_fn: GenerateFn) -> None:
        validate_deployment_config(deployment)
        self.deployment = deployment
        self.generate_fn = generate_fn

    def handle(self, request: ServingRequest) -> ServingResponse:
        start = time.perf_counter()
        output = self.generate_fn(request.messages, request.generation_config)
        latency_ms = max(0, int((time.perf_counter() - start) * 1000))
        input_tokens = _count_message_tokens(request.messages)
        output_tokens = output.output_tokens or _count_text_tokens(output.answer)
        return ServingResponse(
            request_id=request.request_id,
            answer=output.answer,
            citations=output.citations,
            safety_flags=output.safety_flags,
            needs_human_review=output.needs_human_review,
            model_version=self.deployment.model_version,
            adapter_version=self.deployment.adapter_version,
            rag_index_version=self.deployment.rag_index_version,
            prompt_template_version=self.deployment.prompt_template_version,
            safety_policy_version=self.deployment.safety_policy_version,
            quantization=self.deployment.quantization,
            generation_config_id=request.generation_config.config_id,
            finish_reason=output.finish_reason,
            parse_status=output.parse_status,
            latency_ms=latency_ms,
            token_usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )

    def error_response(
        self,
        request_id: str,
        code: str,
        message: str,
        *,
        retryable: bool,
    ) -> ServingErrorResponse:
        return ServingErrorResponse(
            request_id=request_id,
            error=ServingError(code=code, message=message),
            model_version=self.deployment.model_version,
            retryable=retryable,
        )


def validate_deployment_config(config: DeploymentConfig) -> None:
    required = {
        "model_version": config.model_version,
        "rollback_target": config.rollback_target,
        "rag_index_version": config.rag_index_version,
        "quantization": config.quantization,
        "owner": config.owner,
        "adapter_version": config.adapter_version,
        "prompt_template_version": config.prompt_template_version,
        "safety_policy_version": config.safety_policy_version,
        "tokenizer_version": config.tokenizer_version,
    }
    missing = [field_name for field_name, value in required.items() if not value]
    if missing:
        raise ValueError(f"deployment config missing fields: {sorted(missing)}")
    if config.rollback_target == config.model_version:
        raise ValueError("rollback_target must differ from model_version.")
    if config.max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive.")


def benchmark_requests(
    engine: LocalServingEngine,
    requests: list[ServingRequest],
    *,
    warmup_requests: int = 0,
    concurrency: int = 1,
    peak_memory_mb: int = 0,
) -> BenchmarkResult:
    latencies: list[int] = []
    input_lengths: list[int] = []
    output_lengths: list[int] = []
    total_output_tokens = 0
    error_count = 0

    measured_requests = requests[warmup_requests:]
    for request in measured_requests:
        try:
            response = engine.handle(request)
        except Exception:
            error_count += 1
            continue
        latencies.append(response.latency_ms)
        input_lengths.append(response.token_usage["prompt_tokens"])
        output_lengths.append(response.token_usage["completion_tokens"])
        total_output_tokens += response.token_usage["output_tokens"]

    total_latency_seconds = max(sum(latencies) / 1000, 1e-9)
    return BenchmarkResult(
        request_count=len(measured_requests),
        error_count=error_count,
        p50_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        p99_latency_ms=_percentile(latencies, 99),
        tokens_per_second=total_output_tokens / total_latency_seconds,
        requests_per_second=max(len(latencies) / total_latency_seconds, 0.0),
        peak_memory_mb=peak_memory_mb,
        input_length_distribution=_length_distribution(input_lengths),
        output_length_distribution=_length_distribution(output_lengths),
        warmup_requests=warmup_requests,
        concurrency=concurrency,
    )


def compare_quantization_runs(runs: list[QuantizationRun]) -> list[dict[str, object]]:
    if not runs:
        raise ValueError("at least one quantization run is required.")
    eval_set_ids = {run.eval_set_id for run in runs}
    if len(eval_set_ids) != 1:
        raise ValueError("quantization runs must use the same eval set.")

    rows: list[dict[str, object]] = []
    for run in runs:
        row: dict[str, object] = {
            "version": run.version,
            "precision": run.precision,
            "memory_mb": run.memory_mb,
            "p50_latency_ms": run.benchmark.p50_latency_ms,
            "p95_latency_ms": run.benchmark.p95_latency_ms,
            "p99_latency_ms": run.benchmark.p99_latency_ms,
            "tokens_per_second": run.benchmark.tokens_per_second,
            "peak_memory_mb": run.benchmark.peak_memory_mb or run.memory_mb,
            "error_rate": run.benchmark.error_rate,
        }
        for metric_name, value in sorted(run.eval_metrics.items()):
            row[f"eval_{metric_name}"] = value
        rows.append(row)
    return rows


def write_quantization_report(path: str | Path, runs: list[QuantizationRun]) -> None:
    rows = compare_quantization_runs(runs)
    headers = list(rows[0].keys())
    lines = [
        "# 量化部署对比报告",
        "",
        f"- eval_set_id: `{runs[0].eval_set_id}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row[header]) for header in headers) + " |")
    Path(path).write_text("\n".join(lines) + "\n")


def run_release_gate(
    candidate: ReleaseCandidate,
    config: ReleaseGateConfig | None = None,
) -> ReleaseGateResult:
    gate = config or ReleaseGateConfig()
    failures: list[str] = []

    for name, exists in sorted(candidate.required_files.items()):
        if not exists:
            failures.append(f"missing required file: {name}")

    json_valid_rate = candidate.eval_metrics.get("json_valid_rate", 0.0)
    citation_support_rate = candidate.eval_metrics.get("citation_support_rate", 0.0)
    unsafe_rate = candidate.eval_metrics.get("high_risk_unsafe_answer_rate", 1.0)

    if json_valid_rate < gate.min_json_valid_rate:
        failures.append("json_valid_rate below threshold")
    if citation_support_rate < gate.min_citation_support_rate:
        failures.append("citation_support_rate below threshold")
    if unsafe_rate > gate.max_high_risk_unsafe_answer_rate:
        failures.append("high_risk_unsafe_answer_rate above threshold")
    if candidate.benchmark.p95_latency_ms > gate.max_p95_latency_ms:
        failures.append("p95_latency_ms above threshold")
    if candidate.benchmark.error_rate > gate.max_error_rate:
        failures.append("error_rate above threshold")

    return ReleaseGateResult(passed=not failures, failed_reasons=failures)


def build_safe_log_record(
    *,
    request_id: str,
    input_text: str,
    response: ServingResponse,
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "input_hash": sha256(input_text.encode("utf-8")).hexdigest(),
        "input_length": _count_text_tokens(input_text),
        "model_version": response.model_version,
        "adapter_version": response.adapter_version,
        "rag_index_version": response.rag_index_version,
        "prompt_template_version": response.prompt_template_version,
        "safety_policy_version": response.safety_policy_version,
        "quantization": response.quantization,
        "safety_flags": response.safety_flags,
        "needs_human_review": response.needs_human_review,
        "finish_reason": response.finish_reason,
        "parse_status": response.parse_status,
        "latency_ms": response.latency_ms,
    }


def _count_message_tokens(messages: list[ChatMessage]) -> int:
    return sum(_count_text_tokens(message.content) for message in messages)


def _count_text_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _percentile(values: list[int], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile / 100)
    return float(ordered[index])


def _length_distribution(values: list[int]) -> dict[str, float]:
    return {
        "min": float(min(values)) if values else 0.0,
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": float(max(values)) if values else 0.0,
    }


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
