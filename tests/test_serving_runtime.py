import pytest

from src.data.text_datasets import ChatMessage
from src.serving.runtime import (
    BenchmarkResult,
    CanaryConfig,
    DeploymentConfig,
    DeploymentManifest,
    GeneratedOutput,
    GenerationConfig,
    LocalServingEngine,
    QuantizationRun,
    ReleaseCandidate,
    ReleaseGateConfig,
    RollbackConfig,
    ServingRequest,
    benchmark_requests,
    build_safe_log_record,
    compare_quantization_runs,
    estimate_vram_mb,
    paired_metric_delta,
    run_release_gate,
    validate_deployment_config,
    write_quantization_report,
)


def make_deployment() -> DeploymentConfig:
    return DeploymentConfig(
        model_version="legal-sft-v2-int8",
        rollback_target="legal-sft-v1-fp16",
        rag_index_version="legal-guidelines-2026-05",
        quantization="int8",
        owner="serving-owner",
        adapter_version="legal-lora-v2",
        prompt_template_version="legal-rag-prompt-v4",
        safety_policy_version="legal-safety-v2",
        tokenizer_version="legal-tokenizer-v1",
        max_concurrency=4,
    )


def make_request(request_id: str = "req_001", max_new_tokens: int = 64) -> ServingRequest:
    return ServingRequest(
        request_id=request_id,
        messages=[ChatMessage(role="user", content="分析这段合同风险")],
        generation_config=GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.2),
    )


def make_engine() -> LocalServingEngine:
    def generate(
        messages: list[ChatMessage],
        config: GenerationConfig,
    ) -> GeneratedOutput:
        return GeneratedOutput(
            answer=f"风险提示，最大输出 {config.max_new_tokens} tokens。",
            citations=["contract#chunk_1"],
            safety_flags=["needs_human_review"],
            needs_human_review=True,
            finish_reason="stop",
            parse_status="valid_json",
            output_tokens=8,
        )

    return LocalServingEngine(make_deployment(), generate)


def test_api_response_contains_answer_model_version_latency_and_audit_fields() -> None:
    response = make_engine().handle(make_request())

    assert response.answer
    assert response.model_version == "legal-sft-v2-int8"
    assert response.latency_ms >= 0
    assert response.citations == ["contract#chunk_1"]
    assert response.safety_flags == ["needs_human_review"]
    assert response.needs_human_review is True
    assert response.rag_index_version == "legal-guidelines-2026-05"
    assert response.adapter_version == "legal-lora-v2"
    assert response.prompt_template_version == "legal-rag-prompt-v4"
    assert response.safety_policy_version == "legal-safety-v2"
    assert response.quantization == "int8"
    assert response.generation_config_id == "generation_default"
    assert response.finish_reason == "stop"
    assert response.parse_status == "valid_json"
    assert response.token_usage["prompt_tokens"] == response.token_usage["input_tokens"]
    assert response.token_usage["completion_tokens"] == response.token_usage["output_tokens"]
    assert response.token_usage["total_tokens"] >= response.token_usage["output_tokens"]


def test_api_error_response_schema() -> None:
    error = make_engine().error_response(
        "req_timeout",
        "generation_timeout",
        "request exceeded max latency budget",
        retryable=True,
    )

    data = error.to_dict()

    assert data["request_id"] == "req_timeout"
    assert data["error"]["code"] == "generation_timeout"
    assert data["model_version"] == "legal-sft-v2-int8"
    assert data["retryable"] is True


def test_generation_config_requires_max_output_length() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        GenerationConfig(max_new_tokens=0)

    with pytest.raises(ValueError, match="top_p"):
        GenerationConfig(max_new_tokens=8, top_p=1.5)


def test_benchmark_reports_p50_p95_tokens_per_second_and_error_rate() -> None:
    engine = make_engine()
    requests = [make_request("a"), make_request("b"), make_request("c")]

    benchmark = benchmark_requests(engine, requests, peak_memory_mb=4096)

    assert benchmark.request_count == 3
    assert benchmark.error_count == 0
    assert benchmark.p50_latency_ms >= 0
    assert benchmark.p95_latency_ms >= 0
    assert benchmark.p99_latency_ms >= 0
    assert benchmark.peak_memory_mb == 4096
    assert benchmark.tokens_per_second > 0
    assert benchmark.requests_per_second > 0
    assert benchmark.error_rate == 0
    assert benchmark.input_length_distribution["p50"] > 0
    assert benchmark.output_length_distribution["p50"] > 0


def test_quantization_runs_must_share_eval_set_and_write_report(tmp_path) -> None:
    fp16 = QuantizationRun(
        version="legal-fp16",
        precision="fp16",
        memory_mb=8000,
        benchmark=BenchmarkResult(2, 0, 20, 30, 40),
        eval_metrics={"json_valid": 1.0, "safe_refusal": 0.95},
        eval_set_id="legal_eval_v1",
        baseline_run_id="baseline",
    )
    int8 = QuantizationRun(
        version="legal-int8",
        precision="int8",
        memory_mb=4200,
        benchmark=BenchmarkResult(2, 0, 15, 25, 55),
        eval_metrics={"json_valid": 1.0, "safe_refusal": 0.95},
        eval_set_id="legal_eval_v1",
        baseline_run_id="legal-fp16",
    )

    rows = compare_quantization_runs([fp16, int8])
    path = tmp_path / "quantization_report.md"
    write_quantization_report(path, [fp16, int8])

    assert rows[0]["eval_json_valid"] == 1.0
    assert "legal_eval_v1" in path.read_text()
    assert "legal-int8" in path.read_text()
    delta, passed = paired_metric_delta(
        fp16,
        int8,
        "safe_refusal",
        higher_is_better=True,
        tolerance=0.01,
    )
    assert delta == pytest.approx(0.0)
    assert passed is True

    with pytest.raises(ValueError, match="same eval set"):
        compare_quantization_runs(
            [
                fp16,
                QuantizationRun(
                    version="legal-int4",
                    precision="int4",
                    memory_mb=2600,
                    benchmark=BenchmarkResult(2, 0, 10, 20, 60),
                    eval_metrics={"json_valid": 0.9},
                    eval_set_id="other_eval",
                ),
            ],
        )


def test_deployment_config_requires_rollback_target() -> None:
    validate_deployment_config(make_deployment())

    with pytest.raises(ValueError, match="rollback_target"):
        validate_deployment_config(
            DeploymentConfig(
                model_version="legal-v2",
                rollback_target="",
                rag_index_version="kb-v1",
                quantization="int8",
                owner="serving-owner",
                adapter_version="adapter-v1",
                prompt_template_version="prompt-v1",
                safety_policy_version="safety-v1",
                tokenizer_version="tokenizer-v1",
            ),
        )

    with pytest.raises(ValueError, match="must differ"):
        validate_deployment_config(
            DeploymentConfig(
                model_version="legal-v2",
                rollback_target="legal-v2",
                rag_index_version="kb-v1",
                quantization="int8",
                owner="serving-owner",
                adapter_version="adapter-v1",
                prompt_template_version="prompt-v1",
                safety_policy_version="safety-v1",
                tokenizer_version="tokenizer-v1",
            ),
        )


def make_manifest() -> DeploymentManifest:
    return DeploymentManifest(
        model_version="legal-sft-v2-int8",
        adapter_version="legal-lora-v2",
        rag_index_version="legal-guidelines-2026-05",
        prompt_template_version="legal-rag-prompt-v4",
        safety_policy_version="legal-safety-v2",
        tokenizer_version="legal-tokenizer-v1",
        quantization="int8",
        generation_config_id="generation_default",
        eval_report="reports/eval_report.md",
        risk_report="reports/risk_report.md",
        model_card="reports/model_card.md",
        benchmark_report="reports/benchmark_report.md",
        rollback_target="legal-sft-v1-fp16",
    )


def test_release_gate_passes_with_required_reports_and_metrics() -> None:
    candidate = ReleaseCandidate(
        manifest=make_manifest(),
        eval_metrics={
            "json_valid_rate": 0.99,
            "schema_pass_rate": 0.99,
            "citation_support_rate": 0.95,
            "legal_boundary_pass_rate": 0.98,
            "high_risk_unsafe_answer_rate": 0.0,
            "prompt_injection_followed_rate": 0.0,
        },
        benchmark=BenchmarkResult(10, 0, 100, 250, 50),
        required_files={
            "eval_report": True,
            "risk_report": True,
            "model_card": True,
            "benchmark_report": True,
            "rollback_target": True,
        },
    )

    result = run_release_gate(candidate, ReleaseGateConfig(max_p95_latency_ms=300))

    assert result.passed is True
    assert result.failed_reasons == []


def test_release_gate_blocks_missing_reports_rollback_and_safety_regression() -> None:
    candidate = ReleaseCandidate(
        manifest=make_manifest(),
        eval_metrics={
            "json_valid_rate": 0.99,
            "schema_pass_rate": 0.9,
            "citation_support_rate": 0.95,
            "legal_boundary_pass_rate": 0.8,
            "high_risk_unsafe_answer_rate": 0.2,
            "prompt_injection_followed_rate": 0.1,
        },
        benchmark=BenchmarkResult(10, 1, 100, 2_500, 50),
        required_files={
            "eval_report": False,
            "risk_report": True,
            "model_card": True,
            "benchmark_report": True,
            "rollback_target": False,
        },
    )

    result = run_release_gate(candidate, ReleaseGateConfig(max_p95_latency_ms=300))

    assert result.passed is False
    assert "missing required file: eval_report" in result.failed_reasons
    assert "missing required file: rollback_target" in result.failed_reasons
    assert "schema_pass_rate below threshold" in result.failed_reasons
    assert "legal_boundary_pass_rate below threshold" in result.failed_reasons
    assert "high_risk_unsafe_answer_rate above threshold" in result.failed_reasons
    assert "prompt_injection_followed_rate above threshold" in result.failed_reasons
    assert "p95_latency_ms above threshold" in result.failed_reasons
    assert "error_rate above threshold" in result.failed_reasons


def test_vram_estimate_and_canary_config_capture_release_controls() -> None:
    estimate = estimate_vram_mb(
        weight_memory_mb=4000,
        layers=2,
        batch_size=1,
        kv_heads=2,
        sequence_length=128,
        head_dim=64,
        bytes_per_kv=2,
        activation_workspace_mb=512,
        framework_overhead_mb=256,
    )
    canary = CanaryConfig(
        owner="serving-owner",
        traffic_scope="1% internal",
        stop_conditions=["privacy leak once", "schema_fail_rate > 0.02"],
        alert_channels=["#alerts"],
        rollback_command="deploy rollback legal-sft-v1-fp16",
        post_rollback_smoke_test="pytest tests/test_serving_runtime.py",
        incident_log_path="reports/incidents.md",
    )

    assert estimate > 4000
    assert canary.rollback_command.startswith("deploy rollback")


def test_safe_log_record_does_not_store_raw_sensitive_input() -> None:
    response = make_engine().handle(make_request())
    raw_input = "合同里包含手机号 13800138000 和身份证 110101199001011234"

    record = build_safe_log_record(
        request_id="req_sensitive",
        input_text=raw_input,
        response=response,
    )

    assert "raw_input" not in record
    assert raw_input not in str(record)
    assert "input_hash" in record
    assert record["model_version"] == "legal-sft-v2-int8"
    assert record["adapter_version"] == "legal-lora-v2"


def test_rollback_config_contains_compatible_version_group() -> None:
    rollback = RollbackConfig(
        current_model_version="legal-sft-v2-int8",
        rollback_model_version="legal-sft-v1-fp16",
        current_adapter_version="legal-lora-v2",
        rollback_adapter_version="legal-lora-v1",
        current_rag_index_version="legal-guidelines-2026-05",
        rollback_rag_index_version="legal-guidelines-2026-04",
        prompt_template_version="legal-rag-prompt-v4",
        safety_policy_version="legal-safety-v2",
        generation_config_id="generation_default",
        tokenizer_version="legal-tokenizer-v1",
    )

    data = rollback.to_dict()

    assert data["current_model_version"] == "legal-sft-v2-int8"
    assert data["rollback_rag_index_version"] == "legal-guidelines-2026-04"

    with pytest.raises(ValueError, match="must differ"):
        RollbackConfig(
            current_model_version="same",
            rollback_model_version="same",
            current_adapter_version="adapter-v2",
            rollback_adapter_version="adapter-v1",
            current_rag_index_version="kb-v2",
            rollback_rag_index_version="kb-v1",
            prompt_template_version="prompt-v1",
            safety_policy_version="safety-v1",
            generation_config_id="generation_default",
            tokenizer_version="tokenizer-v1",
        )
