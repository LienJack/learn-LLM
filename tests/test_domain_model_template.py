import json

import pytest

from src.domain_model_template.template import (
    DomainConfig,
    RunManifest,
    check_release_readiness,
    parse_simple_yaml,
    validate_config,
    validate_report_chain,
    validate_run_manifest,
    validate_template_tree,
    write_run_manifest,
)


def make_config() -> DomainConfig:
    return DomainConfig.from_dict(
        {
            "project": {"name": "domain_model_template", "domain": "custom"},
            "base_model": {"model_id": "tiny-base", "revision": "main"},
            "data": {
                "train_path": "data/sft/train.jsonl",
                "val_path": "data/sft/val.jsonl",
                "eval_path": "data/eval/eval.jsonl",
                "split_seed": 42,
            },
            "training": {"method": "lora", "learning_rate": 0.0002},
            "rag": {"index_version": "kb_v1", "top_k": 5},
            "serving": {
                "model_version": "domain-model-v1",
                "rollback_target": "domain-model-v0",
            },
        },
    )


def make_manifest() -> RunManifest:
    return RunManifest(
        run_id="run_v1",
        base_model="tiny-base@main",
        dataset_version="sft_v1",
        eval_dataset_version="eval_v1",
        rag_index_version="kb_v1",
        model_version="domain-model-v1",
        config_files=["configs/data.yaml", "configs/eval.yaml"],
        report_files=[
            "reports/data_quality_report.md",
            "reports/eval_report.md",
            "reports/failure_cases.csv",
            "reports/risk_report.md",
            "reports/model_card.md",
        ],
    )


def write_template_reports(root) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True)
    (reports / "data_quality_report.md").write_text("dataset_version: sft_v1\n")
    (reports / "eval_report.md").write_text("dataset_version: sft_v1\n")
    (reports / "failure_cases.csv").write_text("eval_id,input\n")
    (reports / "risk_report.md").write_text("failure_cases.csv\n")
    (reports / "model_card.md").write_text("eval_report.md\n")


def test_template_tree_contains_required_directories(tmp_path) -> None:
    for directory in ["configs", "data", "scripts", "src", "tests", "reports"]:
        (tmp_path / directory).mkdir()

    validate_template_tree(tmp_path)

    with pytest.raises(ValueError, match="configs"):
        validate_template_tree(tmp_path / "missing")


def test_each_config_can_parse_and_contains_required_fields() -> None:
    text = """
project:
  name: domain_model_template
  domain: custom
base_model:
  model_id: tiny-base
  revision: main
data:
  train_path: data/sft/train.jsonl
  eval_path: data/eval/eval.jsonl
  split_seed: 42
training:
  method: lora
  learning_rate: 0.0002
rag:
  index_version: kb_v1
  top_k: 5
serving:
  model_version: domain-model-v1
  rollback_target: domain-model-v0
"""
    config = DomainConfig.from_dict(parse_simple_yaml(text))

    validate_config(config)
    assert config.project["name"] == "domain_model_template"

    with pytest.raises(ValueError, match="rollback_target"):
        validate_config(
            DomainConfig.from_dict(
                {
                    **config.to_dict(),
                    "serving": {
                        "model_version": "same",
                        "rollback_target": "same",
                    },
                },
            ),
        )


def test_run_manifest_records_model_data_rag_index_and_config_versions(tmp_path) -> None:
    manifest = make_manifest()
    path = tmp_path / "run_manifest.json"

    validate_run_manifest(manifest)
    write_run_manifest(path, manifest)
    loaded = json.loads(path.read_text())

    assert loaded["base_model"] == "tiny-base@main"
    assert loaded["dataset_version"] == "sft_v1"
    assert loaded["rag_index_version"] == "kb_v1"
    assert loaded["config_files"] == ["configs/data.yaml", "configs/eval.yaml"]


def test_report_chain_requires_files_and_version_references(tmp_path) -> None:
    write_template_reports(tmp_path)
    manifest = make_manifest()

    validate_report_chain(tmp_path, manifest)

    (tmp_path / "reports" / "eval_report.md").unlink()
    with pytest.raises(ValueError, match="eval_report"):
        validate_report_chain(tmp_path, manifest)


def test_release_check_blocks_missing_eval_report_or_rollback_target(tmp_path) -> None:
    for directory in ["configs", "data", "scripts", "src", "tests"]:
        (tmp_path / directory).mkdir()
    write_template_reports(tmp_path)

    ready = check_release_readiness(tmp_path, make_config(), make_manifest())
    assert ready.passed is True

    (tmp_path / "reports" / "eval_report.md").unlink()
    missing_report = check_release_readiness(tmp_path, make_config(), make_manifest())
    assert missing_report.passed is False
    assert any("eval_report" in failure for failure in missing_report.failures)

    bad_config = DomainConfig.from_dict(
        {
            **make_config().to_dict(),
            "serving": {"model_version": "domain-model-v1", "rollback_target": ""},
        },
    )
    missing_rollback = check_release_readiness(tmp_path, bad_config, make_manifest())
    assert missing_rollback.passed is False
    assert any("rollback_target" in failure for failure in missing_rollback.failures)
