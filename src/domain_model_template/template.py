from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REQUIRED_TEMPLATE_DIRS = ("configs", "data", "scripts", "src", "tests", "reports")
REQUIRED_CONFIG_SECTIONS = ("project", "base_model", "data", "training", "rag", "serving")
REQUIRED_REPORT_FILES = (
    "data_quality_report.md",
    "eval_report.md",
    "failure_cases.csv",
    "risk_report.md",
    "model_card.md",
)


@dataclass(frozen=True)
class DomainConfig:
    project: dict[str, Any]
    base_model: dict[str, Any]
    data: dict[str, Any]
    training: dict[str, Any]
    rag: dict[str, Any]
    serving: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainConfig:
        missing = sorted(set(REQUIRED_CONFIG_SECTIONS) - data.keys())
        if missing:
            raise ValueError(f"domain config missing sections: {missing}")
        return cls(
            project=dict(data["project"]),
            base_model=dict(data["base_model"]),
            data=dict(data["data"]),
            training=dict(data["training"]),
            rag=dict(data["rag"]),
            serving=dict(data["serving"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    base_model: str
    dataset_version: str
    eval_dataset_version: str
    rag_index_version: str
    model_version: str
    config_files: list[str]
    report_files: list[str]
    git_commit: str = "unknown"

    def __post_init__(self) -> None:
        validate_run_manifest(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReleaseCheckResult:
    passed: bool
    failures: list[str] = field(default_factory=list)


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small nested key/value YAML subset used by this teaching template."""
    root: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, separator, value = raw_line.strip().partition(":")
        if not separator:
            raise ValueError(f"invalid config line: {raw_line}")
        if indent == 0:
            if value.strip():
                root[key] = _parse_scalar(value.strip())
                current_section = None
            else:
                root[key] = {}
                current_section = root[key]
            continue
        if current_section is None:
            raise ValueError(f"nested key without section: {raw_line}")
        current_section[key] = _parse_scalar(value.strip())
    return root


def validate_template_tree(root: str | Path) -> None:
    root_path = Path(root)
    missing = [
        directory
        for directory in REQUIRED_TEMPLATE_DIRS
        if not (root_path / directory).is_dir()
    ]
    if missing:
        raise ValueError(f"template missing directories: {missing}")


def validate_config(config: DomainConfig) -> None:
    required_values = {
        "project.name": config.project.get("name"),
        "project.domain": config.project.get("domain"),
        "base_model.model_id": config.base_model.get("model_id"),
        "base_model.revision": config.base_model.get("revision"),
        "data.train_path": config.data.get("train_path"),
        "data.eval_path": config.data.get("eval_path"),
        "data.split_seed": config.data.get("split_seed"),
        "training.method": config.training.get("method"),
        "training.learning_rate": config.training.get("learning_rate"),
        "rag.index_version": config.rag.get("index_version"),
        "rag.top_k": config.rag.get("top_k"),
        "serving.model_version": config.serving.get("model_version"),
        "serving.rollback_target": config.serving.get("rollback_target"),
    }
    missing = [name for name, value in required_values.items() if value in (None, "")]
    if missing:
        raise ValueError(f"domain config missing required values: {missing}")
    if config.serving["model_version"] == config.serving["rollback_target"]:
        raise ValueError("serving.rollback_target must differ from model_version.")


def validate_run_manifest(manifest: RunManifest) -> None:
    required_values = {
        "run_id": manifest.run_id,
        "base_model": manifest.base_model,
        "dataset_version": manifest.dataset_version,
        "eval_dataset_version": manifest.eval_dataset_version,
        "rag_index_version": manifest.rag_index_version,
        "model_version": manifest.model_version,
    }
    missing = [name for name, value in required_values.items() if not value]
    if missing:
        raise ValueError(f"run manifest missing fields: {missing}")
    if not manifest.config_files:
        raise ValueError("run manifest must include config_files.")
    if not manifest.report_files:
        raise ValueError("run manifest must include report_files.")


def write_run_manifest(path: str | Path, manifest: RunManifest) -> None:
    Path(path).write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n")


def validate_report_chain(root: str | Path, manifest: RunManifest) -> None:
    root_path = Path(root)
    reports_path = root_path / "reports"
    missing_reports = [
        report for report in REQUIRED_REPORT_FILES if not (reports_path / report).is_file()
    ]
    if missing_reports:
        raise ValueError(f"missing required report files: {missing_reports}")
    missing_manifest_reports = [
        report for report in manifest.report_files if not (root_path / report).is_file()
    ]
    if missing_manifest_reports:
        raise ValueError(f"manifest references missing reports: {missing_manifest_reports}")

    eval_report = (reports_path / "eval_report.md").read_text()
    model_card = (reports_path / "model_card.md").read_text()
    risk_report = (reports_path / "risk_report.md").read_text()
    if manifest.dataset_version not in eval_report:
        raise ValueError("eval report must reference dataset version.")
    if "eval_report.md" not in model_card:
        raise ValueError("model card must reference eval_report.md.")
    if "failure_cases.csv" not in risk_report:
        raise ValueError("risk report must reference failure_cases.csv.")


def check_release_readiness(
    root: str | Path,
    config: DomainConfig,
    manifest: RunManifest,
) -> ReleaseCheckResult:
    failures: list[str] = []
    for check in (
        lambda: validate_template_tree(root),
        lambda: validate_config(config),
        lambda: validate_run_manifest(manifest),
        lambda: validate_report_chain(root, manifest),
    ):
        try:
            check()
        except ValueError as exc:
            failures.append(str(exc))
    return ReleaseCheckResult(passed=not failures, failures=failures)


def _parse_scalar(value: str) -> Any:
    if value in {"true", "false"}:
        return value == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip('"')
