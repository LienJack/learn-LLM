from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

RELEASE_STATES = {"no_release", "internal_only", "canary", "release", "rollback"}
MATERIAL_STATUSES = {"present", "missing", "stale"}


@dataclass(frozen=True)
class GraduationAuditManifest:
    model_version: str
    data_version: str
    eval_report: str
    model_card: str
    risk_report: str
    release_gate_report: str
    rollback_plan: str
    owner: str
    candidate_model: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class AuditMaterial:
    material: str
    version: str
    candidate_model: str
    hash: str
    status: str

    def __post_init__(self) -> None:
        if self.status not in MATERIAL_STATUSES:
            raise ValueError(f"status must be one of {sorted(MATERIAL_STATUSES)}.")
        missing = [
            name
            for name, value in asdict(self).items()
            if isinstance(value, str) and not value
        ]
        if missing:
            raise ValueError(f"AuditMaterial missing fields: {sorted(missing)}")


@dataclass(frozen=True)
class InternalOnlyControls:
    allowed_users: list[str]
    traffic_scope: str
    human_review_enforced: bool
    export_disabled: bool
    audit_logging: bool
    expiration_date: str

    def is_enforced(self) -> bool:
        return (
            bool(self.allowed_users)
            and bool(self.traffic_scope)
            and self.human_review_enforced
            and self.export_disabled
            and self.audit_logging
            and bool(self.expiration_date)
        )


@dataclass(frozen=True)
class IncidentResponsePlan:
    severity: str
    first_response_owner: str
    rollback_time: str
    user_impact: str
    added_failure_cases: list[str]
    postmortem_link: str
    next_review_date: str


@dataclass(frozen=True)
class SeverityIssue:
    issue_id: str
    severity: str
    blocking: bool


@dataclass(frozen=True)
class ReleaseDecision:
    state: str
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GraduationAuditResult:
    passed: bool
    missing_fields: list[str]
    missing_files: list[str]


REQUIRED_AUDIT_FIELDS = (
    "model_version",
    "data_version",
    "eval_report",
    "model_card",
    "risk_report",
    "release_gate_report",
    "rollback_plan",
    "owner",
)


def audit_graduation_release(
    manifest: GraduationAuditManifest,
    *,
    project_root: str | Path,
) -> GraduationAuditResult:
    data = manifest.to_dict()
    missing_fields = [field for field in REQUIRED_AUDIT_FIELDS if not data[field]]
    root = Path(project_root)
    missing_files = [
        field
        for field in (
            "eval_report",
            "model_card",
            "risk_report",
            "release_gate_report",
            "rollback_plan",
        )
        if data[field] and not (root / data[field]).exists()
    ]
    return GraduationAuditResult(
        passed=not missing_fields and not missing_files,
        missing_fields=missing_fields,
        missing_files=missing_files,
    )


def decide_release_state(
    issues: list[SeverityIssue],
    *,
    gate_passed: bool,
    internal_only_controls: InternalOnlyControls | None = None,
) -> ReleaseDecision:
    critical = [
        issue.issue_id
        for issue in issues
        if issue.severity == "critical" and issue.blocking
    ]
    if critical:
        return ReleaseDecision(
            "no_release",
            [f"critical blocking issue: {issue}" for issue in critical],
        )
    if not gate_passed:
        return ReleaseDecision("no_release", ["release gate failed"])
    noncritical_boundary = [
        issue.issue_id
        for issue in issues
        if issue.blocking and issue.severity in {"low", "medium", "high"}
    ]
    if noncritical_boundary:
        if internal_only_controls and internal_only_controls.is_enforced():
            return ReleaseDecision(
                "internal_only",
                [
                    f"noncritical boundary issue under controls: {issue}"
                    for issue in noncritical_boundary
                ],
            )
        return ReleaseDecision("no_release", ["internal_only controls missing"])
    return ReleaseDecision("canary", [])


def render_graduation_audit_report(
    result: GraduationAuditResult,
    manifest: GraduationAuditManifest,
) -> str:
    status = "PASS" if result.passed else "BLOCKED"
    lines = [
        "# 毕业发布审计",
        "",
        f"- status: `{status}`",
        f"- model_version: `{manifest.model_version or 'missing'}`",
        f"- data_version: `{manifest.data_version or 'missing'}`",
        f"- owner: `{manifest.owner or 'missing'}`",
        "",
        "## 缺失字段",
    ]
    if result.missing_fields:
        lines.extend(f"- {field}" for field in result.missing_fields)
    else:
        lines.append("- 无")
    lines.extend(["", "## 缺失文件"])
    if result.missing_files:
        lines.extend(f"- {field}" for field in result.missing_files)
    else:
        lines.append("- 无")
    return "\n".join(lines) + "\n"
