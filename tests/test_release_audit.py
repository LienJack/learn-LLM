from src.evaluation.release_audit import (
    AuditMaterial,
    GraduationAuditManifest,
    InternalOnlyControls,
    SeverityIssue,
    audit_graduation_release,
    decide_release_state,
    render_graduation_audit_report,
)


def make_manifest() -> GraduationAuditManifest:
    return GraduationAuditManifest(
        model_version="legal-model-v1",
        data_version="legal-data-v1",
        eval_report="reports/eval_report.md",
        model_card="reports/model_card.md",
        risk_report="reports/risk_report.md",
        release_gate_report="reports/release_gate.md",
        rollback_plan="reports/rollback.md",
        owner="course-owner",
    )


def test_graduation_audit_passes_when_required_documents_exist(tmp_path) -> None:
    for relative_path in [
        "reports/eval_report.md",
        "reports/model_card.md",
        "reports/risk_report.md",
        "reports/release_gate.md",
        "reports/rollback.md",
    ]:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n")

    result = audit_graduation_release(make_manifest(), project_root=tmp_path)

    assert result.passed is True
    assert result.missing_fields == []
    assert result.missing_files == []


def test_graduation_audit_blocks_missing_fields_and_files(tmp_path) -> None:
    manifest = GraduationAuditManifest(
        model_version="",
        data_version="data-v1",
        eval_report="reports/eval_report.md",
        model_card="",
        risk_report="reports/risk_report.md",
        release_gate_report="reports/release_gate.md",
        rollback_plan="reports/rollback.md",
        owner="",
    )

    result = audit_graduation_release(manifest, project_root=tmp_path)
    report = render_graduation_audit_report(result, manifest)

    assert result.passed is False
    assert result.missing_fields == ["model_version", "model_card", "owner"]
    assert "eval_report" in result.missing_files
    assert "BLOCKED" in report
    assert "model_version" in report


def test_severity_aware_release_decision_blocks_critical_and_allows_controlled_internal() -> None:
    controls = InternalOnlyControls(
        allowed_users=["reviewer@example.com"],
        traffic_scope="selected internal reviewers",
        human_review_enforced=True,
        export_disabled=True,
        audit_logging=True,
        expiration_date="2026-06-30",
    )
    internal = decide_release_state(
        [SeverityIssue(issue_id="boundary_001", severity="medium", blocking=True)],
        gate_passed=True,
        internal_only_controls=controls,
    )
    blocked = decide_release_state(
        [SeverityIssue(issue_id="privacy_001", severity="critical", blocking=True)],
        gate_passed=True,
        internal_only_controls=controls,
    )

    assert AuditMaterial(
        material="eval_report",
        version="v1",
        candidate_model="legal-model-v1",
        hash="abc123",
        status="present",
    ).status == "present"
    assert internal.state == "internal_only"
    assert blocked.state == "no_release"
