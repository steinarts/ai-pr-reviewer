from __future__ import annotations

from .models import Finding, Severity, Status

DEFAULT_ALLOWED_SEVERITIES = {Severity.HIGH, Severity.CRITICAL}
DEFAULT_MIN_CONFIDENCE = 0.85


def guard_findings(
    findings: list[Finding],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    allowed_severities: set[Severity] | None = None,
    max_published: int = 3,
) -> tuple[list[Finding], list[Finding]]:
    allowed = allowed_severities or DEFAULT_ALLOWED_SEVERITIES
    accepted: list[Finding] = []
    rejected: list[Finding] = []

    for finding in findings:
        reason = ""
        if finding.severity not in allowed:
            reason = "severity_below_threshold"
        elif finding.confidence < min_confidence:
            reason = "confidence_below_threshold"
        elif finding.style_only:
            reason = "style_only"
        elif not finding.introduced_by_diff:
            reason = "not_introduced_by_diff"
        elif not finding.actionable:
            reason = "not_actionable"
        elif not finding.evidence.strip():
            reason = "missing_evidence"
        elif not finding.consequence.strip():
            reason = "missing_consequence"

        if reason:
            finding.status = Status.REJECTED
            finding.rejection_reason = reason
            rejected.append(finding)
        else:
            finding.status = Status.ACCEPTED
            accepted.append(finding)

    if len(accepted) > max_published:
        overflow = accepted[max_published:]
        accepted = accepted[:max_published]
        for finding in overflow:
            finding.status = Status.REJECTED
            finding.rejection_reason = "max_published_limit"
            rejected.append(finding)

    return accepted, rejected
