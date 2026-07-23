from reviewer.deduplicator import deduplicate_findings
from reviewer.models import Finding, Severity


def _finding(fid: str, conf: float, reviewer: str) -> Finding:
    return Finding(
        id=fid,
        file="x.py",
        line=5,
        category="security",
        severity=Severity.HIGH,
        confidence=conf,
        title="Same root cause",
        evidence="dangerous path",
        consequence="impact",
        suggestion="fix",
        introduced_by_diff=True,
        actionable=True,
        style_only=False,
        reviewer=reviewer,
    )


def test_deduplicate_keeps_highest_confidence() -> None:
    findings = [_finding("1", 0.85, "bug"), _finding("2", 0.95, "security")]
    deduped = deduplicate_findings(findings)

    assert len(deduped) == 1
    assert deduped[0].confidence == 0.95
