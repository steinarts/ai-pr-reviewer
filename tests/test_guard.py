from reviewer.guard import guard_findings
from reviewer.models import Finding, Severity


def test_guard_rejects_low_confidence() -> None:
    finding = Finding(
        id="f1",
        file="a.py",
        line=10,
        category="bug",
        severity=Severity.HIGH,
        confidence=0.4,
        title="Low confidence sample",
        evidence="some evidence",
        consequence="can fail",
        suggestion="add check",
        introduced_by_diff=True,
        actionable=True,
        style_only=False,
        reviewer="bug",
    )

    accepted, rejected = guard_findings([finding])
    assert not accepted
    assert len(rejected) == 1
    assert rejected[0].rejection_reason == "confidence_below_threshold"
