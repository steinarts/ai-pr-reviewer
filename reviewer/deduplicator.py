from __future__ import annotations

from .models import Finding


def deduplicate_findings(findings: list[Finding]) -> list[Finding]:
    by_key: dict[tuple[str, int, str, str, str], Finding] = {}
    for finding in findings:
        key = (
            finding.file,
            finding.line,
            finding.category,
            finding.title,
            finding.evidence,
        )
        existing = by_key.get(key)
        if existing is None or finding.confidence > existing.confidence:
            by_key[key] = finding
        elif existing and finding.reviewer and finding.reviewer not in existing.reviewer:
            existing.reviewer = f"{existing.reviewer},{finding.reviewer}".strip(",")
    return list(by_key.values())
