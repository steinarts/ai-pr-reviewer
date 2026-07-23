from __future__ import annotations

import json
from pathlib import Path

from .models import ReviewResult


def print_summary(result: ReviewResult, proposed_count: int) -> None:
    print("AI PR review completed")
    print(f"Compared: {result.metadata.base}...{result.metadata.head}")
    print(f"Changed files: {result.metadata.changed_files}")
    print(f"Diff lines: {result.metadata.diff_lines}")
    print(f"Reviewers: {', '.join(result.metadata.reviewers)}")
    print(f"Proposed findings: {proposed_count}")
    print(f"Accepted findings: {len(result.accepted_findings)}")
    print(f"Rejected findings: {len(result.rejected_findings)}")

    if result.rejected_findings:
        print("Rejection reasons:")
        for finding in result.rejected_findings:
            print(f"- {finding.id}: {finding.rejection_reason}")


def write_json(result: ReviewResult, output_path: Path) -> None:
    if hasattr(result, "model_dump"):
        payload = result.model_dump(mode="json")
    else:
        payload = result.dict()

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"JSON written to: {output_path}")
