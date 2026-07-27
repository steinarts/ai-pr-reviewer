from __future__ import annotations

import json
from pathlib import Path

from .models import ReviewResult


def print_summary(result: ReviewResult, proposed_count: int) -> None:
    print("AI PR review completed")
    print(f"Compared: {result.metadata.base}...{result.metadata.head}")
    print(f"Changed files: {result.metadata.changed_files}")
    print(f"Diff lines: {result.metadata.diff_lines}")
    print(f"Review mode: {result.metadata.review_mode}")
    print(f"Reviewers: {', '.join(result.metadata.reviewers)}")
    print(f"Reviewable chunks: {result.metadata.reviewable_chunks}")
    print(f"Skipped chunks: {result.metadata.skipped_chunks}")
    print(f"Planned LLM requests: {result.metadata.planned_requests}")
    print(f"Completed LLM requests: {result.metadata.completed_requests}")
    print(f"Failed LLM requests: {result.metadata.failed_requests}")
    print(f"Skipped LLM requests: {result.metadata.skipped_requests}")
    print(f"Total elapsed time: {result.metadata.total_elapsed_seconds:.1f}s")
    print(f"Total time budget: {result.metadata.total_time_budget_seconds:.0f}s")
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
