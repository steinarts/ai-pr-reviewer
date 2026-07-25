from __future__ import annotations

import argparse
from pathlib import Path

from .context_builder import build_context
from .deduplicator import deduplicate_findings
from .git_diff import GitDiffError, collect_diff
from .guard import guard_findings
from .llm_factory import create_llm_client
from .models import ReviewMetadata, ReviewResult, Severity
from .output import print_summary, write_json
from .scouts import run_reviewers

DEFAULT_REVIEWERS = ["bug", "reliability", "security"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local AI PR reviewer")
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", default="review-result.json")
    parser.add_argument(
        "--provider",
        choices=["fake", "ollama"],
        default="fake",
        help="LLM provider (default: fake)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name (required for ollama, e.g., qwen2.5-coder:7b)",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama server host (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use fake LLM (same as --provider fake)",
    )
    parser.add_argument("--max-files", type=int, default=30)
    parser.add_argument("--max-diff-lines", type=int, default=3000)
    parser.add_argument("--max-published", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--severities", nargs="*", default=["high", "critical"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cwd = Path.cwd()
    project_root = Path(__file__).resolve().parents[1]
    prompts_dir = project_root / "prompts"

    try:
        snapshot = collect_diff(
            base=args.base,
            head=args.head,
            cwd=cwd,
            max_files=args.max_files,
            max_diff_lines=args.max_diff_lines,
        )
    except GitDiffError as exc:
        print(f"Git diff error: {exc}")
        return 2

    context = build_context(snapshot=snapshot, head=args.head, cwd=cwd)

    # Determine provider: --dry-run forces fake, otherwise use --provider
    provider = "fake" if args.dry_run else args.provider

    try:
        llm_client = create_llm_client(
            provider=provider,
            model=args.model,
            ollama_host=args.ollama_host,
        )
    except (ValueError, ConnectionError) as e:
        print(f"Error: {e}")
        return 2

    print(f"Comparing commits: {args.base}...{args.head}")
    print(f"Changed files: {len(snapshot.changed_files)}")
    print(f"Diff lines: {snapshot.diff_lines}")
    proposed = []
    deduped = []

    if snapshot.diff_lines == 0 or not snapshot.changed_files:
        print("No reviewable diff content after filtering; skipping reviewers.")
    else:
        print(f"Running reviewers: {', '.join(DEFAULT_REVIEWERS)}")
        proposed = run_reviewers(
            reviewers=DEFAULT_REVIEWERS,
            llm_client=llm_client,
            context=context,
            prompts_dir=prompts_dir,
        )
        deduped = deduplicate_findings(proposed)

    allowed_severities = {Severity(level) for level in args.severities}
    accepted, rejected = guard_findings(
        deduped,
        min_confidence=args.min_confidence,
        allowed_severities=allowed_severities,
        max_published=args.max_published,
    )

    result = ReviewResult(
        metadata=ReviewMetadata(
            base=args.base,
            head=args.head,
            changed_files=len(snapshot.changed_files),
            diff_lines=snapshot.diff_lines,
            reviewers=DEFAULT_REVIEWERS,
        ),
        accepted_findings=accepted,
        rejected_findings=rejected,
    )

    print_summary(result, proposed_count=len(proposed))
    write_json(result, Path(args.output))
    return 0
