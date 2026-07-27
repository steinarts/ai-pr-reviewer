from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from .context_builder import build_context
from .deduplicator import deduplicate_findings
from .git_diff import GitDiffError, collect_diff
from .guard import guard_findings
from .llm_factory import create_llm_client
from .models import ReviewMetadata, ReviewResult, Severity
from .output import print_summary, write_json
from .scouts import REVIEW_MODE_CONSOLIDATED, REVIEW_MODE_SEPARATE, ReviewParseError, run_reviewers

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
        "--llm-timeout",
        type=float,
        default=180.0,
        help="LLM request timeout in seconds (default: 180.0)",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=3500,
        help="Approximate max input tokens per LLM request (default: 3500)",
    )
    parser.add_argument(
        "--test-max-prompt-tokens",
        type=int,
        default=2500,
        help="Approximate max input tokens for test chunks (default: 2500)",
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
    parser.add_argument(
        "--llm-debug",
        action="store_true",
        help="Print per-reviewer LLM diagnostics (model, prompt size, latency, errors)",
    )
    parser.add_argument(
        "--llm-debug-log",
        default="",
        help="Optional path to write JSONL debug events from --llm-debug",
    )
    parser.add_argument(
        "--llm-max-output-tokens",
        type=int,
        default=700,
        help="Approximate max output tokens per LLM request (default: 700)",
    )
    parser.add_argument(
        "--max-findings-per-chunk",
        type=int,
        default=3,
        help="Max findings accepted per LLM request before truncation (default: 3)",
    )
    parser.add_argument(
        "--max-review-seconds",
        type=float,
        default=900.0,
        help="Total wall-clock review budget in seconds (default: 900)",
    )
    parser.add_argument(
        "--review-mode",
        choices=[REVIEW_MODE_CONSOLIDATED, REVIEW_MODE_SEPARATE],
        default=REVIEW_MODE_CONSOLIDATED,
        help="Review strategy: consolidated (default) or separate",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_prompt_tokens <= 0:
        print("Error: --max-prompt-tokens must be greater than zero")
        return 2
    if args.llm_max_output_tokens <= 0:
        print("Error: --llm-max-output-tokens must be greater than zero")
        return 2
    if args.test_max_prompt_tokens <= 0:
        print("Error: --test-max-prompt-tokens must be greater than zero")
        return 2
    if args.max_review_seconds <= 0:
        print("Error: --max-review-seconds must be greater than zero")
        return 2
    if args.llm_timeout <= 0:
        print("Error: --llm-timeout must be greater than zero")
        return 2
    if args.max_findings_per_chunk <= 0:
        print("Error: --max-findings-per-chunk must be greater than zero")
        return 2

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
            llm_timeout=args.llm_timeout,
            llm_max_output_tokens=args.llm_max_output_tokens,
        )
    except (ValueError, ConnectionError) as e:
        print(f"Error: {e}")
        return 2

    print(f"Comparing commits: {args.base}...{args.head}")
    print(f"Changed files: {len(snapshot.changed_files)}")
    print(f"Diff lines: {snapshot.diff_lines}")
    proposed = []
    deduped = []
    reviewer_failures = []
    reviewer_skips = []
    completed_requests = 0
    failed_requests = 0
    planned_requests = 0
    skipped_requests = 0
    reviewable_chunks = 0
    skipped_chunks = 0
    chunk_count = 0
    total_elapsed_seconds = 0.0
    total_time_budget_seconds = args.max_review_seconds

    active_reviewers = (
        DEFAULT_REVIEWERS if args.review_mode == REVIEW_MODE_SEPARATE else ["consolidated"]
    )

    debug_log_path = Path(args.llm_debug_log) if args.llm_debug_log else None
    debug_sink: Callable[[dict[str, object]], None] | None = None
    if args.llm_debug:
        if debug_log_path is not None:
            debug_log_path.parent.mkdir(parents=True, exist_ok=True)

        def emit_debug(event: dict[str, object]) -> None:
            line = json.dumps(event, ensure_ascii=False)
            print(f"[llm-debug] {line}")
            if debug_log_path is not None:
                with debug_log_path.open("a", encoding="utf-8") as file:
                    file.write(f"{line}\n")

        debug_sink = emit_debug

    if snapshot.diff_lines == 0 or not snapshot.changed_files:
        print("No reviewable diff content after filtering; skipping reviewers.")
    else:
        print(f"Review mode: {args.review_mode}")
        print(f"Running reviewers: {', '.join(active_reviewers)}")
        try:
            proposed = run_reviewers(
                reviewers=DEFAULT_REVIEWERS,
                llm_client=llm_client,
                context=context,
                prompts_dir=prompts_dir,
                max_prompt_tokens=args.max_prompt_tokens,
                review_mode=args.review_mode,
                llm_timeout_seconds=args.llm_timeout,
                max_review_seconds=args.max_review_seconds,
                test_max_prompt_tokens=args.test_max_prompt_tokens,
                max_findings_per_request=args.max_findings_per_chunk,
                debug_sink=debug_sink,
                progress_sink=print,
            )
            reviewer_failures = proposed.reviewer_failures
            reviewer_skips = proposed.reviewer_skips
            completed_requests = proposed.completed_requests
            failed_requests = proposed.failed_requests
            planned_requests = proposed.planned_requests
            skipped_requests = proposed.skipped_requests
            reviewable_chunks = proposed.reviewable_chunks
            skipped_chunks = proposed.skipped_chunks
            chunk_count = proposed.chunk_count
            total_elapsed_seconds = proposed.total_elapsed_seconds
            total_time_budget_seconds = proposed.total_time_budget_seconds
            proposed = proposed.findings
        except (ReviewParseError, TimeoutError, ConnectionError, ValueError) as exc:
            print(f"LLM review setup error: {exc}")
            return 2
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
            reviewers=active_reviewers,
            review_mode=args.review_mode,
            reviewer_failures=reviewer_failures,
            reviewer_skips=reviewer_skips,
            completed_requests=completed_requests,
            failed_requests=failed_requests,
            planned_requests=planned_requests,
            skipped_requests=skipped_requests,
            reviewable_chunks=reviewable_chunks,
            skipped_chunks=skipped_chunks,
            chunk_count=chunk_count,
            total_elapsed_seconds=total_elapsed_seconds,
            total_time_budget_seconds=total_time_budget_seconds,
        ),
        accepted_findings=accepted,
        rejected_findings=rejected,
    )

    print_summary(result, proposed_count=len(proposed))
    if failed_requests > 0:
        print("Review completed with partial results.")
        print(f"Successful LLM requests: {completed_requests}")
        print(f"Failed LLM requests: {failed_requests}")
        print(f"Skipped LLM requests: {skipped_requests}")
        print(f"Reviewer failures: {len(reviewer_failures)}")
    write_json(result, Path(args.output))
    if completed_requests == 0 and (failed_requests > 0 or skipped_requests > 0):
        return 2
    return 0
