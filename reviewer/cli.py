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
from .verifier import LLMFindingVerifier, VerificationContext

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
    parser.add_argument(
        "--verify-findings",
        action="store_true",
        help="Enable second-step skeptical verification for candidate findings",
    )
    parser.add_argument(
        "--verification-model",
        default="",
        help="Verification model (defaults to review model)",
    )
    parser.add_argument(
        "--verification-timeout-seconds",
        type=float,
        default=60.0,
        help="Per-finding verification timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--verification-total-budget-seconds",
        type=float,
        default=180.0,
        help="Total verification wall-clock budget in seconds (default: 180)",
    )
    parser.add_argument(
        "--verification-max-findings",
        type=int,
        default=5,
        help="Max candidate findings to verify per review (default: 5)",
    )
    parser.add_argument(
        "--verification-min-confidence",
        type=float,
        default=0.8,
        help="Minimum verifier confidence required for valid verdicts (default: 0.8)",
    )
    parser.add_argument(
        "--verification-fail-policy",
        choices=["unverified", "reject"],
        default="unverified",
        help="Behavior on verifier timeout/parse failure (default: unverified)",
    )
    parser.add_argument(
        "--verification-uncertain-policy",
        choices=["unverified", "reject"],
        default="unverified",
        help="Behavior on uncertain verifier verdicts (default: unverified)",
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
    if args.verification_timeout_seconds <= 0:
        print("Error: --verification-timeout-seconds must be greater than zero")
        return 2
    if args.verification_total_budget_seconds <= 0:
        print("Error: --verification-total-budget-seconds must be greater than zero")
        return 2
    if args.verification_max_findings <= 0:
        print("Error: --verification-max-findings must be greater than zero")
        return 2
    if args.verification_min_confidence < 0.0 or args.verification_min_confidence > 1.0:
        print("Error: --verification-min-confidence must be between 0.0 and 1.0")
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
    candidate_findings = []
    verified_findings = []
    verification_rejected_findings = []
    verification_requests_completed = 0
    verification_requests_failed = 0
    verification_requests_skipped = 0
    verification_valid_count = 0
    verification_invalid_count = 0
    verification_unverified_count = 0
    verification_uncertain_count = 0
    verification_skipped_count = 0
    verification_elapsed_seconds = 0.0

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
    candidate_findings = deduped
    verified_findings = candidate_findings

    verification_model = args.verification_model or str(
        getattr(llm_client, "model", "") or args.model
    )
    if args.verify_findings and candidate_findings:
        verification_client = llm_client
        if (
            provider == "ollama"
            and verification_model
            and verification_model != str(getattr(llm_client, "model", ""))
        ):
            try:
                verification_client = create_llm_client(
                    provider=provider,
                    model=verification_model,
                    ollama_host=args.ollama_host,
                    llm_timeout=args.verification_timeout_seconds,
                    llm_max_output_tokens=args.llm_max_output_tokens,
                )
            except (ValueError, ConnectionError) as exc:
                print(f"Verification setup error: {exc}")
                return 2

        verifier = LLMFindingVerifier(verification_client, debug_sink=debug_sink)
        changed_lines_by_file = {
            diff_file.path.as_posix(): set(diff_file.changed_lines)
            for diff_file in snapshot.changed_files
        }
        verification_context = VerificationContext(
            base=args.base,
            head=args.head,
            diff_text=context.diff_text,
            file_contexts=context.file_contexts,
            changed_lines_by_file=changed_lines_by_file,
            provider=provider,
            review_model=str(getattr(llm_client, "model", "") or args.model),
            verification_model=verification_model,
            timeout_seconds=args.verification_timeout_seconds,
            total_budget_seconds=args.verification_total_budget_seconds,
            max_findings=args.verification_max_findings,
            min_confidence=args.verification_min_confidence,
            fail_policy=args.verification_fail_policy,
            uncertain_policy=args.verification_uncertain_policy,
        )
        verification_result = verifier.verify(candidate_findings, verification_context)
        verified_findings = verification_result.verified_findings
        verification_rejected_findings = verification_result.verification_rejected_findings
        verification_requests_completed = verification_result.completed_requests
        verification_requests_failed = verification_result.failed_requests
        verification_requests_skipped = verification_result.skipped_requests
        verification_valid_count = verification_result.valid_count
        verification_invalid_count = verification_result.invalid_count
        verification_unverified_count = verification_result.unverified_count
        verification_uncertain_count = verification_result.uncertain_count
        verification_skipped_count = verification_result.skipped_count
        verification_elapsed_seconds = verification_result.elapsed_seconds

    accepted, rejected = guard_findings(
        verified_findings,
        min_confidence=args.min_confidence,
        allowed_severities=allowed_severities,
        max_published=args.max_published,
    )
    rejected = [*verification_rejected_findings, *rejected]

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
            verification_enabled=args.verify_findings,
            verification_model=verification_model,
            verification_fail_policy=args.verification_fail_policy,
            verification_uncertain_policy=args.verification_uncertain_policy,
            verification_requests_planned=min(
                len(candidate_findings), args.verification_max_findings
            )
            if args.verify_findings
            else 0,
            verification_requests_completed=verification_requests_completed,
            verification_requests_failed=verification_requests_failed,
            verification_requests_skipped=verification_requests_skipped,
            verification_valid_count=verification_valid_count,
            verification_invalid_count=verification_invalid_count,
            verification_uncertain_count=verification_uncertain_count,
            verification_unverified_count=verification_unverified_count,
            verification_skipped_count=verification_skipped_count,
            verification_elapsed_seconds=verification_elapsed_seconds,
        ),
        candidate_findings=candidate_findings,
        verified_findings=verified_findings,
        verification_rejected_findings=verification_rejected_findings,
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
