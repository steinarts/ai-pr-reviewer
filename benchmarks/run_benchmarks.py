from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, sleep

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarks.schema import (
    CandidateCaseRecord,
    CandidateDataset,
    CandidateRunConfig,
    CaseGroundTruth,
    CasePaths,
    load_candidate_dataset,
    load_ground_truth,
)
from benchmarks.scoring import PARSE_ERROR_TYPES, safe_divide, score_case
from reviewer.context_builder import build_context
from reviewer.deduplicator import deduplicate_findings
from reviewer.git_diff import GitDiffError, collect_diff
from reviewer.guard import guard_findings
from reviewer.llm_factory import create_llm_client
from reviewer.models import Finding, ReviewMetadata, ReviewResult, Severity
from reviewer.scouts import REVIEW_MODE_CONSOLIDATED, REVIEW_MODE_SEPARATE, run_reviewers
from reviewer.verifier import LLMFindingVerifier, VerificationContext

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT / "benchmarks"
CASES_DIR = BENCHMARKS_DIR / "cases"
DEFAULT_OUTPUT = BENCHMARKS_DIR / "results" / "benchmark-result.json"


@dataclass(slots=True)
class CaseExecution:
    case_id: str
    ground_truth: CaseGroundTruth
    success: bool
    error: str
    elapsed_seconds: float
    result: ReviewResult | None
    candidate_findings: list[Finding] = field(default_factory=list)
    verified_findings: list[Finding] = field(default_factory=list)
    verification_rejected_findings: list[Finding] = field(default_factory=list)
    scored_findings: list[Finding] = field(default_factory=list)
    accepted_findings: list[Finding] = field(default_factory=list)
    rejected_findings: list[Finding] = field(default_factory=list)
    raw_events: list[dict[str, object]] = field(default_factory=list)
    source_identifier: str = ""


@dataclass(slots=True)
class AggregateMetrics:
    cases: int
    clean_cases: int
    expected_defects: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    clean_case_false_positive_rate: float
    valid_file_and_line_rate: float
    parse_success_rate: float
    elapsed_seconds: float
    correct_findings_per_minute: float


def _run_git(args: list[str], cwd: Path) -> None:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "git command failed")


def _copy_tree(src: Path, dst: Path) -> None:
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _sync_worktree(repo: Path, desired: Path) -> None:
    existing_paths = {
        p.relative_to(repo).as_posix() for p in repo.rglob("*") if ".git" not in p.parts
    }
    desired_paths = {p.relative_to(desired).as_posix() for p in desired.rglob("*")}

    for rel in sorted(existing_paths - desired_paths, reverse=True):
        target = repo / rel
        if target.is_file():
            target.unlink()

    for rel in sorted(existing_paths - desired_paths, reverse=True):
        target = repo / rel
        if target.is_dir():
            try:
                target.rmdir()
            except OSError:
                pass

    _copy_tree(desired, repo)


def _materialize_case_repo(case: CasePaths, keep_temp: bool) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix=f"bench-{case.case_id}-"))
    repo = temp_root / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    _copy_tree(case.base_dir, repo)
    _run_git(["init", "-b", "main"], repo)
    _run_git(["config", "user.email", "bench@example.com"], repo)
    _run_git(["config", "user.name", "Benchmark Runner"], repo)
    _run_git(["add", "."], repo)
    _run_git(["commit", "--allow-empty", "-m", "base"], repo)
    _run_git(["tag", "base"], repo)

    _sync_worktree(repo, case.head_dir)
    _run_git(["add", "-A"], repo)
    _run_git(["commit", "--allow-empty", "-m", "head"], repo)
    _run_git(["tag", "head"], repo)

    if keep_temp:
        print(f"Keeping temp repo for {case.case_id}: {repo}")
    return repo


def _cleanup_repo(repo: Path, keep_temp: bool) -> None:
    if keep_temp:
        return

    target = repo.parent

    def onerror(func, path, exc_info):  # type: ignore[no-untyped-def]
        _ = exc_info
        os.chmod(path, stat.S_IWRITE)
        func(path)

    for _ in range(10):
        if not target.exists():
            return
        try:
            shutil.rmtree(target, onerror=onerror)
            return
        except OSError:
            sleep(0.05)


def _collect_cases(selected_case: str | None) -> list[CasePaths]:
    cases: list[CasePaths] = []
    for case_dir in sorted(CASES_DIR.iterdir()):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name
        if selected_case and case_id != selected_case:
            continue
        base_dir = case_dir / "base"
        head_dir = case_dir / "head"
        expected_file = case_dir / "expected.json"
        if not base_dir.exists() or not head_dir.exists() or not expected_file.exists():
            continue
        cases.append(
            CasePaths(
                case_id=case_id,
                case_dir=case_dir,
                base_dir=base_dir,
                head_dir=head_dir,
                expected_file=expected_file,
            )
        )
    return cases


def _build_result(
    metadata: ReviewMetadata,
    candidate_findings: list[Finding],
    verified_findings: list[Finding],
    verification_rejected_findings: list[Finding],
) -> ReviewResult:
    accepted, rejected = guard_findings(
        verified_findings,
        min_confidence=0.85,
        allowed_severities={Severity.HIGH, Severity.CRITICAL},
        max_published=3,
    )
    return ReviewResult(
        metadata=metadata,
        candidate_findings=candidate_findings,
        verified_findings=verified_findings,
        verification_rejected_findings=verification_rejected_findings,
        accepted_findings=accepted,
        rejected_findings=[*verification_rejected_findings, *rejected],
    )


def _build_source_identifier(diff_text: str, *, base: str, head: str) -> str:
    digest = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
    return f"{base}..{head}:{digest[:16]}"


def _finding_key(finding: Finding) -> tuple[str, str, int, str, str]:
    return (
        finding.id,
        finding.file,
        finding.line,
        finding.category,
        finding.title,
    )


def _actual_verifier_verdict(execution: CaseExecution) -> str | None:
    if not execution.candidate_findings:
        return None

    first_candidate = execution.candidate_findings[0]
    key = _finding_key(first_candidate)
    for finding in execution.verification_rejected_findings:
        if _finding_key(finding) == key:
            return finding.verification_verdict or "invalid"
    for finding in execution.verified_findings:
        if _finding_key(finding) == key:
            return finding.verification_verdict or None
    return None


def _compute_delta(
    before: dict[str, float | int],
    after: dict[str, float | int],
    *,
    runtime_overhead_seconds: float,
) -> dict[str, float | int]:
    before_tp = int(before["true_positives"])
    before_fp = int(before["false_positives"])
    before_fn = int(before["false_negatives"])
    after_tp = int(after["true_positives"])
    after_fp = int(after["false_positives"])
    after_fn = int(after["false_negatives"])
    before_precision = float(before["precision"])
    before_recall = float(before["recall"])
    after_precision = float(after["precision"])
    after_recall = float(after["recall"])

    return {
        "tp_removed": before_tp - after_tp,
        "fp_removed": before_fp - after_fp,
        "new_fn": after_fn - before_fn,
        "precision_change": after_precision - before_precision,
        "recall_change": after_recall - before_recall,
        "runtime_overhead_seconds": runtime_overhead_seconds,
    }


def _build_per_finding_effects(execution: CaseExecution) -> list[dict[str, object]]:
    if not execution.success:
        return []

    candidate_score = score_case(execution.ground_truth, execution.candidate_findings)
    candidate_tp_indexes = {finding_index for _, finding_index in candidate_score.matched_pairs}

    verified_keys = {_finding_key(item) for item in execution.verified_findings}
    accepted_keys = {_finding_key(item) for item in execution.accepted_findings}
    rejected_by_key = {_finding_key(item): item for item in execution.rejected_findings}
    verified_by_key = {_finding_key(item): item for item in execution.verified_findings}

    effects: list[dict[str, object]] = []
    for index, candidate in enumerate(execution.candidate_findings):
        key = _finding_key(candidate)
        candidate_classification = "TP" if index in candidate_tp_indexes else "FP"
        seen_after_verification = key in verified_keys

        resolved = verified_by_key.get(key) or rejected_by_key.get(key) or candidate
        rejection_code = ""
        if key in rejected_by_key:
            rejection_code = rejected_by_key[key].rejection_reason

        if key in accepted_keys:
            final_status = "published"
        elif key in rejected_by_key and rejection_code.startswith("verification"):
            final_status = "verification_rejected"
        elif key in rejected_by_key:
            final_status = "guard_rejected"
        elif seen_after_verification:
            final_status = "verified_not_published"
        else:
            final_status = "dropped_after_verification"

        if candidate_classification == "FP" and not seen_after_verification:
            effect = "improved"
        elif candidate_classification == "TP" and not seen_after_verification:
            effect = "harmed"
        else:
            effect = "neutral"

        effects.append(
            {
                "case_id": execution.case_id,
                "finding_id": candidate.id,
                "file": candidate.file,
                "line": candidate.line,
                "candidate_classification": candidate_classification,
                "verifier_verdict": resolved.verification_verdict,
                "confidence": resolved.verification_confidence,
                "rejection_code": rejection_code,
                "final_status": final_status,
                "verification_effect": effect,
            }
        )

    return effects


def _build_candidate_dataset(
    executions: list[CaseExecution],
    *,
    provider: str,
    model: str,
    review_mode: str,
    llm_timeout: float,
    max_review_seconds: float,
    reviewer_prompt_fingerprints: dict[str, str],
    deterministic_seed: int | None,
    sampling_params: dict[str, object],
    run_id: str,
    timestamp_utc: str,
) -> CandidateDataset:
    run_config = CandidateRunConfig(
        provider=provider,
        model=model,
        review_mode=review_mode,
        llm_timeout=llm_timeout,
        max_review_seconds=max_review_seconds,
        reviewers=["bug", "reliability", "security"],
        reviewer_prompt_fingerprints=reviewer_prompt_fingerprints,
        deterministic_seed=deterministic_seed,
        sampling_params=sampling_params,
    )

    case_records = [
        CandidateCaseRecord(
            case_id=execution.case_id,
            source_identifier=execution.source_identifier,
            expected=execution.ground_truth,
            candidate_findings=execution.candidate_findings,
        )
        for execution in executions
    ]

    return CandidateDataset(
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        run_config=run_config,
        cases=case_records,
    )


def _sample_config_for_logging(
    *,
    provider: str,
    sampling_seed: int | None,
    sampling_temperature: float,
    sampling_top_p: float | None,
) -> dict[str, object]:
    supported = provider == "ollama"
    return {
        "provider_supports_seed": supported,
        "seed": sampling_seed,
        "temperature": sampling_temperature,
        "top_p": sampling_top_p,
        "seed_applied": supported and sampling_seed is not None,
    }


def _prompt_fingerprints(prompts_dir: Path) -> dict[str, str]:
    reviewers = ["bug", "reliability", "security", "consolidated", "guard"]
    fingerprints: dict[str, str] = {}
    for reviewer in reviewers:
        path = prompts_dir / f"{reviewer}_reviewer.md"
        if reviewer == "guard":
            path = prompts_dir / "guard.md"
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        fingerprints[reviewer] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return fingerprints


def _run_case(
    case: CasePaths,
    *,
    provider: str,
    model: str,
    review_mode: str,
    llm_timeout: float,
    max_review_seconds: float,
    verify_findings: bool,
    verification_model: str,
    verification_timeout_seconds: float,
    verification_total_budget_seconds: float,
    verification_max_findings: int,
    verification_min_confidence: float,
    verification_fail_policy: str,
    verification_uncertain_policy: str,
    sampling_seed: int | None,
    sampling_temperature: float,
    sampling_top_p: float | None,
    keep_temp: bool,
    verbose: bool,
    replay_candidate_findings: list[Finding] | None = None,
    replay_source_identifier: str = "",
    replay_review_model: str = "",
) -> CaseExecution:
    ground_truth = load_ground_truth(case.expected_file)
    start = perf_counter()
    repo: Path | None = None

    try:
        repo = _materialize_case_repo(case, keep_temp=keep_temp)
        events: list[dict[str, object]] = []
        snapshot = collect_diff(
            base="base",
            head="head",
            cwd=repo,
            max_files=100,
            max_diff_lines=20000,
        )
        context = build_context(snapshot=snapshot, head="head", cwd=repo)
        source_identifier = _build_source_identifier(snapshot.diff_text, base="base", head="head")

        llm_client = None
        runner_result = None
        if replay_candidate_findings is None:
            llm_client = create_llm_client(
                provider=provider,
                model=model if provider != "fake" else None,
                llm_timeout=llm_timeout,
                deterministic_seed=sampling_seed,
                temperature=sampling_temperature,
                top_p=sampling_top_p,
            )

            runner_result = run_reviewers(
                reviewers=["bug", "reliability", "security"],
                llm_client=llm_client,
                context=context,
                prompts_dir=ROOT / "prompts",
                max_prompt_tokens=3500,
                review_mode=review_mode,
                llm_timeout_seconds=llm_timeout,
                max_review_seconds=max_review_seconds,
                test_max_prompt_tokens=2500,
                max_findings_per_request=3,
                debug_sink=lambda event: events.append(event),
                progress_sink=print if verbose else None,
            )

        if runner_result is None:
            candidate_findings = deduplicate_findings(
                [item.model_copy(deep=True) for item in (replay_candidate_findings or [])]
            )
            review_model_name = replay_review_model or model
            events.append(
                {
                    "event": "candidate_replay_loaded",
                    "case_id": case.case_id,
                    "candidate_count": len(candidate_findings),
                    "source_identifier": replay_source_identifier,
                }
            )
            metadata = ReviewMetadata(
                base="base",
                head="head",
                changed_files=len(snapshot.changed_files),
                diff_lines=snapshot.diff_lines,
                reviewers=["candidate_replay"],
                review_mode=review_mode,
                completed_requests=0,
                failed_requests=0,
                planned_requests=0,
                skipped_requests=0,
                reviewable_chunks=0,
                skipped_chunks=0,
                chunk_count=0,
                total_elapsed_seconds=0.0,
                total_time_budget_seconds=max_review_seconds,
            )
        else:
            metadata = ReviewMetadata(
                base="base",
                head="head",
                changed_files=len(snapshot.changed_files),
                diff_lines=snapshot.diff_lines,
                reviewers=["consolidated"]
                if review_mode == REVIEW_MODE_CONSOLIDATED
                else ["bug", "reliability", "security"],
                review_mode=review_mode,
                reviewer_failures=runner_result.reviewer_failures,
                reviewer_skips=runner_result.reviewer_skips,
                completed_requests=runner_result.completed_requests,
                failed_requests=runner_result.failed_requests,
                planned_requests=runner_result.planned_requests,
                skipped_requests=runner_result.skipped_requests,
                reviewable_chunks=runner_result.reviewable_chunks,
                skipped_chunks=runner_result.skipped_chunks,
                chunk_count=runner_result.chunk_count,
                total_elapsed_seconds=runner_result.total_elapsed_seconds,
                total_time_budget_seconds=runner_result.total_time_budget_seconds,
            )
            candidate_findings = deduplicate_findings(runner_result.findings)
            review_model_name = str(getattr(llm_client, "model", "") or model)
        verified_findings = candidate_findings
        verification_rejected_findings: list[Finding] = []
        verifier_result = None

        if verify_findings and candidate_findings:
            if llm_client is None:
                llm_client = create_llm_client(
                    provider=provider,
                    model=model if provider != "fake" else None,
                    llm_timeout=llm_timeout,
                    deterministic_seed=sampling_seed,
                    temperature=sampling_temperature,
                    top_p=sampling_top_p,
                )

            verification_client = llm_client
            selected_verification_model = verification_model or review_model_name
            if (
                provider == "ollama"
                and selected_verification_model
                and selected_verification_model != str(getattr(llm_client, "model", ""))
            ):
                verification_client = create_llm_client(
                    provider=provider,
                    model=selected_verification_model,
                    llm_timeout=verification_timeout_seconds,
                    deterministic_seed=sampling_seed,
                    temperature=sampling_temperature,
                    top_p=sampling_top_p,
                )

            changed_lines_by_file = {
                diff_file.path.as_posix(): set(diff_file.changed_lines)
                for diff_file in snapshot.changed_files
            }
            verifier = LLMFindingVerifier(
                verification_client, debug_sink=lambda event: events.append(event)
            )
            verifier_context = VerificationContext(
                base="base",
                head="head",
                diff_text=context.diff_text,
                file_contexts=context.file_contexts,
                changed_lines_by_file=changed_lines_by_file,
                provider=provider,
                review_model=review_model_name,
                verification_model=selected_verification_model,
                timeout_seconds=verification_timeout_seconds,
                total_budget_seconds=verification_total_budget_seconds,
                max_findings=verification_max_findings,
                min_confidence=verification_min_confidence,
                fail_policy=verification_fail_policy,
                uncertain_policy=verification_uncertain_policy,
            )
            verifier_result = verifier.verify(candidate_findings, verifier_context)
            verified_findings = verifier_result.verified_findings
            verification_rejected_findings = verifier_result.verification_rejected_findings

            metadata.verification_enabled = True
            metadata.verification_model = selected_verification_model
            metadata.verification_fail_policy = verification_fail_policy
            metadata.verification_uncertain_policy = verification_uncertain_policy
            metadata.verification_requests_planned = min(
                len(candidate_findings),
                verification_max_findings,
            )
            metadata.verification_requests_completed = verifier_result.completed_requests
            metadata.verification_requests_failed = verifier_result.failed_requests
            metadata.verification_requests_skipped = verifier_result.skipped_requests
            metadata.verification_valid_count = verifier_result.valid_count
            metadata.verification_invalid_count = verifier_result.invalid_count
            metadata.verification_uncertain_count = verifier_result.uncertain_count
            metadata.verification_unverified_count = verifier_result.unverified_count
            metadata.verification_skipped_count = verifier_result.skipped_count
            metadata.verification_elapsed_seconds = verifier_result.elapsed_seconds
        else:
            metadata.verification_enabled = verify_findings
            metadata.verification_model = verification_model or str(review_model_name or model)
            metadata.verification_fail_policy = verification_fail_policy
            metadata.verification_uncertain_policy = verification_uncertain_policy

        result = _build_result(
            metadata=metadata,
            candidate_findings=candidate_findings,
            verified_findings=verified_findings,
            verification_rejected_findings=verification_rejected_findings,
        )
        elapsed = perf_counter() - start
        return CaseExecution(
            case_id=case.case_id,
            ground_truth=ground_truth,
            success=True,
            error="",
            elapsed_seconds=elapsed,
            result=result,
            candidate_findings=candidate_findings,
            verified_findings=verified_findings,
            verification_rejected_findings=verification_rejected_findings,
            scored_findings=candidate_findings,
            accepted_findings=result.accepted_findings,
            rejected_findings=result.rejected_findings,
            raw_events=events,
            source_identifier=source_identifier,
        )
    except (RuntimeError, GitDiffError, ValueError, ConnectionError, TimeoutError) as exc:
        elapsed = perf_counter() - start
        return CaseExecution(
            case_id=case.case_id,
            ground_truth=ground_truth,
            success=False,
            error=f"{exc.__class__.__name__}: {exc}",
            elapsed_seconds=elapsed,
            result=None,
            candidate_findings=[],
            verified_findings=[],
            verification_rejected_findings=[],
            scored_findings=[],
            accepted_findings=[],
            rejected_findings=[],
            raw_events=[],
            source_identifier="",
        )
    finally:
        if repo is not None:
            _cleanup_repo(repo, keep_temp=keep_temp)


def _aggregate(executions: list[CaseExecution]) -> AggregateMetrics:
    case_scores = [
        score_case(item.ground_truth, item.scored_findings) for item in executions if item.success
    ]

    tp = sum(item.true_positives for item in case_scores)
    fp = sum(item.false_positives for item in case_scores)
    fn = sum(item.false_negatives for item in case_scores)

    clean_scores = [item for item in case_scores if item.clean]
    clean_cases_with_fp = sum(1 for item in clean_scores if item.false_positives > 0)

    total_expected = sum(len(item.ground_truth.expected_findings) for item in executions)
    total_elapsed = sum(item.elapsed_seconds for item in executions)

    parse_failures = 0
    planned_requests = 0
    total_raw_findings = 0
    invalid_location_rejections = 0
    for execution in executions:
        if not execution.result:
            continue
        planned_requests += execution.result.metadata.planned_requests
        parse_failures += sum(
            1
            for failure in execution.result.metadata.reviewer_failures
            if failure.error_type in PARSE_ERROR_TYPES
        )
        for event in execution.raw_events:
            if event.get("event") == "reviewer_complete":
                total_raw_findings += int(event.get("raw_findings_count", 0))
            if event.get("event") == "reviewer_rejections":
                counts = event.get("rejection_counts", {})
                if isinstance(counts, dict):
                    invalid_location_rejections += int(counts.get("file_not_in_chunk", 0))
                    invalid_location_rejections += int(counts.get("line_not_changed", 0))

    valid_location_rate = safe_divide(
        max(0, total_raw_findings - invalid_location_rejections),
        total_raw_findings,
    )
    parse_success_rate = safe_divide(max(0, planned_requests - parse_failures), planned_requests)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return AggregateMetrics(
        cases=len(executions),
        clean_cases=len(clean_scores),
        expected_defects=total_expected,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        clean_case_false_positive_rate=safe_divide(clean_cases_with_fp, len(clean_scores)),
        valid_file_and_line_rate=valid_location_rate,
        parse_success_rate=parse_success_rate,
        elapsed_seconds=total_elapsed,
        correct_findings_per_minute=safe_divide(tp, safe_divide(total_elapsed, 60.0)),
    )


def _aggregate_for_level(
    executions: list[CaseExecution],
    *,
    level: str,
) -> dict[str, float | int]:
    selector = {
        "candidate": lambda item: item.candidate_findings,
        "verified": lambda item: item.verified_findings,
        "published": lambda item: item.accepted_findings,
    }[level]

    case_scores = [
        score_case(item.ground_truth, selector(item)) for item in executions if item.success
    ]
    tp = sum(item.true_positives for item in case_scores)
    fp = sum(item.false_positives for item in case_scores)
    fn = sum(item.false_negatives for item in case_scores)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _serialize_finding(finding: Finding) -> dict[str, object]:
    return finding.model_dump(mode="json")


def _print_summary(model: str, metrics: AggregateMetrics) -> None:
    print(f"Model: {model}")
    print(f"Cases: {metrics.cases}")
    print(f"Clean cases: {metrics.clean_cases}")
    print(f"Expected defects: {metrics.expected_defects}")
    print(f"True positives: {metrics.true_positives}")
    print(f"False positives: {metrics.false_positives}")
    print(f"False negatives: {metrics.false_negatives}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall: {metrics.recall:.4f}")
    print(f"F1: {metrics.f1:.4f}")
    print(f"Clean-case false-positive rate: {metrics.clean_case_false_positive_rate:.4f}")
    print(f"Valid location rate: {metrics.valid_file_and_line_rate:.4f}")
    print(f"Parse success rate: {metrics.parse_success_rate:.4f}")
    print(f"Elapsed seconds: {metrics.elapsed_seconds:.3f}")
    print(f"Correct findings per minute: {metrics.correct_findings_per_minute:.4f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic benchmark suite")
    parser.add_argument("--case", default="", help="Run only one case by case_id")
    parser.add_argument("--provider", choices=["fake", "ollama"], default="fake")
    parser.add_argument("--model", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--review-mode",
        choices=[REVIEW_MODE_CONSOLIDATED, REVIEW_MODE_SEPARATE],
        default=REVIEW_MODE_CONSOLIDATED,
    )
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--max-review-seconds", type=float, default=900.0)
    parser.add_argument("--verify-findings", action="store_true")
    parser.add_argument("--verification-model", default="")
    parser.add_argument("--verification-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--verification-total-budget-seconds", type=float, default=180.0)
    parser.add_argument("--verification-max-findings", type=int, default=5)
    parser.add_argument("--verification-min-confidence", type=float, default=0.8)
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument("--sampling-temperature", type=float, default=0.0)
    parser.add_argument("--sampling-top-p", type=float, default=None)
    parser.add_argument("--candidate-findings-output", default="")
    parser.add_argument("--candidate-findings-input", default="")
    parser.add_argument(
        "--verification-fail-policy",
        choices=["unverified", "reject"],
        default="unverified",
    )
    parser.add_argument(
        "--verification-uncertain-policy",
        choices=["unverified", "reject"],
        default="unverified",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    verification_only_mode = bool(args.candidate_findings_input)

    candidate_dataset = None
    candidate_cases_by_id: dict[str, CandidateCaseRecord] = {}
    if verification_only_mode:
        dataset_path = Path(args.candidate_findings_input)
        if not dataset_path.exists():
            print(f"Error: candidate dataset not found: {dataset_path}")
            return 2
        candidate_dataset = load_candidate_dataset(dataset_path)
        candidate_cases_by_id = {item.case_id: item for item in candidate_dataset.cases}
        if not args.model:
            args.model = candidate_dataset.run_config.model

    if args.provider == "ollama" and not (args.model or args.verification_model):
        print("Error: --model is required for --provider ollama")
        return 2

    cases = _collect_cases(args.case or None)
    if not cases:
        print("No benchmark cases found.")
        return 2

    executions: list[CaseExecution] = []
    verify_findings = args.verify_findings or verification_only_mode
    for case in cases:
        if args.verbose:
            print(f"Running case: {case.case_id}")

        replay_findings = None
        replay_source_identifier = ""
        replay_review_model = ""
        if verification_only_mode:
            replay_case = candidate_cases_by_id.get(case.case_id)
            if replay_case is None:
                execution = CaseExecution(
                    case_id=case.case_id,
                    ground_truth=load_ground_truth(case.expected_file),
                    success=False,
                    error="Candidate dataset missing this case_id",
                    elapsed_seconds=0.0,
                    result=None,
                )
                executions.append(execution)
                continue
            replay_findings = replay_case.candidate_findings
            replay_source_identifier = replay_case.source_identifier
            replay_review_model = candidate_dataset.run_config.model if candidate_dataset else ""

        execution = _run_case(
            case,
            provider=args.provider,
            model=args.model,
            review_mode=args.review_mode,
            llm_timeout=args.llm_timeout,
            max_review_seconds=args.max_review_seconds,
            verify_findings=verify_findings,
            verification_model=args.verification_model,
            verification_timeout_seconds=args.verification_timeout_seconds,
            verification_total_budget_seconds=args.verification_total_budget_seconds,
            verification_max_findings=args.verification_max_findings,
            verification_min_confidence=args.verification_min_confidence,
            verification_fail_policy=args.verification_fail_policy,
            verification_uncertain_policy=args.verification_uncertain_policy,
            sampling_seed=args.sampling_seed,
            sampling_temperature=args.sampling_temperature,
            sampling_top_p=args.sampling_top_p,
            keep_temp=args.keep_temp,
            verbose=args.verbose,
            replay_candidate_findings=replay_findings,
            replay_source_identifier=replay_source_identifier,
            replay_review_model=replay_review_model,
        )
        executions.append(execution)
        if args.verbose and not execution.success:
            print(f"Case failed: {case.case_id} -> {execution.error}")

    metrics = _aggregate(executions)
    candidate_metrics = _aggregate_for_level(executions, level="candidate")
    verified_metrics = _aggregate_for_level(executions, level="verified")
    published_metrics = _aggregate_for_level(executions, level="published")
    verification_runtime_overhead = sum(
        execution.result.metadata.verification_elapsed_seconds
        for execution in executions
        if execution.result is not None
    )
    delta_after_verification = _compute_delta(
        candidate_metrics,
        verified_metrics,
        runtime_overhead_seconds=verification_runtime_overhead,
    )
    delta_after_published = _compute_delta(
        candidate_metrics,
        published_metrics,
        runtime_overhead_seconds=verification_runtime_overhead,
    )

    case_payloads = []
    per_finding_effects: list[dict[str, object]] = []
    for execution in executions:
        case_score = score_case(execution.ground_truth, execution.scored_findings)
        per_finding_effects.extend(_build_per_finding_effects(execution))
        replay_case = (
            candidate_cases_by_id.get(execution.case_id) if verification_only_mode else None
        )
        actual_verifier_verdict = _actual_verifier_verdict(execution)
        case_payloads.append(
            {
                "case_id": execution.case_id,
                "clean": execution.ground_truth.clean,
                "success": execution.success,
                "error": execution.error,
                "elapsed_seconds": execution.elapsed_seconds,
                "source_identifier": execution.source_identifier,
                "replay_source": replay_case.source.model_dump(mode="json")
                if replay_case and replay_case.source
                else None,
                "replay_expected_verdict": replay_case.expected_verdict if replay_case else None,
                "replay_actual_verifier_verdict": actual_verifier_verdict,
                "replay_verdict_matches_expectation": (
                    actual_verifier_verdict == replay_case.expected_verdict
                    if replay_case and replay_case.expected_verdict and actual_verifier_verdict
                    else None
                ),
                "expected": execution.ground_truth.model_dump(mode="json"),
                "candidate_findings": [
                    _serialize_finding(item) for item in execution.candidate_findings
                ],
                "verified_findings": [
                    _serialize_finding(item) for item in execution.verified_findings
                ],
                "verification_rejected_findings": [
                    _serialize_finding(item) for item in execution.verification_rejected_findings
                ],
                "accepted_findings": [
                    _serialize_finding(item) for item in execution.accepted_findings
                ],
                "scored_findings": [_serialize_finding(item) for item in execution.scored_findings],
                "rejected_findings": [
                    _serialize_finding(item) for item in execution.rejected_findings
                ],
                "metadata": execution.result.metadata.model_dump(mode="json")
                if execution.result
                else {},
                "debug_events": execution.raw_events,
                "score": {
                    **asdict(case_score),
                    "forbidden_hallucinations_count": len(case_score.forbidden_hallucinations),
                }
                if execution.success
                else {},
            }
        )

    stage_metrics = {
        "before_verification": candidate_metrics,
        "after_verification": verified_metrics,
        "after_publishing": published_metrics,
        "delta_after_verification": delta_after_verification,
        "delta_after_publishing": delta_after_published,
    }

    payload = {
        "model": args.model,
        "provider": args.provider,
        "review_mode": args.review_mode,
        "benchmark_mode": "verification_only" if verification_only_mode else "end_to_end",
        "run_id": args.run_id
        or (
            candidate_dataset.run_id
            if candidate_dataset
            else datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        ),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "metrics": asdict(metrics),
        "end_to_end_metrics": None if verification_only_mode else stage_metrics,
        "verification_only_metrics": stage_metrics if verification_only_mode else None,
        "metrics_presentation": {
            "candidate_model_metrics": "metrics_by_stage.candidate",
            "verified_metrics": "metrics_by_stage.verified",
            "published_metrics": "metrics_by_stage.published",
            "note": "Do not interpret candidate recall as published recall.",
        },
        "metrics_by_stage": {
            "candidate": candidate_metrics,
            "verified": verified_metrics,
            "published": published_metrics,
        },
        "verification_delta": delta_after_verification,
        "candidate_dataset": {
            "input_path": args.candidate_findings_input or None,
            "input_run_id": candidate_dataset.run_id if candidate_dataset else None,
        },
        "sampling": _sample_config_for_logging(
            provider=args.provider,
            sampling_seed=args.sampling_seed,
            sampling_temperature=args.sampling_temperature,
            sampling_top_p=args.sampling_top_p,
        ),
        "quality_gate_recommendations": {
            "precision_min": 0.80,
            "clean_case_false_positive_rate_max": 0.10,
            "valid_location_rate_min": 0.95,
            "parse_success_rate_min": 0.99,
        },
        "per_finding_effects": per_finding_effects,
        "cases": case_payloads,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if args.candidate_findings_output:
        run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        timestamp_utc = datetime.now(UTC).isoformat()
        sampling_params = _sample_config_for_logging(
            provider=args.provider,
            sampling_seed=args.sampling_seed,
            sampling_temperature=args.sampling_temperature,
            sampling_top_p=args.sampling_top_p,
        )
        candidate_dataset_payload = _build_candidate_dataset(
            executions,
            provider=args.provider,
            model=args.model,
            review_mode=args.review_mode,
            llm_timeout=args.llm_timeout,
            max_review_seconds=args.max_review_seconds,
            reviewer_prompt_fingerprints=_prompt_fingerprints(ROOT / "prompts"),
            deterministic_seed=args.sampling_seed,
            sampling_params=sampling_params,
            run_id=run_id,
            timestamp_utc=timestamp_utc,
        )
        candidate_output = Path(args.candidate_findings_output)
        candidate_output.parent.mkdir(parents=True, exist_ok=True)
        candidate_output.write_text(
            json.dumps(
                candidate_dataset_payload.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"Candidate findings written to: {candidate_output}")

    _print_summary(args.model or "fake", metrics)
    print(f"Result written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
