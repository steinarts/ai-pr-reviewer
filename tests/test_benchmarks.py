from __future__ import annotations

import json
import subprocess
from pathlib import Path

from benchmarks.run_benchmarks import (
    AggregateMetrics,
    CaseExecution,
    _aggregate,
    _cleanup_repo,
    _materialize_case_repo,
    run,
)
from benchmarks.schema import CaseGroundTruth, CasePaths, ExpectedFinding, load_ground_truth
from benchmarks.scoring import (
    detect_forbidden_concepts,
    finding_matches_expected,
    normalize_concept_keywords,
    safe_divide,
    score_case,
)
from reviewer.models import Finding, ReviewMetadata, ReviewResult, Severity, Status
from reviewer.verifier import VerificationResult


def _finding(
    *,
    file: str,
    line: int,
    category: str,
    title: str,
    evidence: str,
) -> Finding:
    return Finding(
        id=f"{category}-{line}",
        file=file,
        line=line,
        category=category,
        severity=Severity.HIGH,
        confidence=0.95,
        title=title,
        evidence=evidence,
        consequence="c",
        suggestion="s",
        introduced_by_diff=True,
        actionable=True,
        style_only=False,
        reviewer=category,
        status=Status.ACCEPTED,
    )


def _ground_truth(clean: bool, expected: list[ExpectedFinding]) -> CaseGroundTruth:
    return CaseGroundTruth(
        case_id="case",
        clean=clean,
        expected_findings=expected,
        forbidden_concepts=["SQL injection", "command injection"],
    )


def _write_case(
    tmp_path: Path,
    *,
    case_id: str,
    base_content: str,
    head_content: str,
    expected_payload: dict[str, object],
) -> CasePaths:
    case_dir = tmp_path / case_id
    base_dir = case_dir / "base"
    head_dir = case_dir / "head"
    (base_dir / "src").mkdir(parents=True, exist_ok=True)
    (head_dir / "src").mkdir(parents=True, exist_ok=True)
    (base_dir / "src" / "target.py").write_text(base_content, encoding="utf-8")
    (head_dir / "src" / "target.py").write_text(head_content, encoding="utf-8")

    expected_file = case_dir / "expected.json"
    expected_file.write_text(json.dumps(expected_payload), encoding="utf-8")

    return CasePaths(
        case_id=case_id,
        case_dir=case_dir,
        base_dir=base_dir,
        head_dir=head_dir,
        expected_file=expected_file,
    )


def _write_candidate_dataset(
    path: Path,
    *,
    case_id: str,
    expected_payload: dict[str, object],
    findings: list[Finding],
) -> None:
    payload = {
        "schema_version": 1,
        "run_id": "fixed-run",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "run_config": {
            "provider": "fake",
            "model": "fake-reviewer",
            "review_mode": "consolidated",
            "llm_timeout": 30.0,
            "max_review_seconds": 120.0,
            "reviewers": ["bug", "reliability", "security"],
            "max_prompt_tokens": 3500,
            "test_max_prompt_tokens": 2500,
            "max_findings_per_request": 3,
            "deterministic_seed": 123,
            "sampling_params": {
                "temperature": 0.0,
                "seed": 123,
            },
        },
        "cases": [
            {
                "case_id": case_id,
                "source_identifier": "base..head:deadbeefcafebabe",
                "expected": expected_payload,
                "candidate_findings": [item.model_dump(mode="json") for item in findings],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _expected(
    *,
    file: str,
    line_start: int,
    line_end: int,
    category: str,
    concept: str,
    concept_id: str | None = None,
) -> ExpectedFinding:
    return ExpectedFinding(
        file=file,
        line_start=line_start,
        line_end=line_end,
        category=category,
        severity=["medium", "high"],
        concept=concept,
        concept_id=concept_id,
    )


def test_ground_truth_schema_parsing(tmp_path: Path) -> None:
    payload = {
        "case_id": "real_timeout_not_propagated",
        "clean": False,
        "expected_findings": [
            {
                "file": "src/client.py",
                "line_start": 10,
                "line_end": 20,
                "category": "reliability",
                "severity": ["medium", "high"],
                "concept": "configured timeout is not propagated",
                "concept_id": "timeout_not_propagated",
            }
        ],
        "forbidden_concepts": ["database"],
    }
    expected_file = tmp_path / "expected.json"
    expected_file.write_text(json.dumps(payload), encoding="utf-8")

    parsed = load_ground_truth(expected_file)
    assert parsed.case_id == "real_timeout_not_propagated"
    assert len(parsed.expected_findings) == 1
    assert parsed.expected_findings[0].concept_id == "timeout_not_propagated"


def test_line_tolerance_matching() -> None:
    expected = ExpectedFinding(
        file="src/app.py",
        line_start=10,
        line_end=20,
        category="bug",
        severity=["high"],
        concept="timeout bug",
    )
    near = _finding(
        file="src/app.py",
        line=25,
        category="bug",
        title="timeout bug",
        evidence="timeout bug",
    )
    far = _finding(
        file="src/app.py",
        line=27,
        category="bug",
        title="timeout bug",
        evidence="timeout bug",
    )

    assert finding_matches_expected(expected, near, line_tolerance=5)
    assert not finding_matches_expected(expected, far, line_tolerance=5)


def test_category_matching() -> None:
    expected = ExpectedFinding(
        file="src/app.py",
        line_start=5,
        line_end=8,
        category="security",
        severity=["high"],
        concept="command injection",
    )
    finding = _finding(
        file="src/app.py",
        line=6,
        category="reliability",
        title="command injection",
        evidence="command injection",
    )

    assert not finding_matches_expected(expected, finding)


def test_reliability_expected_matches_bug_finding() -> None:
    expected = _expected(
        file="src/storage.py",
        line_start=5,
        line_end=8,
        category="reliability",
        concept="opened file handle can leak on early return",
    )
    finding = _finding(
        file="src/storage.py",
        line=5,
        category="bug",
        title="File handle leak",
        evidence="The file handle is opened but not closed if should_skip is True.",
    )

    assert finding_matches_expected(expected, finding)


def test_bug_expected_matches_reliability_finding() -> None:
    expected = _expected(
        file="src/storage.py",
        line_start=5,
        line_end=8,
        category="bug",
        concept="opened file handle can leak on early return",
    )
    finding = _finding(
        file="src/storage.py",
        line=6,
        category="reliability",
        title="Leak on early return",
        evidence="Open handle is not closed on should_skip branch.",
    )

    assert finding_matches_expected(expected, finding)


def test_security_does_not_match_bug_even_with_same_concept() -> None:
    expected = _expected(
        file="src/runner.py",
        line_start=4,
        line_end=6,
        category="security",
        concept="command injection via shell true",
        concept_id="command_injection",
    )
    finding = _finding(
        file="src/runner.py",
        line=5,
        category="bug",
        title="Command injection",
        evidence="shell=True with untrusted input",
    )

    assert not finding_matches_expected(expected, finding)


def test_security_matches_security() -> None:
    expected = _expected(
        file="src/runner.py",
        line_start=4,
        line_end=6,
        category="security",
        concept="command injection via shell true",
        concept_id="command_injection",
    )
    finding = _finding(
        file="src/runner.py",
        line=5,
        category="security",
        title="Untrusted input in shell command",
        evidence="User supplied input reaches subprocess with shell=True.",
    )

    assert finding_matches_expected(expected, finding)


def test_unknown_category_requires_exact_match() -> None:
    expected = _expected(
        file="src/calc.py",
        line_start=10,
        line_end=15,
        category="performance",
        concept="quadratic loop",
    )
    same = _finding(
        file="src/calc.py",
        line=12,
        category="performance",
        title="Quadratic loop",
        evidence="nested loop causes quadratic behavior",
    )
    other = _finding(
        file="src/calc.py",
        line=12,
        category="bug",
        title="Quadratic loop",
        evidence="nested loop causes quadratic behavior",
    )

    assert finding_matches_expected(expected, same)
    assert not finding_matches_expected(expected, other)


def test_generic_single_word_overlap_is_not_enough() -> None:
    expected = _expected(
        file="src/storage.py",
        line_start=5,
        line_end=8,
        category="reliability",
        concept="opened file handle can leak on early return",
    )
    finding = _finding(
        file="src/storage.py",
        line=6,
        category="bug",
        title="File naming issue",
        evidence="File naming convention differs from style guide.",
    )

    assert not finding_matches_expected(expected, finding)


def test_resource_leak_finding_matches() -> None:
    expected = _expected(
        file="src/storage.py",
        line_start=5,
        line_end=8,
        category="reliability",
        concept="opened file handle can leak on early return",
        concept_id="resource_leak",
    )
    finding = _finding(
        file="src/storage.py",
        line=7,
        category="bug",
        title="File handle leak",
        evidence="Opened handle is not closed on early return path.",
    )

    assert finding_matches_expected(expected, finding)


def test_timeout_finding_matches_across_bug_reliability() -> None:
    expected = _expected(
        file="src/client.py",
        line_start=10,
        line_end=20,
        category="reliability",
        concept="configured timeout is not propagated",
        concept_id="timeout_not_propagated",
    )
    finding = _finding(
        file="src/client.py",
        line=14,
        category="bug",
        title="Timeout is ignored in blocking call",
        evidence="Configured timeout is not used when invoking the HTTP client.",
    )

    assert finding_matches_expected(expected, finding)


def test_generic_shell_timeout_file_comments_do_not_match_aliases() -> None:
    command_expected = _expected(
        file="src/runner.py",
        line_start=4,
        line_end=6,
        category="security",
        concept="command injection via shell true",
        concept_id="command_injection",
    )
    timeout_expected = _expected(
        file="src/client.py",
        line_start=10,
        line_end=20,
        category="reliability",
        concept="configured timeout is not propagated",
        concept_id="timeout_not_propagated",
    )
    leak_expected = _expected(
        file="src/storage.py",
        line_start=5,
        line_end=8,
        category="reliability",
        concept="opened file handle can leak on early return",
        concept_id="resource_leak",
    )

    shell_only = _finding(
        file="src/runner.py",
        line=5,
        category="security",
        title="Uses shell command",
        evidence="A shell command is executed.",
    )
    timeout_only = _finding(
        file="src/client.py",
        line=12,
        category="bug",
        title="Timeout configured",
        evidence="Timeout value is configured to 30 seconds.",
    )
    file_only = _finding(
        file="src/storage.py",
        line=6,
        category="bug",
        title="File operations present",
        evidence="Function handles file paths and writes output.",
    )

    assert not finding_matches_expected(command_expected, shell_only)
    assert not finding_matches_expected(timeout_expected, timeout_only)
    assert not finding_matches_expected(leak_expected, file_only)


def test_wrong_file_does_not_match() -> None:
    expected = _expected(
        file="src/storage.py",
        line_start=5,
        line_end=8,
        category="reliability",
        concept="opened file handle can leak on early return",
    )
    finding = _finding(
        file="src/other.py",
        line=6,
        category="bug",
        title="File handle leak",
        evidence="Handle not closed on early return.",
    )

    assert not finding_matches_expected(expected, finding)


def test_line_outside_tolerance_does_not_match() -> None:
    expected = _expected(
        file="src/storage.py",
        line_start=5,
        line_end=8,
        category="reliability",
        concept="opened file handle can leak on early return",
    )
    finding = _finding(
        file="src/storage.py",
        line=20,
        category="bug",
        title="File handle leak",
        evidence="Handle not closed on early return.",
    )

    assert not finding_matches_expected(expected, finding)


def test_concept_normalization() -> None:
    keywords = normalize_concept_keywords("SQL injection in shell command")
    assert "sql_injection" in keywords
    assert "command" in keywords


def test_one_to_one_matching() -> None:
    expected = [
        ExpectedFinding(
            file="src/client.py",
            line_start=10,
            line_end=20,
            category="reliability",
            severity=["high"],
            concept="timeout not propagated",
        )
    ]
    findings = [
        _finding(
            file="src/client.py",
            line=12,
            category="reliability",
            title="timeout not propagated",
            evidence="timeout",
        ),
        _finding(
            file="src/client.py",
            line=13,
            category="reliability",
            title="timeout not propagated",
            evidence="timeout",
        ),
    ]

    scored = score_case(_ground_truth(clean=False, expected=expected), findings)
    assert scored.true_positives == 1
    assert scored.false_positives == 1
    assert scored.false_negatives == 0


def test_false_positive_and_false_negative_calculation() -> None:
    expected = [
        ExpectedFinding(
            file="src/a.py",
            line_start=1,
            line_end=2,
            category="bug",
            severity=["high"],
            concept="real bug",
        ),
        ExpectedFinding(
            file="src/b.py",
            line_start=3,
            line_end=4,
            category="security",
            severity=["high"],
            concept="real security bug",
        ),
    ]
    findings = [
        _finding(file="src/a.py", line=1, category="bug", title="real bug", evidence="real bug"),
        _finding(file="src/x.py", line=9, category="bug", title="noise", evidence="noise"),
    ]

    scored = score_case(_ground_truth(clean=False, expected=expected), findings)
    assert scored.true_positives == 1
    assert scored.false_positives == 1
    assert scored.false_negatives == 1


def test_score_case_regression_bug_reliability_semantic_match() -> None:
    expected = [
        _expected(
            file="src/storage.py",
            line_start=5,
            line_end=8,
            category="reliability",
            concept="opened file handle can leak on early return",
        )
    ]
    findings = [
        _finding(
            file="src/storage.py",
            line=5,
            category="bug",
            title="File handle leak",
            evidence="The file handle is opened but not closed if should_skip is True.",
        )
    ]

    scored = score_case(_ground_truth(clean=False, expected=expected), findings)
    assert scored.true_positives == 1
    assert scored.false_positives == 0
    assert scored.false_negatives == 0
    assert scored.matched_pairs == [(0, 0)]


def test_forbidden_concept_detection() -> None:
    gt = _ground_truth(clean=True, expected=[])
    findings = [
        _finding(
            file="src/a.py",
            line=10,
            category="security",
            title="Possible SQL injection",
            evidence="SQL injection vector",
        )
    ]
    hits = detect_forbidden_concepts(gt, findings)
    assert hits


def test_zero_division_behavior() -> None:
    assert safe_divide(1, 0) == 0.0
    assert safe_divide(0, 0) == 0.0


def test_clean_case_false_positive_rate() -> None:
    metadata = ReviewMetadata(
        base="base", head="head", changed_files=1, diff_lines=1, reviewers=["bug"]
    )
    result = ReviewResult(metadata=metadata, accepted_findings=[], rejected_findings=[])

    clean_gt = _ground_truth(clean=True, expected=[])
    noisy_clean = _ground_truth(clean=True, expected=[])
    finding = _finding(file="src/a.py", line=1, category="bug", title="noise", evidence="noise")

    executions = [
        CaseExecution(
            case_id="clean-ok",
            ground_truth=clean_gt,
            success=True,
            error="",
            elapsed_seconds=1.0,
            result=result,
            scored_findings=[],
            accepted_findings=[],
            rejected_findings=[],
            raw_events=[],
        ),
        CaseExecution(
            case_id="clean-fp",
            ground_truth=noisy_clean,
            success=True,
            error="",
            elapsed_seconds=1.0,
            result=result,
            scored_findings=[finding],
            accepted_findings=[finding],
            rejected_findings=[],
            raw_events=[],
        ),
    ]

    metrics = _aggregate(executions)
    assert metrics.clean_case_false_positive_rate == 0.5


def test_aggregate_scores_rejected_but_semantically_correct_findings() -> None:
    metadata = ReviewMetadata(
        base="base", head="head", changed_files=1, diff_lines=1, reviewers=["reliability"]
    )
    result = ReviewResult(metadata=metadata, accepted_findings=[], rejected_findings=[])

    expected = [
        _expected(
            file="src/client.py",
            line_start=12,
            line_end=18,
            category="reliability",
            concept="configured timeout is not propagated",
            concept_id="timeout_not_propagated",
        )
    ]

    finding = _finding(
        file="src/client.py",
        line=14,
        category="bug",
        title="Configured timeout ignored",
        evidence="Configured timeout is not forwarded to the HTTP client call.",
    )
    finding.actionable = False

    execution = CaseExecution(
        case_id="timeout-case",
        ground_truth=_ground_truth(clean=False, expected=expected),
        success=True,
        error="",
        elapsed_seconds=0.5,
        result=result,
        scored_findings=[finding],
        accepted_findings=[],
        rejected_findings=[finding],
        raw_events=[],
    )

    metrics = _aggregate([execution])
    assert metrics.true_positives == 1
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0


def test_temporary_git_repository_creation_and_cleanup(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    base = case_dir / "base"
    head = case_dir / "head"
    (base / "src").mkdir(parents=True)
    (head / "src").mkdir(parents=True)

    (base / "src" / "module.py").write_text("x = 1\n", encoding="utf-8")
    (head / "src" / "module.py").write_text("x = 2\n", encoding="utf-8")

    expected_file = case_dir / "expected.json"
    expected_file.write_text(
        json.dumps(
            {
                "case_id": "temp-case",
                "clean": True,
                "expected_findings": [],
                "forbidden_concepts": [],
            }
        ),
        encoding="utf-8",
    )

    case = CasePaths(
        case_id="temp-case",
        case_dir=case_dir,
        base_dir=base,
        head_dir=head,
        expected_file=expected_file,
    )

    repo = _materialize_case_repo(case, keep_temp=False)
    assert repo.exists()

    tags = subprocess.run(
        ["git", "tag", "--list"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "base" in tags
    assert "head" in tags

    _cleanup_repo(repo, keep_temp=False)
    assert not repo.exists()


def test_continuation_after_one_failed_case(monkeypatch, tmp_path: Path) -> None:
    calls = {"count": 0}

    fake_case = CasePaths(
        case_id="a",
        case_dir=tmp_path,
        base_dir=tmp_path,
        head_dir=tmp_path,
        expected_file=tmp_path / "expected.json",
    )

    gt = CaseGroundTruth(
        case_id="a",
        clean=True,
        expected_findings=[],
        forbidden_concepts=[],
    )

    metadata = ReviewMetadata(
        base="base", head="head", changed_files=1, diff_lines=1, reviewers=["bug"]
    )
    result = ReviewResult(metadata=metadata, accepted_findings=[], rejected_findings=[])

    def fake_collect(_case: str | None) -> list[CasePaths]:
        return [fake_case, fake_case]

    def fake_run_case(_case: CasePaths, **kwargs) -> CaseExecution:  # type: ignore[no-untyped-def]
        calls["count"] += 1
        if calls["count"] == 1:
            return CaseExecution(
                case_id="failed",
                ground_truth=gt,
                success=False,
                error="boom",
                elapsed_seconds=0.1,
                result=None,
                scored_findings=[],
                accepted_findings=[],
                rejected_findings=[],
                raw_events=[],
            )
        return CaseExecution(
            case_id="ok",
            ground_truth=gt,
            success=True,
            error="",
            elapsed_seconds=0.1,
            result=result,
            scored_findings=[],
            accepted_findings=[],
            rejected_findings=[],
            raw_events=[],
        )

    def fake_aggregate(_executions: list[CaseExecution]) -> AggregateMetrics:
        return AggregateMetrics(
            cases=2,
            clean_cases=2,
            expected_defects=0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            clean_case_false_positive_rate=0.0,
            valid_file_and_line_rate=1.0,
            parse_success_rate=1.0,
            elapsed_seconds=0.2,
            correct_findings_per_minute=0.0,
        )

    monkeypatch.setattr("benchmarks.run_benchmarks._collect_cases", fake_collect)
    monkeypatch.setattr("benchmarks.run_benchmarks._run_case", fake_run_case)
    monkeypatch.setattr("benchmarks.run_benchmarks._aggregate", fake_aggregate)

    output = tmp_path / "result.json"
    exit_code = run(["--provider", "fake", "--output", str(output)])

    assert exit_code == 0
    assert calls["count"] == 2
    assert output.exists()


def test_verification_only_mode_does_not_call_reviewer(monkeypatch, tmp_path: Path) -> None:
    expected_payload = {
        "case_id": "replay_case",
        "clean": True,
        "expected_findings": [],
        "forbidden_concepts": [],
    }
    case = _write_case(
        tmp_path,
        case_id="replay_case",
        base_content="value = 1\n",
        head_content="value = 2\n",
        expected_payload=expected_payload,
    )
    candidate = _finding(
        file="src/target.py",
        line=1,
        category="bug",
        title="Candidate issue",
        evidence="Candidate evidence",
    )
    candidate_path = tmp_path / "candidates.json"
    _write_candidate_dataset(
        candidate_path,
        case_id="replay_case",
        expected_payload=expected_payload,
        findings=[candidate],
    )

    monkeypatch.setattr("benchmarks.run_benchmarks._collect_cases", lambda _case: [case])

    def fail_if_called(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("run_reviewers should not be called")

    monkeypatch.setattr("benchmarks.run_benchmarks.run_reviewers", fail_if_called)

    output = tmp_path / "result.json"
    exit_code = run(
        [
            "--provider",
            "fake",
            "--candidate-findings-input",
            str(candidate_path),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert output.exists()


def test_verification_only_same_input_same_candidate_metrics(monkeypatch, tmp_path: Path) -> None:
    expected_payload = {
        "case_id": "replay_case",
        "clean": True,
        "expected_findings": [],
        "forbidden_concepts": [],
    }
    case = _write_case(
        tmp_path,
        case_id="replay_case",
        base_content="value = 1\n",
        head_content="value = 2\n",
        expected_payload=expected_payload,
    )
    candidate = _finding(
        file="src/target.py",
        line=1,
        category="bug",
        title="Candidate issue",
        evidence="Candidate evidence",
    )
    candidate_path = tmp_path / "candidates.json"
    _write_candidate_dataset(
        candidate_path,
        case_id="replay_case",
        expected_payload=expected_payload,
        findings=[candidate],
    )

    monkeypatch.setattr("benchmarks.run_benchmarks._collect_cases", lambda _case: [case])
    monkeypatch.setattr("benchmarks.run_benchmarks.run_reviewers", lambda *args, **kwargs: None)

    output_a = tmp_path / "result-a.json"
    output_b = tmp_path / "result-b.json"
    assert (
        run(
            [
                "--provider",
                "fake",
                "--candidate-findings-input",
                str(candidate_path),
                "--output",
                str(output_a),
            ]
        )
        == 0
    )
    assert (
        run(
            [
                "--provider",
                "fake",
                "--candidate-findings-input",
                str(candidate_path),
                "--output",
                str(output_b),
            ]
        )
        == 0
    )

    payload_a = json.loads(output_a.read_text(encoding="utf-8"))
    payload_b = json.loads(output_b.read_text(encoding="utf-8"))
    assert payload_a["metrics_by_stage"]["candidate"] == payload_b["metrics_by_stage"]["candidate"]


def test_verifier_reduces_fp_without_reviewer_changes(monkeypatch, tmp_path: Path) -> None:
    expected_payload = {
        "case_id": "fp_case",
        "clean": True,
        "expected_findings": [],
        "forbidden_concepts": [],
    }
    case = _write_case(
        tmp_path,
        case_id="fp_case",
        base_content="value = 1\n",
        head_content="value = 2\n",
        expected_payload=expected_payload,
    )
    candidate = _finding(
        file="src/target.py",
        line=1,
        category="bug",
        title="False positive",
        evidence="False positive evidence",
    )
    candidate_path = tmp_path / "candidates.json"
    _write_candidate_dataset(
        candidate_path,
        case_id="fp_case",
        expected_payload=expected_payload,
        findings=[candidate],
    )

    monkeypatch.setattr("benchmarks.run_benchmarks._collect_cases", lambda _case: [case])
    monkeypatch.setattr("benchmarks.run_benchmarks.run_reviewers", lambda *args, **kwargs: None)

    class RejectAllVerifier:
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            pass

        def verify(self, findings, context):  # type: ignore[no-untyped-def]
            rejected = []
            for item in findings:
                copy = item.model_copy(deep=True)
                copy.status = Status.REJECTED
                copy.rejection_reason = "verification_failed"
                copy.verification_verdict = "invalid"
                copy.verification_confidence = 0.99
                rejected.append(copy)
            return VerificationResult(
                verified_findings=[],
                verification_rejected_findings=rejected,
                completed_requests=len(findings),
                failed_requests=0,
                skipped_requests=0,
                elapsed_seconds=0.01,
                valid_count=0,
                invalid_count=len(findings),
                uncertain_count=0,
                unverified_count=0,
                skipped_count=0,
                debug_events=[],
            )

    monkeypatch.setattr("benchmarks.run_benchmarks.LLMFindingVerifier", RejectAllVerifier)

    output = tmp_path / "result.json"
    assert (
        run(
            [
                "--provider",
                "fake",
                "--candidate-findings-input",
                str(candidate_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metrics_by_stage"]["candidate"]["false_positives"] == 1
    assert payload["metrics_by_stage"]["verified"]["false_positives"] == 0
    assert payload["verification_only_metrics"]["delta_after_verification"]["fp_removed"] == 1


def test_missing_candidate_tp_not_attributed_to_verifier(monkeypatch, tmp_path: Path) -> None:
    expected_payload = {
        "case_id": "tp_case",
        "clean": False,
        "expected_findings": [
            {
                "file": "src/target.py",
                "line_start": 1,
                "line_end": 1,
                "category": "bug",
                "severity": ["high"],
                "concept": "real defect",
                "concept_id": None,
            }
        ],
        "forbidden_concepts": [],
    }
    case = _write_case(
        tmp_path,
        case_id="tp_case",
        base_content="value = 1\n",
        head_content="value = 2\n",
        expected_payload=expected_payload,
    )
    candidate_path = tmp_path / "candidates.json"
    _write_candidate_dataset(
        candidate_path,
        case_id="tp_case",
        expected_payload=expected_payload,
        findings=[],
    )

    monkeypatch.setattr("benchmarks.run_benchmarks._collect_cases", lambda _case: [case])
    monkeypatch.setattr("benchmarks.run_benchmarks.run_reviewers", lambda *args, **kwargs: None)

    output = tmp_path / "result.json"
    assert (
        run(
            [
                "--provider",
                "fake",
                "--candidate-findings-input",
                str(candidate_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metrics_by_stage"]["candidate"]["false_negatives"] == 1
    assert payload["metrics_by_stage"]["verified"]["false_negatives"] == 1
    assert payload["verification_only_metrics"]["delta_after_verification"]["new_fn"] == 0


def test_delta_metrics_correct(monkeypatch, tmp_path: Path) -> None:
    expected_payload = {
        "case_id": "mixed_case",
        "clean": False,
        "expected_findings": [
            {
                "file": "src/target.py",
                "line_start": 1,
                "line_end": 1,
                "category": "bug",
                "severity": ["high"],
                "concept": "real defect",
                "concept_id": None,
            }
        ],
        "forbidden_concepts": [],
    }
    case = _write_case(
        tmp_path,
        case_id="mixed_case",
        base_content="value = 1\n",
        head_content="value = 2\n",
        expected_payload=expected_payload,
    )
    tp = _finding(
        file="src/target.py",
        line=1,
        category="bug",
        title="Real defect",
        evidence="real defect evidence",
    )
    fp = _finding(
        file="src/target.py",
        line=1,
        category="security",
        title="Noise",
        evidence="noise evidence",
    )
    fp.id = "fp"
    tp.id = "tp"
    candidate_path = tmp_path / "candidates.json"
    _write_candidate_dataset(
        candidate_path,
        case_id="mixed_case",
        expected_payload=expected_payload,
        findings=[tp, fp],
    )

    monkeypatch.setattr("benchmarks.run_benchmarks._collect_cases", lambda _case: [case])
    monkeypatch.setattr("benchmarks.run_benchmarks.run_reviewers", lambda *args, **kwargs: None)

    class KeepTpDropFpVerifier:
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            pass

        def verify(self, findings, context):  # type: ignore[no-untyped-def]
            kept = []
            rejected = []
            for item in findings:
                copy = item.model_copy(deep=True)
                if copy.id == "tp":
                    copy.verification_verdict = "valid"
                    copy.verification_confidence = 0.99
                    kept.append(copy)
                else:
                    copy.status = Status.REJECTED
                    copy.rejection_reason = "verification_failed"
                    copy.verification_verdict = "invalid"
                    copy.verification_confidence = 0.99
                    rejected.append(copy)
            return VerificationResult(
                verified_findings=kept,
                verification_rejected_findings=rejected,
                completed_requests=len(findings),
                failed_requests=0,
                skipped_requests=0,
                elapsed_seconds=0.01,
                valid_count=1,
                invalid_count=1,
                uncertain_count=0,
                unverified_count=0,
                skipped_count=0,
                debug_events=[],
            )

    monkeypatch.setattr("benchmarks.run_benchmarks.LLMFindingVerifier", KeepTpDropFpVerifier)

    output = tmp_path / "result.json"
    assert (
        run(
            [
                "--provider",
                "fake",
                "--candidate-findings-input",
                str(candidate_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    delta = payload["verification_only_metrics"]["delta_after_verification"]
    assert delta["tp_removed"] == 0
    assert delta["fp_removed"] == 1
    assert delta["new_fn"] == 0
    assert delta["precision_change"] == 0.5
    assert delta["recall_change"] == 0.0
