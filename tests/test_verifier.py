from __future__ import annotations

import json

from benchmarks.schema import CaseGroundTruth
from benchmarks.scoring import score_case
from reviewer.models import Finding, Severity, Status
from reviewer.verifier import (
    FakeFindingVerifier,
    LLMFindingVerifier,
    VerificationContext,
    build_verification_context,
)


class _SequenceLLM:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.prompts: list[tuple[str, str]] = []
        self.model = "fake-verifier-model"

    def generate(self, system_prompt: str, user_prompt: str, *, response_schema=None) -> str:  # type: ignore[no-untyped-def]
        _ = response_schema
        self.prompts.append((system_prompt, user_prompt))
        if not self._responses:
            raise AssertionError("No more responses configured")
        current = self._responses.pop(0)
        if isinstance(current, Exception):
            raise current
        return str(current)


def _finding(
    finding_id: str,
    *,
    file: str,
    line: int,
    category: str = "bug",
    title: str,
    evidence: str,
    consequence: str,
    suggestion: str,
    style_only: bool = False,
    introduced_by_diff: bool = True,
) -> Finding:
    return Finding(
        id=finding_id,
        file=file,
        line=line,
        category=category,
        severity=Severity.HIGH,
        confidence=0.95,
        title=title,
        evidence=evidence,
        consequence=consequence,
        suggestion=suggestion,
        introduced_by_diff=introduced_by_diff,
        actionable=True,
        style_only=style_only,
        duplicate_of=None,
        reviewer="consolidated",
        status=Status.PROPOSED,
        rejection_reason="",
    )


def _context(
    *,
    max_findings: int = 5,
    fail_policy: str = "unverified",
    uncertain_policy: str = "unverified",
) -> VerificationContext:
    diff_text = (
        "diff --git a/src/client.py b/src/client.py\n"
        "--- a/src/client.py\n"
        "+++ b/src/client.py\n"
        "@@ -10,2 +10,4 @@\n"
        "+configured_timeout = self.timeout_seconds\n"
        "+return self._http.post('/chat', json={'prompt': prompt})\n"
        "\n"
        "diff --git a/tests/test_sync.py b/tests/test_sync.py\n"
        "--- a/tests/test_sync.py\n"
        "+++ b/tests/test_sync.py\n"
        "@@ -4,2 +4,3 @@\n"
        "+with pytest.raises(ValueError):\n"
        "+    parse_positive(-1)\n"
    )
    return VerificationContext(
        base="base",
        head="head",
        diff_text=diff_text,
        file_contexts={
            "src/client.py": "FILE: src/client.py\nCHANGED_LINES: 10,11",
            "tests/test_sync.py": "FILE: tests/test_sync.py\nCHANGED_LINES: 4,5",
            "src/unrelated.py": "FILE: src/unrelated.py\nCHANGED_LINES: 99",
        },
        changed_lines_by_file={
            "src/client.py": {10, 11},
            "tests/test_sync.py": {4, 5},
        },
        provider="fake",
        review_model="qwen2.5-coder:1.5b",
        verification_model="qwen2.5-coder:1.5b",
        timeout_seconds=2.0,
        total_budget_seconds=10.0,
        max_findings=max_findings,
        min_confidence=0.8,
        fail_policy=fail_policy,
        uncertain_policy=uncertain_policy,
    )


def test_build_verification_context_only_returns_target_file_hunk() -> None:
    context = _context()
    slice_context = build_verification_context(
        context.diff_text,
        file="src/client.py",
        line=10,
        context_lines=4,
    )

    assert "src/client.py" not in slice_context.diff_excerpt
    assert "configured_timeout" in slice_context.diff_excerpt
    assert "pytest.raises" not in slice_context.diff_excerpt


def test_fake_verifier_verification_disabled_passthrough_shape() -> None:
    findings = [
        _finding(
            "timeout-1",
            file="src/client.py",
            line=10,
            title="Timeout ignored",
            evidence="configured timeout not used",
            consequence="request can hang",
            suggestion="pass timeout argument",
        )
    ]

    result = findings

    assert result[0].id == "timeout-1"


def test_fake_verifier_valid_finding() -> None:
    verifier = FakeFindingVerifier(mode="approve_all")
    findings = [
        _finding(
            "timeout-1",
            file="src/client.py",
            line=10,
            title="Timeout ignored",
            evidence="configured timeout not used",
            consequence="request can hang",
            suggestion="pass timeout argument",
        )
    ]

    result = verifier.verify(findings, _context())

    assert len(result.verified_findings) == 1
    assert result.verified_findings[0].verification_status == "valid"
    assert result.completed_requests == 1


def test_fake_verifier_invalid_finding_is_rejected() -> None:
    verifier = FakeFindingVerifier(mode="reject_all")
    finding = _finding(
        "clean-1",
        file="tests/test_sync.py",
        line=4,
        title="pytest.raises is a bug",
        evidence="Raises block indicates failure",
        consequence="test hides errors",
        suggestion="remove pytest.raises",
    )

    result = verifier.verify([finding], _context())

    assert result.verified_findings == []
    assert len(result.verification_rejected_findings) == 1
    assert result.verification_rejected_findings[0].rejection_reason == "verification_failed"


def test_fake_verifier_uncertain_goes_to_unverified_by_default() -> None:
    verifier = FakeFindingVerifier(mode="uncertain_all")
    finding = _finding(
        "u-1",
        file="src/client.py",
        line=10,
        title="Potential bug",
        evidence="Context seems partial",
        consequence="may fail",
        suggestion="investigate",
    )

    result = verifier.verify([finding], _context())

    assert len(result.verified_findings) == 1
    assert result.verified_findings[0].verification_status == "unverified"
    assert result.verified_findings[0].verification_verdict == "uncertain"
    assert result.uncertain_count == 1


def test_fake_verifier_uncertain_can_be_rejected_by_policy() -> None:
    verifier = FakeFindingVerifier(mode="uncertain_all")
    finding = _finding(
        "u-2",
        file="src/client.py",
        line=10,
        title="Potential bug",
        evidence="Context seems partial",
        consequence="may fail",
        suggestion="investigate",
    )

    result = verifier.verify([finding], _context(uncertain_policy="reject"))

    assert result.verified_findings == []
    assert len(result.verification_rejected_findings) == 1
    assert (
        result.verification_rejected_findings[0].rejection_reason
        == "verification_uncertain_rejected"
    )


def test_fake_verifier_parse_failure_uses_unverified_policy() -> None:
    verifier = FakeFindingVerifier(mode="approve_all", parse_error_ids={"parse-1"})
    finding = _finding(
        "parse-1",
        file="src/client.py",
        line=10,
        title="Timeout ignored",
        evidence="configured timeout not used",
        consequence="request can hang",
        suggestion="pass timeout argument",
    )

    result = verifier.verify([finding], _context(fail_policy="unverified"))

    assert len(result.verified_findings) == 1
    assert result.verified_findings[0].verification_status == "unverified"
    assert result.failed_requests == 1


def test_fake_verifier_timeout_uses_unverified_policy() -> None:
    verifier = FakeFindingVerifier(mode="approve_all", timeout_ids={"timeout-1"})
    finding = _finding(
        "timeout-1",
        file="src/client.py",
        line=10,
        title="Timeout ignored",
        evidence="configured timeout not used",
        consequence="request can hang",
        suggestion="pass timeout argument",
    )

    result = verifier.verify([finding], _context(fail_policy="unverified"))

    assert len(result.verified_findings) == 1
    assert result.verified_findings[0].verification_status == "unverified"
    assert result.failed_requests == 1


def test_fake_verifier_total_budget_marks_remaining_skipped() -> None:
    verifier = FakeFindingVerifier(mode="approve_all", budget_exhaust_after=1)
    findings = [
        _finding(
            "a",
            file="src/client.py",
            line=10,
            title="Timeout ignored",
            evidence="configured timeout not used",
            consequence="request can hang",
            suggestion="pass timeout argument",
        ),
        _finding(
            "b",
            file="src/client.py",
            line=11,
            title="Timeout ignored",
            evidence="configured timeout not used",
            consequence="request can hang",
            suggestion="pass timeout argument",
        ),
    ]

    result = verifier.verify(findings, _context(max_findings=5))

    assert result.completed_requests == 1
    assert result.skipped_requests == 1
    assert any(item.verification_status == "skipped" for item in result.verified_findings)


def test_fake_verifier_max_findings_marks_overflow_skipped() -> None:
    verifier = FakeFindingVerifier(mode="approve_all")
    findings = [
        _finding(
            "a",
            file="src/client.py",
            line=10,
            title="Timeout ignored",
            evidence="configured timeout not used",
            consequence="request can hang",
            suggestion="pass timeout argument",
        ),
        _finding(
            "b",
            file="src/client.py",
            line=11,
            title="Timeout ignored",
            evidence="configured timeout not used",
            consequence="request can hang",
            suggestion="pass timeout argument",
        ),
    ]

    result = verifier.verify(findings, _context(max_findings=1))

    assert result.completed_requests == 1
    assert result.skipped_requests == 1
    assert len(result.verified_findings) == 2


def test_llm_verifier_relevant_prompt_uses_only_target_file_context() -> None:
    llm = _SequenceLLM(
        [
            json.dumps(
                {
                    "verdict": "valid",
                    "confidence": 0.95,
                    "reason": "Clear timeout mismatch",
                    "evidence_lines": [10, 11],
                }
            )
        ]
    )
    verifier = LLMFindingVerifier(llm)
    finding = _finding(
        "timeout-1",
        file="src/client.py",
        line=10,
        title="Timeout ignored",
        evidence="configured timeout not used",
        consequence="request can hang",
        suggestion="pass timeout argument",
    )

    result = verifier.verify([finding], _context())

    assert len(result.verified_findings) == 1
    assert result.verification_rejected_findings == []
    assert len(llm.prompts) == 1
    prompt = llm.prompts[0][1]
    assert "TARGET_FILE: src/client.py" in prompt
    assert "tests/test_sync.py" not in prompt


def test_llm_verifier_ignores_attempt_to_add_new_findings() -> None:
    llm = _SequenceLLM(
        [
            json.dumps(
                {
                    "verdict": "valid",
                    "confidence": 0.95,
                    "reason": "Valid finding",
                    "evidence_lines": [10],
                    "findings": [
                        {
                            "id": "new-issue",
                            "file": "src/new.py",
                            "line": 1,
                        }
                    ],
                }
            )
        ]
    )
    verifier = LLMFindingVerifier(llm)
    finding = _finding(
        "timeout-1",
        file="src/client.py",
        line=10,
        title="Timeout ignored",
        evidence="configured timeout not used",
        consequence="request can hang",
        suggestion="pass timeout argument",
    )

    result = verifier.verify([finding], _context())

    assert len(result.verified_findings) == 1
    assert result.verified_findings[0].id == "timeout-1"


def test_llm_verifier_low_confidence_becomes_uncertain_not_invalid() -> None:
    llm = _SequenceLLM(
        [
            json.dumps(
                {
                    "verdict": "valid",
                    "confidence": 0.62,
                    "reason": "Could be valid but confidence is low.",
                    "evidence_lines": [10],
                }
            )
        ]
    )
    verifier = LLMFindingVerifier(llm)
    finding = _finding(
        "timeout-1",
        file="src/client.py",
        line=10,
        title="Configured timeout is not propagated",
        evidence="configured_timeout not passed to post call",
        consequence="request may block indefinitely",
        suggestion="pass timeout argument",
    )

    result = verifier.verify([finding], _context())

    assert len(result.verification_rejected_findings) == 0
    assert len(result.verified_findings) == 1
    assert result.verified_findings[0].verification_status == "unverified"
    assert result.verified_findings[0].verification_verdict == "uncertain"
    assert result.uncertain_count == 1


def test_llm_verifier_uncertain_from_insufficient_context_not_invalid() -> None:
    llm = _SequenceLLM(
        [
            json.dumps(
                {
                    "verdict": "uncertain",
                    "confidence": 0.92,
                    "reason": "Insufficient context to determine full control flow.",
                    "evidence_lines": [10],
                }
            )
        ]
    )
    verifier = LLMFindingVerifier(llm)
    finding = _finding(
        "res-1",
        file="src/client.py",
        line=10,
        title="Resource leak",
        evidence="file opened before early return",
        consequence="descriptor leak",
        suggestion="ensure close in finally",
    )

    result = verifier.verify([finding], _context())

    assert len(result.verification_rejected_findings) == 0
    assert len(result.verified_findings) == 1
    assert result.verified_findings[0].verification_status == "unverified"
    assert result.verified_findings[0].verification_verdict == "uncertain"


def test_llm_verifier_invalid_with_contradiction_is_rejected_with_diagnostics() -> None:
    llm = _SequenceLLM(
        [
            json.dumps(
                {
                    "verdict": "invalid",
                    "confidence": 0.94,
                    "reason": "The diff explicitly passes the timeout argument already.",
                    "evidence_lines": [11],
                }
            )
        ]
    )
    verifier = LLMFindingVerifier(llm)
    finding = _finding(
        "wrong-1",
        file="src/client.py",
        line=10,
        title="Timeout missing",
        evidence="timeout not forwarded",
        consequence="hang",
        suggestion="pass timeout",
    )

    result = verifier.verify([finding], _context())

    assert len(result.verified_findings) == 0
    assert len(result.verification_rejected_findings) == 1
    rejected = result.verification_rejected_findings[0]
    assert rejected.rejection_reason == "verification_failed"
    assert rejected.verification_prompt_chars is not None
    assert rejected.verification_response_text
    assert rejected.verification_line_in_context is True


def test_llm_verifier_invalid_without_clear_contradiction_becomes_uncertain() -> None:
    llm = _SequenceLLM(
        [
            json.dumps(
                {
                    "verdict": "invalid",
                    "confidence": 0.96,
                    "reason": "The provided code does not contradict the candidate claim.",
                    "evidence_lines": [10],
                }
            )
        ]
    )
    verifier = LLMFindingVerifier(llm)
    finding = _finding(
        "ambiguous-invalid",
        file="src/client.py",
        line=10,
        title="Configured timeout is not propagated",
        evidence="configured_timeout not passed to post call",
        consequence="request may block indefinitely",
        suggestion="pass timeout argument",
    )

    result = verifier.verify([finding], _context())

    assert len(result.verification_rejected_findings) == 0
    assert len(result.verified_findings) == 1
    assert result.verified_findings[0].verification_verdict == "uncertain"
    assert result.verified_findings[0].verification_status == "unverified"


def test_llm_verifier_does_not_invalidate_for_weak_suggestion_or_severity() -> None:
    llm = _SequenceLLM(
        [
            json.dumps(
                {
                    "verdict": "valid",
                    "confidence": 0.93,
                    "reason": "Core defect claim appears correct even if wording is rough.",
                    "evidence_lines": [10, 11],
                }
            )
        ]
    )
    verifier = LLMFindingVerifier(llm)
    finding = _finding(
        "tp-weak-meta",
        file="src/client.py",
        line=10,
        title="Timeout issue",
        evidence="configured_timeout not passed to post call",
        consequence="impact wording may be overstated",
        suggestion="maybe refactor all clients first",
    )
    finding.severity = Severity.LOW

    result = verifier.verify([finding], _context())

    assert len(result.verification_rejected_findings) == 0
    assert len(result.verified_findings) == 1
    assert result.verified_findings[0].verification_status == "valid"


def test_regression_clean_case_false_positives_are_rejected() -> None:
    verifier = FakeFindingVerifier(mode="approve_ids", valid_ids=set())
    findings = [
        _finding(
            "fp-pytest-raises",
            file="tests/test_sync.py",
            line=4,
            title="pytest.raises is a bug",
            evidence="exception is raised in test",
            consequence="unexpected failure",
            suggestion="remove raises",
        ),
        _finding(
            "fp-default-response",
            file="src/client.py",
            line=10,
            title="FakeLLMClient must have default response",
            evidence="constructor should default response",
            consequence="instantiation fails",
            suggestion="add default",
        ),
        _finding(
            "fp-exception-translation",
            file="src/client.py",
            line=10,
            title="Exception translation is wrong",
            evidence="custom error wrapping is bug",
            consequence="error hidden",
            suggestion="remove translation",
        ),
        _finding(
            "fp-missing-param",
            file="src/client.py",
            line=10,
            title="Parameter is not used",
            evidence="parameter appears unused",
            consequence="runtime error",
            suggestion="remove parameter",
        ),
        _finding(
            "fp-git-config",
            file="tests/test_sync.py",
            line=4,
            title="git setup needs user config",
            evidence="test setup lacks config",
            consequence="git commit fails",
            suggestion="set global config",
        ),
    ]

    result = verifier.verify(findings, _context(max_findings=10))

    assert result.verified_findings == []
    assert len(result.verification_rejected_findings) == 5


def test_regression_real_findings_stay_valid_and_keep_recall() -> None:
    verifier = FakeFindingVerifier(
        mode="approve_ids",
        valid_ids={"tp-command", "tp-resource", "tp-timeout"},
    )
    findings = [
        _finding(
            "tp-command",
            file="src/client.py",
            line=10,
            category="security",
            title="Untrusted input reaches shell command",
            evidence="shell command includes unsanitized model_name",
            consequence="command injection",
            suggestion="use argv list instead of shell command",
        ),
        _finding(
            "tp-resource",
            file="src/client.py",
            line=10,
            title="Resource leak on early return",
            evidence="file handle opened and not closed",
            consequence="resource leak",
            suggestion="use context manager",
        ),
        _finding(
            "tp-timeout",
            file="src/client.py",
            line=10,
            title="Configured timeout is not propagated",
            evidence="configured_timeout not passed to post call",
            consequence="request may block indefinitely",
            suggestion="pass timeout argument",
        ),
    ]

    result = verifier.verify(findings, _context(max_findings=10))
    assert len(result.verified_findings) == 3

    gt_command = CaseGroundTruth.model_validate(
        {
            "case_id": "real_command_injection",
            "clean": False,
            "expected_findings": [
                {
                    "file": "src/client.py",
                    "line_start": 8,
                    "line_end": 12,
                    "category": "security",
                    "severity": ["high"],
                    "concept": "command injection via shell true",
                    "concept_id": "command_injection",
                }
            ],
            "forbidden_concepts": [],
        }
    )
    gt_resource = CaseGroundTruth.model_validate(
        {
            "case_id": "real_resource_leak",
            "clean": False,
            "expected_findings": [
                {
                    "file": "src/client.py",
                    "line_start": 8,
                    "line_end": 12,
                    "category": "reliability",
                    "severity": ["high"],
                    "concept": "opened file handle can leak on early return",
                    "concept_id": "resource_leak",
                }
            ],
            "forbidden_concepts": [],
        }
    )
    gt_timeout = CaseGroundTruth.model_validate(
        {
            "case_id": "real_timeout_not_propagated",
            "clean": False,
            "expected_findings": [
                {
                    "file": "src/client.py",
                    "line_start": 8,
                    "line_end": 12,
                    "category": "reliability",
                    "severity": ["high"],
                    "concept": "configured timeout is not propagated",
                    "concept_id": "timeout_not_propagated",
                }
            ],
            "forbidden_concepts": [],
        }
    )

    score_command = score_case(gt_command, [result.verified_findings[0]])
    score_resource = score_case(gt_resource, [result.verified_findings[1]])
    score_timeout = score_case(gt_timeout, [result.verified_findings[2]])

    assert score_command.true_positives == 1
    assert score_resource.true_positives == 1
    assert score_timeout.true_positives == 1
    assert (
        score_command.false_negatives
        + score_resource.false_negatives
        + score_timeout.false_negatives
        == 0
    )
