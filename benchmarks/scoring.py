from __future__ import annotations

import re
from dataclasses import dataclass

from reviewer.models import Finding

from .schema import CaseGroundTruth, ExpectedFinding

PARSE_ERROR_TYPES = {"InvalidJSONError", "SchemaValidationError", "ReviewParseError"}

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "any",
    "api",
    "as",
    "be",
    "by",
    "code",
    "can",
    "could",
    "configured",
    "concrete",
    "defect",
    "file",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "line",
    "function",
    "method",
    "missing",
    "not",
    "of",
    "on",
    "or",
    "the",
    "to",
    "value",
    "variable",
    "with",
}

_CATEGORY_EQUIVALENTS: dict[str, set[str]] = {
    "bug": {"bug", "reliability"},
    "reliability": {"bug", "reliability"},
    "security": {"security"},
}

_SPECIAL_CONCEPT_TERMS = {
    "sql_injection",
    "command_injection",
    "resource_leak",
}

_CONCEPT_SYNONYMS = {
    "propagated": "propagate",
    "propagation": "propagate",
    "forwarded": "propagate",
    "forwarding": "propagate",
    "forwards": "propagate",
    "ignored": "ignore",
    "ignores": "ignore",
    "leaking": "leak",
    "leaks": "leak",
}


_CONCEPT_ID_GROUPS: dict[str, list[set[str]]] = {
    "command_injection": [
        {"shell", "shell_true", "subprocess", "command"},
        {
            "command_injection",
            "shell_injection",
            "untrusted_input",
            "user_supplied_input",
            "malicious_input",
            "arbitrary_code_execution",
            "injection",
            "untrusted",
            "malicious",
            "arbitrary",
        },
    ],
    "resource_leak": [
        {"resource", "file_handle", "handle", "socket", "connection", "descriptor"},
        {
            "resource_leak",
            "leak",
            "not_closed",
            "unclosed",
            "early_return",
            "resource_exhaustion",
            "exhaustion",
        },
    ],
    "timeout_not_propagated": [
        {"timeout", "configured_timeout"},
        {
            "not_used",
            "not_passed",
            "not_propagated",
            "not_forwarded",
            "ignored",
            "ignore",
            "unused",
        },
    ],
}


@dataclass(slots=True)
class CaseScore:
    case_id: str
    clean: bool
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_pairs: list[tuple[int, int]]
    unmatched_finding_indexes: list[int]
    unmatched_expected_indexes: list[int]
    forbidden_hallucinations: list[str]


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def normalize_concept_keywords(text: str) -> set[str]:
    normalized = text.lower()
    normalized = normalized.replace("sql injection", "sql_injection")
    normalized = normalized.replace("command injection", "command_injection")
    normalized = normalized.replace("shell injection", "shell_injection")
    normalized = normalized.replace("resource leak", "resource_leak")
    normalized = normalized.replace("resource exhaustion", "resource_exhaustion")
    normalized = normalized.replace("file handle", "file_handle")
    normalized = normalized.replace("not closed", "not_closed")
    normalized = normalized.replace("early return", "early_return")
    normalized = normalized.replace("untrusted input", "untrusted_input")
    normalized = normalized.replace("user supplied input", "user_supplied_input")
    normalized = normalized.replace("malicious input", "malicious_input")
    normalized = normalized.replace("arbitrary code execution", "arbitrary_code_execution")
    normalized = normalized.replace("shell=true", "shell_true")
    normalized = normalized.replace("configured timeout", "configured_timeout")
    normalized = normalized.replace("not used", "not_used")
    normalized = normalized.replace("not passed", "not_passed")
    normalized = normalized.replace("not propagated", "not_propagated")
    normalized = normalized.replace("not forwarded", "not_forwarded")
    normalized = normalized.replace("timeout", "timeout")
    tokens = re.findall(r"[a-z0-9_]+", normalized)
    normalized_tokens = [_CONCEPT_SYNONYMS.get(token, token) for token in tokens]
    return {token for token in normalized_tokens if token not in _STOP_WORDS and len(token) >= 3}


def _line_match(line: int, line_start: int, line_end: int, tolerance: int) -> bool:
    return (line_start - tolerance) <= line <= (line_end + tolerance)


def _normalize_category(value: str) -> str:
    return value.strip().lower()


def _category_match(expected: ExpectedFinding, finding: Finding) -> bool:
    expected_category = _normalize_category(expected.category)
    finding_category = _normalize_category(finding.category)
    compatible_categories = _CATEGORY_EQUIVALENTS.get(expected_category, {expected_category})
    return finding_category in compatible_categories


def _concept_match(expected: ExpectedFinding, finding: Finding) -> bool:
    finding_text = " ".join(
        [
            finding.category,
            finding.title,
            finding.evidence,
            finding.consequence,
            finding.suggestion,
        ]
    )
    finding_terms = normalize_concept_keywords(finding_text)

    expected_concept_id = (expected.concept_id or "").strip().lower()
    if expected_concept_id:
        groups = _CONCEPT_ID_GROUPS.get(expected_concept_id)
        if groups:
            return all(bool(group & finding_terms) for group in groups)

    expected_terms = normalize_concept_keywords(expected.concept)
    if not expected_terms:
        return True

    overlap = expected_terms & finding_terms

    if overlap & _SPECIAL_CONCEPT_TERMS:
        return True

    required_overlap = 1 if len(expected_terms) == 1 else 2
    return len(overlap) >= required_overlap


def finding_matches_expected(
    expected: ExpectedFinding,
    finding: Finding,
    line_tolerance: int = 5,
) -> bool:
    if finding.file != expected.file:
        return False
    if not _line_match(finding.line, expected.line_start, expected.line_end, line_tolerance):
        return False
    if not _category_match(expected, finding):
        return False
    if not _concept_match(expected, finding):
        return False
    return True


def detect_forbidden_concepts(ground_truth: CaseGroundTruth, findings: list[Finding]) -> list[str]:
    hits: list[str] = []
    if not ground_truth.clean:
        return hits

    for finding in findings:
        haystack = " ".join(
            [
                finding.category,
                finding.title,
                finding.evidence,
                finding.consequence,
                finding.suggestion,
            ]
        ).lower()
        for concept in ground_truth.forbidden_concepts:
            if concept.lower() in haystack:
                hits.append(f"{finding.id}:{concept}")
    return hits


def score_case(
    ground_truth: CaseGroundTruth,
    published_findings: list[Finding],
    line_tolerance: int = 5,
) -> CaseScore:
    matched_pairs: list[tuple[int, int]] = []
    used_finding_indexes: set[int] = set()

    for expected_index, expected in enumerate(ground_truth.expected_findings):
        for finding_index, finding in enumerate(published_findings):
            if finding_index in used_finding_indexes:
                continue
            if finding_matches_expected(expected, finding, line_tolerance=line_tolerance):
                matched_pairs.append((expected_index, finding_index))
                used_finding_indexes.add(finding_index)
                break

    unmatched_expected_indexes = [
        index
        for index in range(len(ground_truth.expected_findings))
        if index not in {expected_index for expected_index, _ in matched_pairs}
    ]
    unmatched_finding_indexes = [
        index for index in range(len(published_findings)) if index not in used_finding_indexes
    ]

    forbidden_hallucinations = detect_forbidden_concepts(ground_truth, published_findings)

    return CaseScore(
        case_id=ground_truth.case_id,
        clean=ground_truth.clean,
        true_positives=len(matched_pairs),
        false_positives=len(unmatched_finding_indexes),
        false_negatives=len(unmatched_expected_indexes),
        matched_pairs=matched_pairs,
        unmatched_finding_indexes=unmatched_finding_indexes,
        unmatched_expected_indexes=unmatched_expected_indexes,
        forbidden_hallucinations=forbidden_hallucinations,
    )
