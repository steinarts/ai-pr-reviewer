from __future__ import annotations

import json

import pytest

from reviewer.models import FindingsPayload, ReviewContext
from reviewer.scouts import (
    InvalidJSONError,
    ReviewParseError,
    SchemaValidationError,
    _build_prompts,
    _parse_findings,
)


def test_build_prompts_include_test_safety_rules() -> None:
    context = ReviewContext(base="main", head="HEAD", diff_text="", file_contexts={})
    system_prompt, user_prompt = _build_prompts("reliability", "template", context)

    assert "Tests describe expected behavior" in system_prompt
    assert "pytest.raises(...) is expected to raise" in system_prompt
    assert "Do not summarize test names" in system_prompt
    assert "Return no finding rather than speculate" in system_prompt
    assert "Confidence must be between 0.0 and 1.0" in system_prompt
    assert "REVIEWER: reliability" in user_prompt


def test_parse_rejects_confidence_out_of_range() -> None:
    bad_payload = {
        "findings": [
            {
                "id": "x1",
                "file": "tests/test_ollama_client.py",
                "line": 30,
                "category": "error",
                "severity": "high",
                "confidence": 100.0,
                "title": "Bad confidence",
                "evidence": "example",
                "consequence": "example",
                "suggestion": "example",
                "introduced_by_diff": True,
                "actionable": True,
                "style_only": False,
                "duplicate_of": None,
                "reviewer": "bug",
                "status": "proposed",
                "rejection_reason": "",
            }
        ]
    }

    with pytest.raises(SchemaValidationError, match="Invalid JSON output"):
        _parse_findings(json.dumps(bad_payload), reviewer="bug")


def test_parse_accepts_confidence_between_0_and_1() -> None:
    payload = {
        "findings": [
            {
                "id": "x2",
                "file": "app.py",
                "line": 10,
                "category": "bug",
                "severity": "high",
                "confidence": 0.91,
                "title": "Concrete issue",
                "evidence": "e",
                "consequence": "c",
                "suggestion": "s",
                "introduced_by_diff": True,
                "actionable": True,
                "style_only": False,
                "duplicate_of": None,
                "reviewer": "bug",
                "status": "proposed",
                "rejection_reason": "",
            }
        ]
    }

    findings = _parse_findings(json.dumps(payload), reviewer="bug")
    assert len(findings) == 1
    assert 0.0 <= findings[0].confidence <= 1.0


def test_no_finding_example_for_pytest_raises_is_valid_empty_payload() -> None:
    # Regression example: this should lead to no findings, not a fabricated bug.
    snippet = """
with pytest.raises(ValueError):
    create_llm_client(provider="ollama", model="")
"""
    assert "pytest.raises" in snippet

    payload = FindingsPayload.model_validate_json('{"findings": []}')
    assert payload.findings == []


def test_parse_invalid_json_uses_invalid_json_error() -> None:
    with pytest.raises(InvalidJSONError, match="Invalid JSON output"):
        _parse_findings("{not-json", reviewer="bug")


def test_parse_validation_error_is_review_parse_error_subclass() -> None:
    bad_payload = {
        "findings": [
            {
                "id": "x3",
                "file": "app.py",
                "line": 1,
                "category": "bug",
                "severity": "high",
                "confidence": "not-a-number",
                "title": "Wrong type",
                "evidence": "e",
                "consequence": "c",
                "suggestion": "s",
                "introduced_by_diff": True,
                "actionable": True,
                "style_only": False,
                "duplicate_of": None,
                "reviewer": "bug",
                "status": "proposed",
                "rejection_reason": "",
            }
        ]
    }

    with pytest.raises(ReviewParseError):
        _parse_findings(json.dumps(bad_payload), reviewer="bug")
