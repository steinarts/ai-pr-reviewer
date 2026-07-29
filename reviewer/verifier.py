from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

from .llm_client import LLMClient
from .models import Finding, Status, VerificationStatus

_ALLOWED_VERDICTS = {"valid", "invalid", "uncertain"}
_DEFAULT_CONTEXT_LINES = 14


class VerificationParseError(ValueError):
    """Raised when verifier output cannot be parsed as valid verification JSON."""


@dataclass(slots=True)
class VerificationContext:
    base: str
    head: str
    diff_text: str
    file_contexts: dict[str, str]
    changed_lines_by_file: dict[str, set[int]]
    provider: str
    review_model: str
    verification_model: str
    timeout_seconds: float
    total_budget_seconds: float
    max_findings: int
    min_confidence: float
    fail_policy: str = "unverified"
    uncertain_policy: str = "unverified"
    context_lines: int = _DEFAULT_CONTEXT_LINES


@dataclass(slots=True)
class VerificationResult:
    verified_findings: list[Finding]
    verification_rejected_findings: list[Finding]
    completed_requests: int
    failed_requests: int
    skipped_requests: int
    elapsed_seconds: float
    valid_count: int
    invalid_count: int
    uncertain_count: int
    unverified_count: int
    skipped_count: int
    debug_events: list[dict[str, object]] = field(default_factory=list)


@dataclass(slots=True)
class VerificationSlice:
    file: str
    line: int
    diff_excerpt: str
    found_line_in_diff: bool
    context_line_start: int | None
    context_line_end: int | None


class FindingVerifier(Protocol):
    def verify(
        self,
        findings: list[Finding],
        context: VerificationContext,
    ) -> VerificationResult:
        ...


def _parse_verification_payload(raw_output: str) -> tuple[str, float, str, list[int]]:
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise VerificationParseError("Verifier output is not valid JSON") from exc

    if not isinstance(parsed, dict):
        raise VerificationParseError("Verifier output must be a JSON object")

    verdict = parsed.get("verdict")
    if verdict not in _ALLOWED_VERDICTS:
        raise VerificationParseError("Verifier output must contain verdict=valid|invalid|uncertain")

    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)):
        raise VerificationParseError("Verifier confidence must be a number")
    confidence = float(confidence)
    if confidence < 0.0 or confidence > 1.0:
        raise VerificationParseError("Verifier confidence must be in [0, 1]")

    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise VerificationParseError("Verifier reason must be a non-empty string")

    evidence_lines = parsed.get("evidence_lines")
    if not isinstance(evidence_lines, list) or any(not isinstance(item, int) for item in evidence_lines):
        raise VerificationParseError("Verifier evidence_lines must be a list[int]")

    return verdict, confidence, reason.strip(), evidence_lines


def _extract_file_sections(diff_text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_file: str | None = None
    current_lines: list[str] = []

    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current_file is not None and current_lines:
                sections.setdefault(current_file, []).append("".join(current_lines))
            current_file = None
            current_lines = [line]
            continue

        current_lines.append(line)
        if line.startswith("+++ b/"):
            current_file = line.removeprefix("+++ b/").strip()

    if current_file is not None and current_lines:
        sections.setdefault(current_file, []).append("".join(current_lines))

    return sections


def _hunk_contains_line(hunk_header: str, line: int, context_lines: int) -> bool:
    match = re.search(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", hunk_header)
    if not match:
        return False

    new_start = int(match.group(1))
    new_count = int(match.group(2) or "1")
    new_end = new_start + max(0, new_count - 1)
    return (new_start - context_lines) <= line <= (new_end + context_lines)


def _split_hunks(section_text: str) -> list[tuple[str, list[str]]]:
    lines = section_text.splitlines(keepends=True)
    hunk_starts = [idx for idx, value in enumerate(lines) if value.startswith("@@")]
    if not hunk_starts:
        return []

    hunks: list[tuple[str, list[str]]] = []
    for idx, start in enumerate(hunk_starts):
        end = hunk_starts[idx + 1] if idx + 1 < len(hunk_starts) else len(lines)
        header = lines[start]
        body = lines[start + 1 : end]
        hunks.append((header, body))
    return hunks


def build_verification_context(
    diff_text: str,
    *,
    file: str,
    line: int,
    context_lines: int = _DEFAULT_CONTEXT_LINES,
) -> VerificationSlice:
    sections = _extract_file_sections(diff_text)
    file_sections = sections.get(file, [])
    if not file_sections:
        return VerificationSlice(
            file=file,
            line=line,
            diff_excerpt="",
            found_line_in_diff=False,
            context_line_start=None,
            context_line_end=None,
        )

    best_excerpt_parts: list[str] = []
    found_line_in_diff = False
    global_start: int | None = None
    global_end: int | None = None

    for section_text in file_sections:
        hunks = _split_hunks(section_text)
        for header, body in hunks:
            if not _hunk_contains_line(header, line, context_lines):
                continue

            rendered: list[tuple[int | None, str]] = []
            match = re.search(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", header)
            if not match:
                continue
            new_line = int(match.group(1))

            for raw in body:
                if raw.startswith("+") and not raw.startswith("+++"):
                    rendered.append((new_line, raw))
                    if new_line == line:
                        found_line_in_diff = True
                    new_line += 1
                elif raw.startswith("-") and not raw.startswith("---"):
                    rendered.append((None, raw))
                else:
                    rendered.append((new_line, raw))
                    if new_line == line:
                        found_line_in_diff = True
                    new_line += 1

            near_indexes = [
                idx
                for idx, (line_no, _raw) in enumerate(rendered)
                if line_no is not None and abs(line_no - line) <= context_lines
            ]
            if near_indexes:
                start = max(0, min(near_indexes) - context_lines)
                end = min(len(rendered), max(near_indexes) + context_lines + 1)
            else:
                start = 0
                end = len(rendered)

            line_nos = [line_no for line_no, _raw in rendered[start:end] if line_no is not None]
            if line_nos:
                local_start = min(line_nos)
                local_end = max(line_nos)
                global_start = local_start if global_start is None else min(global_start, local_start)
                global_end = local_end if global_end is None else max(global_end, local_end)

            excerpt_lines = [header, *(raw for _line_no, raw in rendered[start:end])]
            best_excerpt_parts.append("".join(excerpt_lines))

    if not best_excerpt_parts:
        return VerificationSlice(
            file=file,
            line=line,
            diff_excerpt="",
            found_line_in_diff=found_line_in_diff,
            context_line_start=None,
            context_line_end=None,
        )

    return VerificationSlice(
        file=file,
        line=line,
        diff_excerpt="\n".join(best_excerpt_parts),
        found_line_in_diff=found_line_in_diff,
        context_line_start=global_start,
        context_line_end=global_end,
    )


def _build_verification_prompts(
    finding: Finding,
    context: VerificationContext,
    slice_context: VerificationSlice,
) -> tuple[str, str]:
    system_prompt = (
        "You are verifying a candidate code review finding.\n"
        "Do not discover or report new issues.\n"
        "Your task is to decide whether the core defect claim is contradicted, supported, or uncertain.\n"
        "Focus on defect existence only.\n"
        "Do not reject just because title/severity/consequence/suggestion wording is imperfect.\n"
        "Reject the finding only if the provided code clearly contradicts the candidate claim.\n"
        "If the claim may be correct but the available context is insufficient, return uncertain.\n"
        "False positives are worse than false negatives.\n"
        "Use invalid only when there is concrete contradictory evidence.\n"
        "Return JSON only with this exact schema:"
        ' {"verdict":"valid|invalid|uncertain","confidence":0.0,"reason":"...","evidence_lines":[1,2]}\n'
    )

    file_context = context.file_contexts.get(finding.file, "")
    user_prompt = (
        f"BASE: {context.base}\n"
        f"HEAD: {context.head}\n"
        f"PROVIDER: {context.provider}\n"
        f"REVIEW_MODEL: {context.review_model}\n"
        f"VERIFICATION_MODEL: {context.verification_model}\n"
        "CANDIDATE_FINDING:\n"
        f"{json.dumps(finding.model_dump(mode='json'), ensure_ascii=False)}\n\n"
        f"TARGET_FILE: {finding.file}\n"
        f"TARGET_LINE: {finding.line}\n"
        f"LINE_PRESENT_IN_DIFF: {slice_context.found_line_in_diff}\n\n"
        f"DIFF_HUNK_CONTEXT:\n{slice_context.diff_excerpt}\n\n"
        f"FILE_CONTEXT:\n{file_context}\n"
    )
    return system_prompt, user_prompt


def _verification_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["valid", "invalid", "uncertain"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": {"type": "string"},
            "evidence_lines": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": ["verdict", "confidence", "reason", "evidence_lines"],
        "additionalProperties": False,
    }


def _is_eligible_for_verification(finding: Finding, changed_lines_by_file: dict[str, set[int]]) -> tuple[bool, str]:
    if finding.file not in changed_lines_by_file:
        return False, "invalid_file"
    changed_lines = changed_lines_by_file[finding.file]
    if changed_lines and finding.line not in changed_lines:
        return False, "invalid_line"
    if finding.style_only:
        return False, "style_only"
    if not finding.introduced_by_diff:
        return False, "not_introduced_by_diff"
    if not finding.evidence.strip():
        return False, "missing_evidence"
    if not finding.consequence.strip():
        return False, "missing_consequence"
    if not finding.suggestion.strip():
        return False, "missing_suggestion"
    return True, ""


def _reason_has_clear_contradiction(reason: str) -> bool:
    normalized = reason.strip().lower()
    contradiction_markers = (
        "clearly false",
        "contradicts",
        "contradiction",
        "already passed",
        "already passes",
        "passes the timeout argument",
        "already uses",
        "already closes",
        "already handled",
        "already validates",
        "is already",
        "not true",
        "does not happen",
        "no such issue",
    )
    non_contradiction_markers = (
        "does not contradict",
        "may be correct",
        "insufficient context",
        "cannot determine",
        "uncertain",
    )
    if any(marker in normalized for marker in non_contradiction_markers):
        return False
    return any(marker in normalized for marker in contradiction_markers)


def _with_verification_metadata(
    finding: Finding,
    *,
    status: VerificationStatus,
    verdict: str,
    confidence: float | None,
    reason: str,
    evidence_lines: list[int],
    model: str,
    elapsed_ms: int | None,
    context_line_start: int | None,
    context_line_end: int | None,
    line_in_context: bool | None,
    prompt_chars: int | None,
    response_text: str,
) -> Finding:
    updated = finding.model_copy(deep=True)
    updated.verification_status = status
    updated.verification_verdict = verdict
    updated.verification_confidence = confidence
    updated.verification_reason = reason
    updated.verification_evidence_lines = evidence_lines
    updated.verification_model = model
    updated.verification_elapsed_ms = elapsed_ms
    updated.verification_context_line_start = context_line_start
    updated.verification_context_line_end = context_line_end
    updated.verification_line_in_context = line_in_context
    updated.verification_prompt_chars = prompt_chars
    updated.verification_response_text = response_text
    return updated


class LLMFindingVerifier:
    def __init__(
        self,
        llm_client: LLMClient,
        *,
        debug_sink: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._debug_sink = debug_sink

    def _emit(self, event: dict[str, object], events: list[dict[str, object]]) -> None:
        events.append(event)
        if self._debug_sink is not None:
            self._debug_sink(event)

    def _invoke(self, system_prompt: str, user_prompt: str, timeout_seconds: float) -> str:
        generate_hard_timeout = getattr(self._llm_client, "generate_hard_timeout", None)
        if callable(generate_hard_timeout):
            return generate_hard_timeout(
                system_prompt,
                user_prompt,
                response_schema=_verification_schema(),
                timeout_seconds=timeout_seconds,
            )
        return self._llm_client.generate(
            system_prompt,
            user_prompt,
            response_schema=_verification_schema(),
        )

    def verify(
        self,
        findings: list[Finding],
        context: VerificationContext,
    ) -> VerificationResult:
        overall_start = perf_counter()
        events: list[dict[str, object]] = []
        verified_findings: list[Finding] = []
        verification_rejected_findings: list[Finding] = []

        completed_requests = 0
        failed_requests = 0
        skipped_requests = 0
        valid_count = 0
        invalid_count = 0
        uncertain_count = 0
        unverified_count = 0
        skipped_count = 0

        eligible: list[Finding] = []
        for finding in findings:
            is_eligible, reason = _is_eligible_for_verification(finding, context.changed_lines_by_file)
            if is_eligible:
                eligible.append(finding)
                continue

            skipped = _with_verification_metadata(
                finding,
                status=VerificationStatus.SKIPPED,
                verdict="",
                confidence=None,
                reason=f"verification skipped: {reason}",
                evidence_lines=[],
                model=context.verification_model,
                elapsed_ms=0,
                context_line_start=None,
                context_line_end=None,
                line_in_context=None,
                prompt_chars=None,
                response_text="",
            )
            verified_findings.append(skipped)
            skipped_requests += 1
            skipped_count += 1
            self._emit(
                {
                    "event": "verification_skipped",
                    "finding_id": finding.id,
                    "file": finding.file,
                    "line": finding.line,
                    "reason": reason,
                },
                events,
            )

        planned_to_verify = eligible[: context.max_findings]
        overflow = eligible[context.max_findings :]
        for finding in overflow:
            skipped = _with_verification_metadata(
                finding,
                status=VerificationStatus.SKIPPED,
                verdict="",
                confidence=None,
                reason="verification skipped: max_findings_limit",
                evidence_lines=[],
                model=context.verification_model,
                elapsed_ms=0,
                context_line_start=None,
                context_line_end=None,
                line_in_context=None,
                prompt_chars=None,
                response_text="",
            )
            verified_findings.append(skipped)
            skipped_requests += 1
            skipped_count += 1
            self._emit(
                {
                    "event": "verification_skipped",
                    "finding_id": finding.id,
                    "file": finding.file,
                    "line": finding.line,
                    "reason": "max_findings_limit",
                },
                events,
            )

        for index, finding in enumerate(planned_to_verify, start=1):
            elapsed_total = perf_counter() - overall_start
            remaining = context.total_budget_seconds - elapsed_total
            if remaining <= 0:
                pending = planned_to_verify[index - 1 :]
                self._emit(
                    {
                        "event": "verification_budget_exhausted",
                        "remaining_findings": len(pending),
                        "verification_elapsed_seconds": elapsed_total,
                    },
                    events,
                )
                for tail in pending:
                    skipped = _with_verification_metadata(
                        tail,
                        status=VerificationStatus.SKIPPED,
                        verdict="",
                        confidence=None,
                        reason="verification skipped: total_budget_exhausted",
                        evidence_lines=[],
                        model=context.verification_model,
                        elapsed_ms=0,
                        context_line_start=None,
                        context_line_end=None,
                        line_in_context=None,
                        prompt_chars=None,
                        response_text="",
                    )
                    verified_findings.append(skipped)
                    skipped_requests += 1
                    skipped_count += 1
                    self._emit(
                        {
                            "event": "verification_skipped",
                            "finding_id": tail.id,
                            "file": tail.file,
                            "line": tail.line,
                            "reason": "total_budget_exhausted",
                        },
                        events,
                    )
                break

            slice_context = build_verification_context(
                context.diff_text,
                file=finding.file,
                line=finding.line,
                context_lines=context.context_lines,
            )
            if not slice_context.found_line_in_diff:
                self._emit(
                    {
                        "event": "verification_skipped",
                        "finding_id": finding.id,
                        "file": finding.file,
                        "line": finding.line,
                        "reason": "line_not_found_in_diff_hunk",
                    },
                    events,
                )

            system_prompt, user_prompt = _build_verification_prompts(finding, context, slice_context)
            prompt_chars = len(system_prompt) + len(user_prompt)

            started = perf_counter()
            self._emit(
                {
                    "event": "verification_start",
                    "finding_id": finding.id,
                    "file": finding.file,
                    "line": finding.line,
                    "request_index": index,
                    "request_count": len(planned_to_verify),
                    "remaining_budget_seconds": max(0.0, remaining),
                    "context_line_start": slice_context.context_line_start,
                    "context_line_end": slice_context.context_line_end,
                    "line_in_context": slice_context.found_line_in_diff,
                    "diff_excerpt": slice_context.diff_excerpt,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                },
                events,
            )

            try:
                raw = self._invoke(
                    system_prompt,
                    user_prompt,
                    timeout_seconds=min(context.timeout_seconds, remaining),
                )
                verdict, confidence, reason, evidence_lines = _parse_verification_payload(raw)
                elapsed_ms = int((perf_counter() - started) * 1000)
                completed_requests += 1

                if verdict == "valid" and confidence >= context.min_confidence:
                    valid = _with_verification_metadata(
                        finding,
                        status=VerificationStatus.VALID,
                        verdict=verdict,
                        confidence=confidence,
                        reason=reason,
                        evidence_lines=evidence_lines,
                        model=context.verification_model,
                        elapsed_ms=elapsed_ms,
                        context_line_start=slice_context.context_line_start,
                        context_line_end=slice_context.context_line_end,
                        line_in_context=slice_context.found_line_in_diff,
                        prompt_chars=prompt_chars,
                        response_text=raw,
                    )
                    verified_findings.append(valid)
                    valid_count += 1
                    self._emit(
                        {
                            "event": "verification_complete",
                            "finding_id": finding.id,
                            "file": finding.file,
                            "line": finding.line,
                            "verdict": "valid",
                            "confidence": confidence,
                            "elapsed_ms": elapsed_ms,
                        },
                        events,
                    )
                    continue

                invalid_has_contradiction = _reason_has_clear_contradiction(reason)

                if verdict == "invalid" and confidence >= context.min_confidence and invalid_has_contradiction:
                    invalid = _with_verification_metadata(
                        finding,
                        status=VerificationStatus.INVALID,
                        verdict=verdict,
                        confidence=confidence,
                        reason=reason,
                        evidence_lines=evidence_lines,
                        model=context.verification_model,
                        elapsed_ms=elapsed_ms,
                        context_line_start=slice_context.context_line_start,
                        context_line_end=slice_context.context_line_end,
                        line_in_context=slice_context.found_line_in_diff,
                        prompt_chars=prompt_chars,
                        response_text=raw,
                    )
                    invalid.status = Status.REJECTED
                    invalid.rejection_reason = "verification_failed"
                    verification_rejected_findings.append(invalid)
                    invalid_count += 1
                    self._emit(
                        {
                            "event": "verification_rejected",
                            "finding_id": finding.id,
                            "file": finding.file,
                            "line": finding.line,
                            "verdict": verdict,
                            "confidence": confidence,
                            "elapsed_ms": elapsed_ms,
                            "reason": "verification_failed",
                            "context_line_start": slice_context.context_line_start,
                            "context_line_end": slice_context.context_line_end,
                            "line_in_context": slice_context.found_line_in_diff,
                            "prompt_chars": prompt_chars,
                            "response_text": raw,
                        },
                        events,
                    )
                    continue

                if verdict == "invalid" and confidence >= context.min_confidence and not invalid_has_contradiction:
                    reason = (
                        "Invalid verdict downgraded to uncertain because reason did not "
                        f"show clear contradiction: {reason}"
                    )
                    verdict = "uncertain"

                uncertain_reason = reason
                if confidence < context.min_confidence:
                    uncertain_reason = (
                        f"Low verification confidence {confidence:.2f} (< {context.min_confidence:.2f}): "
                        f"{reason}"
                    )

                if context.uncertain_policy == "reject":
                    uncertain_rejected = _with_verification_metadata(
                        finding,
                        status=VerificationStatus.INVALID,
                        verdict="uncertain",
                        confidence=confidence,
                        reason=uncertain_reason,
                        evidence_lines=evidence_lines,
                        model=context.verification_model,
                        elapsed_ms=elapsed_ms,
                        context_line_start=slice_context.context_line_start,
                        context_line_end=slice_context.context_line_end,
                        line_in_context=slice_context.found_line_in_diff,
                        prompt_chars=prompt_chars,
                        response_text=raw,
                    )
                    uncertain_rejected.status = Status.REJECTED
                    uncertain_rejected.rejection_reason = "verification_uncertain_rejected"
                    verification_rejected_findings.append(uncertain_rejected)
                    invalid_count += 1
                else:
                    uncertain_item = _with_verification_metadata(
                        finding,
                        status=VerificationStatus.UNVERIFIED,
                        verdict="uncertain",
                        confidence=confidence,
                        reason=uncertain_reason,
                        evidence_lines=evidence_lines,
                        model=context.verification_model,
                        elapsed_ms=elapsed_ms,
                        context_line_start=slice_context.context_line_start,
                        context_line_end=slice_context.context_line_end,
                        line_in_context=slice_context.found_line_in_diff,
                        prompt_chars=prompt_chars,
                        response_text=raw,
                    )
                    verified_findings.append(uncertain_item)
                    unverified_count += 1
                uncertain_count += 1
                self._emit(
                    {
                        "event": "verification_uncertain",
                        "finding_id": finding.id,
                        "file": finding.file,
                        "line": finding.line,
                        "verdict": verdict,
                        "confidence": confidence,
                        "elapsed_ms": elapsed_ms,
                        "reason": uncertain_reason,
                    },
                    events,
                )
            except TimeoutError as exc:
                failed_requests += 1
                elapsed_ms = int((perf_counter() - started) * 1000)
                self._emit(
                    {
                        "event": "verification_timeout",
                        "finding_id": finding.id,
                        "file": finding.file,
                        "line": finding.line,
                        "elapsed_ms": elapsed_ms,
                        "error": str(exc),
                    },
                    events,
                )
                if context.fail_policy == "reject":
                    rejected = _with_verification_metadata(
                        finding,
                        status=VerificationStatus.INVALID,
                        verdict="",
                        confidence=None,
                        reason=f"Verifier timeout: {exc}",
                        evidence_lines=[],
                        model=context.verification_model,
                        elapsed_ms=elapsed_ms,
                        context_line_start=slice_context.context_line_start,
                        context_line_end=slice_context.context_line_end,
                        line_in_context=slice_context.found_line_in_diff,
                        prompt_chars=prompt_chars,
                        response_text="",
                    )
                    rejected.status = Status.REJECTED
                    rejected.rejection_reason = "verification_timeout"
                    verification_rejected_findings.append(rejected)
                    invalid_count += 1
                else:
                    unverified = _with_verification_metadata(
                        finding,
                        status=VerificationStatus.UNVERIFIED,
                        verdict="uncertain",
                        confidence=None,
                        reason=f"Verifier timeout: {exc}",
                        evidence_lines=[],
                        model=context.verification_model,
                        elapsed_ms=elapsed_ms,
                        context_line_start=slice_context.context_line_start,
                        context_line_end=slice_context.context_line_end,
                        line_in_context=slice_context.found_line_in_diff,
                        prompt_chars=prompt_chars,
                        response_text="",
                    )
                    verified_findings.append(unverified)
                    uncertain_count += 1
                    unverified_count += 1
            except (ConnectionError, ValueError, VerificationParseError) as exc:
                failed_requests += 1
                elapsed_ms = int((perf_counter() - started) * 1000)
                self._emit(
                    {
                        "event": "verification_parse_failure",
                        "finding_id": finding.id,
                        "file": finding.file,
                        "line": finding.line,
                        "elapsed_ms": elapsed_ms,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                    events,
                )
                if context.fail_policy == "reject":
                    rejected = _with_verification_metadata(
                        finding,
                        status=VerificationStatus.INVALID,
                        verdict="",
                        confidence=None,
                        reason=f"Verifier failure: {exc}",
                        evidence_lines=[],
                        model=context.verification_model,
                        elapsed_ms=elapsed_ms,
                        context_line_start=slice_context.context_line_start,
                        context_line_end=slice_context.context_line_end,
                        line_in_context=slice_context.found_line_in_diff,
                        prompt_chars=prompt_chars,
                        response_text="",
                    )
                    rejected.status = Status.REJECTED
                    rejected.rejection_reason = "verification_failed"
                    verification_rejected_findings.append(rejected)
                    invalid_count += 1
                else:
                    unverified = _with_verification_metadata(
                        finding,
                        status=VerificationStatus.UNVERIFIED,
                        verdict="uncertain",
                        confidence=None,
                        reason=f"Verifier failure: {exc}",
                        evidence_lines=[],
                        model=context.verification_model,
                        elapsed_ms=elapsed_ms,
                        context_line_start=slice_context.context_line_start,
                        context_line_end=slice_context.context_line_end,
                        line_in_context=slice_context.found_line_in_diff,
                        prompt_chars=prompt_chars,
                        response_text="",
                    )
                    verified_findings.append(unverified)
                    uncertain_count += 1
                    unverified_count += 1

        return VerificationResult(
            verified_findings=verified_findings,
            verification_rejected_findings=verification_rejected_findings,
            completed_requests=completed_requests,
            failed_requests=failed_requests,
            skipped_requests=skipped_requests,
            elapsed_seconds=perf_counter() - overall_start,
            valid_count=valid_count,
            invalid_count=invalid_count,
            uncertain_count=uncertain_count,
            unverified_count=unverified_count,
            skipped_count=skipped_count,
            debug_events=events,
        )


class FakeFindingVerifier:
    """Deterministic verifier used by tests and offline benchmarking."""

    def __init__(
        self,
        *,
        mode: str = "approve_all",
        valid_ids: set[str] | None = None,
        parse_error_ids: set[str] | None = None,
        timeout_ids: set[str] | None = None,
        budget_exhaust_after: int | None = None,
    ) -> None:
        self.mode = mode
        self.valid_ids = valid_ids or set()
        self.parse_error_ids = parse_error_ids or set()
        self.timeout_ids = timeout_ids or set()
        self.budget_exhaust_after = budget_exhaust_after

    def _decide(self, finding: Finding) -> tuple[str, float, str, list[int]]:
        if finding.id in self.parse_error_ids:
            raise VerificationParseError("synthetic parse error")
        if finding.id in self.timeout_ids:
            raise TimeoutError("synthetic timeout")

        if self.mode == "reject_all":
            return "invalid", 0.95, "Fake verifier rejected finding.", [finding.line]
        if self.mode == "uncertain_all":
            return "uncertain", 0.65, "Fake verifier is uncertain.", [finding.line]
        if self.mode == "approve_ids":
            if finding.id in self.valid_ids:
                return "valid", 0.96, "Fake verifier approved this finding id.", [finding.line]
            return "invalid", 0.94, "Fake verifier rejected non-whitelisted finding id.", [finding.line]
        return "valid", 0.96, "Fake verifier approved finding.", [finding.line]

    def verify(
        self,
        findings: list[Finding],
        context: VerificationContext,
    ) -> VerificationResult:
        start = perf_counter()
        verified_findings: list[Finding] = []
        verification_rejected_findings: list[Finding] = []
        events: list[dict[str, object]] = []

        completed_requests = 0
        failed_requests = 0
        skipped_requests = 0
        valid_count = 0
        invalid_count = 0
        uncertain_count = 0
        unverified_count = 0
        skipped_count = 0

        for index, finding in enumerate(findings):
            if index >= context.max_findings:
                skipped = _with_verification_metadata(
                    finding,
                    status=VerificationStatus.SKIPPED,
                    verdict="",
                    confidence=None,
                    reason="verification skipped: max_findings_limit",
                    evidence_lines=[],
                    model=context.verification_model,
                    elapsed_ms=0,
                    context_line_start=None,
                    context_line_end=None,
                    line_in_context=None,
                    prompt_chars=None,
                    response_text="",
                )
                verified_findings.append(skipped)
                skipped_requests += 1
                skipped_count += 1
                events.append(
                    {
                        "event": "verification_skipped",
                        "finding_id": finding.id,
                        "reason": "max_findings_limit",
                    }
                )
                continue

            if self.budget_exhaust_after is not None and completed_requests >= self.budget_exhaust_after:
                skipped = _with_verification_metadata(
                    finding,
                    status=VerificationStatus.SKIPPED,
                    verdict="",
                    confidence=None,
                    reason="verification skipped: total_budget_exhausted",
                    evidence_lines=[],
                    model=context.verification_model,
                    elapsed_ms=0,
                    context_line_start=None,
                    context_line_end=None,
                    line_in_context=None,
                    prompt_chars=None,
                    response_text="",
                )
                verified_findings.append(skipped)
                skipped_requests += 1
                skipped_count += 1
                events.append(
                    {
                        "event": "verification_budget_exhausted",
                        "finding_id": finding.id,
                    }
                )
                continue

            try:
                verdict, confidence, reason, evidence_lines = self._decide(finding)
                completed_requests += 1
                if verdict == "valid" and confidence >= context.min_confidence:
                    verified = _with_verification_metadata(
                        finding,
                        status=VerificationStatus.VALID,
                        verdict=verdict,
                        confidence=confidence,
                        reason=reason,
                        evidence_lines=evidence_lines,
                        model=context.verification_model,
                        elapsed_ms=1,
                        context_line_start=None,
                        context_line_end=None,
                        line_in_context=True,
                        prompt_chars=10,
                        response_text=reason,
                    )
                    verified_findings.append(verified)
                    valid_count += 1
                    events.append({"event": "verification_complete", "finding_id": finding.id})
                elif verdict == "invalid" and confidence >= context.min_confidence:
                    rejected = _with_verification_metadata(
                        finding,
                        status=VerificationStatus.INVALID,
                        verdict=verdict,
                        confidence=confidence,
                        reason=reason,
                        evidence_lines=evidence_lines,
                        model=context.verification_model,
                        elapsed_ms=1,
                        context_line_start=None,
                        context_line_end=None,
                        line_in_context=True,
                        prompt_chars=10,
                        response_text=reason,
                    )
                    rejected.status = Status.REJECTED
                    rejected.rejection_reason = "verification_failed"
                    verification_rejected_findings.append(rejected)
                    invalid_count += 1
                    events.append({"event": "verification_rejected", "finding_id": finding.id})
                else:
                    uncertain_reason = reason
                    if confidence < context.min_confidence:
                        uncertain_reason = (
                            f"Low verification confidence {confidence:.2f} (< {context.min_confidence:.2f}): "
                            f"{reason}"
                        )
                    uncertain_count += 1
                    if context.uncertain_policy == "reject":
                        rejected = _with_verification_metadata(
                            finding,
                            status=VerificationStatus.INVALID,
                            verdict="uncertain",
                            confidence=confidence,
                            reason=uncertain_reason,
                            evidence_lines=evidence_lines,
                            model=context.verification_model,
                            elapsed_ms=1,
                            context_line_start=None,
                            context_line_end=None,
                            line_in_context=True,
                            prompt_chars=10,
                            response_text=reason,
                        )
                        rejected.status = Status.REJECTED
                        rejected.rejection_reason = "verification_uncertain_rejected"
                        verification_rejected_findings.append(rejected)
                        invalid_count += 1
                    else:
                        unverified = _with_verification_metadata(
                            finding,
                            status=VerificationStatus.UNVERIFIED,
                            verdict="uncertain",
                            confidence=confidence,
                            reason=uncertain_reason,
                            evidence_lines=evidence_lines,
                            model=context.verification_model,
                            elapsed_ms=1,
                            context_line_start=None,
                            context_line_end=None,
                            line_in_context=True,
                            prompt_chars=10,
                            response_text=reason,
                        )
                        verified_findings.append(unverified)
                        unverified_count += 1
                    events.append({"event": "verification_uncertain", "finding_id": finding.id})
            except TimeoutError as exc:
                failed_requests += 1
                events.append(
                    {
                        "event": "verification_timeout",
                        "finding_id": finding.id,
                        "error": str(exc),
                    }
                )
                if context.fail_policy == "reject":
                    rejected = _with_verification_metadata(
                        finding,
                        status=VerificationStatus.INVALID,
                        verdict="",
                        confidence=None,
                        reason=f"Verifier timeout: {exc}",
                        evidence_lines=[],
                        model=context.verification_model,
                        elapsed_ms=1,
                        context_line_start=None,
                        context_line_end=None,
                        line_in_context=None,
                        prompt_chars=None,
                        response_text="",
                    )
                    rejected.status = Status.REJECTED
                    rejected.rejection_reason = "verification_timeout"
                    verification_rejected_findings.append(rejected)
                    invalid_count += 1
                else:
                    unverified = _with_verification_metadata(
                        finding,
                        status=VerificationStatus.UNVERIFIED,
                        verdict="uncertain",
                        confidence=None,
                        reason=f"Verifier timeout: {exc}",
                        evidence_lines=[],
                        model=context.verification_model,
                        elapsed_ms=1,
                        context_line_start=None,
                        context_line_end=None,
                        line_in_context=None,
                        prompt_chars=None,
                        response_text="",
                    )
                    verified_findings.append(unverified)
                    uncertain_count += 1
                    unverified_count += 1
            except VerificationParseError as exc:
                failed_requests += 1
                events.append(
                    {
                        "event": "verification_parse_failure",
                        "finding_id": finding.id,
                        "error": str(exc),
                    }
                )
                if context.fail_policy == "reject":
                    rejected = _with_verification_metadata(
                        finding,
                        status=VerificationStatus.INVALID,
                        verdict="",
                        confidence=None,
                        reason=f"Verifier failure: {exc}",
                        evidence_lines=[],
                        model=context.verification_model,
                        elapsed_ms=1,
                        context_line_start=None,
                        context_line_end=None,
                        line_in_context=None,
                        prompt_chars=None,
                        response_text="",
                    )
                    rejected.status = Status.REJECTED
                    rejected.rejection_reason = "verification_failed"
                    verification_rejected_findings.append(rejected)
                    invalid_count += 1
                else:
                    unverified = _with_verification_metadata(
                        finding,
                        status=VerificationStatus.UNVERIFIED,
                        verdict="uncertain",
                        confidence=None,
                        reason=f"Verifier failure: {exc}",
                        evidence_lines=[],
                        model=context.verification_model,
                        elapsed_ms=1,
                        context_line_start=None,
                        context_line_end=None,
                        line_in_context=None,
                        prompt_chars=None,
                        response_text="",
                    )
                    verified_findings.append(unverified)
                    uncertain_count += 1
                    unverified_count += 1

        return VerificationResult(
            verified_findings=verified_findings,
            verification_rejected_findings=verification_rejected_findings,
            completed_requests=completed_requests,
            failed_requests=failed_requests,
            skipped_requests=skipped_requests,
            elapsed_seconds=perf_counter() - start,
            valid_count=valid_count,
            invalid_count=invalid_count,
            uncertain_count=uncertain_count,
            unverified_count=unverified_count,
            skipped_count=skipped_count,
            debug_events=events,
        )
