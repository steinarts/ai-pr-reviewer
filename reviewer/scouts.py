from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError

from .llm_client import LLMClient, load_prompt
from .models import Finding, FindingsPayload, ReviewContext, ReviewerFailure, ReviewerSkip
from .token_utils import estimate_tokens

REVIEW_MODE_CONSOLIDATED = "consolidated"
REVIEW_MODE_SEPARATE = "separate"

FILE_CLASS_SOURCE = "source"
FILE_CLASS_SECURITY = "security_sensitive"
FILE_CLASS_RELIABILITY = "reliability_infra"
FILE_CLASS_TEST = "test"
FILE_CLASS_CONFIG = "config"
FILE_CLASS_DOCUMENTATION = "documentation"
FILE_CLASS_GENERATED = "generated"

FILE_CLASS_PRIORITY = {
    FILE_CLASS_SOURCE: 1,
    FILE_CLASS_SECURITY: 2,
    FILE_CLASS_RELIABILITY: 3,
    FILE_CLASS_TEST: 4,
    FILE_CLASS_CONFIG: 5,
    FILE_CLASS_DOCUMENTATION: 6,
    FILE_CLASS_GENERATED: 7,
}

NON_REVIEWABLE_CLASSES = {FILE_CLASS_CONFIG, FILE_CLASS_DOCUMENTATION, FILE_CLASS_GENERATED}


class ReviewParseError(ValueError):
    """Raised when model output cannot be parsed as FindingsPayload JSON."""


class InvalidJSONError(ReviewParseError):
    """Raised when model output is not valid JSON."""


class SchemaValidationError(ReviewParseError):
    """Raised when JSON does not match FindingsPayload schema."""


class SemanticValidationError(ReviewParseError):
    """Raised when parsed findings fail semantic validation."""


@dataclass(slots=True)
class ReviewChunk:
    sections: list[tuple[str, str]]
    file_contexts: dict[str, str]
    file_class: str
    priority: int


@dataclass(slots=True)
class PlannedRequest:
    reviewer: str
    chunk_index: int
    chunk_count: int
    chunk: ReviewChunk


@dataclass(slots=True)
class RunReviewersResult:
    findings: list[Finding]
    reviewer_failures: list[ReviewerFailure]
    reviewer_skips: list[ReviewerSkip]
    completed_requests: int
    failed_requests: int
    planned_requests: int
    skipped_requests: int
    chunk_count: int
    reviewable_chunks: int
    skipped_chunks: int
    total_elapsed_seconds: float
    total_time_budget_seconds: float
    review_mode: str


COMMON_REVIEW_RULES = """Hard requirements:
- Report only defects introduced by added or modified lines in this diff.
- Do not report existing code when introduced_by_diff would be false.
- Tests describe expected behavior and are not themselves defects.
- Code inside pytest.raises(...) is expected to raise.
- Assertions that validate correct behavior must not be reported as missing behavior.
- Do not summarize test names, docstrings, assertions, or comments as findings.
- Do not report preventive recommendations without concrete evidence of an actual bug.
- Do not speculate. Return no finding rather than speculate.
- Reject hypothetical phrasing like: "If this does not work, it may cause...".
- A finding must include concrete evidence that current implementation
  is wrong, incomplete, unsafe, or inconsistent.
- Before returning a finding, verify the suggested fix is not already implemented in the shown diff.
- Return at most 3 findings.
- Keep evidence, consequence, and suggestion concise (max 2 short sentences each).
- Do not report style-only issues (formatting, lint, naming, line length, E501, whitespace).
- If no concrete defect exists, return {"findings": []}.
- Confidence must be between 0.0 and 1.0.
- file and line must point to an added or modified line, not merely a test description.
"""


def _build_prompts(
    reviewer: str,
    template: str,
    context: ReviewContext,
    *,
    review_mode: str = REVIEW_MODE_SEPARATE,
) -> tuple[str, str]:
    """Build system and user prompts for the reviewer."""
    if review_mode == REVIEW_MODE_CONSOLIDATED:
        specialty = "bug, reliability, and security"
        category_rule = (
            "- Every finding category must be exactly one of: bug, reliability, security.\n"
        )
    else:
        specialty = f"{reviewer}"
        category_rule = ""

    system_prompt = (
        f"You are an expert code reviewer specializing in {specialty} issues.\n"
        "Analyze the provided Git diff.\n"
        'Return ONLY valid JSON with schema {"findings": [...]}\n'
        "Do not return any text outside the JSON.\n\n"
        f"{COMMON_REVIEW_RULES}"
        f"{category_rule}"
    )

    user_prompt = (
        f"REVIEWER: {reviewer}\n"
        f"BASE: {context.base}\n"
        f"HEAD: {context.head}\n\n"
        f"{template}\n\n"
        f"DIFF:\n{context.diff_text}\n\n"
        f"FILE_CONTEXTS:\n{json.dumps(context.file_contexts, ensure_ascii=False)}"
    )

    return system_prompt, user_prompt


def _parse_findings(raw_output: str, reviewer: str) -> list[Finding]:
    preview = raw_output.strip().replace("\n", " ")[:280]

    try:
        parsed_json = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise InvalidJSONError(
            f"Invalid JSON output from reviewer '{reviewer}'. "
            f'Expected schema {{"findings": [...]}}. '
            f"Model output preview: {preview!r}"
        ) from exc

    try:
        payload = FindingsPayload.model_validate(parsed_json)
        return payload.findings
    except ValidationError as exc:
        raise SchemaValidationError(
            f"Invalid JSON output from reviewer '{reviewer}'. "
            f'Expected schema {{"findings": [...]}}. '
            f"Model output preview: {preview!r}"
        ) from exc


def _is_test_file_path(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/")
    name = normalized.rsplit("/", maxsplit=1)[-1]
    return (
        normalized.startswith("tests/")
        or "/tests/" in normalized
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _is_documentation_path(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/").lower()
    name = normalized.rsplit("/", maxsplit=1)[-1]
    return (
        name in {"readme.md", "changelog.md", "contributing.md"}
        or normalized.startswith("docs/")
        or normalized.endswith(".md")
        or normalized.endswith(".rst")
        or normalized.endswith(".txt")
    )


def _is_generated_path(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/").lower()
    return (
        "/generated/" in normalized
        or "/dist/" in normalized
        or "/build/" in normalized
        or normalized.endswith(".min.js")
        or normalized.endswith(".min.css")
    )


def _is_config_path(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/").lower()
    name = normalized.rsplit("/", maxsplit=1)[-1]
    return (
        name == ".gitignore"
        or name.endswith(".lock")
        or name in {"pyproject.toml", "mypy.ini", "setup.cfg", "tox.ini", ".editorconfig"}
        or normalized.endswith(".toml")
        or normalized.endswith(".yaml")
        or normalized.endswith(".yml")
        or normalized.endswith(".ini")
        or normalized.endswith(".cfg")
        or normalized.endswith(".json")
    )


def _is_security_sensitive_path(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/").lower()
    return any(
        marker in normalized
        for marker in ("auth", "security", "crypto", "token", "secret", "permission")
    )


def _is_reliability_infra_path(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/").lower()
    return any(
        marker in normalized
        for marker in ("queue", "worker", "retry", "cache", "pipeline", "database", "db")
    )


def _classify_file(file_path: str) -> str:
    if _is_generated_path(file_path):
        return FILE_CLASS_GENERATED
    if _is_documentation_path(file_path):
        return FILE_CLASS_DOCUMENTATION
    if _is_config_path(file_path):
        return FILE_CLASS_CONFIG
    if _is_test_file_path(file_path):
        return FILE_CLASS_TEST
    if _is_security_sensitive_path(file_path):
        return FILE_CLASS_SECURITY
    if _is_reliability_infra_path(file_path):
        return FILE_CLASS_RELIABILITY
    return FILE_CLASS_SOURCE


def _parse_changed_lines_map(diff_text: str) -> dict[str, set[int]]:
    changed: dict[str, set[int]] = {}
    current_file: str | None = None
    current_new_line = 0
    hunk_re = re.compile(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")

    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current_file = line.removeprefix("+++ b/").strip()
            changed.setdefault(current_file, set())
            continue

        if line.startswith("@@"):
            match = hunk_re.search(line)
            if match:
                current_new_line = int(match.group(1))
            continue

        if current_file is None:
            continue

        if line.startswith("+") and not line.startswith("+++"):
            changed[current_file].add(current_new_line)
            current_new_line += 1
            continue

        if line.startswith("-") and not line.startswith("---"):
            continue

        if not line.startswith("\\"):
            current_new_line += 1

    return changed


def _looks_like_style_only(finding: Finding) -> bool:
    style_terms = (
        "e501",
        "line too long",
        "pep8",
        "lint",
        "format",
        "formatting",
        "whitespace",
        "trailing whitespace",
        "naming convention",
    )
    text = " ".join(
        [
            finding.id,
            finding.title,
            finding.category,
            finding.evidence,
            finding.suggestion,
        ]
    ).lower()
    return any(term in text for term in style_terms)


def _is_test_intent_restatement(finding: Finding) -> bool:
    text = " ".join([finding.title, finding.evidence, finding.consequence]).lower()
    markers = (
        "pytest.raises",
        "expected to raise",
        "assertion",
        "assert ",
        "test verifies",
        "test checks",
        "test expects",
    )
    return any(marker in text for marker in markers)


def _validate_findings_for_chunk(
    findings: list[Finding],
    *,
    chunk_files: list[str],
    changed_lines_by_file: dict[str, set[int]],
    review_mode: str,
) -> tuple[list[Finding], list[tuple[Finding, str]]]:
    valid: list[Finding] = []
    rejected: list[tuple[Finding, str]] = []
    chunk_file_set = set(chunk_files)

    for finding in findings:
        if finding.file not in chunk_file_set:
            rejected.append((finding, "file_not_in_chunk"))
            continue

        changed_lines = changed_lines_by_file.get(finding.file, set())
        if changed_lines and finding.line not in changed_lines:
            rejected.append((finding, "line_not_changed"))
            continue

        if not 0.0 <= finding.confidence <= 1.0:
            rejected.append((finding, "invalid_confidence"))
            continue

        if (
            not finding.evidence.strip()
            or not finding.consequence.strip()
            or not finding.suggestion.strip()
        ):
            rejected.append((finding, "invalid_evidence"))
            continue

        if finding.style_only or _looks_like_style_only(finding):
            rejected.append((finding, "style_only"))
            continue

        if _is_test_file_path(finding.file) and _is_test_intent_restatement(finding):
            rejected.append((finding, "test_intent_restated"))
            continue

        if review_mode == REVIEW_MODE_CONSOLIDATED and finding.category not in {
            "bug",
            "reliability",
            "security",
        }:
            rejected.append((finding, "invalid_category"))
            continue

        valid.append(finding)

    return valid, rejected


def _chunk_files(chunk: ReviewChunk) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for file_path, _ in chunk.sections:
        if file_path in seen:
            continue
        seen.add(file_path)
        ordered.append(file_path)
    return ordered


def _chunk_diff_text(chunk: ReviewChunk) -> str:
    return "".join(section for _, section in chunk.sections)


def _estimate_chunk_content_tokens(diff_text: str, file_contexts: dict[str, str]) -> int:
    content = (
        f"DIFF:\n{diff_text}\n\nFILE_CONTEXTS:\n{json.dumps(file_contexts, ensure_ascii=False)}"
    )
    return estimate_tokens(content)


def _extract_file_path(section_text: str) -> str:
    for line in section_text.splitlines():
        if line.startswith("+++ b/"):
            return line.removeprefix("+++ b/").strip()

    match = re.search(r"^diff --git a/(.+) b/(.+)$", section_text, flags=re.MULTILINE)
    if match:
        return match.group(2).strip()

    return "unknown"


def _split_diff_sections(diff_text: str) -> list[tuple[str, str]]:
    if not diff_text.strip():
        return []

    sections: list[str] = []
    current: list[str] = []
    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git ") and current:
            sections.append("".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("".join(current))

    return [(_extract_file_path(section), section) for section in sections]


def _split_section_by_hunks(section_text: str) -> list[str]:
    lines = section_text.splitlines(keepends=True)
    hunk_indices = [idx for idx, line in enumerate(lines) if line.startswith("@@")]
    if not hunk_indices:
        return [section_text]

    header = lines[: hunk_indices[0]]
    parts: list[str] = []
    for idx, start in enumerate(hunk_indices):
        end = hunk_indices[idx + 1] if idx + 1 < len(hunk_indices) else len(lines)
        parts.append("".join(header + lines[start:end]))
    return parts


def _split_section_by_line_blocks(section_text: str, block_lines: int = 180) -> list[str]:
    lines = section_text.splitlines(keepends=True)
    if len(lines) <= block_lines:
        return [section_text]

    return [
        "".join(lines[start : start + block_lines]) for start in range(0, len(lines), block_lines)
    ]


def _compress_context_block(context_block: str) -> str:
    if not context_block.strip():
        return ""
    kept: list[str] = []
    for line in context_block.splitlines():
        if line.startswith(("FILE:", "STATUS:", "CHANGED_LINES:", "SYMBOLS:")):
            kept.append(line)
    return "\n".join(kept)


def _split_section_for_content_budget(
    file_path: str,
    section_text: str,
    context_block: str,
    content_budget_tokens: int,
) -> list[str]:
    def fits(text: str) -> bool:
        contexts = {file_path: context_block} if context_block else {}
        return _estimate_chunk_content_tokens(text, contexts) <= content_budget_tokens

    is_test_file = _is_test_file_path(file_path)
    if fits(section_text):
        return [section_text]

    by_hunk = _split_section_by_hunks(section_text)
    if len(by_hunk) > 1:
        expanded: list[str] = []
        for part in by_hunk:
            if fits(part):
                expanded.append(part)
                continue
            if is_test_file:
                expanded.append(part)
                continue

            by_lines = _split_section_by_line_blocks(part)
            if len(by_lines) == 1:
                expanded.append(part)
            else:
                expanded.extend(by_lines)
        return expanded

    if is_test_file:
        return [section_text]
    return _split_section_by_line_blocks(section_text)


def _build_base_chunks(
    context: ReviewContext,
    content_budget_tokens: int,
    test_content_budget_tokens: int,
) -> list[ReviewChunk]:
    sections = _split_diff_sections(context.diff_text)
    if not sections:
        return []

    chunks: list[ReviewChunk] = []
    current_sections: list[tuple[str, str]] = []
    current_contexts: dict[str, str] = {}
    current_class: str | None = None

    for file_path, section_text in sections:
        file_class = _classify_file(file_path)
        is_test_file = file_class == FILE_CLASS_TEST
        file_budget_tokens = test_content_budget_tokens if is_test_file else content_budget_tokens
        context_block = _compress_context_block(context.file_contexts.get(file_path, ""))

        parts = _split_section_for_content_budget(
            file_path=file_path,
            section_text=section_text,
            context_block=context_block,
            content_budget_tokens=file_budget_tokens,
        )

        for part in parts:
            if current_sections and current_class is not None and current_class != file_class:
                chunks.append(
                    ReviewChunk(
                        sections=current_sections,
                        file_contexts=current_contexts,
                        file_class=current_class,
                        priority=FILE_CLASS_PRIORITY[current_class],
                    )
                )
                current_sections = []
                current_contexts = {}

            candidate_sections = [*current_sections, (file_path, part)]
            candidate_contexts = dict(current_contexts)
            if context_block:
                candidate_contexts[file_path] = context_block

            candidate_tokens = _estimate_chunk_content_tokens(
                diff_text="".join(section for _, section in candidate_sections),
                file_contexts=candidate_contexts,
            )

            if current_sections and candidate_tokens > file_budget_tokens:
                assert current_class is not None
                chunks.append(
                    ReviewChunk(
                        sections=current_sections,
                        file_contexts=current_contexts,
                        file_class=current_class,
                        priority=FILE_CLASS_PRIORITY[current_class],
                    )
                )
                current_sections = [(file_path, part)]
                current_contexts = {file_path: context_block} if context_block else {}
                current_class = file_class
            else:
                current_sections = candidate_sections
                current_contexts = candidate_contexts
                current_class = file_class

    if current_sections and current_class is not None:
        chunks.append(
            ReviewChunk(
                sections=current_sections,
                file_contexts=current_contexts,
                file_class=current_class,
                priority=FILE_CLASS_PRIORITY[current_class],
            )
        )

    return chunks


def _refine_chunk_for_request_budget(
    chunk: ReviewChunk,
    estimate_request_tokens: Callable[[ReviewChunk], int],
    prompt_budget_tokens: int,
) -> tuple[list[ReviewChunk], list[int]]:
    estimate = estimate_request_tokens(chunk)
    if estimate <= prompt_budget_tokens:
        return [chunk], []

    if len(chunk.sections) > 1:
        refined: list[ReviewChunk] = []
        warnings: list[int] = []
        for file_path, section_text in chunk.sections:
            context_block = chunk.file_contexts.get(file_path, "")
            sub_chunk = ReviewChunk(
                sections=[(file_path, section_text)],
                file_contexts={file_path: context_block} if context_block else {},
                file_class=_classify_file(file_path),
                priority=FILE_CLASS_PRIORITY[_classify_file(file_path)],
            )
            sub_refined, sub_warnings = _refine_chunk_for_request_budget(
                sub_chunk,
                estimate_request_tokens,
                prompt_budget_tokens,
            )
            refined.extend(sub_refined)
            warnings.extend(sub_warnings)
        return refined, warnings

    file_path, section_text = chunk.sections[0]
    context_block = chunk.file_contexts.get(file_path, "")
    is_test_file = _is_test_file_path(file_path)

    by_hunk = _split_section_by_hunks(section_text)
    if len(by_hunk) > 1:
        refined: list[ReviewChunk] = []
        warnings: list[int] = []
        for part in by_hunk:
            sub_chunk = ReviewChunk(
                sections=[(file_path, part)],
                file_contexts={file_path: context_block} if context_block else {},
                file_class=chunk.file_class,
                priority=chunk.priority,
            )
            sub_refined, sub_warnings = _refine_chunk_for_request_budget(
                sub_chunk,
                estimate_request_tokens,
                prompt_budget_tokens,
            )
            refined.extend(sub_refined)
            warnings.extend(sub_warnings)
        return refined, warnings

    if is_test_file:
        return [chunk], [estimate]

    by_lines = _split_section_by_line_blocks(section_text)
    if len(by_lines) > 1:
        refined = []
        warnings = []
        for part in by_lines:
            sub_chunk = ReviewChunk(
                sections=[(file_path, part)],
                file_contexts={file_path: context_block} if context_block else {},
                file_class=chunk.file_class,
                priority=chunk.priority,
            )
            sub_refined, sub_warnings = _refine_chunk_for_request_budget(
                sub_chunk,
                estimate_request_tokens,
                prompt_budget_tokens,
            )
            refined.extend(sub_refined)
            warnings.extend(sub_warnings)
        return refined, warnings

    return [chunk], [estimate]


def _is_comment_or_blank_line(line: str) -> bool:
    stripped = line.strip()
    return (
        not stripped
        or stripped.startswith("#")
        or stripped.startswith("//")
        or stripped.startswith("*")
        or stripped.startswith('"""')
        or stripped.startswith("'''")
    )


def _chunk_has_reviewable_code(chunk: ReviewChunk) -> bool:
    if chunk.file_class in NON_REVIEWABLE_CLASSES:
        return False

    chunk_diff = _chunk_diff_text(chunk)
    changed_lines_map = _parse_changed_lines_map(chunk_diff)
    if not any(changed_lines_map.values()):
        return False

    added_lines = [
        line.removeprefix("+")
        for line in chunk_diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    if not added_lines:
        return False

    return any(not _is_comment_or_blank_line(line) for line in added_lines)


def _load_templates(prompts_dir: Path, review_mode: str, reviewers: list[str]) -> dict[str, str]:
    if review_mode == REVIEW_MODE_CONSOLIDATED:
        bug = load_prompt(prompts_dir, "bug")
        reliability = load_prompt(prompts_dir, "reliability")
        security = load_prompt(prompts_dir, "security")
        combined = (
            "Consolidated review. Analyze only concrete defects introduced in the diff.\n\n"
            "BUG REVIEW RULES:\n"
            f"{bug}\n\n"
            "RELIABILITY REVIEW RULES:\n"
            f"{reliability}\n\n"
            "SECURITY REVIEW RULES:\n"
            f"{security}"
        )
        return {"consolidated": combined}

    return {reviewer: load_prompt(prompts_dir, reviewer) for reviewer in reviewers}


def _invoke_llm(
    llm_client: LLMClient,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict[str, object],
    llm_timeout_seconds: float,
) -> str:
    hard_timeout_method = getattr(llm_client, "generate_hard_timeout", None)
    if callable(hard_timeout_method):
        return hard_timeout_method(
            system_prompt,
            user_prompt,
            response_schema=response_schema,
            timeout_seconds=llm_timeout_seconds,
        )

    return llm_client.generate(system_prompt, user_prompt, response_schema=response_schema)


def run_reviewers(
    reviewers: list[str],
    llm_client: LLMClient,
    context: ReviewContext,
    prompts_dir: Path,
    max_prompt_tokens: int,
    review_mode: str = REVIEW_MODE_CONSOLIDATED,
    llm_timeout_seconds: float = 180.0,
    max_review_seconds: float = 900.0,
    test_max_prompt_tokens: int = 2500,
    max_findings_per_request: int = 3,
    debug_sink: Callable[[dict[str, object]], None] | None = None,
    progress_sink: Callable[[str], None] | None = None,
) -> RunReviewersResult:
    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be greater than zero")
    if llm_timeout_seconds <= 0:
        raise ValueError("llm_timeout_seconds must be greater than zero")
    if max_review_seconds <= 0:
        raise ValueError("max_review_seconds must be greater than zero")
    if test_max_prompt_tokens <= 0:
        raise ValueError("test_max_prompt_tokens must be greater than zero")
    if max_findings_per_request <= 0:
        raise ValueError("max_findings_per_request must be greater than zero")

    findings_schema = FindingsPayload.model_json_schema()
    schema_text = json.dumps(findings_schema, ensure_ascii=False)

    templates = _load_templates(prompts_dir, review_mode, reviewers)
    active_reviewers = ["consolidated"] if review_mode == REVIEW_MODE_CONSOLIDATED else reviewers

    max_overhead_tokens = 1
    for reviewer, template in templates.items():
        empty_context = ReviewContext(
            base=context.base,
            head=context.head,
            diff_text="",
            file_contexts={},
        )
        system_prompt, user_prompt = _build_prompts(
            reviewer,
            template,
            empty_context,
            review_mode=review_mode,
        )
        overhead = (
            estimate_tokens(system_prompt)
            + estimate_tokens(user_prompt)
            + estimate_tokens(schema_text)
        )
        max_overhead_tokens = max(max_overhead_tokens, overhead)

    content_budget_tokens = max(1, max_prompt_tokens - max_overhead_tokens)
    test_content_budget_tokens = max(
        1, min(test_max_prompt_tokens, max_prompt_tokens) - max_overhead_tokens
    )

    base_chunks = _build_base_chunks(
        context=context,
        content_budget_tokens=content_budget_tokens,
        test_content_budget_tokens=test_content_budget_tokens,
    )

    sorted_chunks: list[tuple[int, ReviewChunk]] = sorted(
        list(enumerate(base_chunks, start=1)),
        key=lambda item: (item[1].priority, item[0]),
    )

    findings: list[Finding] = []
    reviewer_failures: list[ReviewerFailure] = []
    reviewer_skips: list[ReviewerSkip] = []
    completed_requests = 0
    failed_requests = 0
    skipped_requests = 0

    provider = llm_client.__class__.__name__
    model = str(getattr(llm_client, "model", ""))
    expected_errors = (ReviewParseError, TimeoutError, ConnectionError, ValueError)
    configured_timeout_seconds = llm_timeout_seconds

    request_plan: list[PlannedRequest] = []
    reviewable_chunks = 0
    skipped_chunks = 0

    def estimate_request_tokens(reviewer: str, template: str, chunk: ReviewChunk) -> int:
        chunk_context = ReviewContext(
            base=context.base,
            head=context.head,
            diff_text=_chunk_diff_text(chunk),
            file_contexts=chunk.file_contexts,
        )
        system_prompt, user_prompt = _build_prompts(
            reviewer,
            template,
            chunk_context,
            review_mode=review_mode,
        )
        return (
            estimate_tokens(system_prompt)
            + estimate_tokens(user_prompt)
            + estimate_tokens(schema_text)
        )

    for chunk_index, base_chunk in sorted_chunks:
        if not _chunk_has_reviewable_code(base_chunk):
            skipped_chunks += 1
            reviewer_skips.append(
                ReviewerSkip(
                    reviewer="scheduler",
                    chunk_index=chunk_index,
                    reason="no_reviewable_code",
                    message=f"chunk class={base_chunk.file_class}",
                )
            )
            if debug_sink is not None:
                debug_sink(
                    {
                        "event": "chunk_skipped",
                        "chunk_index": chunk_index,
                        "chunk_count": len(base_chunks),
                        "file_class": base_chunk.file_class,
                        "reason": "no_reviewable_code",
                        "skip_reason": "no_reviewable_code",
                    }
                )
            continue

        reviewable_chunks += 1

        for reviewer in active_reviewers:
            template = templates[reviewer]
            prompt_budget = (
                min(test_max_prompt_tokens, max_prompt_tokens)
                if base_chunk.file_class == FILE_CLASS_TEST
                else max_prompt_tokens
            )

            split_chunks, oversize_estimates = _refine_chunk_for_request_budget(
                chunk=base_chunk,
                estimate_request_tokens=(
                    lambda chunk, reviewer=reviewer, template=template: estimate_request_tokens(
                        reviewer,
                        template,
                        chunk,
                    )
                ),
                prompt_budget_tokens=prompt_budget,
            )

            for estimated in oversize_estimates:
                if debug_sink is not None:
                    debug_sink(
                        {
                            "event": "reviewer_warning",
                            "reviewer": reviewer,
                            "provider": provider,
                            "model": model,
                            "chunk_index": chunk_index,
                            "chunk_count": len(base_chunks),
                            "prompt_budget_tokens": prompt_budget,
                            "estimated_prompt_tokens": estimated,
                            "warning": (
                                "Chunk exceeds prompt budget even after splitting "
                                "minimal diff sections. "
                                f"Estimated tokens: {estimated}, budget: {prompt_budget}."
                            ),
                        }
                    )

            for refined_chunk in split_chunks:
                request_plan.append(
                    PlannedRequest(
                        reviewer=reviewer,
                        chunk_index=chunk_index,
                        chunk_count=len(base_chunks),
                        chunk=refined_chunk,
                    )
                )

    planned_requests = len(request_plan)
    overall_start = perf_counter()

    for request_index, request in enumerate(request_plan, start=1):
        total_elapsed = perf_counter() - overall_start
        remaining_budget = max_review_seconds - total_elapsed

        if remaining_budget <= 0:
            for pending in request_plan[request_index - 1 :]:
                skipped_requests += 1
                reviewer_skips.append(
                    ReviewerSkip(
                        reviewer=pending.reviewer,
                        chunk_index=pending.chunk_index,
                        reason="total_time_budget_exceeded",
                        message="Review time budget exhausted before request start.",
                    )
                )
                if debug_sink is not None:
                    debug_sink(
                        {
                            "event": "request_skipped",
                            "reviewer": pending.reviewer,
                            "chunk_index": pending.chunk_index,
                            "chunk_count": pending.chunk_count,
                            "reason": "total_time_budget_exceeded",
                            "skip_reason": "total_time_budget_exceeded",
                            "total_elapsed_seconds": total_elapsed,
                            "remaining_review_budget_seconds": 0.0,
                        }
                    )
            break

        reviewer = request.reviewer
        chunk_index = request.chunk_index
        chunk = request.chunk
        template = templates[reviewer]

        chunk_diff = _chunk_diff_text(chunk)
        chunk_context = ReviewContext(
            base=context.base,
            head=context.head,
            diff_text=chunk_diff,
            file_contexts=chunk.file_contexts,
        )

        system_prompt, user_prompt = _build_prompts(
            reviewer,
            template,
            chunk_context,
            review_mode=review_mode,
        )

        system_chars = len(system_prompt)
        user_chars = len(user_prompt)
        estimated_prompt_tokens = (
            estimate_tokens(system_prompt)
            + estimate_tokens(user_prompt)
            + estimate_tokens(schema_text)
        )

        chunk_files = _chunk_files(chunk)
        request_started_at = perf_counter()
        request_deadline = request_started_at + llm_timeout_seconds

        if progress_sink is not None:
            progress_sink(
                f"Running {reviewer} reviewer, chunk {chunk_index}/{request.chunk_count} "
                f"(request {request_index}/{planned_requests})..."
            )

        if debug_sink is not None:
            debug_sink(
                {
                    "event": "reviewer_start",
                    "reviewer": reviewer,
                    "provider": provider,
                    "model": model,
                    "chunk_index": chunk_index,
                    "chunk_count": request.chunk_count,
                    "chunk_file_count": len(chunk_files),
                    "chunk_files": chunk_files,
                    "file_class": chunk.file_class,
                    "prompt_budget_tokens": min(test_max_prompt_tokens, max_prompt_tokens)
                    if chunk.file_class == FILE_CLASS_TEST
                    else max_prompt_tokens,
                    "estimated_prompt_tokens": estimated_prompt_tokens,
                    "configured_timeout_seconds": configured_timeout_seconds,
                    "system_prompt_chars": system_chars,
                    "user_prompt_chars": user_chars,
                    "total_prompt_chars": system_chars + user_chars,
                    "diff_chars": len(chunk_diff),
                    "request_started_at": request_started_at,
                    "request_deadline": request_deadline,
                    "total_elapsed_seconds": total_elapsed,
                    "remaining_review_budget_seconds": remaining_budget,
                }
            )

        try:
            raw_output = _invoke_llm(
                llm_client,
                system_prompt,
                user_prompt,
                findings_schema,
                llm_timeout_seconds=min(llm_timeout_seconds, remaining_budget),
            )
            parsed = _parse_findings(raw_output, reviewer)
            changed_lines_by_file = _parse_changed_lines_map(chunk_diff)
            valid_findings, rejected_findings = _validate_findings_for_chunk(
                parsed,
                chunk_files=chunk_files,
                changed_lines_by_file=changed_lines_by_file,
                review_mode=review_mode,
            )
            if len(valid_findings) > max_findings_per_request:
                overflow = valid_findings[max_findings_per_request:]
                valid_findings = valid_findings[:max_findings_per_request]
                rejected_findings.extend((finding, "output_limit_exceeded") for finding in overflow)

            elapsed_ms = int((perf_counter() - request_started_at) * 1000)
            completed_requests += 1
            findings.extend(valid_findings)

            if progress_sink is not None:
                progress_sink(
                    f"{reviewer.capitalize()} reviewer chunk {chunk_index} "
                    f"completed in {elapsed_ms / 1000:.1f}s"
                )

            if debug_sink is not None:
                debug_sink(
                    {
                        "event": "reviewer_complete",
                        "reviewer": reviewer,
                        "provider": provider,
                        "model": model,
                        "chunk_index": chunk_index,
                        "chunk_count": request.chunk_count,
                        "chunk_file_count": len(chunk_files),
                        "chunk_files": chunk_files,
                        "file_class": chunk.file_class,
                        "prompt_budget_tokens": max_prompt_tokens,
                        "estimated_prompt_tokens": estimated_prompt_tokens,
                        "system_prompt_chars": system_chars,
                        "user_prompt_chars": user_chars,
                        "total_prompt_chars": system_chars + user_chars,
                        "diff_chars": len(chunk_diff),
                        "elapsed_ms": elapsed_ms,
                        "response_chars": len(raw_output),
                        "findings_count": len(valid_findings),
                        "raw_findings_count": len(parsed),
                        "total_elapsed_seconds": perf_counter() - overall_start,
                        "remaining_review_budget_seconds": max(
                            0.0,
                            max_review_seconds - (perf_counter() - overall_start),
                        ),
                    }
                )

            if rejected_findings and debug_sink is not None:
                rejection_counts: dict[str, int] = {}
                for _, reason in rejected_findings:
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                debug_sink(
                    {
                        "event": "reviewer_rejections",
                        "reviewer": reviewer,
                        "provider": provider,
                        "model": model,
                        "chunk_index": chunk_index,
                        "chunk_count": request.chunk_count,
                        "rejection_counts": rejection_counts,
                    }
                )
        except expected_errors as exc:
            elapsed_ms = int((perf_counter() - request_started_at) * 1000)
            failed_requests += 1
            reviewer_failures.append(
                ReviewerFailure(
                    reviewer=reviewer,
                    chunk_index=chunk_index,
                    error_type=exc.__class__.__name__,
                    message=str(exc),
                )
            )

            if progress_sink is not None:
                progress_sink(
                    f"{reviewer.capitalize()} reviewer chunk {chunk_index} "
                    f"failed: {exc.__class__.__name__}"
                )
                progress_sink("Continuing with remaining review requests.")

            if debug_sink is not None:
                debug_sink(
                    {
                        "event": "reviewer_error",
                        "reviewer": reviewer,
                        "provider": provider,
                        "model": model,
                        "chunk_index": chunk_index,
                        "chunk_count": request.chunk_count,
                        "chunk_file_count": len(chunk_files),
                        "chunk_files": chunk_files,
                        "file_class": chunk.file_class,
                        "prompt_budget_tokens": max_prompt_tokens,
                        "estimated_prompt_tokens": estimated_prompt_tokens,
                        "configured_timeout_seconds": configured_timeout_seconds,
                        "system_prompt_chars": system_chars,
                        "user_prompt_chars": user_chars,
                        "total_prompt_chars": system_chars + user_chars,
                        "diff_chars": len(chunk_diff),
                        "elapsed_ms": elapsed_ms,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "total_elapsed_seconds": perf_counter() - overall_start,
                        "remaining_review_budget_seconds": max(
                            0.0,
                            max_review_seconds - (perf_counter() - overall_start),
                        ),
                    }
                )

    return RunReviewersResult(
        findings=findings,
        reviewer_failures=reviewer_failures,
        reviewer_skips=reviewer_skips,
        completed_requests=completed_requests,
        failed_requests=failed_requests,
        planned_requests=planned_requests,
        skipped_requests=skipped_requests,
        chunk_count=len(base_chunks),
        reviewable_chunks=reviewable_chunks,
        skipped_chunks=skipped_chunks,
        total_elapsed_seconds=perf_counter() - overall_start,
        total_time_budget_seconds=max_review_seconds,
        review_mode=review_mode,
    )
