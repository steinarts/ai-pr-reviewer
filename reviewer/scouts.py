from __future__ import annotations

import json
from pathlib import Path

from .llm_client import LLMClient, load_prompt
from .models import Finding, FindingsPayload, ReviewContext


def _build_prompt(reviewer: str, template: str, context: ReviewContext) -> str:
    return (
        f"REVIEWER: {reviewer}\n"
        f"BASE: {context.base}\n"
        f"HEAD: {context.head}\n"
        'RULES: Return JSON only with schema {"findings": [...]}\n\n'
        f"{template}\n\n"
        f"DIFF:\n{context.diff_text}\n\n"
        f"FILE_CONTEXTS:\n{json.dumps(context.file_contexts, ensure_ascii=False)}"
    )


def _parse_findings(raw_output: str) -> list[Finding]:
    try:
        payload = FindingsPayload.model_validate_json(raw_output)
        return payload.findings
    except Exception:
        return []


def run_reviewers(
    reviewers: list[str],
    llm_client: LLMClient,
    context: ReviewContext,
    prompts_dir: Path,
) -> list[Finding]:
    findings: list[Finding] = []
    for reviewer in reviewers:
        template = load_prompt(prompts_dir, reviewer)
        prompt = _build_prompt(reviewer, template, context)
        raw_output = llm_client.review(prompt)
        findings.extend(_parse_findings(raw_output))
    return findings
