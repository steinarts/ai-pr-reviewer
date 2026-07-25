from __future__ import annotations

import json
from pathlib import Path

from .llm_client import LLMClient, load_prompt
from .models import Finding, FindingsPayload, ReviewContext


def _build_prompts(reviewer: str, template: str, context: ReviewContext) -> tuple[str, str]:
    """Build system and user prompts for the reviewer.

    Returns:
        (system_prompt, user_prompt)
    """
    system_prompt = (
        f"You are an expert code reviewer specializing in {reviewer} issues.\n"
        f"Analyze the provided Git diff and file contexts.\n"
        'Return ONLY valid JSON with schema {{"findings": [...]}}\n'
        "Do not return any text outside the JSON."
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
        system_prompt, user_prompt = _build_prompts(reviewer, template, context)
        raw_output = llm_client.generate(system_prompt, user_prompt)
        findings.extend(_parse_findings(raw_output))
    return findings
