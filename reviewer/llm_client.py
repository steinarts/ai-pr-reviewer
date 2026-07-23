from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from .models import Severity


class LLMClient(Protocol):
    def review(self, prompt: str) -> str:
        ...


class FakeLLMClient:
    def review(self, prompt: str) -> str:
        reviewer = "bug"
        if "REVIEWER: reliability" in prompt:
            reviewer = "reliability"
        elif "REVIEWER: security" in prompt:
            reviewer = "security"

        payload = {
            "findings": [
                {
                    "id": f"{reviewer}-1",
                    "file": "dbFunctions.py",
                    "line": 84,
                    "category": reviewer,
                    "severity": Severity.HIGH.value,
                    "confidence": 0.92,
                    "title": "Databaseforbindelsen kan bli stående åpen",
                    "evidence": "Forbindelsen lukkes bare på normal kodevei.",
                    "consequence": "Dette kan bidra til låsing av SQLite-databasen.",
                    "suggestion": "Bruk context manager eller try/finally.",
                    "introduced_by_diff": True,
                    "actionable": True,
                    "style_only": False,
                    "duplicate_of": None,
                    "reviewer": reviewer,
                    "status": "proposed",
                    "rejection_reason": "",
                },
                {
                    "id": f"{reviewer}-2",
                    "file": "dbFunctions.py",
                    "line": 90,
                    "category": reviewer,
                    "severity": Severity.LOW.value,
                    "confidence": 0.60,
                    "title": "Minor naming suggestion",
                    "evidence": "Variable name could be clearer.",
                    "consequence": "Readability only.",
                    "suggestion": "Rename variable for readability.",
                    "introduced_by_diff": True,
                    "actionable": False,
                    "style_only": True,
                    "duplicate_of": None,
                    "reviewer": reviewer,
                    "status": "proposed",
                    "rejection_reason": "",
                },
            ]
        }
        return json.dumps(payload, ensure_ascii=False)


class OpenAILLMClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("AI_REVIEW_MODEL", "")
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    def review(self, prompt: str) -> str:
        raise NotImplementedError("OpenAI client is not implemented in phase 1-2")


def load_prompt(prompts_dir: Path, reviewer: str) -> str:
    prompt_path = prompts_dir / f"{reviewer}_reviewer.md"
    if not prompt_path.exists():
        return "Return JSON only."
    return prompt_path.read_text(encoding="utf-8")
