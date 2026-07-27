from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .models import Severity


class LLMClient(Protocol):
    """Protocol for LLM clients (FakeLLMClient, OllamaLLMClient, etc.)."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        """Generate a response given system and user prompts."""
        ...


class FakeLLMClient:
    """Fake LLM client for testing without API calls."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        """Return a fake JSON response with findings."""
        # Determine reviewer from the system prompt
        reviewer = "bug"
        if "reliability" in system_prompt.lower():
            reviewer = "reliability"
        elif "security" in system_prompt.lower():
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


def load_prompt(prompts_dir: Path, reviewer: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = prompts_dir / f"{reviewer}_reviewer.md"
    if not prompt_path.exists():
        return "Return JSON only."
    return prompt_path.read_text(encoding="utf-8")
