from __future__ import annotations

import json
from typing import Protocol


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

    @staticmethod
    def _reviewer_from_user_prompt(user_prompt: str) -> str:
        supported_reviewers = {"bug", "reliability", "security", "consolidated"}
        for line in user_prompt.splitlines():
            if not line.startswith("REVIEWER:"):
                continue
            reviewer = line.partition(":")[2].strip().lower()
            if reviewer in supported_reviewers:
                return reviewer
            break
        return "bug"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        reviewer = self._reviewer_from_user_prompt(user_prompt)
        payload = {"reviewer": reviewer, "findings": []}
        return json.dumps(payload)
