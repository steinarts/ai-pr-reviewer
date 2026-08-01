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
        payload = {"reviewer": reviewer, "findings": []}
        return json.dumps(payload)
