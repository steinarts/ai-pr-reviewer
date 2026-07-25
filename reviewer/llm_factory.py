from __future__ import annotations

from .llm_client import FakeLLMClient, LLMClient
from .ollama_client import OllamaLLMClient


def create_llm_client(
    provider: str,
    model: str | None = None,
    ollama_host: str = "http://localhost:11434",
) -> LLMClient:
    """Create an LLM client based on provider.

    Args:
        provider: "fake" or "ollama"
        model: Model name (required for ollama, ignored for fake)
        ollama_host: Ollama server host (only for ollama)

    Returns:
        LLMClient instance

    Raises:
        ValueError: If provider is unknown or required args are missing
    """
    if provider == "fake":
        return FakeLLMClient()
    elif provider == "ollama":
        if not model:
            raise ValueError("--model is required when using --provider ollama")
        return OllamaLLMClient(model=model, host=ollama_host)
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose from: fake, ollama")
