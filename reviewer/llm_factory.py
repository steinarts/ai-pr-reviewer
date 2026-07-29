from __future__ import annotations

from .llm_client import FakeLLMClient, LLMClient
from .ollama_client import OllamaLLMClient


def create_llm_client(
    provider: str,
    model: str | None = None,
    ollama_host: str = "http://localhost:11434",
    llm_timeout: float = 300.0,
    llm_max_output_tokens: int = 700,
    deterministic_seed: int | None = None,
    temperature: float = 0.0,
    top_p: float | None = None,
) -> LLMClient:
    """Create an LLM client based on provider.

    Args:
        provider: "fake" or "ollama"
        model: Model name (required for ollama, ignored for fake)
        ollama_host: Ollama server host (only for ollama)
        llm_timeout: Request timeout in seconds (only for ollama)
        llm_max_output_tokens: Approximate max output tokens per request (ollama)

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
        return OllamaLLMClient(
            model=model,
            host=ollama_host,
            timeout=llm_timeout,
            max_output_tokens=llm_max_output_tokens,
            seed=deterministic_seed,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose from: fake, ollama")
