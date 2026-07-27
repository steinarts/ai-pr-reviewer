from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length using a simple chars/4 heuristic."""
    return max(1, len(text) // 4)
