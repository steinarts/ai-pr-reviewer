from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from queue import Empty
from time import perf_counter
from typing import Any


def _worker_entry(
    queue: multiprocessing.queues.Queue,
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    try:
        result = target(*args, **kwargs)
        queue.put({"ok": True, "result": result})
    except BaseException as exc:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            }
        )


def run_with_hard_timeout(
    target: Callable[..., Any],
    *args: Any,
    timeout_seconds: float,
    timeout_error_message: str,
    kwargs: dict[str, Any] | None = None,
    shutdown_grace_seconds: float = 0.2,
) -> Any:
    """Run a blocking target in a worker process with a hard wall-clock timeout."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")

    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.queues.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_worker_entry, args=(queue, target, args, kwargs or {}), daemon=True)

    start = perf_counter()
    proc.start()
    proc.join(timeout=timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=shutdown_grace_seconds)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=shutdown_grace_seconds)
        elapsed = perf_counter() - start
        raise TimeoutError(f"{timeout_error_message} (elapsed={elapsed:.3f}s)")

    try:
        payload = queue.get(timeout=0.2)
    except Empty as exc:
        raise RuntimeError("Worker process exited without returning a result payload.") from exc

    if payload.get("ok"):
        return payload.get("result")

    error_type = str(payload.get("error_type", "RuntimeError"))
    message = str(payload.get("error_message", "Unknown worker error"))

    if error_type == "ConnectionError":
        raise ConnectionError(message)
    if error_type in {"ReadTimeout", "TimeoutException", "TimeoutError"}:
        raise TimeoutError(message)
    if error_type in {"ValueError", "OllamaProviderError", "ResponseError"}:
        raise ValueError(message)
    raise RuntimeError(f"{error_type}: {message}")
