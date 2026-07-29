class BlockingHTTP:
    def post(
        self, path: str, json: dict[str, object], timeout: float | None = None
    ) -> dict[str, object]:
        return {"path": path, "json": json, "timeout": timeout}


class ProviderClient:
    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        self._http = BlockingHTTP()

    def generate(self, prompt: str) -> dict[str, object]:
        configured_timeout = self.timeout_seconds
        _ = configured_timeout
        # Defect: configured timeout is ignored in the blocking call below.
        return self._http.post(
            "/chat",
            json={"prompt": prompt},
        )
