class BlockingHTTP:
    def post(self, path: str, json: dict[str, object], timeout: float) -> dict[str, object]:
        return {"path": path, "json": json, "timeout": timeout}


class ProviderClient:
    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        self._http = BlockingHTTP()

    def generate(self, prompt: str) -> dict[str, object]:
        return self._http.post(
            "/chat",
            json={"prompt": prompt},
            timeout=self.timeout_seconds,
        )
