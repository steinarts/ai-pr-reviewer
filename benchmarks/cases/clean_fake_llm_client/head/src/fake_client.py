class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self._response
