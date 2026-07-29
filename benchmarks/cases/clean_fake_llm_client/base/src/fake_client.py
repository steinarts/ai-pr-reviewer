class FakeLLMClient:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "{}"
