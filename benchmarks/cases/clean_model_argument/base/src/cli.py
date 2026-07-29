class ProviderClient:
    def __init__(self, model: str) -> None:
        self.model = model


def create_client() -> ProviderClient:
    return ProviderClient(model="baseline")
