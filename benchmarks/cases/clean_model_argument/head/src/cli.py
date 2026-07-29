class ProviderClient:
    def __init__(self, model: str) -> None:
        self.model = model


def create_client(model_name: str) -> ProviderClient:
    return ProviderClient(model=model_name)
