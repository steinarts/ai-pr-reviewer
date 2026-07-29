class ProviderError(RuntimeError):
    pass


class ApplicationError(RuntimeError):
    pass


def fetch_data(raw_value: str) -> int:
    return int(raw_value)
