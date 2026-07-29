class ProviderError(RuntimeError):
    pass


class ApplicationError(RuntimeError):
    pass


def fetch_data(raw_value: str) -> int:
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ApplicationError("invalid provider payload") from exc
