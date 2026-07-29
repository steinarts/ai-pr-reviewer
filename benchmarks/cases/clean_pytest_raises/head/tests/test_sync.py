import pytest


def parse_positive(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError("negative")
    return parsed


def test_parse_positive_rejects_negative_value() -> None:
    with pytest.raises(ValueError):
        parse_positive("-1")
