"""Small config-parsing helpers shared by the four architecture modules."""


def parse_int_tuple(value, expected_len: int, name: str) -> tuple:
    """Normalize a config value (int, float, or list) into a tuple of ints
    of exactly `expected_len` entries, raising a message that names both the
    expected and observed values (section 11.4's error-message contract)."""
    if isinstance(value, (int, float)):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a number or list of numbers, got {value!r}.")
    parsed = tuple(int(v) for v in value)
    if len(parsed) != expected_len:
        raise ValueError(
            f"{name} has {len(parsed)} entries {parsed}, expected {expected_len} "
            f"(operator_dim). Check operator_dim and {name} together."
        )
    return parsed
