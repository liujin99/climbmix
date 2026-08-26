"""
Token count estimation utilities.

Centralizes the char→token estimation heuristic that was duplicated
across 5 files, plus human-readable token-count parsing ("2B", "10M").
"""

import re

import numpy as np
import numpy.typing as npt


DEFAULT_CHARS_PER_TOKEN = 4

_TOKEN_COUNT_RE = re.compile(r'^([0-9]*\.?[0-9]+)\s*([kKmMbB])?$')
_TOKEN_SUFFIX_MULTIPLIERS = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}


def parse_token_count(value) -> int:
    """Parse a token count from "2B", "10M", "500K", "1.5B" or a plain integer.

    Case-insensitive suffix; plain integers pass through unchanged
    (backwards compatible). Raises ValueError on invalid input.
    """
    if isinstance(value, bool):
        raise ValueError(f"Invalid token count: {value!r}")
    if isinstance(value, (int, np.integer)):
        result = int(value)
        if result < 0:
            raise ValueError(f"Token count must be non-negative, got {result}")
        return result

    s = str(value).strip()
    m = _TOKEN_COUNT_RE.match(s)
    if not m:
        raise ValueError(
            f"Invalid token count: {value!r} (expected e.g. '2B', '10M', '500K' or an integer)"
        )
    number, suffix = m.groups()
    result = float(number)
    if suffix is not None:
        result *= _TOKEN_SUFFIX_MULTIPLIERS[suffix.upper()]
    result = int(result)
    if result < 0:
        raise ValueError(f"Token count must be non-negative, got {result}")
    return result


def estimate_tokens_from_chars(
    char_count: int,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> int:
    return max(1, int(char_count / chars_per_token))


def estimate_tokens_from_text(
    text: str,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> int:
    return estimate_tokens_from_chars(len(text), chars_per_token)


def estimate_token_counts_array(
    char_counts: npt.NDArray[np.int64],
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> npt.NDArray[np.int64]:
    return np.maximum(char_counts // chars_per_token, 1).astype(np.int64)
