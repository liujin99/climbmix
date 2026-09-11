"""
Token count estimation utilities.

Centralizes the char→token estimation heuristic that was duplicated
across 5 files, plus human-readable token-count parsing ("2B", "10M").
"""

import re

import numpy as np
import numpy.typing as npt


DEFAULT_CHARS_PER_TOKEN = 4

_TOKEN_COUNT_RE = re.compile(r'^([0-9]*\.?[0-9]+)\s*([kKmMgGtTbB][iI]?)?$')
_TOKEN_SUFFIX_MULTIPLIERS = {
    'K': 1_000, 'M': 1_000_000, 'G': 1_000_000_000, 'T': 1_000_000_000_000,
    'B': 1_000_000_000,  # legacy decimal billion
    'KI': 1_024, 'MI': 1_048_576, 'GI': 1_073_741_824, 'TI': 1_099_511_627_776,
    'BI': 1_073_741_824,
}


def parse_token_count(value) -> int:
    """Parse a token count from "2B", "10M", "500K", "1.5B", "500Mi" or a
    plain integer.

    Case-insensitive suffix. Decimal suffixes (K/M/B, powers of 1000) are
    the conventional config format (e.g. "2B", "640M") — budget ratios stay
    exact and readable at any scale. Binary suffixes (Ki/Mi/Gi/Ti/Bi,
    powers of 1024) align exactly with step-count arithmetic: the d20/d28
    total_batch_size is 1,048,576 = 2^20 (server-measured 2026-09-11), so
    "N Mi" = N steps exactly ("1000Mi" = the prod1/prod2 exact 1000-step
    pair, kept for legacy replication). Plain integers pass through
    unchanged (backwards compatible). Raises ValueError on invalid input.
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
