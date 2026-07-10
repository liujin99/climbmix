"""
Token count estimation utilities.

Centralizes the char→token estimation heuristic that was duplicated
across 5 files.
"""

import numpy as np
import numpy.typing as npt


DEFAULT_CHARS_PER_TOKEN = 4


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
