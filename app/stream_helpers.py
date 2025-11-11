"""Utilities for streaming message content without losing formatting."""

from __future__ import annotations

import re
from typing import Iterable, Sequence, Tuple

_WHITESPACE_TOKEN_PATTERN = re.compile(r"\s+|\S+")


def _iter_preserving_whitespace(text: str) -> Iterable[str]:
    """Yield each contiguous run of whitespace or non-whitespace characters."""

    for match in _WHITESPACE_TOKEN_PATTERN.finditer(text):
        segment = match.group(0)
        if segment:
            yield segment


def diff_stream_segments(current: str, latest: str) -> Tuple[str, Tuple[str, ...]]:
    """Return the updated message and delta segments without dropping whitespace."""

    if not latest or len(latest) <= len(current):
        return current, tuple()

    delta = latest[len(current) :]
    return latest, tuple(_iter_preserving_whitespace(delta))


def validate_stream_consistency(streamed: str, final: str) -> bool:
    """Ensure streamed content matches what is persisted or returned later."""

    return streamed == final

