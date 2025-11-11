"""Utilities for streaming message content without losing formatting."""

from __future__ import annotations

import re
from typing import Iterable, Tuple

_WHITESPACE_TOKEN_PATTERN = re.compile(r"\s+|\S+")


def _iter_preserving_whitespace(text: str) -> Iterable[str]:
    """Yield each contiguous run of whitespace or non-whitespace characters."""

    for match in _WHITESPACE_TOKEN_PATTERN.finditer(text):
        segment = match.group(0)
        if segment:
            yield segment


def diff_stream_segments(current: str, latest: str) -> Tuple[str, Tuple[str, ...]]:
    """Return updated full text plus delta segments.

    This expects ``latest`` to represent the *entire* response accumulated so far
    (e.g., from conversation state). If the backend only provides the newest
    delta, prefer :func:`split_stream_segments` instead.
    """

    if not latest or len(latest) <= len(current):
        return current, tuple()

    delta = latest[len(current) :]
    return latest, tuple(_iter_preserving_whitespace(delta))


def split_stream_segments(delta: str) -> Tuple[str, ...]:
    """Split a delta chunk into whitespace-preserving segments."""

    if not delta:
        return tuple()
    return tuple(_iter_preserving_whitespace(delta))


def validate_stream_consistency(streamed: str, final: str) -> bool:
    """Ensure streamed content matches what is persisted or returned later."""

    return streamed == final
