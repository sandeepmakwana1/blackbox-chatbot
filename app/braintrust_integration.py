"""
Braintrust logging integration helpers.

This module configures Braintrust telemetry for LangChain/LangGraph runs when
the relevant environment variables are provided. All imports are guarded so the
application can run without Braintrust installed.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import BRAINTRUST_API_KEY, BRAINTRUST_ENABLED, BRAINTRUST_PROJECT

LOGGER = logging.getLogger("braintrust-integration")

_initialized = False
_handler = None


def setup_braintrust_logging() -> Optional[object]:
    """
    Configure the global Braintrust callback handler if enabled.

    Returns the instantiated handler when successful, or None if Braintrust is
    disabled/misconfigured/unavailable.
    """
    global _initialized, _handler

    if not BRAINTRUST_ENABLED:
        LOGGER.debug("Braintrust logging disabled via configuration.")
        return None

    if _initialized:
        return _handler

    if not BRAINTRUST_API_KEY:
        LOGGER.warning(
            "Braintrust logging requested but BRAINTRUST_API_KEY is not set."
        )
        return None

    try:
        from braintrust import init_logger
        from braintrust_langchain import BraintrustCallbackHandler, set_global_handler
    except ImportError as exc:
        LOGGER.error(
            "Braintrust packages are not available. "
            "Install 'braintrust' and 'braintrust-langchain' to enable logging: %s",
            exc,
        )
        return None

    try:
        init_logger(project=BRAINTRUST_PROJECT, api_key=BRAINTRUST_API_KEY)
        _handler = BraintrustCallbackHandler()
        set_global_handler(_handler)
        _initialized = True
        LOGGER.info("Braintrust logging initialized for project '%s'", BRAINTRUST_PROJECT)
        return _handler
    except Exception:  # pragma: no cover - defensive logging
        LOGGER.exception("Failed to initialize Braintrust logging.")
        return None
