"""
Braintrust logging integration helpers.

This module configures Braintrust telemetry for LangChain/LangGraph runs when
the relevant environment variables are provided. All imports are guarded so the
application can run without Braintrust installed.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence, Union
from uuid import UUID

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
        from braintrust_langchain import (
            BraintrustCallbackHandler,
            set_global_handler,
        )
    except ImportError as exc:
        LOGGER.error(
            "Braintrust packages are not available. "
            "Install 'braintrust' and 'braintrust-langchain' to enable logging: %s",
            exc,
        )
        return None

    class SafeBraintrustCallbackHandler(BraintrustCallbackHandler):
        """Guard against context token mismatches in async executions."""

        def _end_span(
            self,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            input: Optional[Any] = None,
            output: Optional[Any] = None,
            expected: Optional[Any] = None,
            error: Optional[str] = None,
            tags: Optional[Sequence[str]] = None,
            scores: Optional[Mapping[str, Union[int, float]]] = None,
            metadata: Optional[Mapping[str, Any]] = None,
            metrics: Optional[Mapping[str, Union[int, float]]] = None,
            dataset_record_id: Optional[str] = None,
        ) -> Any:
            if run_id not in self.spans:
                return

            if run_id in self.skipped_runs:
                self.skipped_runs.discard(run_id)
                return

            span = self.spans.pop(run_id)

            if getattr(self, "root_run_id", None) == run_id:
                self.root_run_id = None

            try:
                span.log(
                    input=input,
                    output=output,
                    expected=expected,
                    error=error,
                    tags=tags,
                    scores=scores,
                    metadata={
                        **({"tags": tags} if tags else {}),
                        **(metadata or {}),
                    },
                    metrics=metrics,
                    dataset_record_id=dataset_record_id,
                )
            except Exception:
                LOGGER.exception("Failed to log Braintrust span data.")

            try:
                span.unset_current()
            except ValueError as exc:
                LOGGER.debug(
                    "Ignoring Braintrust context token mismatch when unsetting span: %s",
                    exc,
                )
            except Exception:
                LOGGER.exception("Failed to unset Braintrust span context.")

            try:
                span.end()
            except Exception:
                LOGGER.exception("Failed to end Braintrust span.")

    try:
        init_logger(project=BRAINTRUST_PROJECT, api_key=BRAINTRUST_API_KEY)
        _handler = SafeBraintrustCallbackHandler()
        set_global_handler(_handler)
        _initialized = True
        LOGGER.info(
            "Braintrust logging initialized for project '%s'", BRAINTRUST_PROJECT
        )
        return _handler
    except Exception:  # pragma: no cover - defensive logging
        LOGGER.exception("Failed to initialize Braintrust logging.")
        return None
