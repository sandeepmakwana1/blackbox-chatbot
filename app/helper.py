from dotenv import load_dotenv

load_dotenv()

import logging
import os
import time
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

from langchain_core.messages import AIMessage
from langchain_core.messages.utils import trim_messages
from openai import APIConnectionError, APITimeoutError, OpenAIError, RateLimitError

from app.config import DEFAULT_MODELS, MAX_TOKENS_FOR_TRIM
from app.schema import ConversationState
from app.constants import ContextType
from common.datastore import S3StoreService
from common.usage import calculate_and_store_playground_usage

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("langgraph-playground")


def _resolve_bucket_name() -> Optional[str]:
    bucket = os.getenv("AWS_BATCH_S3_BUCKET")
    if bucket:
        return bucket
    prefix = os.getenv("SSM_PREFIX")
    if prefix:
        return f"{prefix}-batch-processing"
    LOGGER.error(
        "Batch context requested but neither AWS_BATCH_S3_BUCKET nor SSM_PREFIX is configured."
    )
    return None


def get_context_data(source_id, key):
    bucket_name = _resolve_bucket_name()
    if not bucket_name:
        raise ValueError(
            "Bucket name could not be resolved for context data retrieval."
        )
    if key in (
        ContextType.VALIDATION_LEGAL,
        ContextType.VALIDATION_TECHNICAL,
        ContextType.VALIDATION_CHECKLIST,
    ):
        key = ContextType.VALIDATION_RESULTS
    try:
        data = S3StoreService.get(
            bucket_name=bucket_name, source_id=source_id, stage_name=key
        )
    except Exception as e:
        LOGGER.error(
            f"Error retrieving context data from S3: {e}",
            extra={"bucket": bucket_name, "source_id": source_id, "stage_name": key},
        )
        data = ""
    return data


def update_token_tracking(
    state: ConversationState,
    response: AIMessage,
    model_name: str,
    additional_tokens: int = 0,
    *,
    stage_name: str = "playground_chat",
    source_id: Optional[str] = None,
    request_id: Optional[str] = None,
    message_text: Optional[str] = None,
    cb: Optional[Any] = None,
) -> Dict[str, Any]:
    """Track token usage across the conversation with consistent error handling."""
    prior = state.get("tokens")
    if prior is None:
        prior = {}
    prior_total = prior.get("total_tokens", 0)
    prior_current = prior.get("current_tokens", 0)

    # Extract usage metadata with fallback
    usage = getattr(response, "usage_metadata", None)
    # Prefer callback measurements when available; fall back if zeros
    if cb is not None:
        cb_input = getattr(cb, "prompt_tokens", 0) or 0
        cb_output = getattr(cb, "completion_tokens", 0) or 0
        cb_cached = getattr(cb, "prompt_tokens_details", {}).get("cached_tokens", 0)
        cb_reasoning = getattr(cb, "completion_tokens_details", {}).get(
            "reasoning_tokens", 0
        )
        cb_total = getattr(cb, "total_tokens", None)
        # Only override when callback captured anything meaningful
        if cb_input or cb_output or cb_total:
            usage = {
                "input_tokens": cb_input,
                "output_tokens": cb_output,
                "cached_tokens": cb_cached,
                "reasoning_tokens": cb_reasoning,
                "total_tokens": cb_total,
            }
    if usage:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
    else:
        # Best-effort fallback using content length estimation
        try:
            content = (
                serialize_content_to_string(response.content)
                if hasattr(response, "content")
                else ""
            )
            output_tokens = max(1, len(content) // 4)  # Rough estimation
            input_est = 0
            if message_text:
                input_est = max(1, len(str(message_text)) // 4)
            input_tokens = input_est
            total_tokens = input_tokens + output_tokens
        except Exception as e:
            LOGGER.warning(f"Error estimating tokens from content: {e}")
            input_tokens = output_tokens = total_tokens = 1  # Minimal fallback

    main_tokens = total_tokens
    total_request_tokens = main_tokens + additional_tokens
    new_total_tokens = prior_total + total_request_tokens
    new_current_tokens = prior_current + main_tokens

    token_info = {
        "total_tokens": new_total_tokens,
        "current_tokens": new_current_tokens,
        "last_request_tokens": total_request_tokens,
        "last_request_breakdown": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "main_tokens": main_tokens,
            "additional_tokens": additional_tokens,
        },
        "model": model_name,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if additional_tokens > 0:
        token_info["breakdown"] = {
            "main_response_tokens": main_tokens,
            "summarization_tokens": additional_tokens,
            "main_model": model_name,
            "summary_model": DEFAULT_MODELS["summarize"],
        }

    LOGGER.info(
        f"Token tracking: {total_request_tokens} tokens this request (input: {input_tokens}, output: {output_tokens}, additional: {additional_tokens}), {new_total_tokens} total"
    )

    # Emit usage to shared logger when identifiers are available
    if source_id:

        class _UsageAdapter:
            def __init__(self, prompt_tokens: int, completion_tokens: int):
                self.prompt_tokens = prompt_tokens
                self.completion_tokens = completion_tokens
                self.prompt_tokens_details = {"cached_tokens": 0}
                self.completion_tokens_details = {"reasoning_tokens": 0}
                self.message = message_text

        try:
            adapter = _UsageAdapter(input_tokens, output_tokens)
            unique_request = (
                request_id
                or state.get("thread_id")
                or state.get("user_id")
                or f"msg-{int(time.time() * 1000)}"
            )
            calculate_and_store_playground_usage(
                LOGGER,
                adapter,
                source_id,
                stage_name,
                unique_request,
                message_text=message_text,
            )
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("Failed to log usage for playground: %s", e)

    return token_info


def get_trimmer_object(
    token_counter_model,
    max_tokens=MAX_TOKENS_FOR_TRIM,
    strategy="last",
    include_system=True,
    allow_partial=False,
    start_on="human",
):
    trimmer = trim_messages(
        max_tokens=max_tokens,
        strategy=strategy,
        token_counter=token_counter_model,
        include_system=include_system,
        allow_partial=allow_partial,
        start_on=start_on,
    )
    return trimmer


def handle_openai_error(
    error: Exception, context: str = "OpenAI operation"
) -> AIMessage:
    """
    Handle OpenAI API errors and return a graceful AIMessage response.

    Args:
        error: The exception that occurred
        context: Context description for logging

    Returns:
        AIMessage with user-friendly error message
    """
    if isinstance(error, RateLimitError):
        message = (
            "I'm experiencing high demand right now. Please try again in a few moments."
        )
        LOGGER.warning(f"{context} - Rate limit exceeded: {error}")
    elif isinstance(error, APITimeoutError):
        message = "The request took too long to process. Please try again."
        LOGGER.warning(f"{context} - Request timeout: {error}")
    elif isinstance(error, APIConnectionError):
        message = "I'm having trouble connecting to my services. Please check your connection and try again."
        LOGGER.error(f"{context} - Connection error: {error}")
    elif isinstance(error, OpenAIError):
        message = "I'm experiencing a technical issue. Please try again in a moment."
        LOGGER.error(f"{context} - OpenAI API error: {error}")
    else:
        message = "I'm unable to process your request at the moment. Please try again."
        LOGGER.error(f"{context} - Unexpected error: {error}")

    return AIMessage(content=message)


def safe_ai_invoke(func, *args, context: str = "AI operation", **kwargs):
    """
    Safely invoke an AI function with error handling.

    Args:
        func: The function to invoke
        *args: Positional arguments for the function
        context: Context description for error logging
        **kwargs: Keyword arguments for the function

    Returns:
        Either the function result or an error AIMessage
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return handle_openai_error(e, context)


async def safe_ai_invoke_async(func, *args, context: str = "AI operation", **kwargs):
    """
    Safely invoke an async AI function with error handling.

    Args:
        func: The async function to invoke
        *args: Positional arguments for the function
        context: Context description for error logging
        **kwargs: Keyword arguments for the function

    Returns:
        Either the function result or an error AIMessage
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        return handle_openai_error(e, context)


def serialize_content_to_string(content: Any) -> str:
    """
    Convert various content formats (string, list, dict) into a clean string representation.

    Args:
        content: Content to serialize (can be string, list, dict, or other)

    Returns:
        Clean string representation of the content
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle list of strings or dictionaries
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                # Handle dict with 'text' or 'content' key
                text = item.get("text", item.get("content", ""))
                if text:
                    text_parts.append(str(text))
            else:
                text_parts.append(str(item))
        return "".join(text_parts)
    elif isinstance(content, dict):
        # Handle dictionary content
        text = content.get("text", content.get("content", ""))
        return str(text) if text else str(content)
    else:
        return str(content)
