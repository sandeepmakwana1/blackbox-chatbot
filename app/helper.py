from dotenv import load_dotenv

load_dotenv()

import logging
import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

from langchain_core.messages import AIMessage
from langchain_core.messages.utils import trim_messages
from openai import APIConnectionError, APITimeoutError, OpenAIError, RateLimitError

# Your app-specific prompts/config
from app.chat_manager import ChatManager
from app.config import DEFAULT_MODELS, MAX_TOKENS_FOR_TRIM
from app.constants import BATCH_CONTEXT_PATHS
from app.schema import ConversationState

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("langgraph-playground")


_chat_manager_instance: Optional[ChatManager] = None


def get_chat_manager() -> ChatManager:
    """Return a singleton ChatManager instance for shared DB access."""
    global _chat_manager_instance
    if _chat_manager_instance is None:
        _chat_manager_instance = ChatManager()
    return _chat_manager_instance


def _normalize_source_id(raw_source_id) -> Optional[int]:
    if not raw_source_id:
        return None
    try:
        return int(raw_source_id)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=256)
def _get_batch_info(numeric_source_id: int) -> Tuple[bool, Optional[str]]:
    try:
        manager = get_chat_manager()
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.error(
            "Failed to initialize ChatManager for batch status lookup: %s",
            exc,
            exc_info=True,
        )
        return False, None

    is_batch = False
    batch_id: Optional[str] = None
    try:
        with manager.get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT batch_id FROM track WHERE source_id = %s LIMIT 1",
                    (numeric_source_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    batch_id = None
                elif isinstance(row, (list, tuple)):
                    batch_id = row[0]
                elif isinstance(row, dict):
                    batch_id = row.get("batch_id")
                else:
                    batch_id = getattr(row, "batch_id", None)
                batch_id = str(batch_id) if batch_id else None
                is_batch = bool(batch_id)
    except Exception as exc:
        LOGGER.error(
            "Failed to determine batch status for source_id=%s: %s",
            numeric_source_id,
            exc,
            exc_info=True,
        )
        is_batch = False
        batch_id = None

    return is_batch, batch_id


def get_batch_context(source_id) -> Tuple[bool, Optional[str], Optional[int]]:
    numeric_source_id = _normalize_source_id(source_id)
    if numeric_source_id is None:
        return False, None, None
    is_batch, batch_id = _get_batch_info(numeric_source_id)
    return is_batch, batch_id, numeric_source_id


def _resolve_batch_bucket() -> Optional[str]:
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
    is_batch, batch_id, numeric_source_id = get_batch_context(source_id)
    print("is_batch =========>", is_batch)
    print("batch_id =========>", batch_id)
    print("numeric_source_id =========>", numeric_source_id)

    redis_data = ""
    if not is_batch:
        try:
            from common import RedisService

            if source_id:
                redis_data = (
                    RedisService.fetch_rfp_data_from_redis(source_id, key) or ""
                )
        except Exception as exc:
            LOGGER.error(
                "Failed to get data from Redis for source_id=%s, key=%s: %s",
                source_id,
                key,
                exc,
                exc_info=True,
            )
    else:
        bucket_name = _resolve_batch_bucket()
        print("bucket_name =========>", bucket_name)
        if not bucket_name or not batch_id or numeric_source_id is None:
            return True, ""

        s3_suffix = BATCH_CONTEXT_PATHS.get(str(key))
        print("s3_suffix =========>", s3_suffix)
        if not s3_suffix:
            LOGGER.warning(
                "No batch S3 mapping defined for key=%s (source_id=%s)",
                key,
                source_id,
            )
            return True, ""

        try:
            from common.datastore import (
                S3DataNotFoundError,
                S3OperationError,
                get_s3_data,
            )
        except Exception as exc:
            LOGGER.error(
                "Failed to import S3 datastore helpers for batch context lookup: %s",
                exc,
                exc_info=True,
            )
            return True, ""

        s3_key = f"{batch_id}/{numeric_source_id}/{s3_suffix}"
        try:
            redis_data = get_s3_data(s3_key, bucket_name)
            print("s3 =========>", redis_data)
        except S3DataNotFoundError:
            LOGGER.warning(
                "Batch context not found in S3 (source_id=%s, key=%s, s3_key=%s)",
                numeric_source_id,
                key,
                s3_key,
            )
            redis_data = ""
        except S3OperationError as exc:
            LOGGER.error(
                "S3 operation failed for batch context (source_id=%s, key=%s, s3_key=%s): %s",
                numeric_source_id,
                key,
                s3_key,
                exc,
                exc_info=True,
            )
            redis_data = ""
        except Exception as exc:  # pragma: no cover - unexpected failure handling
            LOGGER.error(
                "Unexpected error when fetching batch context from S3 (source_id=%s, key=%s, s3_key=%s): %s",
                numeric_source_id,
                key,
                s3_key,
                exc,
                exc_info=True,
            )
            redis_data = ""
    return is_batch, redis_data


def update_token_tracking(
    state: ConversationState,
    response: AIMessage,
    model_name: str,
    additional_tokens: int = 0,
) -> Dict[str, Any]:
    """Track token usage across the conversation with consistent error handling."""
    prior = state.get("tokens")
    if prior is None:
        prior = {}
    prior_total = prior.get("total_tokens", 0)
    prior_current = prior.get("current_tokens", 0)

    # Extract usage metadata with fallback
    usage = getattr(response, "usage_metadata", None)
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
            input_tokens = 0  # We don't have input token info in fallback
            output_tokens = max(
                1, len(content) // 4
            )  # Rough estimation: 4 chars per token
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
