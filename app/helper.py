from dotenv import load_dotenv

load_dotenv()

import logging
from datetime import datetime
from typing import Dict, Any

from langchain_core.messages import AIMessage
from langchain_core.messages.utils import trim_messages
from openai import OpenAIError, RateLimitError, APITimeoutError, APIConnectionError

# Your app-specific prompts/config
from app.config import DEFAULT_MODELS, MAX_TOKENS_FOR_TRIM
from app.schema import ConversationState

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("langgraph-app")


def update_token_tracking(
    state: ConversationState,
    response: AIMessage,
    model_name: str,
    additional_tokens: int = 0,
) -> Dict:
    """Track token usage across the conversation with consistent error handling."""
    prior = state.get("tokens", {})
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
