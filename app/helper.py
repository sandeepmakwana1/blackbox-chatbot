
# fixed_langgraph_app.py

from dotenv import load_dotenv
load_dotenv()

import logging


from langchain_core.messages import AIMessage
from langchain_core.messages.utils import trim_messages

# Your app-specific prompts/config
from app.config import DEFAULT_MODELS, MAX_TOKENS_FOR_TRIM
from app.schema import ConversationState
from typing import Dict

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("langgraph-app")


def update_token_tracking(
    state: ConversationState, response: AIMessage, model_name: str, additional_tokens: int = 0
) -> Dict:
    """Track token usage across the conversation (counts only)."""
    prior = state.get("tokens", {})
    prior_total = prior.get("total_tokens", 0)

    usage = getattr(response, "usage_metadata", None)
    if usage:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
    else:
        # Best-effort fallback
        input_tokens = 0
        output_tokens = max(1, len(response.content) // 4)

    main_tokens = input_tokens + output_tokens
    total_request_tokens = main_tokens + additional_tokens
    new_total_tokens = prior_total + total_request_tokens

    token_info = {
        "total_tokens": new_total_tokens,
        "last_request_tokens": total_request_tokens,
        "current_tokens": main_tokens + prior.get("current_tokens", 0),
        "model": model_name,
    }
    if additional_tokens > 0:
        token_info["breakdown"] = {
            "main_response_tokens": main_tokens,
            "summarization_tokens": additional_tokens,
            "main_model": model_name,
            "summary_model": DEFAULT_MODELS["summarize"],
        }

    LOGGER.info(f"Token tracking: {total_request_tokens} tokens this request, {new_total_tokens} total")
    return token_info


def get_trimmer_object(token_counter_model, max_tokens=MAX_TOKENS_FOR_TRIM, strategy="last", include_system=True, allow_partial=False, start_on="human"):
    trimmer = trim_messages(
            max_tokens=max_tokens,
            strategy=strategy,
            token_counter=token_counter_model,
            include_system=include_system,
            allow_partial=allow_partial,
            start_on=start_on,
        )
    return trimmer