from dotenv import load_dotenv
load_dotenv()

import logging
from typing import Dict, Optional, Sequence


from langchain_core.messages import (
    BaseMessage, AIMessage
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate, HumanMessagePromptTemplate
)

from app.prompt import PlaygroundChatSummaryPrompt
from app.config import DEFAULT_MODELS
from langchain_openai import ChatOpenAI
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("langgraph-app")
from app.helper import get_trimmer_object

async def summarize_history(history: Sequence[BaseMessage], previous_summary = "") -> tuple[str, Optional[Dict]]:
    """Summarize long histories; return concise, actionable context and token info."""
    system_prompt = SystemMessagePromptTemplate.from_template(
        PlaygroundChatSummaryPrompt.system_prompt
    )
    user_prompt = HumanMessagePromptTemplate.from_template(
        PlaygroundChatSummaryPrompt.user_prompt
    )
    prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

    model_summarize = ChatOpenAI(model=DEFAULT_MODELS["summarize"], temperature=0.2)
    trimmer = get_trimmer_object(token_counter_model=model_summarize)
    history = trimmer.invoke(history)
    msgs = prompt.format_messages(history=list(history), previous_summary=previous_summary)

    # FIX: 'self' was undefined; use the global summarize model
    summary_msg: AIMessage = await model_summarize.ainvoke(msgs)

    usage = getattr(summary_msg, "usage_metadata", None)
    if usage:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens
        usage_info = {
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": DEFAULT_MODELS["summarize"],
        }
    else:
        # Fallback estimate if provider doesn't return usage
        est_out = max(1, len(summary_msg.content) // 4)
        est_in = sum(len(getattr(m, "content", "")) for m in history) // 4
        usage_info = {
            "total_tokens": est_in + est_out,
            "input_tokens": est_in,
            "output_tokens": est_out,
            "model": DEFAULT_MODELS["summarize"],
        }

    LOGGER.info(f"Summarization used {usage_info['total_tokens']} tokens (model: {usage_info['model']})")
    return summary_msg.content.strip(), usage_info