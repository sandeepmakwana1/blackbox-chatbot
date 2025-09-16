import logging
from datetime import datetime
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages,
)
from app.schema import ConversationState
from app.config import SUMMARY_TRIGGER_COUNT, MAX_TOKENS_FOR_SUMMARY, TOOLS
from app.summary_agent import summarize_history
from app.config import DEFAULT_MODELS, MAX_TOKENS_FOR_TRIM, SUMMARY_TRIGGER_COUNT
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from app.prompt import (
    PlaygroundChatPrompt,
    PlaygroundRFPContextChatPrompt,
    PlaygroundProposalSectionContextChatPrompt,
)
from app.helper import get_trimmer_object, update_token_tracking
from langchain_openai import ChatOpenAI
from app.constents import ContextType

LOGGER = logging.getLogger("langgraph-nodes")


# summary Nodes
def route_summarize(state: ConversationState) -> str:
    msgs = state.get("messages", [])
    current_token = state.get("tokens", {}).get("current_tokens", 0)
    if len(msgs) >= SUMMARY_TRIGGER_COUNT and current_token >= MAX_TOKENS_FOR_SUMMARY:
        return "summarize"
    return "chat"


# helper
def get_data_from_redis(source_id, key):
    redis_data = ""
    try:
        from common import RedisService

        if source_id:
            redis_data = RedisService.fetch_rfp_data_from_redis(source_id, key) or ""
    except Exception as e:
        print(e)
    return redis_data


async def summarize_node(state: ConversationState):
    history = list(state["messages"])
    if len(history) <= 1:
        return {}

    summary_text, usage = await summarize_history(
        history[:-1], state.get("summary_context", [])
    )
    summary_note = SystemMessage(
        content=f"Summary of prior conversation:\n{summary_text}"
    )

    # Improved token tracking with consistent format
    prior_tokens = state.get("tokens", {"total_tokens": 0, "current_tokens": 0})
    summary_tokens = (usage or {}).get("total_tokens", 0)

    # Reset current tokens since we're summarizing (trimming history)
    merged_tokens = {
        "total_tokens": prior_tokens.get("total_tokens", 0) + summary_tokens,
        "current_tokens": 0,  # Reset since we're summarizing
        "last_request_tokens": summary_tokens,
        "summarization_tokens": summary_tokens,
        "last_request_breakdown": {
            "summarization_tokens": summary_tokens,
            "main_tokens": 0,
        },
        "model": DEFAULT_MODELS["summarize"],
        "timestamp": datetime.utcnow().isoformat(),
    }

    LOGGER.info(
        f"Summarization completed: {summary_tokens} tokens used, conversation reset"
    )
    return {"summary_context": summary_note, "tokens": merged_tokens}


# chat Nodes
async def chat_node(state: ConversationState):
    messages = list(state.get("messages", []))
    language = state.get("language", "English")
    tool = state.get("tool", "")
    contexts = state.get("contexts", [])
    summary = state.get("summary_context")

    # Use new tool and contexts parameters
    use_web = tool == "web"
    has_contexts = bool(contexts)

    # Determine which prompt to use based on contexts
    if has_contexts:
        if len(contexts) == 1 and contexts[0] == "rfp_context":
            chat_system_prompt = PlaygroundRFPContextChatPrompt.system_prompt
            chat_user_prompt = PlaygroundRFPContextChatPrompt.user_prompt
        else:
            chat_system_prompt = (
                PlaygroundProposalSectionContextChatPrompt.system_prompt
            )
            chat_user_prompt = PlaygroundProposalSectionContextChatPrompt.user_prompt
    else:
        chat_system_prompt = PlaygroundChatPrompt.system_prompt
        chat_user_prompt = PlaygroundChatPrompt.user_prompt

    system_prompt = SystemMessagePromptTemplate.from_template(chat_system_prompt)
    user_prompt = HumanMessagePromptTemplate.from_template(chat_user_prompt)
    prompt_template = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

    model_chat = ChatOpenAI(
        model=DEFAULT_MODELS["chat"], temperature=0.7, streaming=True
    )
    trimmed = get_trimmer_object(token_counter_model=model_chat).invoke(messages)

    # Enable web search if tool is "web"
    if use_web:
        model_chat = model_chat.bind_tools([TOOLS["web_search_preview"]])

    # Common prompt args
    prompt_args = {
        "language": language,
        "messages": trimmed,
        "summary_context": summary,
    }

    # Inject RFP context only for context-* modes
    source_id = (state.get("user_id") or "").split("_")[-1]
    context_mapping = {
        ContextType.COST_INFRASTRUCTURE: "COST INFRASTRUCTURE",
        ContextType.COST_LICENSE: "COST LICENSE",
        ContextType.COST_HR: "COST Hourly Wages",
        ContextType.VALIDATION_LEGAL: "VALIDATION LEGAL CHECK DATA",
        ContextType.VALIDATION_TECHNICAL: "VALIDATION TECHNICAL CHECK DATA",
        ContextType.TABLE_OF_CONTENT: "TABLE OF CONTENT",
        ContextType.DEEP_RESEARCH: "DEEP RESEARCH",
        ContextType.USER_PREFERENCE: "USER PREFERENCE",
    }
    prompt_text = " "
    if has_contexts:
        for context in contexts:
            if "content" in context:
                continue  # TODO: NEED TO IMPLEMENT
            # Handle each context type in a more concise way
            elif context in context_mapping:
                if data := get_data_from_redis(source_id=source_id, key=context):
                    section_header = context_mapping[context]
                    prompt_text += f"""
                    [{section_header}]
                    {data}
                    """
        prompt_args["section_context"] = prompt_text
        prompt_args["rfp_context"] = get_data_from_redis(source_id, "rfp_text")
    msgs = prompt_template.format_messages(**prompt_args)

    response: AIMessage = await model_chat.ainvoke(msgs)

    token_info = update_token_tracking(
        state=state,
        response=response,
        model_name=DEFAULT_MODELS["chat"],
        additional_tokens=0,
    )
    return {"messages": [response], "summary_context": summary, "tokens": token_info}
