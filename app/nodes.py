import logging
from datetime import datetime
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, trim_messages
)
from app.schema import ConversationState
from app.config import SUMMARY_TRIGGER_COUNT, MAX_TOKENS_FOR_SUMMARY, TOOLS
from app.summary_agent import summarize_history
from app.config import DEFAULT_MODELS, MAX_TOKENS_FOR_TRIM, SUMMARY_TRIGGER_COUNT
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, trim_messages
)
from langchain_core.prompts import (
    ChatPromptTemplate, MessagesPlaceholder,
    SystemMessagePromptTemplate, HumanMessagePromptTemplate
)

from app.prompt import PlaygroundChatPrompt , PlaygroundContextChatPrompt
from app.helper import get_trimmer_object , update_token_tracking
from langchain_openai import ChatOpenAI

LOGGER = logging.getLogger("langgraph-nodes")


# summary Nodes 
def route_summarize(state: ConversationState) -> str:
    msgs = state.get("messages", [])
    current_token = state.get("tokens", {}).get("current_tokens", 0)
    if (len(msgs) >= SUMMARY_TRIGGER_COUNT and current_token >= MAX_TOKENS_FOR_SUMMARY):
        return "summarize" 
    return "chat"


async def summarize_node(state: ConversationState):
    history = list(state["messages"])
    if len(history) <= 1:
        return {}
    
    summary_text, usage = await summarize_history(history[:-1], state.get("summary_context",[]))
    summary_note = SystemMessage(content=f"Summary of prior conversation:\n{summary_text}")
    
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
            "main_tokens": 0
        },
        "model": DEFAULT_MODELS["summarize"],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    LOGGER.info(f"Summarization completed: {summary_tokens} tokens used, conversation reset")
    return {"summary_context": summary_note, "tokens": merged_tokens}


# chat Nodes
async def chat_node(state: ConversationState):
    base = list(state["messages"])
    language = state.get("language", "English")
    conversation_type = state.get("conversation_type")

    if conversation_type.lower() == "context-chat" or conversation_type.lower() == "context-web":
        chat_system_prompt = PlaygroundContextChatPrompt.system_prompt
        chat_user_prompt = PlaygroundContextChatPrompt.user_prompt
    else:
        chat_system_prompt = PlaygroundChatPrompt.system_prompt
        chat_user_prompt = PlaygroundChatPrompt.user_prompt


    system_prompt = SystemMessagePromptTemplate.from_template(
        chat_system_prompt
    )
    user_prompt = HumanMessagePromptTemplate.from_template(
        chat_user_prompt
    )
    prompt_template = ChatPromptTemplate.from_messages(
    [system_prompt, user_prompt])

    model_chat = ChatOpenAI(model=DEFAULT_MODELS["chat"], temperature=0.7, streaming=True)
    
    # Check conversation type for web search functionality
    if conversation_type == "web" or conversation_type == "context-web":
        model_chat = model_chat.bind_tools([TOOLS["web_search_preview"]])
    
    trimmed = get_trimmer_object(token_counter_model=model_chat).invoke(base)

    if conversation_type.lower() == "context-chat" or conversation_type.lower() == "context-web":
        from common import RedisService
        source_id = state.get("thread_id","").split("_")
        if source_id:
            source_id = source_id[0]
            rfp_context = RedisService.fetch_rfp_data_from_redis(source_id, "rfp_text")
        else:
            rfp_context = ""

        msgs = prompt_template.format_messages(
                language=language,
                messages=trimmed,
                summary_context = state.get("summary_context"),
                rfp_context = rfp_context
            )
    else:
        msgs = prompt_template.format_messages(
            language=language,
            messages=trimmed,
            summary_context = state.get("summary_context")
        )
    response: AIMessage = await model_chat.ainvoke(msgs)

    token_info = update_token_tracking(
        state=state, response=response, model_name=DEFAULT_MODELS["chat"], additional_tokens=0
    )
    return {"messages": [response], "summary_context": state.get("summary_context", []), "tokens": token_info}



