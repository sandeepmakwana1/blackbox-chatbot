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

from app.prompt import PlaygroundChatPrompt
from app.helper import get_trimmer_object , update_token_tracking
from langchain_openai import ChatOpenAI


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
    # working = [summary_note, history[-1]]

    
    # FIX: merge token_tracking instead of overwriting it
    prior_tt = state.get("tokens", {"total_tokens": 0})
    prior_tt["current_tokens"] = 0
    merged_tt = {**prior_tt, "summarization_tokens": (usage or {}).get("total_tokens", 0)}

    return {"summary_context": summary_note, "token": merged_tt}


# chat Nodes
async def chat_node(state: ConversationState):
    base = list(state["messages"])
    language = state.get("language", "English")
    
    system_prompt = SystemMessagePromptTemplate.from_template(
        PlaygroundChatPrompt.system_prompt
    )
    user_prompt = HumanMessagePromptTemplate.from_template(
    PlaygroundChatPrompt.user_prompt
    )
    prompt_template = ChatPromptTemplate.from_messages(
    [system_prompt, user_prompt])
    model_chat = ChatOpenAI(model=DEFAULT_MODELS["chat"], temperature=0.7, streaming=True)
    
    # Check conversation type for web search functionality
    conversation_type = state.get("conversation_type")
    if conversation_type == "web":
        model_chat = model_chat.bind_tools([TOOLS["web_search_preview"]])
    
    trimmed = get_trimmer_object(token_counter_model=model_chat).invoke(base)
    msgs = prompt_template.format_messages(
        language=language,
        messages=trimmed,
        summary_context = state.get("summary_context") 
    )
    response: AIMessage = await model_chat.ainvoke(msgs)

    addl = state.get("tokens", {}).get("total_tokens", 0)
    token_info = update_token_tracking(
        state=state, response=response, model_name=DEFAULT_MODELS["chat"], additional_tokens=addl
    )
    return {"messages": [response], "summary_context": state.get("summary_context", []), "tokens": token_info}



