import logging
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

from app.config import (
    DEFAULT_MODELS,
    MAX_TOKENS_FOR_SUMMARY,
    SUMMARY_TRIGGER_COUNT,
    TOOLS,
)
from app.constants import ContextType
from app.helper import (
    get_context_data,
    get_trimmer_object,
    update_token_tracking,
)
from app.prompt import (
    PlaygroundChatPrompt,
    PlaygroundProposalSectionContextChatPrompt,
    PlaygroundRFPContextChatPrompt,
)
from app.schema import ConversationState
from app.summary_agent import summarize_history

LOGGER = logging.getLogger("langgraph-nodes")


# summary Nodes
def route_summarize(state: ConversationState) -> str:
    msgs = state.get("messages", [])
    tokens = state.get("tokens")
    current_token = tokens.get("current_tokens", 0) if tokens else 0
    if len(msgs) >= SUMMARY_TRIGGER_COUNT and current_token >= MAX_TOKENS_FOR_SUMMARY:
        return "summarize"
    return "chat"


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
    prior_total = prior_tokens.get("total_tokens", 0) if prior_tokens else 0
    merged_tokens = {
        "total_tokens": prior_total + summary_tokens,
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
    metadata = state.get("metadata") or {}

    # Append human-visible attachment context for any uploaded files
    file_urls = []
    if isinstance(metadata, dict):
        files = metadata.get("files")
        if isinstance(files, list):
            for file in files:
                url = file.get("url") if isinstance(file, dict) else None
                if url:
                    file_urls.append(url)

    if file_urls:
        # Attach images to the most recent human message so text and images stay together
        last_human_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], HumanMessage):
                last_human_idx = idx
                break

        def _coerce_to_blocks(content):
            if isinstance(content, list):
                return list(content)
            if isinstance(content, str):
                return [{"type": "text", "text": content}]
            return [{"type": "text", "text": str(content)}]

        image_blocks = [
            {"type": "image_url", "image_url": {"url": url}} for url in file_urls
        ]

        if last_human_idx is not None:
            last_msg = messages[last_human_idx]
            content_blocks = _coerce_to_blocks(getattr(last_msg, "content", ""))
            content_blocks.extend(image_blocks)
            messages[last_human_idx] = HumanMessage(
                content=content_blocks,
                additional_kwargs=getattr(last_msg, "additional_kwargs", {}),
                response_metadata=getattr(last_msg, "response_metadata", {}),
                id=getattr(last_msg, "id", None),
            )
        else:
            # Fallback: create a new human message if none exist
            content_blocks = [{"type": "text", "text": "Attached images for context."}]
            content_blocks.extend(image_blocks)
            messages.append(HumanMessage(content=content_blocks))

    # Use new tool and contexts parameters
    use_web = tool == "web"
    has_contexts = bool(contexts)

    # Determine which prompt to use based on contexts
    if has_contexts:
        if contexts and len(contexts) == 1 and contexts[0] == "rfp_context":
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
    prompt_template = ChatPromptTemplate.from_messages([system_prompt, messages[-1]])

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
        ContextType.VALIDATION_CHECKLIST: "VALIDATION CHECKLIST",
        ContextType.TABLE_OF_CONTENT: "TABLE OF CONTENT",
        ContextType.DEEP_RESEARCH: "DEEP RESEARCH",
        ContextType.USER_PREFERENCE: "USER PREFERENCE",
        ContextType.CONTENT: "PROPOSAL SECTIONS CONTENT",
    }
    prompt_text = " "
    if has_contexts and contexts is not None:
        for context in contexts:
            if context and "content" == context:
                data = get_context_data(source_id=source_id, key=ContextType.CONTENT)
                if data:
                    prompt_text += f"""
                    [{context_mapping[ContextType.CONTENT]}]
                    {data}
                    """
            elif context in context_mapping:
                data = get_context_data(source_id=source_id, key=context)
                if data:
                    section_header = context_mapping[context]
                    prompt_text += f"""
                    [{section_header}]
                    {data}
                    """
        prompt_args["section_context"] = prompt_text
        rfp_context = get_context_data(source_id, "rfp_text")
        prompt_args["rfp_context"] = rfp_context or ""
    msgs = prompt_template.format_messages(**prompt_args)
    print(msgs)

    response: AIMessage = await model_chat.ainvoke(msgs)

    token_info = update_token_tracking(
        state=state,
        response=response,
        model_name=DEFAULT_MODELS["chat"],
        additional_tokens=0,
    )
    return {"messages": [response], "summary_context": summary, "tokens": token_info}
