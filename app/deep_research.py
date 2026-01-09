"""
Deep Research Module

This module contains all the LangGraph nodes and utilities for deep research functionality.
Extracted from 5_deep_research_chat_bot_support.ipynb and refactored for production use.
"""

import logging
import time
from typing import Dict, Optional, List
from openai import OpenAI
from langgraph.types import interrupt, Command
from app.helper import get_context_data

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

from app.schema import ConversationState
from app.config import DEFAULT_MODELS
from app.helper import (
    update_token_tracking,
    safe_ai_invoke_async,
    handle_openai_error,
    serialize_content_to_string,
)
from app.prompt import (
    PlaygroundResearchPlainPrompt,
    DeepResearchPromptGenreatorPrompt,
)
from app.store import add_record, delete_record
from app.constants import ContextType

__all__ = [
    "deepresearch_plaining_node",
    "deep_research_prompt",
    "deep_research_run_node",
    "route_deep_research",
    "wait_for_user_input",
    "close_deep_research",
    "rejected_node",
]

LOGGER = logging.getLogger("deep-research")


def _message_to_text(msg: BaseMessage) -> str:
    """Convert a message into a compact ``Role: content`` string."""

    if msg is None:
        return ""

    role = "Assistant"
    if isinstance(msg, HumanMessage):
        role = "User"
    elif isinstance(msg, SystemMessage):
        role = "System"

    content = (
        serialize_content_to_string(getattr(msg, "content", msg)).strip()
        if hasattr(msg, "content")
        else str(msg).strip()
    )
    return f"{role}: {content}" if content else ""


def _build_conversation_context(
    state: ConversationState, max_turns: int = 8, max_chars: int = 5000
) -> str:
    """Assemble summary + recent turns so deep research can reuse prior context."""

    context_parts: List[str] = []

    summary_context = state.get("summary_context")
    if summary_context:
        if isinstance(summary_context, (list, tuple)):
            summary_text = "\n".join(
                serialize_content_to_string(getattr(msg, "content", msg)).strip()
                for msg in summary_context
                if msg
            )
        else:
            summary_text = serialize_content_to_string(
                getattr(summary_context, "content", summary_context)
            ).strip()

        if summary_text:
            context_parts.append(f"Summary Context:\n{summary_text}")

    messages = list(state.get("messages", []))
    if messages:
        recent_history = []
        for msg in messages[-max_turns:]:
            formatted = _message_to_text(msg)
            if formatted:
                recent_history.append(formatted)

        if recent_history:
            context_parts.append("Recent Conversation:\n" + "\n".join(recent_history))

    context_text = "\n\n".join(context_parts).strip()
    if context_text and len(context_text) > max_chars:
        context_text = context_text[-max_chars:]

    return context_text


async def deepresearch_plaining_node(state: ConversationState) -> Dict:
    """
    Initial deep research planning node that asks clarifying questions.

    This node uses the research planning prompt to ask the user for more details
    about their research requirements before proceeding with the actual research.
    """
    base = list(state["messages"])
    language = state.get("language", "English")
    conversation_context = _build_conversation_context(state)
    contexts = state.get("contexts", [])
    has_contexts = bool(contexts)

    system_prompt = SystemMessagePromptTemplate.from_template(
        PlaygroundResearchPlainPrompt.system_prompt
    )
    user_prompt = HumanMessagePromptTemplate.from_template(
        PlaygroundResearchPlainPrompt.user_prompt
    )
    prompt_template = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

    model_chat = ChatOpenAI(
        model=DEFAULT_MODELS["research_plain"],
        temperature=0.7,
        streaming=False,  # Disable streaming so routing can work properly
    )

    last_message = base[-1] if base else None
    user_query = (
        serialize_content_to_string(getattr(last_message, "content", last_message))
        if last_message is not None
        else ""
    )

    rfp_context = ""
    if has_contexts:
        source_id = (state.get("user_id") or "").split("_")[-1]
        rfp_context = get_context_data(source_id, "rfp_text")

    msgs = prompt_template.format_messages(
        language=language,
        user_query=user_query,
        conversation_context=conversation_context or "",
        rfp_context=rfp_context,
    )

    # Use safe async invoke with proper error handling
    with get_openai_callback() as cb:
        response = await safe_ai_invoke_async(
            model_chat.ainvoke,
            msgs,
            context="Deep research planning",
        )
        source_id = (state.get("user_id") or "").split("_")[-1]
        token_info = update_token_tracking(
            state=state,
            response=response,
            model_name=DEFAULT_MODELS["research_plain"],
            additional_tokens=0,
            stage_name=ContextType.PLAYGROUND_DEEP_RESEARCH_PLAN,
            source_id=source_id,
            request_id=state.get("thread_id") or state.get("user_id"),
            cb=cb,
        )

    # If response is an error message, return it
    if (
        isinstance(response, AIMessage)
        and "unable to process" in response.content.lower()
    ):
        return {"messages": [response], "tokens": state.get("tokens", {})}

    return {
        "messages": [response],
        "tokens": token_info,
        "research_plan": response.content,
    }


async def deep_research_prompt(state: ConversationState) -> Dict:
    """
    Generate detailed research instructions based on the research plan.

    This node takes the research plan from state and creates comprehensive
    instructions for the deep research execution.
    """
    # Use the research plan from the previous node
    query = state.get("research_plan", "")
    contexts = state.get("contexts", [])
    has_contexts = bool(contexts)

    if not query:
        # Fallback to extracting from messages
        messages = state.get("messages", [])
        if messages:
            query = (
                messages[-1].content
                if hasattr(messages[-1], "content")
                else str(messages[-1])
            )

    if not query:
        return {
            "research_plan": "No research query available",
            "messages": [
                AIMessage(content="No research query available for processing")
            ],
        }

    llm = ChatOpenAI(model=DEFAULT_MODELS["research_plain"], temperature=0.7)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                DeepResearchPromptGenreatorPrompt.system_prompt
            ),
            HumanMessagePromptTemplate.from_template(
                DeepResearchPromptGenreatorPrompt.user_prompt
            ),
        ]
    )

    conversation_context = _build_conversation_context(state)
    rfp_context = ""
    if has_contexts:
        source_id = (state.get("user_id") or "").split("_")[-1]
        rfp_context = get_context_data(source_id, "rfp_text")

    messages = prompt.format_messages(
        query=query,
        conversation_context=conversation_context or "",
        rfp_context=rfp_context,
    )

    # Use safe async invoke with proper error handling
    with get_openai_callback() as cb:
        response = await safe_ai_invoke_async(
            llm.ainvoke,
            messages,
            context="Research plan generation",
        )
        token_info = update_token_tracking(
            state=state,
            response=response,
            model_name=DEFAULT_MODELS["research_plain"],
            additional_tokens=0,
            stage_name=ContextType.PLAYGROUND_DEEP_RESEARCH_PROMPT,
            source_id=source_id,
            request_id=state.get("thread_id") or state.get("user_id"),
            cb=cb,
        )

    # If response is an error message, handle appropriately
    if (
        isinstance(response, AIMessage)
        and "unable to process" in response.content.lower()
    ):
        return {
            "research_plan": "Unable to generate research plan at this time. Please try again.",
            "messages": [response],
        }

    LOGGER.info(f"Generated enhanced research plan for query: {query[:100]}...")
    return {
        "research_plan": response.content,
        "tokens": token_info,
        # "messages": [AIMessage(content="Research plan generated. Proceeding with execution...")]
    }


async def deep_research_run_node(state: ConversationState) -> Dict:
    """
    Execute the deep research using OpenAI's response API.

    This node submits the research plan to OpenAI's response API for
    background processing and tracks the request in the store.
    """
    query = state.get("research_plan", "")
    if not query:
        error_msg = (
            "Unable to execute your plan at this time! Check your query and try again"
        )
        LOGGER.warning("deep_research_run_node called without research_plan")
        return {"messages": [AIMessage(content=error_msg)]}

    # Prepare the research system prompt
    deep_research_system_prompt = (
        query.strip("```markdown")
        + """

                Rules / Final Instructions
                Boundaries:
                Never guess or fabricate information.
                If you have any query don't ask with user, do the research your self.                
                Always Do:
                Prioritize actionable insights.
                Format results cleanly with headers and bullet points where helpful.
                Cite sources wherever possible.
                Tailor content for proposal teams, capture managers, and solution engineers.

                Never Do:
                Never include metadata, tokens, or irrelevant information.
                Never overgeneralize across agencies.
                Never rely heavily on stale data (>4 years old) unless specifically required, and flag it when used.
                Never reuse boilerplate content across responses; ensure uniqueness.
                """
    )

    try:
        # Get webhook secret from environment
        from app.config import OPENAI_WEBHOOK_SECRET, OPENAI_RESPONSE_MODEL

        client = OpenAI(
            webhook_secret=OPENAI_WEBHOOK_SECRET,
        )

        deep_research_call = client.responses.create(
            model=OPENAI_RESPONSE_MODEL,
            input=[
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": deep_research_system_prompt,
                        }
                    ],
                },
            ],
            tools=[
                {"type": "web_search_preview"},
            ],
            background=True,
            metadata={
                "thread_id": str(state.get("thread_id")),
                "user_id": str(state.get("user_id")),
            },
        )

        # Store the research request
        record = {
            "thread_id": state.get("thread_id"),
            "user_id": state.get("user_id"),
            "status": "pending",
            "research_plan": state.get("research_plan"),
            "start_at": time.time(),
            "response_id": deep_research_call.id
            if hasattr(deep_research_call, "id")
            else None,
        }

        # Clean up any existing record and add new one
        thread_id = state.get("thread_id")
        if thread_id:
            delete_record(thread_id)
        add_record(record)

        LOGGER.info(f"Deep research initiated for thread {state.get('thread_id')}")

        # Create a message that will trigger frontend research monitoring
        # research_message = AIMessage(
        #     content="Research plan generated and Send to the Deep research service for execution"
        # )

        # Return both the message and the flag so it appears in streaming
        return {
            # "messages": [research_message],
            "research_initiated": True,  # Flag to indicate research has started
            "tokens": state.get("tokens", {}),  # Preserve token tracking
        }

    except Exception as e:
        LOGGER.error(f"Error in deep_research_run_node: {e}")
        error_msg = (
            "Unable to execute your plan at this time! Check your query and try again"
        )
        return Command(
            goto="rejected_path", update={"messages": [AIMessage(content=error_msg)]}
        )


async def wait_for_user_input(state: ConversationState) -> Dict:
    """
    Interrupt node that waits for user clarification.
    """
    LOGGER.info("wait_for_user_input node called!")
    msgs = state.get("messages", [])
    if not msgs:
        LOGGER.warning("No messages to process in wait_for_user_input")
        return {"messages": [AIMessage(content="No messages to process")]}

    last_message = msgs[-1]

    # Check if this is a user message (indicating we're resuming)
    if isinstance(last_message, HumanMessage):
        LOGGER.info(
            "Resume detected - user provided clarification, continuing to research prompt generation"
        )
        return {"messages": []}

    # Check if this message contains questions that need clarification
    if hasattr(last_message, "content") and last_message.content:
        content = last_message.content.lower()

        # If the message contains questions, interrupt and wait for user input
        if any(
            question_indicator in content
            for question_indicator in [
                "could you clarify",
                "?",
                "please specify",
                "would you like",
                "preferences",
            ]
        ):
            LOGGER.info(
                " Deep research requires user clarification, calling interrupt()..."
            )
            interrupt("Waiting for user clarification on research requirements")
    # If we reach here, no questions were found, continue normally
    LOGGER.info("No questions found, continuing normally")
    return {"messages": []}


async def route_deep_research(state: ConversationState) -> str:
    """
    Route the deep research flow based on message content.

    This function determines if the planning response contains questions.
    """
    try:
        msgs = state.get("messages", [])
        if not msgs:
            return "close_deep_research"

        last_message = msgs[-1]

        # Check if this is a planning response asking questions
        if hasattr(last_message, "content") and last_message.content:
            content = last_message.content.lower()

            if any(
                question_indicator in content
                for question_indicator in [
                    "could you clarify",
                    "?",
                    "please specify",
                    "would you like",
                    "preferences",
                ]
            ):
                LOGGER.info("Deep research routing to user input wait...")
                return "wait_for_user_input"
            else:
                # Planning response doesn't ask questions, proceed directly
                return "deep_research_prompt"

        return "close_deep_research"

    except Exception as e:
        LOGGER.error(f"Error in route_deep_research: {e}")
        return "close_deep_research"


def close_deep_research(state: ConversationState) -> Dict:
    """
    Close the deep research session with a friendly message.
    """
    research_plan = state.get("research_plan", "")
    response: Dict = {"research_initiated": False}

    if not research_plan:
        content = (
            "Deep Research session ended. "
            "You can switch to regular chat mode or start a new deep research task."
        )

        response_message = AIMessage(content=content)
        response["messages"] = [response_message]
    return response


def rejected_node(state: ConversationState) -> Dict:
    """
    Terminal node for rejected or error paths.
    """
    return {}
