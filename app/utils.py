__all__ = [
    "ChatService",
    "ConversationState",
    "HumanMessage",
    "SystemMessage",
    "AIMessage",
    "LOGGER",
]

import os
import logging
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Optional, Sequence, Annotated, Any

# LangGraph / LangChain
from typing_extensions import TypedDict
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.message import add_messages

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.prompt import PlaygroundChatPrompt
from app.schema import ConversationState
from app.config import (
    DEFAULT_MODELS,
    SUMMARY_TRIGGER_COUNT,
    TOOLS,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
)
from app.helper import (
    get_trimmer_object,
    update_token_tracking,
    serialize_content_to_string,
)
from app.stream_helpers import (
    diff_stream_segments,
    validate_stream_consistency,
    split_stream_segments,
    merge_cumulative_or_delta,
)
from app.summary_agent import summarize_history
from app.nodes import route_summarize, summarize_node, chat_node
from app.graph_builder import build_graph, validate_conversation_type
from app.constants import ConversationType
from app.deep_agent import DeepAgentRunner

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("langgraph-app")


def _latest_persisted_ai_content(messages: Sequence[BaseMessage]) -> str:
    """Return the most recent AI message content as a string for validation."""

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, "content"):
            return serialize_content_to_string(msg.content)
    return ""


def _sync_stream_with_persisted(
    final_state: ConversationState, streamed_text: str
) -> str:
    """Ensure the streamed text matches what the graph persisted/logged."""

    try:
        messages = final_state.values.get("messages", [])
    except AttributeError:
        return streamed_text

    persisted = _latest_persisted_ai_content(messages)
    if persisted and not validate_stream_consistency(streamed_text, persisted):
        LOGGER.debug("Streamed content differed from persisted state; using persisted")
        return persisted
    return streamed_text


# -----------------------------------------------------------------------------
# Core Orchestrator
# -----------------------------------------------------------------------------


class ChatService:
    """
    Central orchestrator that builds and runs graphs for:
      - conversation
    """

    def __init__(self, db_uri: str = None):
        """Initialize ChatService with PostgreSQL persistence."""
        # Build DB URI with better SSL handling
        if db_uri is None:
            # Check if we're in a local development environment
            is_local = POSTGRES_HOST in ["localhost", "127.0.0.1", "postgres"]
            ssl_mode = "disable" if is_local else "require"
            self.db_uri = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}?sslmode={ssl_mode}"
        else:
            self.db_uri = db_uri

        # LLMs
        self.model_chat = ChatOpenAI(
            model=DEFAULT_MODELS["chat"], temperature=0.7, streaming=True
        )
        self.model_summarize = ChatOpenAI(
            model=DEFAULT_MODELS["summarize"], temperature=0.2
        )
        self.web_search_model_chat = self.model_chat.bind(tools=TOOLS)

        # Embeddings
        self.embeddings = OpenAIEmbeddings(model=DEFAULT_MODELS["embed"])

        # services - will be initialized asynchronously
        self.app = None
        self.checkpointer = None
        self.deep_agent_runner: Optional[DeepAgentRunner] = None
        self._checkpointer_cm = None
        self._is_initialized = False
        self._initialization_lock = asyncio.Lock()
        self._current_graph_type = "chat"  # Track current graph type

    # -------------------------- Conversation graph ---------------------------

    async def initialize(self):
        """Initialize the graph with PostgreSQL persistence with proper error handling."""
        async with self._initialization_lock:
            if self._is_initialized:
                return

            try:
                # Clean up existing checkpointer if any
                await self._cleanup_checkpointer()

                # Create new checkpointer with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        self._checkpointer_cm = AsyncPostgresSaver.from_conn_string(
                            self.db_uri
                        )
                        self.checkpointer = await self._checkpointer_cm.__aenter__()
                        await self.checkpointer.setup()
                        break
                    except Exception as e:
                        LOGGER.warning(
                            f"Database connection attempt {attempt + 1}/{max_retries} failed: {e}"
                        )
                        if attempt == max_retries - 1:
                            raise
                        await asyncio.sleep(2**attempt)  # Exponential backoff

                # Build a default chat graph - will be rebuilt per conversation type as needed
                graph = build_graph("chat")
                self.app = graph.compile(checkpointer=self.checkpointer)
                self.deep_agent_runner = DeepAgentRunner(self.checkpointer)
                self._is_initialized = True
                LOGGER.info("ChatService initialized successfully")

            except Exception as e:
                LOGGER.error(f"Failed to initialize ChatService: {e}")
                await self._cleanup_checkpointer()
                raise

    async def _cleanup_checkpointer(self):
        """Clean up existing checkpointer to prevent connection leaks."""
        if self._checkpointer_cm is not None:
            try:
                await self._checkpointer_cm.__aexit__(None, None, None)
            except Exception as e:
                LOGGER.warning(f"Error during checkpointer cleanup: {e}")
            finally:
                self._checkpointer_cm = None
                self.checkpointer = None
                self.deep_agent_runner = None

    async def _validate_connection(self):
        """Validate database connection and reinitialize if needed."""
        try:
            if self.checkpointer is None:
                await self.initialize()
                return

            # Try a simple operation to validate connection
            await self.checkpointer.aget_tuple(
                {"configurable": {"thread_id": "__test__"}}
            )
        except Exception as e:
            LOGGER.info(f"Connection validation failed, reinitializing: {e}")
            await self.initialize()

    async def _ensure_graph_for_conversation_type(self, conversation_type: str):
        """Ensure the graph is built for the specific conversation type."""
        if not validate_conversation_type(conversation_type):
            LOGGER.warning(
                f"Invalid conversation type: {conversation_type}, falling back to chat"
            )
            conversation_type = "chat"

        if conversation_type == ConversationType.DEEP_AGENT:
            if self.deep_agent_runner is None and self.checkpointer is not None:
                self.deep_agent_runner = DeepAgentRunner(self.checkpointer)
            return

        # For deep-research, we need the full graph. For others, we can use a simpler one.
        # But since deep-research graph includes all paths, we'll use it for deep-research
        # and the standard graph for others.
        required_graph_type = (
            "deep-research"
            if conversation_type == ConversationType.DEEP_RESEARCH
            else "standard"
        )

        if self._current_graph_type != required_graph_type:
            LOGGER.info(
                f"Switching graph from {self._current_graph_type} to {required_graph_type}"
            )
            graph = build_graph(conversation_type)
            self.app = graph.compile(checkpointer=self.checkpointer)
            self._current_graph_type = required_graph_type

    async def process_message(
        self,
        message: str,
        thread_id: str,
        user_id: str = "anonymous",
        language: str = "English",
        conversation_type: str = "chat",
        tool: str = "",
        contexts: List[str] = [],
    ) -> Dict[str, Any]:
        """
        Process a single message in a conversation thread.

        Args:
            message: The user's message
            thread_id: Unique identifier for the conversation thread
            user_id: User identifier for multi-tenant scenarios
            language: Response language preference
            conversation_type: Type of conversation (e.g., 'chat', 'research')

        Returns:
            Dict containing the response and metadata
        """
        await self._validate_connection()
        await self._ensure_graph_for_conversation_type(conversation_type)

        if conversation_type == ConversationType.DEEP_AGENT:
            if self.deep_agent_runner is None:
                raise RuntimeError("Deep agent runner is not initialised")
            result = await self.deep_agent_runner.invoke(
                message=message,
                thread_id=thread_id,
                user_id=user_id,
                language=language,
                contexts=contexts,
            )
            return {
                "response": result.get("content", ""),
                "token_usage": result.get("token_usage", {}),
                "thread_id": thread_id,
            }

        config = {"configurable": {"thread_id": thread_id}}

        # Get existing state to preserve token tracking with retry
        try:
            existing_state = await self.app.aget_state(config)
            existing_tokens = existing_state.values.get("tokens", {"total_tokens": 0})
            LOGGER.info(
                f"Thread {thread_id}: Found existing token tracking: {existing_tokens}"
            )
        except Exception as e:
            LOGGER.warning(f"Attempt failed to get state: {e}")
            existing_tokens = {"total_tokens": 0}
            LOGGER.info(
                f"Thread {thread_id}: Using fresh state due to connection issues"
            )

        initial_state: ConversationState = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "conversation_type": conversation_type,
            "tool": tool,
            "contexts": contexts,
            "language": language,
            "thread_id": thread_id,
            "tokens": existing_tokens,
            "research_initiated": False,
        }

        try:
            result = await self.app.ainvoke(initial_state, config)
        except Exception:
            LOGGER.exception("Chat graph invocation failed for thread %s", thread_id)
            raise

        return {
            "response": result["messages"][-1].content,
            "token_usage": result.get("tokens", {}),
            "thread_id": thread_id,
        }

    async def process_message_streaming(
        self,
        message: str,
        thread_id: str,
        user_id: str = "anonymous",
        language: str = "English",
        conversation_type: str = "chat",
        tool: str = "",
        contexts: List[str] = [],
    ):
        """
        Process a message with streaming responses using LangGraph's astream.
        Handles interrupts for deep research user input.

        Args:
            message: The user's message
            thread_id: Unique identifier for the conversation thread
            user_id: User identifier for multi-tenant scenarios
            language: Response language preference
            conversation_type: Type of conversation

        Yields:
            Streaming chunks of the response
        """
        await self._validate_connection()
        await self._ensure_graph_for_conversation_type(conversation_type)

        if conversation_type == ConversationType.DEEP_AGENT:
            if self.deep_agent_runner is None:
                raise RuntimeError("Deep agent runner is not initialised")
            yield {"type": "start", "content": "", "conversation_type": conversation_type}
            async for chunk in self.deep_agent_runner.stream(
                message=message,
                thread_id=thread_id,
                user_id=user_id,
                language=language,
                contexts=contexts,
            ):
                yield chunk
            return

        config = {"configurable": {"thread_id": thread_id}}

        # Get existing state to preserve token tracking and check for interrupts
        existing_state = None
        existing_tokens = {"total_tokens": 0}

        try:
            existing_state = await self.app.aget_state(config)
            existing_tokens = existing_state.values.get("tokens", {"total_tokens": 0})

            # Check if we're resuming from an interrupt
            if existing_state.next and len(existing_state.next) > 0:
                LOGGER.info(
                    f"Resuming interrupted conversation with user input: {message}"
                )
                LOGGER.info(f"Interrupted state next nodes: {existing_state.next}")
                # Resume with the user's clarification by updating the state and resuming
                yield {
                    "type": "start",
                    "content": "",
                    "conversation_type": conversation_type,
                    "tool": tool,
                    "contexts": contexts,
                }
                full_response = ""
                research_initiated_sent = False  # Track notification in resume flow too
                try:
                    # Update the conversation state with user input and resume from where we left off
                    user_clarification = HumanMessage(content=message)

                    # Update the state with the user's response
                    await self.app.aupdate_state(
                        config, {"messages": [user_clarification]}
                    )

                    if conversation_type == "deep-research":
                        # Use the same values-based streaming for deep research resume
                        # Get the current number of messages to track new ones only
                        existing_messages_count = len(
                            existing_state.values.get("messages", [])
                        )

                        # Resume execution with None (no new input, just continue)
                        async for chunk in self.app.astream(
                            None, config, stream_mode="values"
                        ):
                            # Extract messages from the chunk
                            messages = chunk.get("messages", [])

                            # Only process new messages beyond what was already there
                            if messages and len(messages) > existing_messages_count:
                                for msg_idx in range(
                                    existing_messages_count, len(messages)
                                ):
                                    msg = messages[msg_idx]

                                    # Skip HumanMessage to avoid echoing
                                    if isinstance(msg, HumanMessage):
                                        continue

                                    # Process AIMessage content
                                    if isinstance(msg, AIMessage) and hasattr(
                                        msg, "content"
                                    ):
                                        content_str = serialize_content_to_string(
                                            msg.content
                                        )

                                        # Check if this is the research initiation message during resume (only send once)
                                        if (
                                            "Deep research initiated" in content_str
                                            and "background" in content_str
                                            and not research_initiated_sent
                                        ):
                                            research_initiated_sent = True

                                        if content_str:
                                            (
                                                full_response,
                                                segments,
                                            ) = merge_cumulative_or_delta(
                                                full_response, content_str
                                            )
                                            for segment in segments:
                                                yield {
                                                    "type": "chunk",
                                                    "content": segment,
                                                }

                                existing_messages_count = len(messages)

                            # Check for research initiation during resume (only if not already sent)
                            if (
                                chunk.get("research_initiated")
                                and not research_initiated_sent
                            ):
                                yield {
                                    "type": "research_initiated",
                                    "content": "Deep research has been initiated and is running in the background.",
                                    "thread_id": thread_id,
                                }
                                research_initiated_sent = (
                                    True  # Mark as sent to prevent duplicates
                                )
                    else:
                        # Standard streaming for non-deep-research resume
                        async for chunk, _meta in self.app.astream(
                            None, config, stream_mode="messages"
                        ):
                            content = getattr(chunk, "content", None)
                            if content is not None:
                                content_str = serialize_content_to_string(content)

                                if content_str:
                                    full_response, segments = merge_cumulative_or_delta(
                                        full_response, content_str
                                    )
                                    for segment in segments:
                                        yield {"type": "chunk", "content": segment}

                    final_state = await self.app.aget_state(config)
                    token_usage = final_state.values.get("tokens", {})
                    full_response = _sync_stream_with_persisted(
                        final_state, full_response
                    )
                    LOGGER.info(
                        f"Resume completed, final state next nodes: {final_state.next}"
                    )
                    if conversation_type != "deep-research":
                        yield {
                            "type": "complete",
                            "content": full_response,
                            "token_tracking": token_usage,
                            "thread_id": thread_id,
                            "done": True,
                        }
                    return

                except Exception as e:
                    LOGGER.exception("Error resuming interrupted conversation")
                    yield {"type": "error", "message": str(e)}
                    return

        except Exception as e:
            LOGGER.warning(f"Failed to get state: {e}")
            existing_tokens = {"total_tokens": 0}

        # Normal flow - start new conversation or continue existing one
        initial_state: ConversationState = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "conversation_type": conversation_type,
            "tool": tool,
            "contexts": contexts,
            "language": language,
            "thread_id": thread_id,
            "tokens": existing_tokens,
            "research_initiated": False,
        }

        yield {"type": "start", "content": "", "conversation_type": conversation_type}

        full_response = ""
        interrupted = False
        research_initiated_sent = (
            False  # Track if we've already sent the research_initiated notification
        )

        # Track how many messages already existed so we don't replay them
        existing_messages_count = 0
        if existing_state and getattr(existing_state, "values", None):
            existing_messages_count = len(
                existing_state.values.get("messages", [])
            )

        try:
            if conversation_type == "deep-research":
                # For deep research, use values streaming to handle interrupts properly
                # Skip messages that were already present (plus the new human input)
                last_sent_index = existing_messages_count + 1

                async for chunk in self.app.astream(
                    initial_state, config, stream_mode="values"
                ):
                    # Extract messages from the chunk
                    messages = chunk.get("messages", [])
                    # Only process new messages that haven't been sent yet
                    if messages and len(messages) > last_sent_index:
                        # Process only the new messages
                        for msg_idx in range(last_sent_index, len(messages)):
                            msg = messages[msg_idx]

                            # Skip HumanMessage to avoid echoing user input
                            if isinstance(msg, HumanMessage):
                                continue

                            # Process AIMessage content
                            if isinstance(msg, AIMessage) and hasattr(msg, "content"):
                                content_str = serialize_content_to_string(msg.content)

                                if (
                                    "Deep research initiated" in content_str
                                    and "background" in content_str
                                    and not research_initiated_sent
                                ):
                                    research_initiated_sent = (
                                        True  # Mark as sent to prevent duplicates
                                    )

                                # Calculate the new content delta
                                if content_str:
                                    full_response, segments = merge_cumulative_or_delta(
                                        full_response, content_str
                                    )
                                    for segment in segments:
                                        yield {
                                            "type": "chunk",
                                            "content": segment,
                                        }

                        # Update the last sent index
                        last_sent_index = len(messages)

                    # Check if research has been initiated (could be in chunk or in state values) - only if not already sent
                    if (
                        chunk.get("research_initiated")
                        or chunk.get("research_initiated") is True
                    ) and not research_initiated_sent:
                        yield {
                            "type": "research_initiated",
                            "content": "Deep research has been initiated and is running in the background.",
                            "thread_id": thread_id,
                        }
                        research_initiated_sent = (
                            True  # Mark as sent to prevent duplicates
                        )
                        LOGGER.info(
                            f"Sent research_initiated notification for thread {thread_id} (chunk flag)"
                        )

                # Check if the conversation was interrupted (waiting for user input)
                final_state = await self.app.aget_state(config)

                if final_state.next and len(final_state.next) > 0:
                    interrupted = True
                    LOGGER.info(
                        f"Deep research workflow interrupted, waiting for user clarification"
                    )
                    yield {
                        "type": "interrupted",
                        "content": "Waiting for your response to continue...",
                        "thread_id": thread_id,
                        "interrupted": True,
                    }
                else:
                    token_usage = final_state.values.get("tokens", {})
                    full_response = _sync_stream_with_persisted(
                        final_state, full_response
                    )
                    yield {
                        "type": "done",
                        "content": full_response,
                        "token_tracking": token_usage,
                        "thread_id": thread_id,
                        "done": True,
                    }
            else:
                # Standard message streaming for other conversation types
                async for chunk, _meta in self.app.astream(
                    initial_state, config, stream_mode="messages"
                ):
                    content = getattr(chunk, "content", None)
                    if content is not None:
                        content_str = serialize_content_to_string(content)

                        if content_str:
                            full_response, segments = merge_cumulative_or_delta(
                                full_response, content_str
                            )
                            for segment in segments:
                                yield {"type": "chunk", "content": segment}

                final_state = await self.app.aget_state(config)
                token_usage = final_state.values.get("tokens", {})
                full_response = _sync_stream_with_persisted(final_state, full_response)
                yield {
                    "type": "complete",
                    "content": full_response,
                    "token_tracking": token_usage,
                    "thread_id": thread_id,
                    "done": True,
                }

        except Exception as e:
            LOGGER.exception("Streaming error")
            yield {"type": "error", "message": str(e)}
