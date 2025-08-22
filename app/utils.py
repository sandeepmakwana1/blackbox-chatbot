__all__ = ['ChatService', 'ConversationState', 'HumanMessage', 'SystemMessage', 'AIMessage', 'LOGGER']

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

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.prompt import PlaygroundChatPrompt
from app.schema import ConversationState
from app.config import DEFAULT_MODELS, SUMMARY_TRIGGER_COUNT, TOOLS, POSTGRES_USER,POSTGRES_PASSWORD,POSTGRES_HOST,POSTGRES_PORT,POSTGRES_DB
from app.helper import get_trimmer_object, update_token_tracking, serialize_content_to_string
from app.summary_agent import summarize_history
from app.nodes import route_summarize, summarize_node, chat_node

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("langgraph-app")





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
            is_local = POSTGRES_HOST in ['localhost', '127.0.0.1', 'postgres']
            ssl_mode = 'disable' if is_local else 'require'
            self.db_uri = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}?sslmode={ssl_mode}"
        else:
            self.db_uri = db_uri
            
        # LLMs
        self.model_chat = ChatOpenAI(model=DEFAULT_MODELS["chat"], temperature=0.7, streaming=True)
        self.model_summarize = ChatOpenAI(model=DEFAULT_MODELS["summarize"], temperature=0.2)
        self.web_search_model_chat = self.model_chat.bind(tools=TOOLS)

        # Embeddings
        self.embeddings = OpenAIEmbeddings(model=DEFAULT_MODELS["embed"])

        # services - will be initialized asynchronously
        self.app = None
        self.checkpointer = None
        self._checkpointer_cm = None
        self._is_initialized = False
        self._initialization_lock = asyncio.Lock()


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
                        self._checkpointer_cm = AsyncPostgresSaver.from_conn_string(self.db_uri)
                        self.checkpointer = await self._checkpointer_cm.__aenter__()
                        await self.checkpointer.setup()
                        break
                    except Exception as e:
                        LOGGER.warning(f"Database connection attempt {attempt + 1}/{max_retries} failed: {e}")
                        if attempt == max_retries - 1:
                            raise
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                # Build graph using existing nodes
                graph = StateGraph(state_schema=ConversationState)
                graph.add_node("summarize", summarize_node)
                graph.add_node("chat", chat_node)
                
                graph.add_conditional_edges(START, route_summarize, {"summarize": "summarize", "chat": "chat"})
                graph.add_edge("summarize", "chat")
                
                self.app = graph.compile(checkpointer=self.checkpointer)
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
    
    async def _validate_connection(self):
        """Validate database connection and reinitialize if needed."""
        try:
            if self.checkpointer is None:
                await self.initialize()
                return
            
            # Try a simple operation to validate connection
            await self.checkpointer.aget_tuple({"configurable": {"thread_id": "__test__"}})
        except Exception as e:
            LOGGER.info(f"Connection validation failed, reinitializing: {e}")
            await self.initialize()

    async def process_message(
        self, 
        message: str, 
        thread_id: str, 
        user_id: str = "anonymous",
        language: str = "English",
        conversation_type: str = "chat"
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

        config = {"configurable": {"thread_id": thread_id}}
        
        # Get existing state to preserve token tracking with retry
        try:
            existing_state = await self.app.aget_state(config)
            existing_tokens = existing_state.values.get("tokens", {"total_tokens": 0})
            LOGGER.info(f"Thread {thread_id}: Found existing token tracking: {existing_tokens}")
        except Exception as e:
            LOGGER.warning(f"Attempt failed to get state: {e}")
            existing_tokens = {"total_tokens": 0}
            LOGGER.info(f"Thread {thread_id}: Using fresh state due to connection issues")

        initial_state: ConversationState = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "conversation_type": conversation_type,
            "language": language,
            "thread_id": thread_id,
            "tokens": existing_tokens,
        }

        try:
            result = await self.app.ainvoke(initial_state, config)
        except Exception as e:
            LOGGER.warning(f"Failed to invoke: {e}")
        
        return {
            "response": result["messages"][-1].content,
            "token_usage": result.get("tokens", {}),
            "thread_id": thread_id
        }

    async def process_message_streaming(
        self, 
        message: str, 
        thread_id: str, 
        user_id: str = "anonymous",
        language: str = "English",
        conversation_type: str = "chat"
    ):
        """
        Process a message with streaming responses using LangGraph's astream.
        
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
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get existing state to preserve token tracking
        try:
            existing_state = await self.app.aget_state(config)
            existing_tokens = existing_state.values.get("tokens", {"total_tokens": 0})
        except Exception as e:
            LOGGER.warning(f"Failed to get state: {e}")
            existing_tokens = {"total_tokens": 0}

        initial_state: ConversationState = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "conversation_type": conversation_type,
            "language": language,
            "thread_id": thread_id,
            "tokens": existing_tokens,
        }

        yield {"type": "start", "content": ""}

        full_response = ""
        try:
            async for chunk, _meta in self.app.astream(initial_state, config, stream_mode="messages"):
                content = getattr(chunk, "content", None)
                if content is not None:
                    content_str = serialize_content_to_string(content)
                    
                    if content_str.strip():
                        full_response += content_str
                        yield {"type": "chunk", "content": content_str}

            final_state = await self.app.aget_state(config)
            token_usage = final_state.values.get("tokens", {})

            yield {
                "type": "complete",
                "content": full_response,
                "token_tracking": token_usage,
                "thread_id": thread_id,
                "done": True
            }

        except Exception as e:
            LOGGER.exception("Streaming error")
            yield {"type": "error", "message": str(e)}
