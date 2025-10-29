"""
Deep Agent runtime integration.

This module wraps LangChain's `deepagents` package so the application can run the
deep agent workflow alongside existing LangGraph graphs. It also optionally loads
MCP tools based on environment configuration.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph.state import CompiledStateGraph

from app.config import (
    DEFAULT_MODELS,
    DEEP_AGENT_ROOT,
    MCP_SERVER_CONFIG,
    MCP_SERVER_TIMEOUT,
)
from app.helper import serialize_content_to_string
from app.schema import ConversationState

LOGGER = logging.getLogger("deep-agent-runner")


DEFAULT_DEEP_AGENT_PROMPT = """
You are the lead proposal strategist for a capture team. Every thread represents a
long-running project where you must plan your work, break tasks into actionable units,
and deliver polished narrative or analytical outputs.

Guidelines:
- Maintain an explicit plan using the built-in todo list so collaborators can follow your progress.
- Move expansive intermediate results into the filesystem tools to keep context tidy.
- Cite sources or assumptions when synthesising research; flag stale or uncertain data.
- Prefer structured Markdown with headings, tables, and bullet lists for dense content.
- Never fabricate facts—defer to research sub-tasks or note gaps that remain open.
- When the user request could benefit from specialised focus, delegate via subagents to
  avoid polluting the main context.
""".strip()


def _parse_mcp_config() -> Dict[str, Dict[str, Any]]:
    """Parse MCP server configuration from the environment variable."""
    if not MCP_SERVER_CONFIG:
        return {}

    try:
        config = json.loads(MCP_SERVER_CONFIG)
        if not isinstance(config, dict):
            raise ValueError("MCP configuration must be a JSON object.")
        return config  # type: ignore[return-value]
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Invalid MCP server config: %s", exc)
        return {}


class DeepAgentRunner:
    """
    Lazily builds and runs the deep agent graph, including optional MCP tool wiring.
    """

    def __init__(self, checkpointer: Any):
        self._checkpointer = checkpointer
        self._agent: Optional[CompiledStateGraph] = None
        self._build_lock = asyncio.Lock()
        self._mcp_connections = _parse_mcp_config()
        self._mcp_tools: Optional[List[BaseTool]] = None
        self._workspace_root = Path(DEEP_AGENT_ROOT)
        self._workspace_root.mkdir(parents=True, exist_ok=True)

    async def _load_mcp_tools(self) -> List[BaseTool]:
        """Load MCP tools once and cache the result."""
        if not self._mcp_connections:
            return []

        if self._mcp_tools is not None:
            return self._mcp_tools

        client = MultiServerMCPClient(self._mcp_connections)
        tools: List[BaseTool] = []
        try:
            tools = await asyncio.wait_for(
                client.get_tools(), timeout=float(MCP_SERVER_TIMEOUT)
            )
            LOGGER.info("Loaded %d MCP tools for deep agent", len(tools))
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Failed to load MCP tools: %s", exc)
        finally:
            # MultiServerMCPClient does not expose close/cleanup today; rely on GC.
            self._mcp_tools = tools

        return tools

    async def _ensure_agent(self) -> CompiledStateGraph:
        """Build the deep agent graph if it has not been created yet."""
        if self._agent is not None:
            return self._agent

        async with self._build_lock:
            if self._agent is not None:
                return self._agent

            model = ChatOpenAI(
                model=DEFAULT_MODELS["deep_agent"],
                temperature=0.2,
                streaming=True,
            )
            tools = await self._load_mcp_tools()

            try:
                self._agent = create_deep_agent(
                    model=model,
                    tools=tools,
                    system_prompt=DEFAULT_DEEP_AGENT_PROMPT,
                    checkpointer=self._checkpointer,
                )
                LOGGER.info(
                    "Deep agent graph initialised (tools=%d)", len(tools or [])
                )
            except Exception:
                LOGGER.exception("Failed to initialise deep agent graph")
                raise

        return self._agent

    async def invoke(
        self,
        message: str,
        thread_id: str,
        user_id: str,
        language: str,
        *,
        contexts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run the deep agent without streaming."""
        agent = await self._ensure_agent()
        config = {"configurable": {"thread_id": thread_id}}
        state: ConversationState = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "conversation_type": "deep-agent",
            "contexts": contexts or [],
            "language": language,
            "thread_id": thread_id,
        }

        result = await agent.ainvoke(state, config)
        messages = result.get("messages", [])
        response_text = ""
        if messages:
            last_ai = next(
                (msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None
            )
            if last_ai:
                response_text = serialize_content_to_string(last_ai.content)

        final_state = await agent.aget_state(config)
        token_usage = final_state.values.get("tokens", {})

        return {
            "content": response_text,
            "token_usage": token_usage,
            "state": final_state.values,
        }

    async def stream(
        self,
        message: str,
        thread_id: str,
        user_id: str,
        language: str,
        *,
        contexts: Optional[List[str]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream deep agent output, yielding UI-friendly chunks."""
        agent = await self._ensure_agent()
        config = {"configurable": {"thread_id": thread_id}}

        initial_state: ConversationState = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "conversation_type": "deep-agent",
            "contexts": contexts or [],
            "language": language,
            "thread_id": thread_id,
        }

        full_response = ""
        last_todos: Optional[List[Dict[str, Any]]] = None
        last_fs_event_count = 0

        async for chunk in agent.astream(
            initial_state, config, stream_mode="values"
        ):
            messages = chunk.get("messages", [])
            if messages:
                latest_ai = next(
                    (msg for msg in reversed(messages) if isinstance(msg, AIMessage)),
                    None,
                )
                if latest_ai:
                    content_str = serialize_content_to_string(latest_ai.content)
                    if content_str and len(content_str) > len(full_response):
                        delta = content_str[len(full_response) :]
                        full_response = content_str
                        if delta.strip():
                            yield {"type": "chunk", "content": delta}

            todos = chunk.get("todos")
            if todos:
                normalized_todos = list(todos)
                if normalized_todos != (last_todos or []):
                    last_todos = normalized_todos
                    yield {
                        "type": "deep_agent_todos",
                        "content": normalized_todos,
                        "thread_id": thread_id,
                    }

            fs_events = chunk.get("filesystem_events") or []
            if fs_events and len(fs_events) > last_fs_event_count:
                # Only emit new events
                new_events = fs_events[last_fs_event_count:]
                last_fs_event_count = len(fs_events)
                yield {
                    "type": "deep_agent_fs",
                    "content": new_events,
                    "thread_id": thread_id,
                }

        final_state = await agent.aget_state(config)
        token_usage = final_state.values.get("tokens", {})

        yield {
            "type": "complete",
            "content": full_response,
            "token_tracking": token_usage,
            "thread_id": thread_id,
            "done": True,
        }
