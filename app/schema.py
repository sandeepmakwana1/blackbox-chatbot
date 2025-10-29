from pydantic import BaseModel
from typing_extensions import TypedDict
from typing import Annotated, Any, Dict, List, Optional, Sequence
from langchain_core.messages import (
    BaseMessage,
)
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired


class ConversationState(TypedDict):
    """State carried through the graph."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary_context: NotRequired[Sequence[BaseMessage]]
    user_id: str
    conversation_type: str  # "chat" | "search"| "document_qa" | "deep-research" | "deep-agent"
    tool: Optional[str]
    contexts: Optional[List[Optional[str]]]
    language: str
    thread_id: str  # Added for memory integration
    documents_context: Optional[List[Dict]]
    research_plan: Optional[str]  # Changed from Dict to str for deep research plan
    research_status: NotRequired[
        Optional[str]
    ]  # Status of deep research: "pending", "completed", "failed"
    research_result: NotRequired[Optional[str]]  # Final result from deep research
    research_initiated: NotRequired[
        Optional[bool]
    ]  # Flag to indicate research has been initiated
    tokens: Optional[Dict]
    todos: NotRequired[Optional[List[Dict[str, Any]]]]
    filesystem_events: NotRequired[Optional[List[Dict[str, Any]]]]
    delegated_tasks: NotRequired[Optional[List[Dict[str, Any]]]]
    deep_agent_metadata: NotRequired[Optional[Dict[str, Any]]]


class PromptOptimizerInput(BaseModel):
    user_prompt: str


class ChatCreateRequest(BaseModel):
    title: Optional[str] = None
    conversation_type: str = "chat"
    first_message: Optional[str] = None


class ChatRenameRequest(BaseModel):
    title: str
