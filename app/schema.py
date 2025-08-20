from typing_extensions import TypedDict
from typing import Annotated, Dict, List, Optional, Sequence
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
    conversation_type: str  # "chat" | "search" | "research" | "document_qa"
    language: str
    thread_id: str  # Added for memory integration
    documents_context: Optional[List[Dict]]
    search_results: Optional[List[Dict]]
    research_plan: Optional[Dict]
    tokens: Optional[Dict]