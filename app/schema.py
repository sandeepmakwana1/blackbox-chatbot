from pydantic import BaseModel
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
    conversation_type: str  # "chat" | "search"| "document_qa" | "deep-research"
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


class PromptOptimizerInput(BaseModel):
    user_prompt: str


class ChatCreateRequest(BaseModel):
    title: Optional[str] = None
    conversation_type: str = "chat"
    first_message: Optional[str] = None


class ChatRenameRequest(BaseModel):
    title: str


class UploadItem(BaseModel):
    """Metadata returned for a single uploaded file."""

    original_filename: str
    file_size: int
    # s3_key: str
    url: str
    expires_at: int


class UploadResponse(BaseModel):
    """Response payload for uploaded files."""

    files: List[UploadItem]
