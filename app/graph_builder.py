"""
Graph Builder Module

This module provides utilities to build different types of conversation graphs
based on the conversation type, supporting both normal chat and deep research flows.
"""

import logging
from typing import Dict, Any
from langgraph.graph import START, END, StateGraph
from langgraph.constants import START as LANGGRAPH_START

from app.schema import ConversationState
from app.config import CONV_TYPE_DEEP_RESEARCH
from app.nodes import route_summarize, summarize_node, chat_node
from app.deep_research import (
    deepresearch_plaining_node,
    deep_research_prompt,
    deep_research_run_node,
    route_deep_research,
    wait_for_user_input,
    close_deep_research,
    rejected_node
)

__all__ = ['build_graph']

LOGGER = logging.getLogger("graph-builder")


def build_graph(conversation_type: str = "chat") -> StateGraph:
    """
    Build a StateGraph based on the conversation type.
    
    Args:
        conversation_type: The type of conversation ("chat", "web", "deep-research")
        
    Returns:
        A configured StateGraph ready for compilation
    """
    graph = StateGraph(state_schema=ConversationState)
    
    if conversation_type == CONV_TYPE_DEEP_RESEARCH:
        return _build_deep_research_graph(graph)
    else:
        return _build_standard_chat_graph(graph, conversation_type)


def _build_standard_chat_graph(graph: StateGraph, conversation_type: str) -> StateGraph:
    """
    Build a standard chat graph with summarization support.
    
    This handles normal chat, web search, and other non-research conversation types.
    """
    # Add standard nodes
    graph.add_node("summarize", summarize_node)
    graph.add_node("chat", chat_node)
    
    # Add routing logic
    graph.add_conditional_edges(
        START, 
        route_summarize, 
        {
            "summarize": "summarize", 
            "chat": "chat"
        }
    )
    
    # Connect summarize to chat (summarize then continue with chat)
    graph.add_edge("summarize", "chat")
    
    LOGGER.info(f"Built standard chat graph for conversation type: {conversation_type}")
    return graph


def _build_deep_research_graph(graph: StateGraph) -> StateGraph:
    """
    Build a deep research graph with interrupt handling for user input.
    
    This includes research planning, user input handling, and execution.
    """
    # Add all nodes
    graph.add_node("summarize", summarize_node)
    graph.add_node("chat", chat_node)
    graph.add_node("deepresearch_plaining", deepresearch_plaining_node)
    graph.add_node("wait_for_user_input", wait_for_user_input)
    graph.add_node("deep_research_prompt", deep_research_prompt)
    graph.add_node("deep_research_run", deep_research_run_node)
    graph.add_node("close_deep_research", close_deep_research)
    
    # Main routing from START based on conversation type
    def route_conversation_type(state: ConversationState) -> str:
        """Route based on conversation type and context."""
        conv_type = state.get("conversation_type")
        if conv_type == CONV_TYPE_DEEP_RESEARCH:
            return "deepresearch_plaining"
        
        # Fall back to standard chat routing
        return route_summarize(state)
    
    # Set up the routing
    graph.add_conditional_edges(
        START,
        route_conversation_type,
        {
            "deepresearch_plaining": "deepresearch_plaining",
            "summarize": "summarize",
            "chat": "chat"
        }
    )
    
    # Deep research flow with routing that handles interrupts
    graph.add_conditional_edges(
        "deepresearch_plaining",
        route_deep_research,
        {
            "wait_for_user_input": "wait_for_user_input",
            "deep_research_prompt": "deep_research_prompt",
            "close_deep_research": "close_deep_research"
        }
    )
    
    # After waiting for user input, proceed to research prompt generation
    graph.add_edge("wait_for_user_input", "deep_research_prompt")
    
    # Continue the research flow
    graph.add_edge("deep_research_prompt", "deep_research_run")
    graph.add_edge("deep_research_run", "close_deep_research")
    
    # Standard chat flow (still available)
    graph.add_edge("summarize", "chat")
    
    # All flows end at close_deep_research or chat
    graph.add_edge("close_deep_research", END)
    graph.add_edge("chat", END)
    
    LOGGER.info("Built deep research graph with interrupt handling")
    return graph


def get_available_conversation_types() -> Dict[str, str]:
    """
    Get a mapping of available conversation types and their descriptions.
    """
    return {
        "chat": "Standard conversational chat with summarization support",
        "web": "Chat with web search capabilities",
        CONV_TYPE_DEEP_RESEARCH: "Deep research with planning and background execution",
        "search": "Search-focused conversations",
        "document_qa": "Document question and answer sessions"
    }


def validate_conversation_type(conversation_type: str) -> bool:
    """
    Validate if a conversation type is supported.
    """
    return conversation_type in get_available_conversation_types()
