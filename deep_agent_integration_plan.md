
# Deep Agent Integration Approach

## Architecture Goals
- Introduce LangChain Deep Agent as a first-class conversation type alongside existing chat and deep research flows.
- Reuse the current Postgres-backed LangGraph checkpointing so deep-agent runs remain resumable and multi-tenant safe.
- Add optional MCP tool loading so user-scoped tools can be injected without hard-coding providers.
- Preserve the existing streaming contract over WebSockets while surfacing deep agent planner updates (todos, file write events).

## Implementation Steps
1. **Configuration & Dependencies**  
   - Add `deepagents` and `langchain-mcp-adapters` to `requirements.txt`.  
   - Extend `app/config.py` with a `CONV_TYPE_DEEP_AGENT` flag, defaults for the deep-agent model, and optional MCP environment switches.

2. **Schema & Constants**  
   - Broaden `ConversationState` in `app/schema.py` so deep-agent specific fields (`todos`, filesystem events, delegated task ids) survive validation.  
   - Register the new conversation type in `app/constants.py` and `app/graph_builder.get_available_conversation_types`.

3. **Deep Agent Runtime**  
   - Introduce `app/deep_agent.py` housing a cached `DeepAgentRunner` that builds `create_deep_agent(...)` with the configured model, optional MCP tools, and shared checkpointer.  
   - Provide utilities to translate deep-agent planner state (todos, file diffs, subagent events) into the streaming payload format consumed by the UI.

4. **ChatService Integration**  
   - Extend `ChatService` to lazily initialize the deep-agent runner once the Postgres checkpointer is ready.  
   - Branch `process_message` / `process_message_streaming` so `conversation_type == "deep-agent"` bypasses the standard LangGraph and delegates to the deep agent while maintaining token tracking.

5. **Surface Streaming Events**  
   - Update the WebSocket pipeline in `app/main.py` to recognize deep-agent specific chunk types (plan updates, todo deltas, background task notices).  
   - Ensure `app/store.py` captures long-running deep-agent tasks similar to the deep research flow.

6. **Housekeeping & Validation**  
   - Adjust helper utilities and prompts as needed for the new conversation type.  
   - Manually verify that existing chat and deep-research flows remain unaffected and that configuration falls back gracefully when MCP env vars are absent.

