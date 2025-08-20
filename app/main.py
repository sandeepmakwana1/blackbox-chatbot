from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

import json
from datetime import datetime
from app.utils import ChatService, ConversationState, HumanMessage, SystemMessage, LOGGER
from app.chat_manager import ChatManager
from pydantic import BaseModel
from typing import Optional

class ChatCreateRequest(BaseModel):
    title: Optional[str] = None
    conversation_type: str = "chat"
    first_message: Optional[str] = None

class ChatRenameRequest(BaseModel):
    title: str



app = FastAPI(title="LangGraph Advanced Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

service = ChatService()
chat_manager = ChatManager()

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)

            user_message = payload.get("message", "")
            conversation_type = payload.get("type", "chat")
            thread_id = payload.get("thread_id")
            language = payload.get("language", "English")
            
            # If no thread_id provided, get or create a default chat
            if not thread_id:
                thread_id = chat_manager.get_or_create_default_chat(user_id)
            
            # Ensure the chat exists in our metadata
            existing_chat = chat_manager.get_chat(user_id, thread_id)
            if not existing_chat:
                # Create chat metadata if it doesn't exist (for backward compatibility)
                chat_manager.create_chat(user_id, user_message, conversation_type)

            # Process the message with streaming
            async for chunk in service.process_message_streaming(
                message=user_message,
                thread_id=thread_id,
                user_id=user_id,
                language=language,
                conversation_type=conversation_type
            ):
                await websocket.send_text(json.dumps(chunk))
            
            # Get current chat state before updating activity
            chat = chat_manager.get_chat(user_id, thread_id)
            current_message_count = chat.get('message_count', 0) if chat else 0
            
            # Update chat activity in metadata after streaming is complete
            chat_manager.update_chat_activity(user_id, thread_id, user_message)
            
            # Auto-rename chat after first message if it has generic title
            if chat and current_message_count == 0 and chat['title'] in ['New Chat', '']:
                new_title = chat_manager.generate_title_from_message(user_message)
                chat_manager.rename_chat(user_id, thread_id, new_title)

    except WebSocketDisconnect:
        LOGGER.info("WebSocket disconnected: user=%s", user_id)
    except Exception as exc:
        LOGGER.exception("WebSocket error")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        finally:
            await websocket.close()

    except WebSocketDisconnect:
        LOGGER.info("WebSocket disconnected: user=%s", user_id)
    except Exception as exc:
        LOGGER.exception("WebSocket error")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        finally:
            await websocket.close()


@app.get("/conversation/{user_id}/{thread_id}")
async def get_conversation_state(user_id: str, thread_id: str):
    """Return the stored state for a given conversation thread."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = await service.app.aget_state(config)
        msgs = []
        for m in state.values.get("messages", []):
            role = "assistant"
            if isinstance(m, HumanMessage):
                role = "human"
            elif isinstance(m, SystemMessage):
                role = "system"
            msgs.append({"role": role, "content": getattr(m, "content", "")})

        return {
            "messages": msgs,
            "token_tracking": state.values.get("token_tracking", {}),
            "conversation_type": state.values.get("conversation_type", "chat"),
        }
    except Exception as e:
        LOGGER.exception("Error getting conversation state")
        return {
            "messages": [],
            "token_tracking": {},
            "conversation_type": "chat",
            "error": "No conversation state found"
        }

# -----------------------------------------------------------------------------
# Chat Management Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/chats/{user_id}")
def get_user_chats(user_id: str):
    """Get all chats for a user"""
    try:
        chats = chat_manager.get_user_chats(user_id)
        return {"status": "success", "chats": chats}
    except Exception as e:
        LOGGER.exception("Error getting user chats")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chats/{user_id}")
def create_chat(user_id: str, request: ChatCreateRequest):
    """Create a new chat"""
    try:
        first_message = request.first_message or ""
        title = request.title
        
        # If no custom title provided, let chat_manager generate one from message
        if not title and first_message:
            thread_id = chat_manager.create_chat(user_id, first_message, request.conversation_type)
        elif title:
            thread_id = chat_manager.create_chat(user_id, first_message, request.conversation_type)
            # Override generated title with custom one
            chat_manager.rename_chat(user_id, thread_id, title)
        else:
            thread_id = chat_manager.create_chat(user_id, "", request.conversation_type)
            
        chat = chat_manager.get_chat(user_id, thread_id)
        return {"status": "success", "thread_id": thread_id, "chat": chat}
    except Exception as e:
        LOGGER.exception("Error creating chat")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/chats/{user_id}/{thread_id}")
def rename_chat(user_id: str, thread_id: str, request: ChatRenameRequest):
    """Rename a chat"""
    try:
        success = chat_manager.rename_chat(user_id, thread_id, request.title)
        if success:
            chat = chat_manager.get_chat(user_id, thread_id)
            return {"status": "success", "chat": chat}
        else:
            raise HTTPException(status_code=404, detail="Chat not found")
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception("Error renaming chat")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chats/{user_id}/{thread_id}")
def delete_chat(user_id: str, thread_id: str):
    """Delete a chat"""
    try:
        success = chat_manager.delete_chat(user_id, thread_id)
        if success:
            return {"status": "success", "message": "Chat deleted"}
        else:
            raise HTTPException(status_code=404, detail="Chat not found")
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception("Error deleting chat")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chats/{user_id}/{thread_id}")
def get_chat_details(user_id: str, thread_id: str):
    """Get chat details and metadata"""
    try:
        chat = chat_manager.get_chat(user_id, thread_id)
        if chat:
            return {"status": "success", "chat": chat}
        else:
            raise HTTPException(status_code=404, detail="Chat not found")
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception("Error getting chat details")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
def healthz():
    return {"ok": True, "time": datetime.utcnow().isoformat()}


# Local run: uvicorn app:app --reload

# cd ui && python -m http.server 3000