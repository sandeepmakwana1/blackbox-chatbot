from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

import json
from datetime import datetime
from app.utils import ChatService, ConversationState, HumanMessage, SystemMessage, LOGGER
from app.chat_manager import ChatManager


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
            
            # Update chat activity in metadata after streaming is complete
            chat_manager.update_chat_activity(user_id, thread_id, user_message)

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


@app.get("/healthz")
def healthz():
    return {"ok": True, "time": datetime.utcnow().isoformat()}




# Local run: uvicorn app:app --reload

# cd ui && python -m http.server 3000