from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai.resources.files import FilesWithRawResponse
from starlette.websockets import WebSocketDisconnect
from contextlib import asynccontextmanager

import json
from datetime import datetime
from app.utils import (
    ChatService,
    ConversationState,
    HumanMessage,
    SystemMessage,
    LOGGER,
)
from langchain_core.messages import AIMessage
from app.helper import serialize_content_to_string
from app.chat_manager import ChatManager
from app.websocket_manager import WebSocketConnectionManager
from app.store import get_record
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI, InvalidWebhookSignatureError
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from app.config import OPENAI_WEBHOOK_SECRET
from app.config import S3_PREFIX, S3_BUCKET, PRESIGN_EXPIRES, AWS_REGION
import os, uuid, mimetypes
from typing import List

import boto3
from botocore.client import Config
from fastapi import FastAPI, UploadFile, File, HTTPException, Form

s3 = boto3.client("s3", region_name=AWS_REGION, config=Config(s3={"addressing_style": "virtual"}))

class ChatCreateRequest(BaseModel):
    title: Optional[str] = None
    conversation_type: str = "chat"
    first_message: Optional[str] = None


class ChatRenameRequest(BaseModel):
    title: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    LOGGER.info("Starting up application...")
    try:
        await service.initialize()
        LOGGER.info("ChatService initialized successfully")
    except Exception as e:
        LOGGER.error(f"Failed to initialize ChatService during startup: {e}")
        # Don't fail startup - service will initialize on first request

    yield

    # Shutdown
    LOGGER.info("Shutting down application...")
    try:
        await service._cleanup_checkpointer()
        LOGGER.info("ChatService cleaned up successfully")
    except Exception as e:
        LOGGER.error(f"Error during shutdown: {e}")


app = FastAPI(title="LangGraph Advanced Chatbot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

service = ChatService()
chat_manager = ChatManager()
websocket_manager = WebSocketConnectionManager()


@app.websocket("/ws/{user_id}/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, thread_id: str):
    """Enhanced WebSocket endpoint with immediate thread binding."""
    await websocket.accept()

    # Immediate registration with provided thread_id
    if not thread_id or thread_id == "default":
        thread_id = chat_manager.get_or_create_default_chat(user_id)

    # Ensure the chat exists in our metadata (create if missing)
    existing_chat = chat_manager.get_chat(user_id, thread_id)
    if not existing_chat:
        chat_manager.create_chat(user_id, "", "chat")

    # Register WebSocket connection immediately
    await websocket_manager.add_connection(thread_id, user_id, websocket)
    current_thread_id = thread_id
    LOGGER.info(f"[WS] Registered user {user_id} to thread {thread_id}")

    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)

            user_message = payload.get("message", "")
            conversation_type = payload.get("type", "chat")
            requested_thread_id = payload.get("thread_id")
            language = payload.get("language", "English")

            # Allow thread switching if client provides different thread_id
            if requested_thread_id and requested_thread_id != current_thread_id:
                # Remove connection from current thread
                await websocket_manager.remove_connection(current_thread_id, user_id)

                # Add connection to new thread
                thread_id = requested_thread_id
                await websocket_manager.add_connection(thread_id, user_id, websocket)
                current_thread_id = thread_id
                LOGGER.info(f"[WS] Thread switched to {thread_id} for user {user_id}")

                # Ensure new chat exists
                existing_chat = chat_manager.get_chat(user_id, thread_id)
                if not existing_chat:
                    chat_manager.create_chat(user_id, "", conversation_type)

            # Skip empty messages
            if not user_message.strip():
                continue

            # Ensure the chat exists in our metadata
            existing_chat = chat_manager.get_chat(user_id, thread_id)
            if not existing_chat:
                chat_manager.create_chat(user_id, user_message, conversation_type)

            # Process the message with streaming
            async for chunk in service.process_message_streaming(
                message=user_message,
                thread_id=thread_id,
                user_id=user_id,
                language=language,
                conversation_type=conversation_type,
            ):
                await websocket.send_text(json.dumps(chunk))

            # Get current chat state and update activity
            chat = chat_manager.get_chat(user_id, thread_id)
            chat_manager.update_chat_activity(user_id, thread_id, user_message)

            # Auto-rename chat after first message if it has generic title
            if chat and chat["title"] in ["New Chat", "", "string"]:
                new_title = chat_manager.generate_title_from_message(user_message)
                chat_manager.rename_chat(user_id, thread_id, new_title)

    except WebSocketDisconnect:
        LOGGER.info(
            "WebSocket disconnected: user=%s thread=%s", user_id, current_thread_id
        )
    except json.JSONDecodeError as e:
        LOGGER.warning(f"Invalid JSON received from user {user_id}: {e}")
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "Invalid JSON format"})
            )
        except Exception:
            pass
    except KeyError as e:
        LOGGER.warning(f"Missing required field in message from user {user_id}: {e}")
        try:
            await websocket.send_text(
                json.dumps(
                    {"type": "error", "message": f"Missing required field: {str(e)}"}
                )
            )
        except Exception:
            pass
    except Exception:
        LOGGER.exception(f"WebSocket error for user {user_id}")
        try:
            error_message = "An internal error occurred. Please try again."
            await websocket.send_text(
                json.dumps({"type": "error", "message": error_message})
            )
        except Exception:
            LOGGER.error(f"Failed to send error message to user {user_id}")
    finally:
        if current_thread_id:
            await websocket_manager.remove_connection(current_thread_id, user_id)
        try:
            await websocket.close()
        except Exception:
            pass


client = OpenAI(
    webhook_secret=OPENAI_WEBHOOK_SECRET,
)


async def update_conversation_state_from_webhook(
    thread_id: str, user_id: str, answer: str, status: str
):
    """
    Update the LangGraph conversation state with the webhook result.

    Args:
        thread_id: The conversation thread ID
        user_id: The user ID
        answer: The response text from OpenAI
        status: Either "completed" or "terminated"
    """
    if not thread_id:
        LOGGER.warning("No thread_id provided for state update, skipping")
        return

    try:
        # Ensure service is initialized and connected
        await service._validate_connection()

        # Create config for the thread
        config = {"configurable": {"thread_id": thread_id}}

        # Try to get existing state
        try:
            existing_state = await service.app.aget_state(config)
            LOGGER.info(f"Retrieved existing state for thread {thread_id}")
        except Exception as e:
            LOGGER.warning(
                f"Could not retrieve existing state for thread {thread_id}: {e}"
            )
            # Continue anyway - aupdate_state can handle missing threads

        # Create assistant message with the result
        assistant_message = AIMessage(content=answer)

        # Prepare state patch - messages will be appended due to add_messages annotation
        patch = {"messages": [assistant_message], "research_status": status}

        # Update the state
        await service.app.aupdate_state(config, patch)

        LOGGER.info(f"Webhook state updated for thread {thread_id}, status={status}")

        # Broadcast the result to connected WebSocket clients
        try:
            # Get current connection stats for debugging
            stats = websocket_manager.get_connection_stats()
            LOGGER.info(f"Current WebSocket stats before broadcast: {stats}")

            webhook_message = {
                "type": "webhook_result",
                "content": answer,
                "thread_id": thread_id,
                "research_status": status,
                "done": True,
                "timestamp": datetime.utcnow().isoformat(),
            }

            LOGGER.info(
                f"Attempting to broadcast webhook result for thread {thread_id}"
            )
            LOGGER.info(f"Webhook message: {webhook_message}")

            broadcast_count = await websocket_manager.broadcast_to_thread(
                thread_id, webhook_message
            )
            LOGGER.info(
                f"Broadcasted webhook result to {broadcast_count} connected clients on thread {thread_id}"
            )

            if broadcast_count == 0:
                LOGGER.warning(
                    f"No connected clients found for thread {thread_id}. Current connections: {stats['threads']}"
                )

        except Exception as broadcast_error:
            LOGGER.error(
                f"Failed to broadcast webhook result for thread {thread_id}: {broadcast_error}"
            )
            import traceback

            LOGGER.error(f"Broadcast error traceback: {traceback.format_exc()}")
            # Don't let broadcast failures affect the webhook processing

    except Exception as e:
        LOGGER.error(f"Failed to update conversation state for thread {thread_id}: {e}")


@app.post("/webhook")
async def webhook(request: Request, bg: BackgroundTasks):
    """
    OpenAI will send a POST request to this endpoint,
    which will store the deep-research data in Redis.
    """
    # logger.info("webhook triggered at %s", time.time())
    body = await request.body()
    headers = request.headers

    try:
        evt = client.webhooks.unwrap(body, headers)
    except InvalidWebhookSignatureError:
        # logger.error("invalid signature")
        raise HTTPException(400, "invalid signature")

    if evt.type == "response.completed":
        rid = evt.data.id

        # run heavy fetch in the background
        async def handle():
            print("start running heavy fetch")
            response = client.responses.retrieve(rid)
            answer = response.output_text
            metadata = response.metadata or {}
            thread_id = metadata.get("thread_id")
            user_id = metadata.get("user_id")

            # Validate required fields
            if not thread_id:
                LOGGER.warning(
                    "No thread_id found in webhook metadata, skipping state update"
                )
                return

            from app.store import update_record
            import time

            # Update Redis record
            update_record(
                thread_id,
                {"answer": answer, "status": "completed", "end_at": time.time()},
            )

            # Update LangGraph conversation state
            await update_conversation_state_from_webhook(
                thread_id=thread_id,
                user_id=user_id or "unknown",
                answer=answer,
                status="completed",
            )

            print(f"Deep research completed for thread {thread_id}")

        bg.add_task(handle)

    else:
        rid = evt.data.id
        print(evt.data, "Failed : evt.data ")

        async def handle_terminate():
            response = client.responses.retrieve(rid)
            answer = response.output_text
            metadata = response.metadata or {}
            thread_id = metadata.get("thread_id")
            user_id = metadata.get("user_id")

            # Validate required fields
            if not thread_id:
                LOGGER.warning(
                    "No thread_id found in terminate webhook metadata, skipping state update"
                )
                return

            from app.store import update_record
            import time

            # Update Redis record
            update_record(
                thread_id,
                {"answer": answer, "status": "terminated", "end_at": time.time()},
            )

            # Update LangGraph conversation state
            await update_conversation_state_from_webhook(
                thread_id=thread_id,
                user_id=user_id or "unknown",
                answer=answer,
                status="terminated",
            )

            print(f"Deep research terminated for thread {thread_id}")

        bg.add_task(handle_terminate)

    return {}  # 200


@app.get("/conversation/{user_id}/{thread_id}")
async def get_conversation_state(user_id: str, thread_id: str):
    """Return the stored state for a given conversation thread."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        await service._validate_connection()
        state = await service.app.aget_state(config)
        msgs = []
        for m in state.values.get("messages", []):
            role = "assistant"
            if isinstance(m, HumanMessage):
                role = "human"
            elif isinstance(m, SystemMessage):
                role = "system"

            # Use centralized content serialization
            content = getattr(m, "content", "")
            content = serialize_content_to_string(content)
            msgs.append({"role": role, "content": content})

        return {
            "messages": msgs,
            "token_tracking": state.values.get("token_tracking", {}),
            "conversation_type": state.values.get("conversation_type", "chat"),
            "research_status": state.values.get("research_status"),
        }
    except Exception as e:
        LOGGER.exception("Error getting conversation state")
        return {
            "messages": [],
            "token_tracking": {},
            "conversation_type": "chat",
            "error": "No conversation state found",
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
            thread_id = chat_manager.create_chat(
                user_id, first_message, request.conversation_type
            )
        elif title:
            thread_id = chat_manager.create_chat(
                user_id, first_message, request.conversation_type
            )
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


@app.get("/api/research/{thread_id}")
def get_research_status(thread_id: str):
    """Get deep research status and result for a thread"""
    try:
        record = get_record(thread_id)
        if record:
            return {
                "thread_id": record.get("thread_id"),
                "user_id": record.get("user_id"),
                "status": record.get("status", "unknown"),
                "research_plan": record.get("research_plan"),
                "answer": record.get("answer"),
                "start_at": record.get("start_at"),
                "completed_at": record.get("completed_at"),
                "response_id": record.get("response_id"),
            }
        else:
            raise HTTPException(status_code=404, detail="Research record not found")
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception("Error getting research status")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/websocket/stats")
def get_websocket_stats():
    """Get WebSocket connection statistics for monitoring"""
    try:
        stats = websocket_manager.get_connection_stats()
        return {"status": "success", "websocket_stats": stats}
    except Exception as e:
        LOGGER.exception("Error getting WebSocket stats")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/test/broadcast/{thread_id}")
async def test_broadcast(thread_id: str, message: dict = None):
    """Test endpoint to manually trigger a webhook broadcast"""
    try:
        test_message = message or {
            "type": "webhook_result",
            "content": "This is a test message from the manual broadcast endpoint!",
            "thread_id": thread_id,
            "research_status": "completed",
            "done": True,
            "timestamp": datetime.utcnow().isoformat(),
        }

        LOGGER.info(
            f"Testing broadcast for thread {thread_id} with message: {test_message}"
        )
        broadcast_count = await websocket_manager.broadcast_to_thread(
            thread_id, test_message
        )

        return {
            "status": "success",
            "thread_id": thread_id,
            "broadcast_count": broadcast_count,
            "message": test_message,
        }
    except Exception as e:
        LOGGER.exception("Error testing broadcast")
        raise HTTPException(status_code=500, detail=str(e))


def make_key(filename: str, user_id: str, thread_id: str) -> str:
    name, ext = os.path.splitext(os.path.basename(filename or "file.bin"))
    return f"{S3_PREFIX}/{user_id}/{thread_id}/{name[:64]}-{uuid.uuid4().hex}{ext.lower()}"

@app.post("/api/upload", response_model=List[str]) 
async def upload_and_presign(
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    thread_id: str = Form(...)
):
    """
    Accepts files, uploads each to S3, and returns a list of pre-signed GET URLs.
    Form field name must be `files`.
    """
    if not files or not thread_id or not user_id:
        print(files, thread_id, user_id)
        raise HTTPException(status_code=400, detail="No files provided or thread_id or user_id")

    urls: List[str] = []
    for f in files:
        key = make_key(f.filename, user_id, thread_id)
        content_type = f.content_type or mimetypes.guess_type(f.filename or "")[0] or "application/octet-stream"
        print(content_type, "==========>")
        try:
            await f.seek(0)
            s3.upload_fileobj(f.file, S3_BUCKET, key, ExtraArgs={"ContentType": content_type})
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": S3_BUCKET, "Key": key},
                ExpiresIn=PRESIGN_EXPIRES,
            )
            urls.append(url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process {f.filename}: {e}")
        finally:
            await f.close()
        
    return urls

@app.get("/api/health")
def health_check():
    """Comprehensive health check with database connection pool status"""
    try:
        # Test database connection
        pool_health = chat_manager._perform_health_check()
        pool_stats = chat_manager.get_pool_stats()

        # Test LangGraph service initialization
        service_status = "initialized" if service._is_initialized else "not_initialized"

        health_status = {
            "status": "healthy" if pool_health else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": {
                    "status": "healthy" if pool_health else "unhealthy",
                    "pool_stats": pool_stats,
                },
                "langgraph": {
                    "status": service_status,
                    "db_uri_configured": bool(service.db_uri),
                },
            },
        }

        status_code = 200 if pool_health else 503
        return health_status

    except Exception as e:
        LOGGER.exception("Health check failed")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }


# Local run: uvicorn app:app --reload

# cd ui && python -m http.server 3000
