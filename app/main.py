import mimetypes
import os
import time
import uuid
from typing import Literal, List

from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from fastapi import (
    FastAPI,
    WebSocket,
    HTTPException,
    Request,
    BackgroundTasks,
    Depends,
    File,
    UploadFile,
    status,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
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
from app.prompt_optimizer import optimize_prompt
from langchain_core.messages import AIMessage
from app.helper import serialize_content_to_string
from app.chat_manager import ChatManager
from app.websocket_manager import WebSocketConnectionManager
from app.store import get_record
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI, InvalidWebhookSignatureError

# Consolidated FastAPI imports above
from app.config import OPENAI_WEBHOOK_SECRET
from app.schema import (
    PromptOptimizerInput,
    ChatCreateRequest,
    ChatRenameRequest,
    UploadResponse,
)
from app.config import (
    ALLOWED_UPLOAD_MIME_TYPES,
    MAX_UPLOAD_SIZE_BYTES,
    TTL_SECONDS,
    UPLOAD_PREFIX,
)
from app.utils import create_s3_client

client = OpenAI(
    webhook_secret=OPENAI_WEBHOOK_SECRET,
)


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


app = FastAPI(title="BlackBox Playground", lifespan=lifespan)
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
            tool = payload.get("tool", "")
            contexts = payload.get("contexts", [])
            metadata = payload.get("metadata", {})

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
                tool=tool,
                contexts=contexts,
                metadata=metadata,
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


def _extract_response_text(response, rid=None):
    i = 0
    while i < 2:
        i += 1
        try:
            """
            Extract concatenated text from any 'message' entries in response['output'].
            This matches the JSON structure you pasted earlier.
            """
            texts = []
            response = response.model_dump()
            for item in response.get("output", []):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if part.get("type") == "output_text":
                            text = part.get("text")
                            if text:
                                texts.append(text)
            if texts:
                return "".join(texts)
        except Exception as e:
            print(f"Attempt {i} to extract response text for {rid} failed: {e}")
        if rid:
            response = client.responses.retrieve(rid)
    return ""


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

            if not answer:
                answer = _extract_response_text(response, rid)
                if not answer:
                    answer = (
                        "Deep Research could not be completed due to an issue with the "
                        "OpenAI(Chat GPT) API service."
                    )
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
async def get_research_status(thread_id: str):
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


@app.post("/api/prompt/optimize")
async def prompt_optimizer(request: PromptOptimizerInput):
    """Optimize a prompt"""
    try:
        optimized_prompt = await optimize_prompt(request.user_prompt)
        return {"status": "success", "result": optimized_prompt}
    except Exception as e:
        LOGGER.exception("Error optimizing prompt")
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


@app.get("/api/health")
async def health_check():
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


# Upload endpoint
@app.post(
    "/api/upload",
    response_model=UploadResponse,
    description="Upload one or more files and receive presigned URLs to access them.",
)
async def upload_files(
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    thread_id: str = Form(...),
):
    """
    Upload images to S3 and return pre-signed URLs for later retrieval.

    Validates file types and sizes, handles S3 connectivity errors, and returns
    per-file metadata including the object key, size, URL, and expiry.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file must be provided.",
        )

    bucket = os.getenv("IMAGE_UPLOAD_BUCKET", "playground-images")
    raw_ttl = os.getenv("TTL_SECONDS", "3600")
    try:
        expires_in = int(raw_ttl) if raw_ttl else TTL_SECONDS
    except (TypeError, ValueError):
        LOGGER.warning(
            "Invalid TTL_SECONDS value '%s'; using default %s",
            raw_ttl,
            TTL_SECONDS,
        )
        expires_in = TTL_SECONDS

    expires_at = int(time.time()) + expires_in

    try:
        s3 = create_s3_client()
    except (BotoCoreError, NoCredentialsError, ClientError) as exc:
        LOGGER.error(
            "Failed to initialize S3 client for global document uploads", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to storage service.",
        ) from exc

    responses = []

    for file in files:
        try:
            filename = os.path.basename(file.filename or "").strip()
            if not filename:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file must include a filename.",
                )

            guessed, _ = mimetypes.guess_type(file.filename or "")
            mime_type = (
                file.content_type or guessed or "application/octet-stream"
            ).lower()
            if mime_type not in ALLOWED_UPLOAD_MIME_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=(
                        f"Unsupported file type '{mime_type}'. "
                        f"Allowed types: {sorted(ALLOWED_UPLOAD_MIME_TYPES)}"
                    ),
                )

            content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
            size = len(content)
            if size == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File '{filename}' is empty.",
                )
            if size > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"File '{filename}' exceeds the maximum allowed size of "
                        f"{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB."
                    ),
                )

            object_key = f"{UPLOAD_PREFIX}/{str(user_id)}/{str(thread_id)}/{uuid.uuid4()}-{filename}"

            try:
                s3.put_object(
                    Bucket=bucket,
                    Key=object_key,
                    Body=content,
                    ContentType=mime_type,
                )
            except (BotoCoreError, NoCredentialsError, ClientError) as exc:
                LOGGER.error(
                    "Failed to upload global document file",
                    exc_info=True,
                    extra={"key": object_key, "bucket": bucket},
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Unable to connect to storage service.",
                ) from exc

            try:
                url = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket, "Key": object_key},
                    ExpiresIn=expires_in,
                )
            except (BotoCoreError, NoCredentialsError, ClientError) as exc:
                LOGGER.error(
                    "Failed to generate presigned URL for global document file",
                    exc_info=True,
                    extra={"key": object_key, "bucket": bucket},
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Unable to generate pre-signed URL.",
                ) from exc

            if not url:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to generate pre-signed URL.",
                )

            responses.append(
                {
                    "original_filename": filename,
                    "file_size": size,
                    # "s3_key": object_key,
                    "url": url,
                    "expires_at": expires_at,
                }
            )
        finally:
            await file.close()

    return {"files": responses}


# Local run: uvicorn app:app --reload

# cd ui && python -m http.server 3000
