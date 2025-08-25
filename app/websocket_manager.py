import asyncio
import json
import logging
from typing import Dict, Set, Optional, Any
from fastapi import WebSocket
from starlette.websockets import WebSocketState

LOGGER = logging.getLogger(__name__)


class WebSocketConnectionManager:
    """
    Manages WebSocket connections for real-time broadcasting of webhook results.
    Tracks connections by thread_id and user_id for targeted messaging.
    """

    def __init__(self):
        # Structure: {thread_id: {user_id: websocket}}
        self.connections: Dict[str, Dict[str, WebSocket]] = {}
        # Track which threads each user is connected to for cleanup
        self.user_threads: Dict[str, Set[str]] = {}

    async def add_connection(self, thread_id: str, user_id: str, websocket: WebSocket):
        """
        Add a WebSocket connection for a specific thread and user.
        
        Args:
            thread_id: The conversation thread ID
            user_id: The user ID
            websocket: The WebSocket connection
        """
        if thread_id not in self.connections:
            self.connections[thread_id] = {}
        
        # If user already has a connection for this thread, close the old one
        if user_id in self.connections[thread_id]:
            old_websocket = self.connections[thread_id][user_id]
            try:
                if old_websocket.client_state == WebSocketState.CONNECTED:
                    await old_websocket.close()
                    LOGGER.info(f"Closed previous WebSocket for user {user_id} on thread {thread_id}")
            except Exception as e:
                LOGGER.warning(f"Error closing old WebSocket connection: {e}")
        
        self.connections[thread_id][user_id] = websocket
        
        # Track user threads for cleanup
        if user_id not in self.user_threads:
            self.user_threads[user_id] = set()
        self.user_threads[user_id].add(thread_id)
        
        LOGGER.info(f"Added WebSocket connection: user {user_id} -> thread {thread_id}")

    async def remove_connection(self, thread_id: str, user_id: str):
        """
        Remove a WebSocket connection for a specific thread and user.
        
        Args:
            thread_id: The conversation thread ID
            user_id: The user ID
        """
        if thread_id in self.connections and user_id in self.connections[thread_id]:
            del self.connections[thread_id][user_id]
            LOGGER.info(f"Removed WebSocket connection: user {user_id} from thread {thread_id}")
            
            # Clean up empty thread containers
            if not self.connections[thread_id]:
                del self.connections[thread_id]
                LOGGER.info(f"Removed empty thread container: {thread_id}")
        
        # Update user thread tracking
        if user_id in self.user_threads:
            self.user_threads[user_id].discard(thread_id)
            if not self.user_threads[user_id]:
                del self.user_threads[user_id]

    def get_connections_for_thread(self, thread_id: str) -> Dict[str, WebSocket]:
        """
        Get all active WebSocket connections for a specific thread.
        
        Args:
            thread_id: The conversation thread ID
            
        Returns:
            Dictionary of {user_id: websocket} for the thread
        """
        return self.connections.get(thread_id, {}).copy()

    async def broadcast_to_thread(self, thread_id: str, message: Dict[str, Any]) -> int:
        """
        Broadcast a message to all connected clients on a specific thread.
        
        Args:
            thread_id: The conversation thread ID
            message: The message to broadcast
            
        Returns:
            Number of successful broadcasts
        """
        connections = self.get_connections_for_thread(thread_id)
        
        if not connections:
            LOGGER.info(f"No active connections found for thread {thread_id}")
            return 0
        
        message_json = json.dumps(message)
        successful_broadcasts = 0
        failed_connections = []
        
        for user_id, websocket in connections.items():
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(message_json)
                    successful_broadcasts += 1
                    LOGGER.info(f"Broadcasted to user {user_id} on thread {thread_id}")
                else:
                    LOGGER.warning(f"WebSocket not connected for user {user_id} on thread {thread_id}")
                    failed_connections.append((user_id, thread_id))
            except Exception as e:
                LOGGER.error(f"Failed to broadcast to user {user_id} on thread {thread_id}: {e}")
                failed_connections.append((user_id, thread_id))
        
        # Clean up failed connections
        for user_id, thread_id in failed_connections:
            await self.remove_connection(thread_id, user_id)
        
        LOGGER.info(f"Broadcast complete: {successful_broadcasts} successful, {len(failed_connections)} failed")
        return successful_broadcasts

    async def cleanup_closed_connections(self):
        """
        Clean up any closed or stale WebSocket connections.
        Should be called periodically or when connections are suspected to be stale.
        """
        closed_connections = []
        
        for thread_id, users in self.connections.items():
            for user_id, websocket in users.items():
                try:
                    if websocket.client_state != WebSocketState.CONNECTED:
                        closed_connections.append((thread_id, user_id))
                except Exception as e:
                    LOGGER.warning(f"Error checking WebSocket state for user {user_id}: {e}")
                    closed_connections.append((thread_id, user_id))
        
        for thread_id, user_id in closed_connections:
            await self.remove_connection(thread_id, user_id)
        
        if closed_connections:
            LOGGER.info(f"Cleaned up {len(closed_connections)} closed connections")

    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current connections for monitoring/debugging.
        
        Returns:
            Dictionary with connection statistics
        """
        total_connections = sum(len(users) for users in self.connections.values())
        
        return {
            "total_threads": len(self.connections),
            "total_connections": total_connections,
            "total_users": len(self.user_threads),
            "threads": {
                thread_id: list(users.keys()) 
                for thread_id, users in self.connections.items()
            }
        }
