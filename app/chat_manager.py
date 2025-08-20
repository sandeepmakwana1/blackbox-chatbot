import os
import re
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

# Production-ready PostgreSQL driver
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import DictCursor

# Configure logging
LOGGER = logging.getLogger("chat-manager-db")
logging.basicConfig(level=logging.INFO)


@dataclass
class ChatMetadata:
    """Chat metadata structure (matches the database schema)"""
    thread_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    last_message_preview: str
    conversation_type: str
    user_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass instance to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMetadata':
        """Creates a ChatMetadata instance from a dictionary."""
        # Convert UUID and datetime objects to strings if they are not already
        data['thread_id'] = str(data.get('thread_id', ''))
        data['created_at'] = data.get('created_at', '').isoformat() if hasattr(data.get('created_at'), 'isoformat') else str(data.get('created_at', ''))
        data['updated_at'] = data.get('updated_at', '').isoformat() if hasattr(data.get('updated_at'), 'isoformat') else str(data.get('updated_at', ''))
        return cls(**data)


class ChatManager:
    """Manages chat metadata, persistence, and operations using a PostgreSQL DB."""

    def __init__(self):
        """
        Initializes the ChatManager and sets up the database connection pool.
        Database connection details are fetched from environment variables.
        """
        try:
            # Fetch DB connection details from environment variables
            db_name = os.getenv("POSTGRES_DB")
            db_user = os.getenv("POSTGRES_USER")
            db_password = os.getenv("POSTGRES_PASSWORD")
            db_host = os.getenv("POSTGRES_HOST", "localhost")
            db_port = os.getenv("POSTGRES_PORT", "5432")

            if not all([db_name, db_user, db_password]):
                raise ValueError("Missing one or more required database environment variables.")

            dsn = f"dbname='{db_name}' user='{db_user}' password='{db_password}' host='{db_host}' port='{db_port}'"
            
            # Create a connection pool (min_conn=1, max_conn=10)
            self.pool = SimpleConnectionPool(1, 10, dsn)
            LOGGER.info("Database connection pool created successfully.")
            self.setup_database()

        except (Exception, psycopg2.Error) as error:
            LOGGER.error("Error while connecting to PostgreSQL", exc_info=error)
            raise

    def setup_database(self):
        """Ensures the required table exists in the database."""
        # The SQL schema is now in a separate `schema.sql` file
        # This function can be used to apply it if needed, or you can apply it manually.
        schema_query = """
        CREATE TABLE IF NOT EXISTS chat_metadata (
            thread_id UUID PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            title VARCHAR(255) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            message_count INTEGER NOT NULL DEFAULT 0,
            last_message_preview TEXT,
            conversation_type VARCHAR(50)
        );
        CREATE INDEX IF NOT EXISTS idx_chat_metadata_user_updated ON chat_metadata(user_id, updated_at DESC);
        """
        try:
            self._execute_query(schema_query)
            LOGGER.info("Database schema verified and is up to date.")
        except Exception as e:
            LOGGER.error(f"Error setting up database schema: {e}")
            raise

    def _execute_query(self, query: str, params: Optional[Tuple] = None, fetch: Optional[str] = None):
        """
        Executes a query using a connection from the pool.
        
        Args:
            query (str): The SQL query to execute.
            params (tuple, optional): Parameters to substitute in the query.
            fetch (str, optional): Type of fetch ('one', 'all').
        
        Returns:
            The result of the query execution or row count for modification queries.
        """
        conn = None
        try:
            conn = self.pool.getconn()
            # Use DictCursor to get results as dictionaries
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query, params)
                
                if fetch == 'one':
                    return cur.fetchone()
                elif fetch == 'all':
                    return cur.fetchall()
                else:
                    # For INSERT, UPDATE, DELETE, return the number of affected rows
                    conn.commit()
                    return cur.rowcount
        except (Exception, psycopg2.Error) as error:
            LOGGER.error("PostgreSQL Error:", exc_info=error)
            if conn:
                conn.rollback() # Rollback on error
            raise
        finally:
            if conn:
                self.pool.putconn(conn) # Always return the connection to the pool

    def generate_title_from_message(self, message: str) -> str:
        """Generate a chat title from the first user message."""
        if not message or not message.strip():
            return "New Chat"
        
        clean_message = re.sub(r'\s+', ' ', message.strip())
        clean_message = re.sub(r'^(hi|hello|hey|can you|could you|please|help me|i need|i want)\s+', '', clean_message.lower(), flags=re.IGNORECASE)
        clean_message = clean_message.capitalize()
        
        if len(clean_message) <= 50:
            return clean_message
            
        truncated = clean_message[:50]
        last_space = truncated.rfind(' ')
        return truncated[:last_space] + "..." if last_space > 30 else truncated[:47] + "..."

    def create_chat(self, user_id: str, first_message: str = "", conversation_type: str = "chat") -> str:
        """Create a new chat in the database and return thread_id."""
        thread_id = uuid.uuid4()
        now = datetime.utcnow()
        title = self.generate_title_from_message(first_message) if first_message else "New Chat"
        
        query = """
            INSERT INTO chat_metadata 
            (thread_id, user_id, title, created_at, updated_at, message_count, last_message_preview, conversation_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            str(thread_id), user_id, title, now, now,
            1 if first_message else 0,
            first_message[:100] if first_message else "",
            conversation_type
        )
        
        self._execute_query(query, params)
        LOGGER.info(f"Created new chat {thread_id} for user {user_id}: '{title}'")
        return str(thread_id)

    def get_user_chats(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all chats for a user, sorted by most recent first."""
        query = "SELECT * FROM chat_metadata WHERE user_id = %s ORDER BY updated_at DESC"
        rows = self._execute_query(query, (user_id,), fetch='all')
        return [ChatMetadata.from_dict(dict(row)).to_dict() for row in rows] if rows else []

    def get_chat(self, user_id: str, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chat by thread_id."""
        query = "SELECT * FROM chat_metadata WHERE user_id = %s AND thread_id = %s"
        row = self._execute_query(query, (user_id, thread_id), fetch='one')
        return ChatMetadata.from_dict(dict(row)).to_dict() if row else None

    def update_chat_activity(self, user_id: str, thread_id: str, last_message: str = "") -> bool:
        """Update chat activity (message count, last message, updated_at)."""
        query = """
            UPDATE chat_metadata
            SET 
                message_count = message_count + 1,
                updated_at = %s,
                last_message_preview = %s
            WHERE user_id = %s AND thread_id = %s
        """
        params = (datetime.utcnow(), last_message[:100], user_id, thread_id)
        affected_rows = self._execute_query(query, params)
        return affected_rows > 0

    def rename_chat(self, user_id: str, thread_id: str, new_title: str) -> bool:
        """Rename a chat."""
        new_title = new_title.strip()[:255] # Use DB column limit
        if not new_title:
            return False
            
        query = """
            UPDATE chat_metadata
            SET title = %s, updated_at = %s
            WHERE user_id = %s AND thread_id = %s
        """
        params = (new_title, datetime.utcnow(), user_id, thread_id)
        affected_rows = self._execute_query(query, params)
        if affected_rows > 0:
            LOGGER.info(f"Renamed chat {thread_id} to '{new_title}'")
        return affected_rows > 0

    def delete_chat(self, user_id: str, thread_id: str) -> bool:
        """Delete a chat from the database."""
        query = "DELETE FROM chat_metadata WHERE user_id = %s AND thread_id = %s"
        affected_rows = self._execute_query(query, (user_id, thread_id))
        if affected_rows > 0:
            LOGGER.info(f"Deleted chat {thread_id} for user {user_id}")
        return affected_rows > 0

    def get_or_create_default_chat(self, user_id: str) -> str:
        """Get the most recent chat or create a new one if none exists."""
        user_chats = self.get_user_chats(user_id)
        if user_chats:
            return user_chats[0]['thread_id']
        else:
            return self.create_chat(user_id)

