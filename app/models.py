"""SQLAlchemy models backing our Postgres tables.

This mirrors the tables created by the chat manager and the LangGraph
Postgres checkpointer so Alembic autogenerate can see them instead of
trying to drop them.
"""

from __future__ import annotations

import uuid
from sqlalchemy import Column, DateTime, Index, Integer, String, Text, desc, text
from sqlalchemy.dialects.postgresql import BYTEA, JSONB, UUID
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class ChatMetadata(Base):
    __tablename__ = "chat_metadata"

    thread_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False)
    title = Column(String(255), nullable=False)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )
    message_count = Column(Integer, nullable=False, server_default=text("0"))
    last_message_preview = Column(Text)
    conversation_type = Column(String(50))

    __table_args__ = (
        Index(
            "idx_chat_metadata_user_updated",
            "user_id",
            desc("updated_at"),
        ),
    )


class CheckpointMigration(Base):
    __tablename__ = "checkpoint_migrations"

    v = Column(Integer, primary_key=True)


class Checkpoint(Base):
    __tablename__ = "checkpoints"

    thread_id = Column(Text, primary_key=True)
    checkpoint_ns = Column(Text, primary_key=True, server_default=text("''"))
    checkpoint_id = Column(Text, primary_key=True)
    parent_checkpoint_id = Column(Text)
    type = Column(Text)
    checkpoint = Column(JSONB, nullable=False)
    metadata = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))

    __table_args__ = (Index("checkpoints_thread_id_idx", "thread_id"),)


class CheckpointBlob(Base):
    __tablename__ = "checkpoint_blobs"

    thread_id = Column(Text, primary_key=True)
    checkpoint_ns = Column(Text, primary_key=True, server_default=text("''"))
    channel = Column(Text, primary_key=True)
    version = Column(Text, primary_key=True)
    type = Column(Text, nullable=False)
    blob = Column(BYTEA)

    __table_args__ = (Index("checkpoint_blobs_thread_id_idx", "thread_id"),)


class CheckpointWrite(Base):
    __tablename__ = "checkpoint_writes"

    thread_id = Column(Text, primary_key=True)
    checkpoint_ns = Column(Text, primary_key=True, server_default=text("''"))
    checkpoint_id = Column(Text, primary_key=True)
    task_id = Column(Text, primary_key=True)
    idx = Column(Integer, primary_key=True)
    channel = Column(Text, nullable=False)
    type = Column(Text)
    blob = Column(BYTEA, nullable=False)
    task_path = Column(Text, nullable=False, server_default=text("''"))

    __table_args__ = (Index("checkpoint_writes_thread_id_idx", "thread_id"),)


# Convenient alias for Alembic env.py
target_metadata = Base.metadata
