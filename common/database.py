"""
Database connection utilities for PostgreSQL.
"""
import threading
from typing import Optional
import psycopg2

from common.logging import get_custom_logger
from common.parameters import get_parameter, get_parameter_path_prefix

logger = get_custom_logger("blackbox.common.database")

# Singleton connection instance and lock
_connection = None
_lock = threading.Lock()


def get_connection(parameter_path_prefix: Optional[str] = None):
    """
    Returns a singleton database connection. Creates the connection on first call
    or if the previous connection has been closed.

    Args:
        parameter_path_prefix: The prefix path for SSM parameters

    Returns:
        A PostgreSQL database connection

    Raises:
        Exception: If the connection cannot be established
    """
    global _connection

    # If we've already got an open connection, return it
    if _connection is not None and _connection.closed == 0:
        return _connection
    parameter_path_prefix = (
        parameter_path_prefix if parameter_path_prefix else get_parameter_path_prefix()
    )

    # Otherwise, acquire lock and (re)create the connection
    with _lock:
        if _connection is not None and _connection.closed == 0:
            return _connection
        try:
            _connection = psycopg2.connect(
                host=get_parameter("db-endpoint", parameter_path_prefix),
                port=get_parameter("db-port", parameter_path_prefix),
                database=get_parameter("db-name", parameter_path_prefix),
                user=get_parameter("db-user", parameter_path_prefix),
                password=get_parameter("db-password", parameter_path_prefix),
            )
            logger.info("Database connection established successfully")
            return _connection
        except psycopg2.OperationalError as e:
            logger.error(f"Database connection failed: {e}", exc_info=True)
            raise
        except psycopg2.Error as e:
            logger.error(f"Database error occurred: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error while connecting to database: {e}", exc_info=True
            )
            raise
