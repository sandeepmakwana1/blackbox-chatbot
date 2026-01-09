"""
Common utilities for AWS Lambda functions.

This package provides shared functionality for AWS Lambda handlers including:
- Logging with request ID tracking
- Execution time measurement
- Parameter retrieval from SSM
- Database connection management
- Datastore operations
"""

from common.database import get_connection
from common.datastore import (
    S3DataNotFoundError,
    S3InvalidJSONError,
    S3OperationError,
    S3StoreService,
)
from common.decorators import measure_execution_time
from common.constants import ContextType

# Import core components for easy access
from common.logging import get_custom_logger, set_request_id
from common.parameters import get_parameter

__all__ = [
    "get_custom_logger",
    "set_request_id",
    "get_parameter",
    "measure_execution_time",
    "get_connection",
    "S3StoreService",
    "S3OperationError",
    "S3DataNotFoundError",
    "S3InvalidJSONError",
    "ContextType",
]
