"""
Common utilities for AWS Lambda functions.

This package provides shared functionality for AWS Lambda handlers including:
- Logging with request ID tracking
- Execution time measurement
- Parameter retrieval from SSM
- Database connection management
- Redis connection management
"""

from common.database import get_connection
from common.datastore import (
    S3DataNotFoundError,
    S3InvalidJSONError,
    S3OperationError,
    check_s3_file_exists,
    get_s3_data,
    store_s3_data,
)
from common.decorators import measure_execution_time

# Import core components for easy access
from common.logging import get_custom_logger, set_request_id
from common.parameters import get_parameter
from common.redis import RedisManager, RedisService

__all__ = [
    "get_custom_logger",
    "set_request_id",
    "get_parameter",
    "measure_execution_time",
    "get_connection",
    "RedisManager",
    "RedisService",
    "get_s3_data",
    "store_s3_data",
    "check_s3_file_exists",
    "S3OperationError",
    "S3DataNotFoundError",
    "S3InvalidJSONError",
]
