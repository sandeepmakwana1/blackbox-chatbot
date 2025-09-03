"""
Utilities for retrieving parameters from AWS SSM Parameter Store.
"""

from contextvars import ContextVar
from typing import Optional

import boto3

# Context variable to store parameter path prefix (scoped per request)
PARAMETER_PATH_PREFIX: ContextVar[str] = ContextVar(
    "PARAMETER_PATH_PREFIX", default="blackbox-rfp"
)


def set_parameter_path_prefix(prefix: str):
    """Set the parameter path prefix for the current context."""
    PARAMETER_PATH_PREFIX.set(prefix or "blackbox-rfp")


def get_parameter_path_prefix() -> str:
    """Get the parameter path prefix for the current context."""
    return PARAMETER_PATH_PREFIX.get()


# Singleton SSM client instance
_ssm_client = None


def _get_ssm_client():
    """Returns a singleton SSM client."""
    global _ssm_client
    if _ssm_client is None:
        _ssm_client = boto3.client("ssm", region_name="us-east-1")
    return _ssm_client


def get_parameter(
    parameter_name: str, parameter_path_prefix: Optional[str] = None
) -> str:
    """
    Retrieve a parameter from SSM Parameter Store.

    Args:
        parameter_name: The name of the parameter to retrieve.
        parameter_path_prefix: Optional override for the parameter path prefix.
            If not provided, uses the context-local default.

    Returns:
        The parameter value as a string.

    Raises:
        Exception: If the parameter cannot be retrieved.
    """
    try:
        prefix = (
            parameter_path_prefix
            if parameter_path_prefix is not None
            else get_parameter_path_prefix()
        )
        response = _get_ssm_client().get_parameter(
            Name=f"/{prefix}/{parameter_name}", WithDecryption=True
        )
        return response["Parameter"]["Value"]
    except Exception as e:
        raise ValueError(f"Failed to get parameter {parameter_name}") from e
