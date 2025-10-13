import json
from contextlib import closing
from typing import Any, Optional

import boto3
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError

from common.logging import get_custom_logger

logger = get_custom_logger(__name__)


class S3OperationError(RuntimeError):
    """Base exception raised when an S3 operation cannot be completed."""


class S3DataNotFoundError(S3OperationError):
    """Raised when the requested key does not exist in the target bucket."""


class S3InvalidJSONError(S3OperationError):
    """Raised when the S3 object content cannot be parsed as JSON."""


class S3ClientSingleton:
    """Simple singleton to provide a shared boto3 S3 client."""

    _instance: Optional["S3ClientSingleton"] = None
    _client: Optional[BaseClient] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._client = boto3.client("s3")
            logger.debug("Initialised singleton S3 client")
        return cls._instance

    @property
    def client(self) -> BaseClient:
        if self._client is None:
            self._client = boto3.client("s3")
            logger.debug("Re-created singleton S3 client after reset")
        return self._client


def _get_s3_client() -> BaseClient:
    return S3ClientSingleton().client


def _ensure_json_bytes(data: Any) -> bytes:
    """Ensure input data is serialised to JSON bytes, validating format."""

    if isinstance(data, bytes):
        text = data.decode("utf-8")
    elif isinstance(data, str):
        text = data
    else:
        try:
            text = json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise S3InvalidJSONError("Provided data is not JSON serialisable") from exc

    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        raise S3InvalidJSONError("Provided data is not valid JSON") from exc

    return text.encode("utf-8")


def get_s3_data(s3_key: str, bucket_name: str) -> str:
    """Return JSON payload stored at ``s3://bucket_name/s3_key`` as a string."""

    client = _get_s3_client()
    try:
        response = client.get_object(Bucket=bucket_name, Key=s3_key)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in {"NoSuchKey", "NotFound", "404"}:
            logger.warning(
                "S3 object not found", extra={"bucket": bucket_name, "key": s3_key}
            )
            raise S3DataNotFoundError(
                f"S3 object not found: s3://{bucket_name}/{s3_key}"
            ) from exc
        logger.exception(
            "Failed to fetch S3 object", extra={"bucket": bucket_name, "key": s3_key}
        )
        raise S3OperationError("Failed to fetch S3 object") from exc
    except BotoCoreError as exc:
        logger.exception(
            "Unexpected boto3 error while fetching S3 object",
            extra={"bucket": bucket_name, "key": s3_key},
        )
        raise S3OperationError(
            "Unexpected boto3 error while fetching S3 object"
        ) from exc

    with closing(response["Body"]) as body:
        payload = body.read().decode("utf-8")

    try:
        json.loads(payload)
    except json.JSONDecodeError as exc:
        logger.error(
            "S3 object does not contain valid JSON",
            extra={"bucket": bucket_name, "key": s3_key},
        )
        raise S3InvalidJSONError(
            f"S3 object is not valid JSON: s3://{bucket_name}/{s3_key}"
        ) from exc

    logger.debug("Fetched %s bytes from s3://%s/%s", len(payload), bucket_name, s3_key)
    return payload


def store_s3_data(s3_key: str, bucket_name: str, data: Any) -> bool:
    """Persist JSON ``data`` to ``s3://bucket_name/s3_key``.

    Returns ``True`` when the object is stored successfully; otherwise ``False``.
    """

    client = _get_s3_client()

    try:
        payload = _ensure_json_bytes(data)
    except S3InvalidJSONError as exc:
        logger.error(
            "Provided data could not be converted to JSON",
            extra={"bucket": bucket_name, "key": s3_key},
        )
        logger.debug("JSON conversion error", exc_info=exc)
        return False

    try:
        client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=payload,
            ContentType="application/json",
        )
        logger.debug("Stored %s bytes to s3://%s/%s", len(payload), bucket_name, s3_key)
        return True
    except ClientError as exc:
        logger.exception(
            "Failed to store data in S3", extra={"bucket": bucket_name, "key": s3_key}
        )
    except BotoCoreError as exc:
        logger.exception(
            "Unexpected boto3 error while storing data in S3",
            extra={"bucket": bucket_name, "key": s3_key},
        )

    return False


def check_s3_file_exists(s3_key: str, bucket_name: str) -> bool:
    """Return ``True`` when the S3 object exists without downloading it."""

    client = _get_s3_client()
    try:
        client.head_object(Bucket=bucket_name, Key=s3_key)
        logger.debug("S3 object exists: s3://%s/%s", bucket_name, s3_key)
        return True
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in {"NoSuchKey", "NotFound", "404"}:
            logger.info(
                "S3 object not found", extra={"bucket": bucket_name, "key": s3_key}
            )
            return False
        logger.exception(
            "Failed to check S3 object existence",
            extra={"bucket": bucket_name, "key": s3_key},
        )
        raise S3OperationError("Failed to check S3 object existence") from exc
    except BotoCoreError as exc:
        logger.exception(
            "Unexpected boto3 error while checking S3 object existence",
            extra={"bucket": bucket_name, "key": s3_key},
        )
        raise S3OperationError(
            "Unexpected boto3 error while checking S3 object existence"
        ) from exc


def get_batch_s3_key(
    batch_id: str, source_id: str, stage_name: str, file_name: str
) -> str:
    """Construct the S3 key for a given batch and source ID."""
    if file_name.lower() == "rfp_text":
        return f"{batch_id}/{source_id}/{file_name}.json"
    return f"{batch_id}/{source_id}/stage/{stage_name}/{file_name}.json"


__all__ = [
    "get_s3_data",
    "store_s3_data",
    "check_s3_file_exists",
    "get_batch_s3_key",
    "S3OperationError",
    "S3DataNotFoundError",
    "S3InvalidJSONError",
]
