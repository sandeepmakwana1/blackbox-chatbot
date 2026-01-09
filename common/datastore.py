import json
from contextlib import closing
from typing import Any, Optional
import os
import boto3
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError

from common.config import S3_KEY_PREFIX, S3_PATH_TEMPLATES
from common.constants import ContextType
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


def get_s3_data(s3_key: str, bucket_name: str, version_id: Optional[str] = None) -> str:
    """Return JSON payload stored at ``s3://bucket_name/s3_key`` as a string.
    Supports optional version_id to fetch specific object versions.
    """

    client = _get_s3_client()
    try:
        # Prepare arguments dynamically
        kwargs = {"Bucket": bucket_name, "Key": s3_key}
        if version_id:
            kwargs["VersionId"] = version_id

        response = client.get_object(**kwargs)

    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in {"NoSuchKey", "NotFound", "404"}:
            logger.warning(
                "S3 object not found", extra={"bucket": bucket_name, "key": s3_key}
            )
            raise S3DataNotFoundError(
                f"S3 object not found: s3://{bucket_name}/{s3_key} (Version: {version_id})"
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

    logger.debug(
        "Fetched %s bytes from s3://%s/%s (Version: %s)",
        len(payload),
        bucket_name,
        s3_key,
        version_id,
    )
    return payload


def destore_s3_data(s3_key: str, bucket_name: str) -> bool:
    """Persist JSON ``data`` to ``s3://bucket_name/s3_key``.

    Returns ``True`` when the object is stored successfully; otherwise ``False``.
    """
    client = _get_s3_client()

    try:
        client.delete_object(Bucket=bucket_name, Key=s3_key)
        logger.debug("Deleted s3://%s/%s", bucket_name, s3_key)
        return True
    except ClientError as exc:
        logger.exception(
            "Failed to delete data in S3", extra={"bucket": bucket_name, "key": s3_key}
        )
    except BotoCoreError as exc:
        logger.exception(
            "Unexpected boto3 error while deleting data in S3",
            extra={"bucket": bucket_name, "key": s3_key},
        )

    return False


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


def check_s3_prefix_exists(s3_prefix: str, bucket_name: str) -> bool:
    """
    Checks if any object exists under the given S3 prefix.
    This is required because head_object does not support wildcards (*).
    """

    client = _get_s3_client()
    try:
        # 1. Use list_objects_v2 with the prefix
        response = client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=s3_prefix,
            MaxKeys=1,  # Optimize: only ask for one key to confirm existence
        )

        # 2. Check the response to see if any content was found
        if response.get("KeyCount", 0) > 0:
            logger.debug(
                "S3 content exists under prefix: s3://%s/%s", bucket_name, s3_prefix
            )
            return True
        else:
            logger.info(
                "S3 content not found under prefix",
                extra={"bucket": bucket_name, "prefix": s3_prefix},
            )
            return False

    except BotoCoreError as exc:
        logger.exception(
            "Unexpected boto3 error while checking S3 prefix existence",
            extra={"bucket": bucket_name, "prefix": s3_prefix},
        )
        # Assuming S3OperationError is defined elsewhere
        raise S3OperationError(
            "Unexpected boto3 error while checking S3 prefix existence"
        ) from exc


def restore_s3_version(s3_key: str, bucket_name: str, version_id: str) -> bool:
    """
    Promotes a specific version to be the latest version.
    Returns True if a new version was created (restored), False if it was already latest.
    """
    client = _get_s3_client()
    try:
        # 1. Check the current latest version
        # head_object is fast and cheap; it tells us the current metadata
        head_response = client.head_object(Bucket=bucket_name, Key=s3_key)
        current_latest_id = head_response.get("VersionId")

        # 2. Compare: If it's already the latest, do nothing
        if current_latest_id == version_id:
            logger.info(
                "Requested version %s is already the latest for %s. No action taken.",
                version_id,
                s3_key,
            )
            return False

        # 3. Restore: Copy the old version to the "current" slot
        copy_source = {"Bucket": bucket_name, "Key": s3_key, "VersionId": version_id}

        # This creates a NEW version at the top of the stack with the old content
        client.copy_object(Bucket=bucket_name, Key=s3_key, CopySource=copy_source)
        logger.info(
            "Successfully restored version %s to top for s3://%s/%s",
            version_id,
            bucket_name,
            s3_key,
        )
        return True

    except ClientError as exc:
        logger.exception(
            "Failed to restore S3 version",
            extra={"bucket": bucket_name, "key": s3_key, "version_id": version_id},
        )
        raise S3OperationError("Failed to restore S3 version") from exc


class S3StoreService:
    """Service class to interact with S3 storage."""

    @classmethod
    def _get_s3_full_key(
        cls, source_id: str | int, stage_name: str, bucket_name: str
    ) -> Optional[str]:
        if not source_id:
            logger.error("Cannot store data to S3: source_id is missing.")
            return None
        if not stage_name:
            logger.error("Cannot store data to S3: stage_name is missing.")
            return None
        if not bucket_name or not bucket_name.strip():
            logger.error("Cannot store data to S3: bucket_name is missing.")
            return None

        template = S3_PATH_TEMPLATES.get(stage_name)
        if not template:
            logger.error(
                "No S3 path template configured for stage_name=%s; skipping store.",
                stage_name,
            )
            return None

        try:
            relative_key = template.format(source_id=str(source_id))
        except (KeyError, ValueError) as exc:
            logger.exception(
                "Failed to resolve S3 key template for stage_name=%s and source_id=%s",
                stage_name,
                source_id,
            )
            return None

        prefix = S3_KEY_PREFIX or ""
        if prefix and not prefix.endswith("/"):
            prefix = f"{prefix}/"

        full_s3_key = f"{prefix}{relative_key.lstrip('/')}"
        return full_s3_key

    @classmethod
    def create(
        cls, *, bucket_name: str, source_id: Any, stage_name: str, data: Any
    ) -> bool:
        """Store ``data`` to S3 using the logical key templates defined in configuration."""

        full_s3_key = cls._get_s3_full_key(source_id, stage_name, bucket_name)
        if full_s3_key is None:
            return False
        is_stored = store_s3_data(full_s3_key, bucket_name, data)
        if not is_stored:
            logger.info(
                "Failed to store data to S3: s3://%s/%s", bucket_name, full_s3_key
            )
            return False
        logger.info(
            "Successfully stored data to S3: s3://%s/%s", bucket_name, full_s3_key
        )
        return is_stored

    @classmethod
    def get(
        cls,
        *,
        bucket_name: str,
        source_id: Any,
        stage_name: str,
        version_id: Optional[str] = None,
    ) -> str:
        """Alias for get_s3_data to provide a simpler function name."""
        s3_key = cls._get_s3_full_key(source_id, stage_name, bucket_name)
        if s3_key is None:
            raise S3OperationError("Cannot determine S3 key to fetch data")
        data = get_s3_data(s3_key, bucket_name, version_id=version_id)
        if not data:
            raise S3DataNotFoundError(
                f"Data not found in S3: s3://{bucket_name}/{s3_key}"
            )
        logger.info(
            "Successfully fetched data from S3: s3://%s/%s", bucket_name, s3_key
        )
        return data

    @classmethod
    def is_exists(
        cls,
        *,
        bucket_name: str,
        source_id: Any,
        stage_name: str,
    ) -> bool:
        """Alias for check_s3_file_exists to provide a simpler function name."""
        s3_key = cls._get_s3_full_key(source_id, stage_name, bucket_name)
        if s3_key is None:
            raise S3OperationError("Cannot determine S3 key to check existence")

        return check_s3_file_exists(s3_key, bucket_name)

    @classmethod
    def get_bucket_name(cls) -> str:
        ssm_prefix = os.getenv("SSM_PREFIX", "blackbox-dev")
        bucket_name = f"{ssm_prefix}-batch-processing"
        return bucket_name

    @classmethod
    def delete(
        cls,
        *,
        bucket_name: str,
        source_id: Any,
        stage_name: str,
    ) -> bool:
        s3_key = cls._get_s3_full_key(source_id, stage_name, bucket_name)
        if s3_key is None:
            raise S3OperationError("Cannot determine S3 key to delete data")
        return destore_s3_data(s3_key, bucket_name)

    @classmethod
    def get_versions(
        cls,
        *,
        bucket_name: str,
        source_id: Any,
        stage_name: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Retrieves the latest 'limit' versions of the file associated with source_id and stage_name.
        """
        from operator import itemgetter  # Import here or at top of file

        # 1. Resolve the Key using existing logic
        s3_key = cls._get_s3_full_key(source_id, stage_name, bucket_name)
        if s3_key is None:
            raise S3OperationError("Cannot determine S3 key to fetch versions")

        client = _get_s3_client()

        try:
            # 2. List Versions
            response = client.list_object_versions(Bucket=bucket_name, Prefix=s3_key)

            if "Versions" not in response:
                return []

            # 3. Sort by Date (Newest First)
            all_versions = sorted(
                response["Versions"], key=itemgetter("LastModified"), reverse=True
            )

            # 4. Limit & Format
            results = []
            for index, v in enumerate(all_versions[:limit]):
                is_latest = index == 0
                results.append(
                    {
                        "version_id": v["VersionId"],
                        "last_modified": v["LastModified"].isoformat(),
                        "is_latest": is_latest,
                        # Adding raw timestamp for potential frontend sorting
                        "_timestamp": v["LastModified"].timestamp(),
                    }
                )

            return results

        except ClientError as exc:
            logger.exception(
                "Failed to list versions",
                extra={"bucket": bucket_name, "key": s3_key},
            )
            raise S3OperationError("Failed to list S3 object versions") from exc
        except Exception as exc:
            logger.exception(
                "Unexpected error listing versions",
                extra={"bucket": bucket_name, "key": s3_key},
            )
            raise S3OperationError("Unexpected error listing versions") from exc

    @classmethod
    def restore(
        cls, *, bucket_name: str, source_id: Any, stage_name: str, version_id: str
    ) -> bool:
        """Restores a previous version to be the current latest version."""
        s3_key = cls._get_s3_full_key(source_id, stage_name, bucket_name)
        if s3_key is None:
            raise S3OperationError("Cannot determine S3 key to restore data")

        return restore_s3_version(s3_key, bucket_name, version_id)


__all__ = [
    "S3OperationError",
    "S3DataNotFoundError",
    "S3InvalidJSONError",
    "S3StoreService",
]
